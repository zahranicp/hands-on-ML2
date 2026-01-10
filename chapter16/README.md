# Custom Keras Subclassed Model (Keras 3 Compatible)

## 📌 Deskripsi Proyek
Proyek ini merupakan implementasi **model Deep Learning berbasis Keras Subclassing API** yang menggunakan:
- **Custom Model (`tf.keras.Model`)**
- **Custom Layer (`tf.keras.layers.Layer`)**
- **Custom Loss Function**
- Format penyimpanan **`.keras` (Keras v3 standard)**

Proyek ini dirancang untuk:
- Memahami mekanisme internal Keras Subclassing
- Menghindari error umum saat `save()` dan `load_model()`
- Menjadi referensi **best practice Keras 3 (Python 3.12 compatible)**

---

## 🚀 Fitur Utama
- ✅ Custom Dense Layer (`MyDense`)
- ✅ Custom Model (`MyModel`)
- ✅ Custom Loss (`custom_mse`)
- ✅ Fully compatible dengan **Keras 3**
- ✅ Aman untuk **save & load model**
- ✅ Anti error:
  - `unexpected keyword argument 'trainable'`
  - `layer count mismatch`
  - `model expected 0 layers`

---

## 🧠 Arsitektur Model
Model menerima input berdimensi `INPUT_DIM` dan memiliki arsitektur:



Input (INPUT_DIM)
↓
MyDense(64, activation="relu")
↓
MyDense(1)
↓
Output (regresi)

```

Model ini digunakan untuk **task regresi sederhana** dengan target berupa jumlah dari fitur input.

---

## 🗂️ Struktur File
```

.
├── README.md
├── custom_model.keras
└── main.py / notebook.ipynb

````

---

## ⚙️ Requirements
Pastikan environment sudah memenuhi spesifikasi berikut:

- Python ≥ **3.10** (direkomendasikan 3.12)
- TensorFlow / Keras ≥ **2.13 / Keras 3**

Install dependensi:
```bash
pip install tensorflow keras numpy
````

---

## 📦 Custom Components

### 1️⃣ Custom Loss

```python
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
```

---

### 2️⃣ Custom Layer

`MyDense` adalah implementasi manual dari Dense layer dengan:

* Weight
* Bias
* Aktivasi opsional

Layer ini mendukung serialisasi melalui `get_config()`.

---

### 3️⃣ Custom Model

`MyModel` merupakan subclass dari `tf.keras.Model` dengan:

* Dukungan penuh `**kwargs`
* Kompatibel dengan `save()` dan `load_model()`
* Aman untuk Keras 3

---

## 🧪 Cara Menjalankan

### 1️⃣ Training & Save Model

```python
model = MyModel()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss=custom_mse,
    metrics=["mae"]
)

# WAJIB: build model
model(tf.zeros((1, INPUT_DIM)))

model.fit(X, y, epochs=3)
model.save("custom_model.keras")
```

---

### 2️⃣ Load Model

```python
loaded_model = tf.keras.models.load_model(
    "custom_model.keras",
    custom_objects={
        "MyModel": MyModel,
        "MyDense": MyDense,
        "custom_mse": custom_mse
    }
)
```

---

### 3️⃣ Inference

```python
pred = loaded_model.predict(X[:5])
print(pred)
```

---

## ⚠️ Catatan Penting (Wajib Dibaca)

### 🔥 Kenapa pakai `.keras` dan bukan `.h5`?

* `.h5` adalah **format legacy**
* Subclassed model **rawan error** jika disimpan sebagai `.h5`
* `.keras` adalah **format resmi Keras 3**

### 🔥 Kenapa model harus dipanggil sebelum save?

```python
model(tf.zeros((1, INPUT_DIM)))
```

Tanpa ini:

* Model dianggap **belum punya layer**
* Akan muncul error saat `load_model()`

---

## ✅ Best Practice yang Digunakan

* ✔ `**kwargs` pada semua custom class
* ✔ `super().__init__(**kwargs)`
* ✔ `get_config()` untuk serialisasi
* ✔ Format `.keras`
* ✔ Build model sebelum save

---

## 🎓 Kegunaan Akademik

Proyek ini cocok untuk:

* Tugas Deep Learning
* UAS Machine Learning
* Studi Keras Subclassing
* Referensi skripsi / riset awal
* Portofolio GitHub

---

## 👤 Author
Zahrani Cahya Priesa
1103223074
TK4603
Disusun untuk keperluan pembelajaran dan akademik.

---

## 📄 Lisensi

Proyek ini bersifat **open-use untuk edukasi**.
Silakan dimodifikasi sesuai kebutuhan.

---

✨ **Happy Coding & Semoga Lancar UAS-nya!** ✨

```

# Brainstorming Fase 2: Perutean Aktif, Circuit Breaker, dan Integrasi LiteLLM

**Evaluator / Desainer:** Claude Opus 4 (Antigravity)  
**Tanggal:** 29 Mei 2026  
**Target Proyek:** `hermes_agent` (Fase 2)  

---

## 1. Analisis & Visi Teknis Fase 2

Fase 2 bertujuan untuk menaikkan kelas `provider_gateway/` dari sistem pemantauan pasif (observabilitas) menjadi **sistem orkestrasi perutean aktif (active smart routing)**.  
Tiga pilar utama yang akan kita bahas dalam brainstorming ini adalah:
1. **Circuit Breaker (Resilience):** Menghentikan panggilan ke provider yang sering gagal secara otomatis untuk menghindari penundaan yang lama bagi pengguna.
2. **Dynamic Routing Engine (Smart Routing):** Memilih provider terbaik berdasarkan strategi (*lowest-cost*, *lowest-latency*, *round-robin*) dengan menyaring kandidat sehat dari Circuit Breaker.
3. **LiteLLM Integration (Optional Backend):** Menyediakan jembatan opsional untuk mendukung 100+ model/provider dengan upaya minimal.

---

## 2. Desain Komponen & Arsitektur

### 2.1 Circuit Breaker (`provider_gateway/circuit_breaker.py`)
Mekanisme ini harus bersifat thread-safe dan memantau kesehatan provider secara real-time.

*   **Pernyataan Status (States):**
    *   `CLOSED`: Status normal. Request diteruskan ke provider.
    *   `OPEN`: Provider terdeteksi rusak karena kegagalan beruntun melebihi batas (`failure_threshold`, default: 5). Request langsung diblokir (*fast-fail*) tanpa memanggil API untuk menghindari timeout panjang.
    *   `HALF_OPEN`: Masa uji coba. Setelah cooldown selesai (`reset_timeout_ms`, default: 60 detik), sirkuit mengizinkan satu request percobaan. Jika sukses, kembali ke `CLOSED`; jika gagal, kembali ke `OPEN` dengan pengali backoff eksponensial.
*   **Keamanan Thread:** Menggunakan `threading.Lock` karena request LLM dapat berjalan di thread latar belakang (misalnya subagent atau tugas asinkron).
*   **Struktur Data:**
    *   `CircuitState` (Enum): `CLOSED`, `OPEN`, `HALF_OPEN`.
    *   `ProviderHealth` (Dataclass): Menyimpan metrik latensi P50, total kegagalan, kegagalan beruntun, tingkat kesalahan, dan tingkat backoff.
    *   `CircuitBreaker` (Class): Mengelola state map `dict[str, ProviderHealth]` dan metode inti: `is_available()`, `record_success()`, dan `record_failure()`.

### 2.2 Routing Engine (`provider_gateway/router.py`)
Mekanisme untuk memilih provider terbaik secara cerdas dari daftar kandidat yang tersedia.

*   **Integrasi Kebijakan (`policy.py`):**
    Kita akan menyatukan daftar kandidat rute yang dihasilkan oleh `build_gateway_policy(agent)` (yang berisi model gateway + fallback chain asli milik agent).
*   **Penyaringan Kesehatan:** Hanya memilih kandidat yang saat ini sehat menurut `CircuitBreaker`. Jika semua sirkuit berstatus `OPEN`, kita gunakan perutean darurat (kembali ke kandidat utama).
*   **Strategi Pemilihan (Routing Strategies):**
    *   `round-robin` (Default): Mengambil kandidat sehat berikutnya dalam barisan setelah provider aktif saat ini.
    *   `lowest-cost`: Memilih provider dengan taksiran biaya per 1 juta token terkecil. Kita bisa memprioritaskan model gratis (cost = 0) atau model dengan estimasi harga termurah.
    *   `lowest-latency`: Memilih provider dengan latensi rata-rata historis (P50) terkecil dari sampel Circuit Breaker.

### 2.3 LiteLLM Backend (`provider_gateway/litellm_backend.py`)
LiteLLM bertindak sebagai dependensi opsional.
*   **Opt-in Safe:** Jika `litellm` belum diinstal oleh pengguna, sistem tidak boleh crash. Kita menggunakan pengaman `ImportError` di tingkat module level.
*   **Fungsi Utama:** Menyediakan metode `complete()`, `estimate_cost()`, dan `list_models()` yang memanfaatkan pustaka `litellm` secara aman jika gateway diaktifkan dengan konfigurasi `backend: litellm`.

---

## 3. Strategi Integrasi Tanpa Merusak Codebase Asli (Zero Regression)

Ini adalah tantangan paling kritis. Hermes memiliki penanganan loop agent yang kompleks di `agent/conversation_loop.py` dan `run_agent.py`. Kita tidak boleh menulis ulang seluruh sistem transportasi.

*   **Seam Integrasi Terbaik: `try_activate_fallback()` di `agent/chat_completion_helpers.py`**
    Saat ini, fungsi ini memilih fallback berikutnya secara linear kaku:
    ```python
    fb = agent._fallback_chain[agent._fallback_index]
    agent._fallback_index += 1
    ```
    Kita akan menyuntikkan logika pintar di sini:
    Jika `provider_gateway.enabled` bernilai `True`:
    1. Ambil seluruh kebijakan rute menggunakan `build_gateway_policy(agent)`.
    2. Jalankan perutean dinamis via `ProviderRouter` untuk menyaring kandidat sehat dan memilih rute terbaik berdasarkan strategi (`routing_strategy`).
    3. Setelah kandidat terpilih (misalnya `ProviderRouteCandidate`), kita mutasikan state agent secara in-place (`agent.model = candidate.model`, `agent.provider = candidate.provider`, `agent.base_url = candidate.base_url`) dan bangun ulang koneksi client persis seperti perilaku fallback bawaan Hermes.
    4. Dengan pendekatan ini, kita **tidak mengubah loop transport internal**, melainkan hanya **meningkatkan kecerdasan pemilihan fallback-nya**! Ini sangat aman dan 100% backward compatible.

---

## 4. Rencana Implementasi Bertahap

Untuk menjamin kualitas bintang 10, kita akan mengimplementasikannya dalam langkah-langkah terstruktur berikut:

### Langkah 2.1: Buat Modul Circuit Breaker (`provider_gateway/circuit_breaker.py`)
- Implementasikan kelas `CircuitBreaker` thread-safe.
- Tambahkan properti perhitungan P50 latency dan error rate.
- Lengkapi dengan unit test khusus di `tests/provider_gateway/test_circuit_breaker.py`.

### Langkah 2.2: Buat Modul Routing Engine (`provider_gateway/router.py`)
- Buat kelas `ProviderRouter` yang menerima instance `CircuitBreaker`.
- Implementasikan metode `select_route` berdasarkan strategi `round-robin`, `lowest-cost`, dan `lowest-latency`.
- Lengkapi dengan unit test khusus di `tests/provider_gateway/test_router.py`.

### Langkah 2.3: Hubungkan Runtime Observability & Circuit Breaker
- Di dalam `provider_gateway/runtime.py`:
  - Saat `record_provider_response_usage` dipanggil (sukses), catat kesuksesan ke `CircuitBreaker` beserta latensinya.
  - Saat `record_provider_error_usage` dipanggil (gagal), catat kegagalan ke `CircuitBreaker`.
- Ini menjamin bahwa Circuit Breaker selalu mendapatkan data kesehatan terbaru secara otomatis dari aktivitas API normal!

### Langkah 2.4: Integrasikan dengan `try_activate_fallback` di `agent/chat_completion_helpers.py`
- Modifikasi `try_activate_fallback` agar mendeteksi konfigurasi gateway aktif.
- Panggil `ProviderRouter` untuk menentukan provider alternatif yang sehat alih-alih melompat secara linear.
- Tambahkan unit test komprehensif untuk memverifikasi perutean aktif.

### Langkah 2.5: Buat Integrasi Opsional LiteLLM Backend
- Daftarkan dependensi opsional di `pyproject.toml`.
- Buat modul `provider_gateway/litellm_backend.py`.
- Tambahkan unit test untuk memverifikasi fungsionalitas mock LiteLLM.

---

## 5. Rencana Pengujian & Verifikasi

Kita akan menulis setidaknya 15+ test baru khusus untuk Fase 2 guna menjamin ketahanan sistem:
- **Test Circuit Breaker:** Uji transisi status `CLOSED` -> `OPEN` setelah 5 kegagalan, uji pemblokiran cepat (*fast-fail*), uji transisi ke `HALF_OPEN` setelah timeout cooldown, uji recovery ke `CLOSED` saat request sukses, serta uji konkurensi multithreading.
- **Test Router:** Uji pemilihan model termurah (*lowest-cost*), uji pemilihan model dengan latensi terendah (*lowest-latency*), uji penanganan ketika semua model mati (fallback darurat).
- **Test Integrasi Fallback:** Uji bahwa loop failover di `try_activate_fallback` benar-benar beralih sesuai instruksi router dinamis dan circuit breaker.
- **Test LiteLLM:** Uji aman dari crash jika pustaka `litellm` tidak diinstal, dan uji mock fungsionalitas penaksiran harga serta pemanggilan API-nya.

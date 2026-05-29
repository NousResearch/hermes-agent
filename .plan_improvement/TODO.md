# TODO: Rencana Penyempurnaan Provider Gateway (Multi-Provider) Hermes Agent

> **Status Terakhir:**
> - **Fondasi Observabilitas:** Selesai 100% (Production-Grade SQLite, WAL Mode, Schema Versioning, Time-Window Query).
> - **Perutean Aktif & CB:** Selesai 100% (Circuit Breaker, Routing Engine, Weighted Scoring, LiteLLM Backend).
> - **Optimasi & Quota:** Selesai 100% (Streaming usage, Semantic Cache, Quota Guard).
> - **Keamanan, Guardrails & Local Server:** Selesai 100% (AES-256-GCM Secure Store, Ollama Discovery, PII Sanitizer & Guardrails, OpenAI Local API Server).
> - **Test Suite:** 99+ skenario uji lulus 100% secara komprehensif ✅.
> - **Desain:** Default-off tetap terjaga penuh. Blast radius runtime terkontrol (tidak ada breaking change).

---

## PETA JALAN & DAFTAR TUGAS (TODO LIST)

### 📌 FASE 1: Fondasi Observabilitas & Pelacakan Penggunaan
*Fase ini meletakkan fondasi penyimpanan, pelaporan, versi skema, dan mitigasi konkurensi database.*

- [x] **Infrastruktur SQLite Aman Konkurensi:**
  - [x] Implementasi helper `_connect()` terpusat dengan mode WAL (*Write-Ahead Logging*).
  - [x] Mengatur `synchronous = NORMAL` untuk kinerja tulis cepat dan aman.
  - [x] Mengatur `busy_timeout = 5000` (5 detik) untuk mencegah kegagalan `SQLITE_BUSY` saat penulisan bersamaan.
- [x] **Schema Versioning & Perlindungan Migrasi Masa Depan:**
  - [x] Menambahkan versi skema tingkat modul `SCHEMA_VERSION = 1` di `usage_tracker.py` and mengekspornya di `__init__.py`.
  - [x] Membuat tabel `provider_usage_schema_version` untuk mencatat tanggal penerapan skema secara idempotent.
  - [x] Verifikasi fungsionalitas re-open database tidak melipatgandakan catatan versi.
- [x] **Time-Window Query & Parameterisasi:**
  - [x] Menambahkan parameter opsional `since` and `until` (Unix timestamp) pada method `summarize_by_provider()`.
  - [x] Query aman dari SQL Injection menggunakan parameterisasi SQL SQLite standard.
- [x] **Cost Tracking Pipeline & Normalisasi:**
  - [x] Menambahkan integrasi pengujian fungsional penaksiran biaya token.
  - [x] Memperbaiki bug parsing input/output token untuk model non-OpenAI (seperti Anthropic `input_tokens`/`output_tokens`).
- [x] **Pengujian Komprehensif (Ekspansi Pengujian):**
  - [x] Meningkatkan cakupan unit test dari 24 menjadi 55 test di seluruh 5 file pengujian (`test_config`, `test_usage_tracker`, `test_runtime`, `test_policy`, `test_status`).
- [x] **CLI Visibility Surface (`/usage`):**
  - [x] Hook status diletakkan pada runtime CLI di `cli.py` agar ringkasan statistik penggunaan dapat diakses tanpa mengaktifkan perutean aktif.

---

### 📌 FASE 2: Perutean Aktif (Active Routing Engine) & Ketahanan (Resilience)
*Tujuan: Mengaktifkan pengalihan otomatis ke provider alternatif dan memantau kesehatan provider secara real-time.*

- [x] **Circuit Breaker Multi-Provider (`circuit_breaker.py`):**
  - [x] Implementasikan state machine thread-safe: `CLOSED`, `OPEN`, `HALF_OPEN`.
  - [x] Mencatat kegagalan berturut-turut (*consecutive failures*) per provider sebelum memblokir request.
  - [x] Mendukung waktu cooldown eksponensial (*exponential backoff cooldown*) sebelum mencoba status `HALF_OPEN`.
  - [x] Buat unit test isolasi untuk skenario kegagalan, sukses, pemulihan, dan multi-threading.
- [x] **Routing Engine & Weighted Scoring (`router.py`):**
  - [x] Membuat algoritma pemilihan rute dinamis berbasis strategi: `round-robin` (perputaran), `lowest-cost` (biaya terendah), dan `lowest-latency` (latensi P50 tercepat).
- [x] **Integrasi dengan Runtime Agent (`agent/chat_completion_helpers.py`):**
  - [x] Hubungkan `ProviderRouter` dengan loop failover Hermes di `try_activate_fallback()` secara modular.
  - [x] Desain terisolasi dan *failsafe* sehingga tetap aman tanpa regression blast radius pada runtime bawaan.
- [x] **Penyedia LiteLLM (Opt-in Multi-Provider Backend):**
  - [x] Tambahkan wrapper adapter di `provider_gateway/litellm_backend.py` secara *import-safe* untuk mendukung backend LiteLLM opsional.

---

### 📌 FASE 3: Optimasi Pesan, Cache Semantik, & Quota Guard (Anggaran Cerdas)
*Tujuan: Menghemat token, biaya, mempercepat respon via cache lokal, dan menegakkan batas anggaran.*

- [x] **Pelacakan Penggunaan untuk Mode Streaming:**
  - [x] Menambahkan penyadapan (*interception*) usage token dan error pada respon streaming (`chat_completions` stream) di `interruptible_streaming_api_call` tanpa overhead koneksi.
- [x] **Semantic Cache Engine (`semantic_cache.py`):**
  - [x] Menyediakan in-memory & SQLite cache berbasis hash SHA-256 riwayat chat secara thread-safe (WAL mode).
  - [x] Menghindari pengiriman request LLM duplikat untuk menghemat biaya operasional secara instan dengan latensi **< 5ms** dan biaya **$0.0** USD.
  - [x] Integrasi preflight & store sukses untuk mode streaming (dengan delta playback) dan non-streaming.
- [x] **Pelacakan Batas Kuota & Quota Guard (`quota_manager.py`):**
  - [x] Menambahkan perhitungan agregat cepat pengeluaran USD harian dan bulanan secara periodik di database SQLite `provider_usage`.
  - [x] Mendukung batas anggaran harian/bulanan di `GatewayConfig` dengan tindakan `block` (raises `QuotaExceededError`) atau `fallback` (otomatis mengalihkan ke model Ollama lokal bebas biaya).

---

### 📌 FASE 4: Ekosistem, Guardrails, & Keamanan
*Tujuan: Menjamin keamanan kredensial dan memperluas dukungan ke edge deployment.*

- [x] **Secure Credential Store:**
  - [x] Jangan gunakan pengodean base64 biasa untuk menyimpan API key tambahan.
  - [x] Hubungkan dengan pustaka `keyring` bawaan OS, interop WSL PowerShell ke Windows Vault Host, atau gunakan enkripsi AES lokal dengan kunci rahasia yang terikat fisik mesin (machine-bound).
- [x] **Dukungan Provider Lokal (Ollama / Local Model Integration):**
  - [x] Mempermudah auto-discovery model Ollama lokal yang berjalan secara default sebagai fallback bebas biaya dan deteksi konteks window (`num_ctx`).
- [x] **Penyaringan Konten & PII Sanitizer (Guardrails):**
  - [x] Deteksi otomatis dan anonimisasi data sensitif (seperti email, IP, API keys, private keys) sebelum dikirim ke server cloud pihak ketiga dengan sliding buffer de-anonimisasi real-time.
- [x] **OpenAI-Compatible Local Endpoint:**
  - [x] Menyediakan local API server di dalam Hermes agar aplikasi atau CLI pihak ketiga lainnya dapat menggunakan sistem perutean multi-provider pintar Hermes.

---

## 🛠️ INSTRUKSI BAGI TIM / AGENT BERIKUTNYA

Seluruh fase telah diselesaikan secara penuh dengan cakupan pengujian komprehensif (99+ unit test lulus 100%). Untuk memverifikasi seluruh komponen gateway, Anda dapat menjalankan pytest:
```bash
uv run --extra dev python -m pytest tests/provider_gateway -q
```
Sistem opt-in provider gateway ini sekarang siap digunakan sepenuhnya di lingkungan produksi secara aman dan efisien!

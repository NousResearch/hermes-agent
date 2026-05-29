# Brainstorming Fase 3: Pelacakan Streaming, Semantic Cache, dan Quota Tracking

**Evaluator / Desainer:** Claude Opus 4 (Antigravity)  
**Tanggal:** 29 Mei 2026  
**Target Proyek:** `hermes_agent` (Fase 3)  

---

## 1. Visi Teknis Fase 3: Efisiensi & Kontrol Anggaran

Setelah Fase 2 sukses memberikan ketahanan perutean aktif yang aman konkurensi (WAL & 73 unit test lulus), Fase 3 akan melangkah maju untuk membawa **efisiensi biaya operasional** dan **kontrol anggaran cerdas** bagi pengguna Hermes Agent. 

Tiga fokus utama Fase 3:
1.  **Pelacakan Penggunaan untuk Streaming (Streaming Usage Tracking):** Menyadap metrik usage token dan biaya dari respon streaming OpenAI-compatible & Anthropic secara andal tanpa mengganggu rendering TUI/CLI.
2.  **Semantic Cache Engine:** Menyimpan histori respon sukses ke database lokal SQLite dan menggunakan pencarian semantik (atau kecocokan hash pesan eksak) untuk menghindari pemanggilan LLM yang berulang untuk pertanyaan yang sama.
3.  **Quota Tracking (Pengaman Anggaran):** Memantau pengeluaran biaya total (dalam USD atau token) per hari/bulan secara dinamis dan memblokir request atau beralih ke provider lokal/gratis (seperti Ollama) jika batas kuota terlampaui.

---

## 2. Desain Arsitektur Komponen Fase 3

### 2.1 Pelacakan Streaming (Streaming Tracking Hook)
Di dalam `interruptible_streaming_api_call` pada `agent/chat_completion_helpers.py`, sistem mengembalikan objek `SimpleNamespace` tiruan (baik `result["response"]` sukses penuh, atau stub parsial) setelah loop stream selesai dikonsumsi.
*   **Taktik Penyadapan:**
    Karena `_call_chat_completions()` secara bawaan mengumpulkan metadata token penggunaan (`usage`) yang dikirim di chunk stream akhir ke dalam properti `usage_obj` dan mengembalikannya di dalam objek respon, kita dapat menyadapnya secara langsung:
    ```python
    if result["response"] is not None:
        try:
            from provider_gateway.runtime import record_provider_response_usage
            record_provider_response_usage(
                agent,
                result["response"],
                latency_seconds=time.time() - api_start_time,
            )
        except Exception:
            pass
    ```
    Ini adalah pendekatan yang sangat bersih karena **tidak memerlukan parser stream baru**, melainkan memanfaatkan akumulator stream bawaan Hermes yang sudah teruji.

### 2.2 Semantic Cache Engine (`provider_gateway/semantic_cache.py`)
Mekanisme ini menyimpan riwayat input pesan dan output respon.

*   **Penyimpanan SQLite Lokal:**
    Kita akan memanfaatkan database lokal `provider_usage.db` dengan membuat tabel baru:
    ```sql
    CREATE TABLE IF NOT EXISTS semantic_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_hash TEXT UNIQUE,
        prompt_text TEXT,
        response_text TEXT,
        model TEXT,
        provider TEXT,
        created_at REAL
    );
    ```
*   **Strategi Kecocokan (Matching Strategy):**
    *   **Kecocokan Eksak (SHA-256 Hash):** Kami meng-hash gabungan string terkompresi dari riwayat pesan chat yang dikirim. Jika hash cocok di database, kembalikan respon instan dalam waktu < 5ms tanpa biaya API (0 token cost).
    *   **Kecocokan Semantik Ringan (Jaccard / Ringan):** Jika diaktifkan, hitung kesamaan kata atau gunakan kemiripan ringan jika model embedding lokal belum terinstal, guna mencegah false hit.

### 2.3 Pelacakan Batas Pengeluaran / Quota Guard (`provider_gateway/quota_manager.py`)
Mencegah agen menghabiskan saldo API di luar kendali pengguna saat loop alat (tool loop) berjalan berulang kali.

*   **Desain Kontrol Kuota:**
    *   Pengaturan kuota baru di `cli-config.yaml`:
        ```yaml
        provider_gateway:
          quota:
            daily_limit_usd: 2.00     # Batas pengeluaran maksimal per hari
            monthly_limit_usd: 30.00  # Batas pengeluaran maksimal per bulan
            on_limit_reached: fallback_to_free # fallback_to_free | block
        ```
    *   **Preflight Check:** Sebelum memanggil API (di awal `interruptible_api_call` dan `interruptible_streaming_api_call`), jalankan query agregasi SQLite:
        ```sql
        SELECT SUM(cost_usd) FROM usage_history WHERE timestamp >= date('now', 'start of day')
        ```
    *   Jika pengeluaran melebihi `daily_limit_usd`, sistem akan:
        *   Memicu fallback otomatis ke model bebas biaya (seperti Ollama lokal atau openrouter free models jika dikonfigurasi).
        *   Atau memblokir request secara total dan menampilkan peringatan yang ramah bagi pengguna di konsol.

---

## 3. Rencana Eksekusi Langkah Demi Langkah (Fase 3)

1.  **Langkah 3.1: Pelacakan Streaming yang Terintegrasi**
    *   Suntikkan pemanggilan `record_provider_response_usage` dan `record_provider_error_usage` di akhir `interruptible_streaming_api_call` pada `agent/chat_completion_helpers.py`.
    *   Tambahkan unit test integrasi streaming di `tests/provider_gateway/test_stream_tracking.py`.
2.  **Langkah 3.2: Implementasikan Semantic Cache Engine**
    *   Buat modul `provider_gateway/semantic_cache.py` dengan fungsionalitas CRUD database lokal `semantic_cache`.
    *   Gunakan sha256 hashing dari pesan riwayat JSON untuk determinisme yang bit-perfect.
    *   Sambungkan ke preflight API call di `interruptible_api_call` dan `interruptible_streaming_api_call` agar cache-hit langsung mengembalikan respon instan.
    *   Tulis unit test menyeluruh di `tests/provider_gateway/test_semantic_cache.py`.
3.  **Langkah 3.3: Implementasikan Quota Manager & Guard**
    *   Buat modul `provider_gateway/quota_manager.py` yang bertugas menghitung total pengeluaran harian/bulanan dari DB SQLite.
    *   Sambungkan ke preflight API call. Lemparkan error kuota atau pemicu fallback darurat jika terlampaui.
    *   Tulis unit test menyeluruh di `tests/provider_gateway/test_quota_manager.py`.

---

## 4. Hasil Brainstorming yang Diyakini
Semua integrasi ini tetap melestarikan sifat **opt-in**.
*   Jika `provider_gateway.enabled` bernilai `False`, seluruh pemantauan streaming, cache semantik, dan kuota tidak akan membebani sistem (nol overhead).
*   Jika diaktifkan, sistem akan menghemat biaya token pengguna secara masif dan memberikan rasa aman finansial saat agen bekerja secara mandiri dalam loop pengerjaan kode panjang.

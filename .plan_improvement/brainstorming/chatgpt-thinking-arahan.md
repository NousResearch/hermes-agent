# Arahan Agent Coding untuk Improve `hermes_agent`

## Tujuan Dokumen

Dokumen ini menjadi panduan kerja untuk agent coding yang akan menyempurnakan `hermes_agent` menjadi **AI agent platform yang lebih kuat untuk multi-provider routing, fallback, kompatibilitas API, kontrol biaya, dan stabilitas operasional**.

Fokus utama bukan hanya menambah jumlah model, tetapi membangun **lapisan orkestrasi model** yang membuat Hermes lebih andal saat dipakai untuk coding workflow harian, baik oleh pengguna individu maupun tim kecil.

---

## Visi Produk

`hermes_agent` harus berkembang dari agent CLI yang fleksibel menjadi **AI control plane** yang:

* bisa mengakses banyak provider dan banyak model lewat satu antarmuka,
* memilih model terbaik secara otomatis berdasarkan tugas, biaya, dan ketersediaan,
* berpindah provider tanpa memutus workflow saat terjadi rate limit, error, atau quota habis,
* memberi transparansi penuh atas token usage, latensi, biaya, dan kesehatan provider,
* tetap mudah dipakai dari CLI, editor, dan tool pihak ketiga.

---

## Prinsip Desain

1. **Stabilitas di atas variasi**
   Lebih baik satu jalur yang stabil daripada banyak provider yang sering gagal.

2. **Model selection harus data-driven**
   Jangan memilih model hanya berdasarkan nama populer. Pilih berdasarkan kemampuan, konteks, biaya, latensi, dan status kesehatan.

3. **Fallback harus otomatis dan aman**
   Pengguna tidak boleh kehilangan sesi kerja hanya karena satu provider down.

4. **Format input/output harus dinormalisasi**
   Semua provider harus lewat adapter yang konsisten agar mudah ditambah dan diuji.

5. **Observability wajib ada**
   Kalau routing gagal, harus jelas gagal karena apa, di mana, dan provider mana yang dipilih.

6. **Konfigurasi harus deklaratif**
   Hindari logic tersebar. Sumber kebenaran harus jelas: config file, registry model, dan policy router.

---

## Target Fitur Utama

### 1) Multi-Provider Model Registry

Bangun registry terpusat untuk semua provider dan model yang didukung.

**Isi registry per model:**

* provider name
* model id
* kategori kemampuan: chat, code, reasoning, vision, audio, embeddings, tool-use
* context window
* streaming support
* tool/function calling support
* multimodal support
* harga / estimasi biaya
* batas rate limit atau kuota jika diketahui
* status kesehatan terakhir
* prioritas fallback

**Tujuan:**

* auto-discovery model,
* filtering model berdasarkan kebutuhan task,
* memudahkan UI/CLI menampilkan daftar model yang relevan,
* memudahkan router memilih alternatif saat model utama gagal.

---

### 2) Routing Policy Engine

Buat engine routing yang bisa mengambil keputusan berdasarkan policy.

**Contoh policy:**

* cheapest-first
* fastest-first
* highest-quality-first
* free-first
* provider-preferred
* privacy-first / local-first
* vision-capable-only
* tool-use-capable-only
* long-context-only

**Requirement:**

* policy dapat diatur per project, per session, atau per request,
* policy dapat di-chain dengan fallback ladder,
* keputusan router harus bisa dijelaskan dalam log.

---

### 3) Auto-Fallback Multi-Level

Jika provider/model gagal, Hermes harus berpindah secara otomatis.

**Trigger fallback:**

* HTTP 429 / rate limit
* HTTP 5xx / upstream error
* auth expired
* timeout
* quota habis
* model tidak tersedia
* unsupported capability untuk task tertentu

**Fallback ladder:**

* model utama
* model cadangan provider yang sama
* provider lain dengan kemampuan serupa
* model murah/free tier
* local model terakhir

**Requirement:**

* fallback tidak mengulang request secara membabi buta,
* retry harus punya backoff,
* setiap fallback harus tercatat.

---

### 4) Unified Adapter Layer

Semua provider harus memiliki adapter yang konsisten.

**Adapter perlu menangani:**

* auth
* request shaping
* response normalization
* error normalization
* streaming normalization
* tool call normalization
* image/file attachment normalization

**Tujuan:**

* menambah provider baru tanpa menyentuh core logic,
* meminimalkan bug karena format API berbeda,
* memudahkan test lintas provider.

---

### 5) Quota, Spend, dan Health Dashboard

Sediakan tampilan atau endpoint yang menunjukkan kondisi operasional.

**Minimal metrics:**

* token masuk/keluar
* request count
* success rate
* error rate
* latency p50/p95
* spend estimasi
* sisa quota / status login
* provider health status
* fallback count
* model usage per session

**Tujuan:**

* pengguna tahu model mana yang hemat dan stabil,
* troubleshooting lebih cepat,
* memudahkan optimasi biaya.

---

### 6) Compression / Token Saver Layer

Tambahkan layer kompresi konteks untuk output tool dan input panjang.

**Sasaran kompresi:**

* diff git
* hasil grep/ripgrep
* struktur folder besar
* log panjang
* hasil web search berukuran besar
* file teks yang redundan

**Aturan:**

* jangan kompres sampai menghilangkan detail penting,
* pertahankan simbol, error message, nama file, dan patch kritikal,
* sediakan mode kompresi yang bisa dimatikan.

---

### 7) Compatibility Gateway / OpenAI-Compatible Endpoint

Hermes sebaiknya menyediakan endpoint yang kompatibel dengan tool lain.

**Gunanya:**

* Codex-like clients,
* editor integrations,
* agent frameworks,
* external automation tools.

**Target:**

* satu endpoint bisa dipakai banyak client,
* model/provider mapping dapat diubah tanpa mengubah client,
* dukungan streaming dan tool-use tetap konsisten.

---

### 8) Credential Management dan Multi-Account Support

Buat sistem yang aman untuk banyak akun/provider.

**Requirement:**

* simpan credential dengan aman,
* dukung beberapa akun per provider,
* support rotasi token / OAuth refresh,
* pilih akun otomatis berdasarkan policy,
* hindari kebocoran secret ke log.

---

### 9) Guardrails dan Safety Controls

Karena Hermes akan jadi layer orkestrasi, guardrail harus ada di router, bukan hanya di model.

**Contoh guardrail:**

* deny model tertentu untuk task tertentu,
* blok provider yang tidak lolos compliance,
* batasi data sensitif keluar ke provider eksternal,
* tandai request berisiko tinggi untuk review manual.

---

### 10) Extensibility untuk Tooling Agent

Hermes harus mudah ditambah tool baru.

**Contoh area ekstensi:**

* browser bridge
* web search
* file system tools
* code execution sandbox
* repo inspection
* issue/PR automation
* MCP-style connectors

**Requirement:**

* tool registry jelas,
* schema input/output terdokumentasi,
* error handling seragam,
* tool permission dapat diatur.

---

## Prioritas Implementasi

### Fase 1 — Fondasi

Fokus pada hal yang paling meningkatkan stabilitas:

* model registry
* adapter layer
* fallback engine
* error normalization
* logging dan metrics dasar

### Fase 2 — Routing Cerdas

Setelah fondasi stabil:

* policy engine
* capability-aware model selection
* quota-aware routing
* spend estimation
* dashboard/status page

### Fase 3 — Efisiensi dan Skalabilitas

Berikutnya:

* compression/token saver
* multi-account management
* compatibility endpoint
* guardrails
* advanced observability

### Fase 4 — Ekosistem

Terakhir:

* plugin/tool ecosystem
* MCP/A2A-style integration
* workspace profiles
* team/admin controls

---

## Definition of Done per Fitur

Setiap fitur dianggap selesai hanya jika memenuhi hal berikut:

* bisa dipakai end-to-end,
* ada test yang relevan,
* ada dokumentasi singkat,
* ada contoh konfigurasi,
* ada error message yang jelas,
* tidak merusak workflow lama,
* lolos lint dan type check bila ada,
* ada fallback atau graceful degradation.

---

## Kriteria Kualitas Kode

Agent coding harus menjaga standar berikut:

* perubahan kecil, modular, dan mudah di-review,
* hindari refactor besar tanpa alasan kuat,
* setiap adapter/provider harus punya interface yang sama,
* hindari hardcode yang sulit dirawat,
* pisahkan logic routing, auth, telemetry, dan UI,
* jangan menambah dependency tanpa manfaat jelas,
* prioritaskan backward compatibility.

---

## Pola Implementasi yang Disarankan

### Struktur Modul

* `core/`

  * session management
  * message pipeline
  * provider abstraction
  * routing policy
* `providers/`

  * adapter per vendor
* `routing/`

  * selection logic
  * fallback logic
  * capability scoring
* `telemetry/`

  * logs
  * metrics
  * tracing
* `config/`

  * schema
  * validation
  * defaults
* `ui/` atau `cli/`

  * status
  * model picker
  * diagnostics

### Alur Request Ideal

1. user membuat request,
2. system membaca policy dan capability requirements,
3. router memilih kandidat model,
4. adapter membentuk request sesuai provider,
5. request dikirim,
6. respons dinormalisasi,
7. metrics dicatat,
8. jika gagal, fallback dipicu,
9. hasil akhir dikirim ke user dengan penjelasan ringkas bila perlu.

---

## Hal yang Sebaiknya Tidak Dikerjakan Dulu

* menambah banyak provider tanpa registry yang rapi,
* membuat UI dashboard besar sebelum telemetry stabil,
* menulis optimasi kompleks sebelum baseline performance terukur,
* membuat kompatibilitas penuh ke semua client sekaligus,
* menambah fitur agent baru tanpa observability.

---

## Acceptance Checklist untuk PR

Sebelum PR di-merge, pastikan:

* tujuan PR jelas,
* ada isu yang diselesaikan,
* ada test minimal,
* ada dokumentasi perubahan jika menyentuh user-facing behavior,
* tidak ada secret atau log sensitif,
* fallback behavior terdokumentasi,
* pesan error mudah dipahami,
* tidak memecah konfigurasi lama.

---

## Output yang Diharapkan dari Agent Coding

Saat bekerja pada `hermes_agent`, agent coding harus menghasilkan:

* kode yang modular,
* perubahan yang aman,
* routing yang lebih pintar,
* dukungan banyak provider yang lebih tahan gagal,
* telemetry yang bisa dipakai untuk debugging,
* pengalaman user yang lebih konsisten.

---

## Ringkasan Eksekutif

Prioritas paling bernilai untuk `hermes_agent` adalah:

1. **model registry yang kaya metadata**,
2. **routing policy + auto-fallback**,
3. **adapter layer yang seragam**,
4. **quota/spend/health observability**,
5. **compression dan compatibility endpoint**.

Kalau lima hal itu matang, Hermes akan jauh lebih dekat ke kategori **AI gateway / control plane** daripada sekadar agent CLI biasa.

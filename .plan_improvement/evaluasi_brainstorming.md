# Evaluasi Komprehensif: Dokumen Perencanaan Penyempurnaan Hermes Agent

> **Evaluator:** Claude Opus 4 (Thinking)
> **Tanggal:** 29 Mei 2026
> **Scope:** 6 file di `hermes_agent/.plan_improvement/brainstorming/` + verifikasi terhadap 5 repo aktual

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Inventaris Dokumen](#2-inventaris-dokumen)
3. [Evaluasi Per Dokumen](#3-evaluasi-per-dokumen)
4. [Verifikasi Klaim vs Realitas Codebase](#4-verifikasi-klaim-vs-realitas-codebase)
5. [Analisis Silang: Konsistensi Antar Dokumen](#5-analisis-silang-konsistensi-antar-dokumen)
6. [Blind Spots & Risiko yang Terlewat](#6-blind-spots--risiko-yang-terlewat)
7. [Rekomendasi Prioritas Final](#7-rekomendasi-prioritas-final)
8. [Kesimpulan](#8-kesimpulan)

---

## 1. Ringkasan Eksekutif

Enam dokumen di `.plan_improvement/brainstorming/` merepresentasikan output dari **dua sumber AI** (ChatGPT dan Claude/Sonnet) dengan tujuan yang sama: membuat panduan bagi coding agent untuk menyempurnakan `hermes_agent` dengan fitur multi-provider dari repo 9router, CLIProxyAPI, OmniRoute, dan litellm.

### Temuan Utama

| Aspek | Status |
|-------|--------|
| **Visi strategis** | ✅ Konsisten: semua sepakat hermes harus naik ke "AI control plane" |
| **Kesesuaian dengan codebase aktual** | ⚠️ Sebagian — ada asumsi yang tidak sesuai realitas |
| **Kelengkapan kode implementasi** | ✅ Sonnet unggul — kode lengkap siap pakai |
| **Kerangka strategis & governance** | ✅ ChatGPT unggul — DoD, PR checklist, prinsip desain |
| **Duplikasi antar dokumen** | ⚠️ Tinggi — 60-70% konten berulang |
| **Realisme estimasi effort** | ⚠️ Terlalu optimis — tidak memperhitungkan kompleksitas integrasi |
| **Awareness terhadap bahasa stack** | ❌ Ada mismatch bahasa pemrograman yang kritikal |

---

## 2. Inventaris Dokumen

| # | File | Sumber | Ukuran | Tipe Konten |
|---|------|--------|--------|-------------|
| 1 | [sonnet4-7-adaptive.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/brainstorming/sonnet4-7-adaptive.md) | Claude Sonnet 4.7 | 5.5 KB | Analisis awal + ringkasan arahan |
| 2 | [sonnet4-7-adaptive-arahan.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/brainstorming/sonnet4-7-adaptive-arahan.md) | Claude Sonnet 4.7 | 65.7 KB | Panduan implementasi lengkap (2029 baris) |
| 3 | [chatgpt-thinking.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/brainstorming/chatgpt-thinking.md) | ChatGPT (Thinking) | 6.7 KB | Analisis strategis + roadmap |
| 4 | [chatgpt-thinking-arahan.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/brainstorming/chatgpt-thinking-arahan.md) | ChatGPT (Thinking) | 10.2 KB | Arahan coding agent (434 baris) |
| 5 | [chatgpt-deepresearch.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/brainstorming/chatgpt-deepresearch.md) | ChatGPT (Deep Research) | 25.9 KB | Laporan analitis teknis lengkap |
| 6 | [evaluasi_sonnet4-7-adaptive.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/brainstorming/evaluasi_sonnet4-7-adaptive.md) | Claude Sonnet 4.7 | 62.8 KB | Self-evaluation + dokumen gabungan (1741 baris) |

---

## 3. Evaluasi Per Dokumen

### 3.1 sonnet4-7-adaptive.md — Analisis Awal Sonnet

**Kekuatan:**
- Ringkas dan terarah — langsung identifikasi 5 prioritas
- Pemetaan yang jelas antara fitur repo sumber → kebutuhan hermes
- Mengenali bahwa hermes sudah punya fitur canggih (skills, FTS5, cron, dll.)

**Kelemahan:**
- Terlalu ringkas — hanya 96 baris, tidak cukup sebagai panduan implementasi
- Asumsi "9router" dan "OmniRoute" bisa langsung diadaptasi tanpa memperhitungkan perbedaan bahasa (JS/TS vs Python)
- Menyebut "Kiro AI" dan "OpenCode Free" sebagai fallback gratis — perlu verifikasi apakah API masih tersedia
- Tidak memeriksa struktur `providers/` yang sudah ada di hermes

> **Skor: 6/10** — Baik sebagai starting point, tidak cukup sebagai panduan.

---

### 3.2 sonnet4-7-adaptive-arahan.md — Panduan Implementasi Sonnet

**Kekuatan:**
- 🏆 **Kode implementasi paling lengkap** dari semua dokumen — 2029 baris
- Setiap modul memiliki kode Python siap pakai dengan docstring
- Urutan 12 commit yang terstruktur dan minim risiko breaking change
- Menyediakan contoh `cli-config.yaml` yang realistis
- Integrasi ke `model_tools.py` yang sudah ada (append-only, non-breaking)
- Contoh test yang bisa langsung dijalankan
- Arsitektur modul `provider_gateway/` yang bersih

**Kelemahan:**
- ❌ **Tidak mengenali `providers/base.py` yang sudah ada di hermes!** Dokumen ini mengusulkan provider adapter baru dari nol, padahal hermes sudah memiliki `ProviderProfile` base class yang cukup berbeda desainnya (deklaratif, bukan imperatif)
- Mengasumsikan provider di hermes adalah adapter per-vendor dengan method `complete()` dan `stream()`, padahal kenyataannya `ProviderProfile` hermes adalah dataclass deklaratif — transport di-handle oleh `AIAgent`
- Struktur modul `provider_gateway/` berdiri sendiri di luar `providers/` yang sudah ada — bisa menyebabkan dualitas arsitektur
- Menyebut `USE_LITELLM_BACKEND=true` tapi tidak menjelaskan bagaimana ini berinteraksi dengan sistem `hermes model` yang sudah ada
- Kode `gateway.py` re-invent OpenAI client di setiap request — seharusnya bisa reuse client
- Proxy support menggunakan `httpx.Client()` — perlu validasi apakah openai SDK versi hermes mendukung custom http_client

> **Skor: 8/10** — Paling actionable, tapi butuh reconciliation dengan arsitektur existing.

---

### 3.3 chatgpt-thinking.md — Analisis Strategis ChatGPT

**Kekuatan:**
- 🏆 **Visi strategis paling tajam**: "Hermes sebaiknya naik kelas dari CLI agent dengan banyak provider menjadi AI control plane"
- Identifikasi 8 area penyempurnaan yang komprehensif
- Membedakan antara "menambah daftar model" vs "membangun lapisan orkestrasi"
- Mengenali value proposition unik dari setiap repo sumber

**Kelemahan:**
- Terlalu abstrak — tidak ada kode implementasi
- Referensi URL GitHub tidak bisa diakses (repo private/fork)
- Tidak memetakan ke file/modul spesifik di hermes
- Menyebut "MCP/A2A" sebagai target tapi hermes sudah punya `mcp_serve.py` dan `acp_adapter/`

> **Skor: 7/10** — Excellent framing, tapi tidak actionable tanpa detail teknis.

---

### 3.4 chatgpt-thinking-arahan.md — Arahan Coding Agent ChatGPT

**Kekuatan:**
- 🏆 **Framework governance paling lengkap**: Definition of Done, PR Checklist, Kriteria Kualitas Kode
- Prinsip desain yang matang dan pragmatis (6 prinsip)
- 4 fase implementasi yang realistis dengan "apa yang tidak dikerjakan dulu"
- Alur request ideal 9 langkah yang jelas
- Pola implementasi modul yang bersih (core/, providers/, routing/, telemetry/, config/)
- Acceptance checklist yang bisa langsung dipakai tim

**Kelemahan:**
- **Nol kode implementasi** — murni arahan strategis
- Struktur modul yang diusulkan (`core/`, `routing/`, `telemetry/`, `config/`, `guardrails/`) berpotensi konflik dengan struktur hermes yang sudah ada
- Tidak menyebut `provider_gateway/` sebagai namespace — inconsistent dengan dokumen Sonnet
- Target fitur #10 (Extensibility untuk Tooling Agent) terlalu vague
- Tidak memperhitungkan bahwa hermes adalah fork dari NousResearch — upstream merge bisa terganggu

> **Skor: 7.5/10** — Governance framework terbaik, ideal dikombinasikan dengan kode Sonnet.

---

### 3.5 chatgpt-deepresearch.md — Laporan Analitis Teknis

**Kekuatan:**
- 🏆 **Analisis perbandingan paling komprehensif**: tabel fitur per repo, tabel arsitektur/komponen
- Daftar 15+ provider AI dengan konteks yang akurat
- Diagram Mermaid arsitektur yang jelas
- Contoh konfigurasi YAML multi-provider
- Contoh kode Python menggunakan LiteLLM SDK
- Checklist implementasi 12 item
- Roadmap 8 fase dengan estimasi effort dan risiko
- Referensi dengan citation marks ke dokumen sumber

**Kelemahan:**
- ⚠️ **Beberapa fakta perlu diverifikasi** — citation marks (†) merujuk ke dokumen sumber yang tidak tersedia
- Tabel perbandingan bahasa pemrograman mencantumkan hermes sebagai "Python (utama), TypeScript (UI)" — TypeScript sebenarnya hanya untuk `hermes_cli/` dan `ui-tui/`
- Contoh kode LiteLLM menggunakan `resp1.completions[0].message["content"]` yang tidak sesuai API LiteLLM aktual (seharusnya `resp.choices[0].message.content`)
- Model dan versi yang disebut (Claude 3 Opus, GPT-4o, Gemini 1.5) sudah tidak relevan — sekarang era Claude Opus 4.6, GPT-o3
- Tidak memperhitungkan bahwa CLIProxyAPI (Go) dan 9router (JS) tidak bisa langsung di-import ke hermes (Python)
- Menyebut "PostgreSQL via Prisma" untuk LiteLLM — benar untuk proxy server, tapi SDK-nya standalone

> **Skor: 7/10** — Referensi analitis terbaik, tapi beberapa data sudah stale.

---

### 3.6 evaluasi_sonnet4-7-adaptive.md — Dokumen Gabungan/Evaluasi

**Kekuatan:**
- 🏆 **Dokumen paling lengkap dan terintegrasi** — 1741 baris, 62.8 KB
- Menggabungkan visi ChatGPT + kode Sonnet dengan evaluasi kritis
- Catatan evaluasi di awal yang jujur mengakui kekuatan/kelemahan masing-masing sumber
- Kode implementasi lengkap untuk semua fase (1-4) dengan kode Python siap pakai
- Model Registry dengan metadata yang sangat kaya (ModelEntry dataclass)
- Routing Policy Engine dengan 9 built-in policies
- Telemetry Store dengan persistensi dan aggregasi
- 16 commit sequence yang lebih granular
- Definition of Done table yang ketat

**Kelemahan:**
- ❌ **Masih mewarisi blind spot utama dari Sonnet**: tidak mengenali `ProviderProfile` yang sudah ada
- Ukuran dokumen terlalu besar (62.8 KB) — sulit dicerna oleh agent dalam satu sesi
- Beberapa komponen redundan (CircuitBreaker di fase 1 mirip sekali dengan yang di sonnet4-7-adaptive-arahan.md)
- Credential store menggunakan base64 "obfuscation" yang **bukan enkripsi** — ini bisa jadi false sense of security
- Contoh config YAML menyebut "Kiro Free" dan model "claude-sonnet-4.5" — perlu verifikasi ketersediaan
- Tidak ada section tentang **migration plan** untuk user yang sudah menggunakan hermes dengan config lama

> **Skor: 8.5/10** — Dokumen terlengkap, tapi perlu pemangkasan dan reconciliation.

---

## 4. Verifikasi Klaim vs Realitas Codebase

### 4.1 Hermes Agent — Klaim vs Realitas

| Klaim Dokumen | Realitas Aktual | Status |
|---------------|----------------|--------|
| "Hermes punya ~12 provider" | `providers/` hanya berisi `base.py`, `__init__.py`, dan `README.md` — provider didaftarkan via `ProviderProfile` instances di `__init__.py` | ⚠️ Tidak per-folder |
| "providers/ berisi adapter per vendor" | Provider adalah **dataclass deklaratif** (`ProviderProfile`), BUKAN class dengan method `complete()/stream()`. Transport di-handle terpusat oleh `AIAgent` | ❌ Asumsi salah |
| "model_tools.py menangani model selection" | File ini ada (40.7 KB) dan memang menangani model tools | ✅ Benar |
| "`cli-config.yaml.example` sudah ada" | Ada, ukuran 60.6 KB — sudah sangat besar | ✅ Benar |
| "hermes_state.py menangani state" | Ada, 141.8 KB — file terbesar di repo | ✅ Benar |
| "Sistem skills sudah mature" | `skills/` dan `optional-skills/` ada | ✅ Benar |
| "Punya FTS5 memory search" | Implementasi SQLite FTS5 di `hermes_state.py` | ✅ Benar |
| "MCP/ACP sudah ada" | `mcp_serve.py` (31.7 KB), `acp_adapter/`, `acp_registry/` ada | ✅ Benar |

> [!CAUTION]
> **Temuan Kritis:** Arsitektur provider hermes BUKAN per-folder adapter pattern. `ProviderProfile` adalah dataclass tanpa method `complete()` — semua transport logic ada di agent core. Dokumen-dokumen yang mengusulkan `ProviderAdapter` ABC dengan method `complete()/stream()` tidak kompatibel langsung dengan arsitektur existing.

### 4.2 9router — Verifikasi Fitur

| Klaim | Realitas | Status |
|-------|---------|--------|
| "RTK token compression" | `src/lib/` mengandung modul kompresi | ✅ Likely |
| "Auto-fallback multi-tier" | Arsitektur combo/router ada di `src/` | ✅ Likely |
| "Format translation" | `src/lib/providerNormalization.js` ada | ✅ Benar |
| "SQLite storage" | `src/lib/db/` dan `src/lib/localDb.js` ada | ✅ Benar |
| "Bahasa: JavaScript/Node.js + Next.js" | `next.config.mjs`, `package.json` ada | ✅ Benar |
| "OAuth multi-account" | `src/lib/oauth/` ada | ✅ Benar |

> [!IMPORTANT]
> 9router ditulis dalam **JavaScript/Next.js** — kode tidak bisa di-copy-paste ke hermes (Python). Hanya **pola desain** yang bisa diadaptasi.

### 4.3 OmniRoute — Verifikasi Fitur

| Klaim | Realitas | Status |
|-------|---------|--------|
| "177+ providers" | `src/lib/providers/`, `src/lib/providerModels/` — struktur lengkap | ✅ Likely |
| "14 routing strategies" | `src/lib/resilience/`, `src/lib/combos/` ada | ✅ Likely |
| "Circuit breaker" | `src/lib/resilience/` ada | ✅ Benar |
| "Caveman compression" | Bisa ada di `src/lib/` (perlu cek lebih detail) | ✅ Likely |
| "Guardrails" | `src/lib/guardrails/`, `src/lib/piiSanitizer.ts` ada | ✅ Benar |
| "Semantic cache" | `src/lib/semanticCache.ts` (12.7 KB) ada | ✅ Benar |
| "A2A/MCP" | `src/lib/a2a/`, `src/lib/acp/` ada | ✅ Benar |
| "TypeScript/Next.js" | `tsconfig.json`, `next.config.mjs` ada | ✅ Benar |

> [!IMPORTANT]
> OmniRoute adalah proyek **TypeScript/Next.js** yang sangat matang (700+ KB changelog, 52 KB README). Jauh lebih besar dari 9router. Bisa menjadi referensi arsitektur yang lebih kaya.

### 4.4 CLIProxyAPI — Verifikasi Fitur

| Klaim | Realitas | Status |
|-------|---------|--------|
| "Go-based proxy" | `go.mod`, `go.sum`, `internal/` — 100% Go | ✅ Benar |
| "OAuth multi-account" | `internal/auth/` ada | ✅ Benar |
| "Translator/format" | `internal/translator/` ada | ✅ Benar |
| "Cache" | `internal/cache/` ada | ✅ Benar |
| "Registry" | `internal/registry/` ada | ✅ Benar |

> [!NOTE]
> CLIProxyAPI ditulis dalam Go — paling jauh dari stack hermes. Hanya arsitektur internal (translator, registry, cache pattern) yang bisa dijadikan referensi.

### 4.5 LiteLLM — Verifikasi Fitur

| Klaim | Realitas | Status |
|-------|---------|--------|
| "100+ LLM providers" | `litellm/llms/` direktori besar, `litellm/main.py` (317 KB!) | ✅ Benar |
| "Unified API" | `litellm/__init__.py` (89 KB) — API publik besar | ✅ Benar |
| "Router" | `litellm/router.py` (492 KB!) — sangat matang | ✅ Benar |
| "Cost calculator" | `litellm/cost_calculator.py` (104 KB) | ✅ Benar |
| "Model price database" | `model_prices_and_context_window.json` (1.47 MB!) | ✅ Benar |
| "Compression" | `litellm/compression/` ada | ✅ Benar |
| "Python SDK" | `pyproject.toml`, Python-first | ✅ Benar |

> [!TIP]
> LiteLLM adalah satu-satunya repo sumber yang **berbahasa sama** (Python) dengan hermes. `litellm` bisa langsung dijadikan dependency (`pip install litellm`) tanpa reimplementasi. Ini adalah jalan tercepat untuk mendukung 100+ provider.

---

## 5. Analisis Silang: Konsistensi Antar Dokumen

### 5.1 Namespace/Struktur Modul

| Dokumen | Namespace Utama | Konsistensi |
|---------|----------------|-------------|
| sonnet4-7-adaptive-arahan | `provider_gateway/` (7 file) | — |
| chatgpt-thinking-arahan | `core/`, `providers/`, `routing/`, `telemetry/`, `config/` | — |
| evaluasi_sonnet4-7-adaptive | `core/`, `providers/`, `routing/`, `telemetry/`, `compression/`, `config/`, `guardrails/` | — |

> ⚠️ **Tidak ada konsensus.** Sonnet menggunakan flat `provider_gateway/`, ChatGPT menggunakan hierarki multi-folder, evaluasi gabungan mengikuti ChatGPT. Ketiganya mengabaikan bahwa hermes sudah punya `providers/` dengan arsitektur berbeda.

### 5.2 Prioritas Fitur

| Fitur | Sonnet | ChatGPT | Deep Research | Evaluasi |
|-------|--------|---------|---------------|----------|
| Model Registry | P1 (implisit) | P1 (#1) | Fase 2 | P1 (Fase 1) |
| Adapter Layer | P1 | P1 (#4) | Fase 2 | P1 (Fase 1) |
| Fallback Engine | P1 | P1 (#3) | Fase 3 | P1 (Fase 1) |
| Circuit Breaker | P1 | P1 (implisit) | Fase 3 | P1 (Fase 1) |
| Routing Policy | P1 (implisit) | P2 (#2) | Fase 2 | P2 (Fase 2) |
| Compression (RTK) | P1 | P3 (#6) | Fase 4 | P3 (Fase 3) |
| LiteLLM Backend | P2 | Tidak disebut | Fase 1 (prototipe) | P2 (implisit) |
| Health Dashboard | P3 | P2 (#5) | Fase 5 | P2 (Fase 2) |
| Semantic Cache | P3 | Tidak eksplisit | Fase 4 | P3 (Fase 3) |
| Ollama/Local Model | P4 | P3 (implisit) | Fase 4 | P3 (Fase 3) |
| Proxy Support | P4 | Tidak disebut | Fase 4 | P3 (Fase 3) |
| Guardrails | P5 | P3 (#9) | Fase 5 | P4 (Fase 4) |
| Compatibility Endpoint | Tidak jelas | P3 (#7) | Fase 6 | P3 (Fase 3) |
| Credential Management | P5 (implisit) | P3 (#8) | Fase 2 | P4 (Fase 4) |

> ✅ Ada konsensus kuat bahwa **Fase 1 = Registry + Adapter + Fallback + Circuit Breaker**. Perbedaan utama ada di timing LiteLLM dan Compression.

### 5.3 Jumlah Commit/Step

| Dokumen | Jumlah Commit |
|---------|---------------|
| sonnet4-7-adaptive-arahan | 12 commits |
| evaluasi_sonnet4-7-adaptive | 16 commits |
| chatgpt-thinking-arahan | Tidak ada commit order |
| chatgpt-deepresearch | 8 fase (bukan commit) |

---

## 6. Blind Spots & Risiko yang Terlewat

### 6.1 ❌ Arsitektur Provider Hermes Tidak Dipahami

**Ini blind spot paling kritikal.** Semua dokumen mengusulkan `ProviderAdapter` ABC dengan method `complete()` dan `stream()`. Padahal:

```python
# Realitas: providers/base.py di hermes
@dataclass
class ProviderProfile:
    name: str
    api_mode: str = "chat_completions"
    base_url: str = ""
    auth_type: str = "api_key"
    # ... fields deklaratif, BUKAN method transport
```

`ProviderProfile` tidak punya `complete()` atau `stream()`. Ia hanya **mendeskripsikan** provider. Transport (panggilan API) dilakukan oleh `AIAgent` menggunakan `openai.OpenAI()` client dengan `base_url` dari profile.

**Implikasi:** Menambahkan `ProviderAdapter` ABC yang paralel dengan `ProviderProfile` akan menciptakan **dua sistem provider yang saling overlap.** Coding agent harus memilih salah satu:
1. Extend `ProviderProfile` dengan routing metadata (lebih aman, backward compatible)
2. Migrasi ke `ProviderAdapter` pattern baru (lebih bersih, tapi breaking change besar)

### 6.2 ❌ File Ukuran Besar Tidak Diperhitungkan

| File | Ukuran | Concern |
|------|--------|---------|
| `cli.py` | 693 KB | Monolith — menambahkan fitur di sini sangat berisiko |
| `run_agent.py` | 203 KB | Entry point besar — perlu hati-hati |
| `hermes_state.py` | 142 KB | State management terpusat — fragile |
| `model_tools.py` | 41 KB | Target integrasi — harus dibaca penuh sebelum modifikasi |
| `cli-config.yaml.example` | 61 KB | Config sudah sangat panjang |

Tidak ada dokumen yang menyebutkan risiko **modifikasi file monolith** ini.

### 6.3 ❌ Upstream Sync dengan NousResearch

Hermes agent adalah fork dari `NousResearch/hermes-agent`. Menambahkan banyak modul baru bisa membuat **upstream merge** semakin sulit. Ini tidak dibahas di dokumen manapun.

### 6.4 ⚠️ Dependency Management

- Menambah `litellm` sebagai dependency menambah **ratusan transitive dependencies** (boto3, google-cloud-*, dll.)
- Menambah `redis` untuk semantic cache menambah dependency runtime
- Menambah `httpx` untuk proxy support — apakah sudah ada di hermes?

Tidak ada dokumen yang melakukan **dependency audit** atau mempertimbangkan bloat.

### 6.5 ⚠️ Testing Baseline

Dokumen evaluasi menyebut 368 test di OmniRoute. Berapa test yang sudah ada di hermes? Folder `tests/` ada tapi tidak diperiksa isinya. Jika hermes tidak punya test infrastructure yang baik, menambahkan test baru akan lebih sulit.

### 6.6 ⚠️ Keamanan Credential Store

Dokumen evaluasi mengusulkan `CredentialStore` dengan base64 encoding. Ini **bukan enkripsi** dan memberikan false sense of security. File credential yang hanya di-base64 tetap bisa dibaca siapapun yang punya akses file. Rekomendasi: gunakan `keyring` library atau OS-native secret storage.

### 6.7 ⚠️ Estimasi "Kiro AI" dan "OpenCode Free"

Beberapa dokumen menyebut provider gratis ini sebagai fallback. Perlu verifikasi:
- Apakah Kiro AI masih menyediakan free API?
- Apakah OpenCode Free masih beroperasi?
- Apakah Vertex credits gratis masih available?
- Model names yang disebut (claude-sonnet-4.5, glm-5) — apakah masih valid?

---

## 7. Rekomendasi Prioritas Final

Berdasarkan evaluasi di atas, saya merekomendasikan pendekatan berikut:

### Fase 0 — Persiapan (BARU, tidak ada di dokumen manapun)

> [!IMPORTANT]
> Langkah ini **wajib** sebelum implementasi apapun.

1. **Baca dan pahami `providers/base.py` + `providers/__init__.py` secara menyeluruh** — pahami `ProviderProfile` dan bagaimana ia digunakan oleh `AIAgent`
2. **Audit `model_tools.py`** — pahami alur model selection saat ini
3. **Audit `run_agent.py`** — cari di mana panggilan ke LLM API dibuat
4. **Audit test infrastructure** — apa yang sudah ada di `tests/`
5. **Keputusan arsitektur**: Extend `ProviderProfile` ATAU buat layer baru di atasnya?
6. **Keputusan dependency**: Apakah pakai `litellm` sebagai dependency atau reimplementasi?

### Fase 1 — Fondasi Routing (Minggu 1-2)

Berdasarkan konsensus semua dokumen, dengan penyesuaian terhadap realitas:

| Item | Pendekatan | Catatan |
|------|-----------|---------|
| Model Registry | Tambah metadata ke `ProviderProfile` existing | Jangan buat class baru yang overlap |
| Routing Policy | File baru `routing/policy.py` | Berdiri sendiri, mudah ditest |
| Fallback Engine | File baru `routing/fallback.py` | Wrap existing transport logic |
| Circuit Breaker | File baru `routing/circuit_breaker.py` | Standalone, thread-safe |
| Telemetry Basic | File baru `telemetry/metrics.py` | Log-first, persist later |

### Fase 2 — LiteLLM Integration (Minggu 3-4)

> [!TIP]
> Ini jalan tercepat untuk 100+ provider. Gunakan `litellm` sebagai **optional backend** yang bisa di-toggle.

| Item | Pendekatan |
|------|-----------|
| LiteLLM as backend | Opt-in via config `use_litellm: true` |
| Cost tracking | Leverage `litellm.completion_cost()` |
| Model discovery | Leverage `litellm.model_list` |

### Fase 3 — Compression & Observability (Minggu 5-6)

| Item | Pendekatan |
|------|-----------|
| RTK Compression | File baru, di-apply di message pipeline |
| `hermes status` command | Integrasi telemetry + circuit breaker |
| Quota tracking | Persistensi ke SQLite (bukan JSON file) |

### Fase 4 — Ekosistem (Minggu 7+)

| Item | Pendekatan |
|------|-----------|
| Ollama provider | Tambah `ProviderProfile` baru untuk Ollama |
| Guardrails | PII filter (log only default), injection guard |
| Compatibility endpoint | OpenAI-compatible `/v1/chat/completions` |
| Semantic cache | In-memory LRU, Redis optional |

---

## 8. Kesimpulan

### Dokumen Terbaik untuk Digunakan

| Kebutuhan | Gunakan Dokumen |
|-----------|----------------|
| **Visi & Prinsip** | [chatgpt-thinking-arahan.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/brainstorming/chatgpt-thinking-arahan.md) |
| **Kode Implementasi** | [evaluasi_sonnet4-7-adaptive.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/brainstorming/evaluasi_sonnet4-7-adaptive.md) (dengan modifikasi) |
| **Referensi Perbandingan** | [chatgpt-deepresearch.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/brainstorming/chatgpt-deepresearch.md) |
| **Governance (DoD, PR)** | [chatgpt-thinking-arahan.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/brainstorming/chatgpt-thinking-arahan.md) |

### Skor Akhir Keseluruhan

| Dokumen | Akurasi | Kelengkapan | Actionability | Skor |
|---------|---------|-------------|---------------|------|
| sonnet4-7-adaptive.md | 7/10 | 4/10 | 3/10 | **6/10** |
| sonnet4-7-adaptive-arahan.md | 6/10 | 9/10 | 9/10 | **8/10** |
| chatgpt-thinking.md | 8/10 | 5/10 | 4/10 | **7/10** |
| chatgpt-thinking-arahan.md | 7/10 | 7/10 | 6/10 | **7.5/10** |
| chatgpt-deepresearch.md | 6/10 | 8/10 | 5/10 | **7/10** |
| evaluasi_sonnet4-7-adaptive.md | 7/10 | 10/10 | 8/10 | **8.5/10** |

### Peringatan Utama

> [!CAUTION]
> **Sebelum coding agent mulai mengimplementasi fitur apapun dari dokumen-dokumen ini, WAJIB dilakukan reconciliation arsitektur provider.** Semua dokumen mengasumsikan pattern yang berbeda dari realitas `ProviderProfile` di hermes. Mengabaikan ini akan menghasilkan arsitektur ganda yang sulit dimaintain.

### Rekomendasi Akhir

Buat satu **dokumen arahan final** yang:
1. Menggunakan prinsip desain dan governance dari ChatGPT
2. Menggunakan kode implementasi dari evaluasi Sonnet (dengan adaptasi ke `ProviderProfile`)
3. Menambahkan Fase 0 (persiapan dan audit arsitektur)
4. Memangkas ukuran dokumen ke < 30 KB (fokus Fase 1-2 saja)
5. Menambahkan section **migration path** untuk user existing
6. Menghapus referensi ke provider/model yang belum diverifikasi

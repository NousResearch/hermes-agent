# Riset Keunggulan Repo Sumber untuk Penyempurnaan Hermes Agent

> **Evaluator:** Claude Opus 4 (Thinking)
> **Tanggal:** 29 Mei 2026
> **Metode:** Pembacaan source code aktual dari setiap repo

---

## Daftar Isi

1. [9router (JavaScript/Next.js)](#1-9router)
2. [OmniRoute (TypeScript/Next.js)](#2-omniroute)
3. [CLIProxyAPI (Go)](#3-cliproxyapi)
4. [LiteLLM (Python)](#4-litellm)
5. [Tabel Ringkasan Perbandingan](#5-tabel-ringkasan-perbandingan)
6. [Rekomendasi Fitur untuk Hermes](#6-rekomendasi-fitur-untuk-hermes)

---

## 1. 9router

**Bahasa:** JavaScript (Node.js + Next.js)
**Lokasi:** `/home/void/lab/git/9router/`
**Fokus:** MITM proxy lokal + OAuth multi-account + dashboard web

### 1.1 Keunggulan Utama

#### A. MITM Proxy Lokal (Fitur Unik)
- **Lokasi:** `src/mitm/manager.js` (852 baris), `src/mitm/server.js` (15.6 KB)
- Sistem MITM proxy yang intercept traffic HTTPS di port 443
- Auto-generate Root CA certificate, auto-install ke system trust store
- DNS hijacking per-tool (Copilot, Gemini CLI, Kiro, dll.) untuk mengarahkan traffic ke proxy lokal
- Mendukung Windows, macOS, Linux
- Auto-restart dengan exponential backoff (5s → 60s, max 5 restarts)
- Health check via `/_mitm_health` endpoint

```
Cara kerja:
1. Generate Root CA cert → install ke system store
2. Spawn HTTPS server di port 443
3. Hijack DNS (hosts file) untuk provider target
4. Intercept request → forward ke router lokal (port 20128)
5. Router pilih provider + apply transformasi
```

**Relevansi untuk Hermes:** ⭐⭐ Rendah — terlalu kompleks, butuh elevated privileges. Tapi konsep *transparent interception* bisa dipakai untuk menangkap request dari coding tools lain.

#### B. Database Schema dengan Migrasi (SQLite)
- **Lokasi:** `src/lib/db/schema.js` (158 baris), `src/lib/db/migrate.js` (13.1 KB)
- Schema deklaratif untuk semua tabel:
  - `providerConnections` — multi-account per provider, dengan priority dan status aktif
  - `proxyPools` — kumpulan proxy SOCKS/HTTP untuk bypass region-lock
  - `apiKeys` — virtual API key management per machine
  - `combos` — routing rule yang bisa dikustomisasi (nama, kind, models)
  - `usageHistory` — log per-request (provider, model, token, cost, status)
  - `usageDaily` — agregasi harian
  - `requestDetails` — detail lengkap request untuk debugging
- WAL mode, PRAGMA optimized, foreign keys, indexes per kolom penting
- Sistem migrasi versi (schema versioning + auto-sync columns)

**Relevansi untuk Hermes:** ⭐⭐⭐⭐ Tinggi — hermes sudah pakai SQLite. Schema untuk `usageHistory` dan `providerConnections` bisa diadaptasi langsung.

#### C. Provider Normalization
- **Lokasi:** `src/lib/providerNormalization.js` (46 baris)
- Normalisasi provider ID (slug-based matching)
- Deteksi model per provider (contoh: xAI Grok detection)
- Normalisasi data spesifik per provider (contoh: Ollama baseUrl)

**Relevansi untuk Hermes:** ⭐⭐⭐ Sedang — pola normalisasi berguna, tapi hermes sudah punya `ProviderProfile.aliases`.

#### D. Usage Tracking per Provider
- **Lokasi:** `src/lib/usage/fetcher.js` (209 baris)
- Fetcher quota aktual dari API provider:
  - **GitHub Copilot:** Chat, completions, premium_interactions quota (paid vs free)
  - **Gemini CLI:** Google Cloud quota
  - **Claude:** Per-request tracking
  - **Codex (OpenAI):** Dashboard-based
  - **Qwen:** Resource URL-based
  - **iFlow:** Per-request
- Mendukung format quota yang berbeda per provider

**Relevansi untuk Hermes:** ⭐⭐⭐⭐⭐ Sangat Tinggi — ini fitur yang sangat bernilai. Hermes belum bisa menampilkan sisa quota dari provider.

#### E. Outbound Proxy Support
- **Lokasi:** `src/lib/network/` (4 file)
- Connection proxy (SOCKS5/HTTP) untuk bypass region restrictions
- Proxy testing sebelum digunakan
- Konfigurasi outbound proxy per-request

**Relevansi untuk Hermes:** ⭐⭐⭐ Sedang — berguna untuk user di region yang dibatasi (China, dll.)

#### F. Tunnel Support
- **Lokasi:** `src/lib/tunnel/` (Cloudflare + Tailscale)
- Ekspos proxy lokal ke internet via Cloudflare tunnel atau Tailscale
- Memungkinkan akses remote ke AI proxy

**Relevansi untuk Hermes:** ⭐⭐ Rendah — lebih relevan untuk server deployment.

---

## 2. OmniRoute

**Bahasa:** TypeScript (Next.js)
**Lokasi:** `/home/void/lab/git/OmniRoute/`
**Fokus:** AI gateway enterprise-grade dengan routing canggih, guardrails, dan observability

### 2.1 Keunggulan Utama

#### A. Intelligent Routing Engine (Fitur Paling Canggih)
- **Lokasi:** `src/lib/combos/intelligentRouting.ts` (193 baris)
- **11-faktor weighted scoring:**

| Faktor | Bobot Default | Deskripsi |
|--------|---------------|-----------|
| `quota` | 0.16 | Sisa kuota provider |
| `health` | 0.20 | Kesehatan provider (dari circuit breaker) |
| `costInv` | 0.16 | Inverse cost — prefer yang lebih murah |
| `latencyInv` | 0.12 | Inverse latency — prefer yang lebih cepat |
| `taskFit` | 0.08 | Kesesuaian model dengan jenis task |
| `stability` | 0.05 | Stabilitas historis provider |
| `tierPriority` | 0.05 | Prioritas tier (free/paid/subscription) |
| `tierAffinity` | 0.05 | Afinitas user ke tier tertentu |
| `specificityMatch` | 0.05 | Kecocokan model spesifik |
| `contextAffinity` | 0.08 | Afinitas konteks (prefer provider yang sama) |
| `resetWindowAffinity` | 0.00 | Kapan kuota reset |

- **5 Router Strategies:** Rules (6-Factor), Cost Optimized, Latency Optimized, SLA-aware, Last Known Good Provider (LKGP)
- **4 Mode Packs:** Ship Fast 🚀, Cost Saver 💰, Quality First 🎯, Offline Friendly ☁️
- **SLA Constraints:** Target P95 latency, max error rate, max cost per 1M tokens
- **Exploration Rate:** Probabilistic discovery (epsilon-greedy) — kadang sengaja pilih provider lain untuk update data

**Relevansi untuk Hermes:** ⭐⭐⭐⭐⭐ Sangat Tinggi — ini inti dari "AI control plane". Bisa disederhanakan untuk fase awal.

#### B. Composite Tiers System
- **Lokasi:** `src/lib/combos/compositeTiers.ts` (201 baris)
- Tier routing bertingkat dengan fallback chain
- Validasi anti-cycle (deteksi circular fallback)
- Setiap tier memiliki `stepId` (model) dan `fallbackTier` (tier backup)
- Contoh: `premium` → model Claude Opus → fallback ke `standard` → model GPT-4o → fallback ke `free` → model gratis

**Relevansi untuk Hermes:** ⭐⭐⭐⭐ Tinggi — desain fallback chain ini lebih fleksibel dari yang ada di dokumen planning.

#### C. Resilience & Circuit Breaker System
- **Lokasi:** `src/lib/resilience/settings.ts` (559 baris)
- Arsitektur resilience yang sangat matang:
  - **Request Queue:** RPM limiter, concurrent request limiter, max wait timeout
  - **Connection Cooldown:** Per auth-type (OAuth vs API key), exponential backoff, upstream retry hint support
  - **Provider Breaker:** Failure threshold + reset timeout, dibedakan per OAuth vs API key
  - **Wait for Cooldown:** Auto-retry dengan max retries + max wait
  - **Quota Preflight:** Pre-check kuota sebelum request (stop at 2% remaining, warn at 20%)
- Legacy compatibility layer untuk migrasi setting lama

**Relevansi untuk Hermes:** ⭐⭐⭐⭐⭐ Sangat Tinggi — system resilience paling mature di antara semua repo.

#### D. Guardrails System
- **Lokasi:** `src/lib/guardrails/` (7 file)
- **Prompt Injection Guard:**
  - Pattern-based detection dengan 3 severity level (low/medium/high)
  - Mode: block, warn, log — configurable
  - Custom patterns support
  - Threshold-based blocking (hanya block jika severity >= threshold)
- **PII Masker:** `src/lib/guardrails/piiMasker.ts` (6.4 KB)
- **Vision Bridge:** Image validation + bridging (14.5 KB helpers)
- **Registry Pattern:** Guardrails bisa didaftar/diaktifkan secara modular

**Relevansi untuk Hermes:** ⭐⭐⭐ Sedang — berguna sebagai layer keamanan opsional. OmniRoute punya implementasi yang lebih mature dari yang diusulkan di dokumen planning.

#### E. Semantic Cache (Two-Tier)
- **Lokasi:** `src/lib/semanticCache.ts` (423 baris)
- **Two-tier:** In-memory LRU + SQLite persistent
- Cache key: SHA-256(model + normalized messages + temperature + top_p)
- Hanya cache request dengan `temperature: 0` (deterministic)
- Fitur lengkap: auto-cleanup timer, invalidate by model/signature/age, cache metrics tracking
- Bypass header: `X-OmniRoute-No-Cache: true`
- Streaming support: cache after assembly, serve as JSON

**Relevansi untuk Hermes:** ⭐⭐⭐⭐ Tinggi — implementasi paling production-ready. Hermes sudah punya SQLite, tinggal tambah tabel.

#### F. Combo Builder
- **Lokasi:** `src/lib/combos/` (7 file, ~68 KB total)
- Sistem konfigurasi routing rules yang sangat fleksibel
- Steps (urutan model/provider), Builder Options, Control Center
- Health testing per combo

**Relevansi untuk Hermes:** ⭐⭐⭐ Sedang — terlalu UI-centric. Tapi konsep "combo" bisa disederhanakan jadi routing preset di config.

---

## 3. CLIProxyAPI

**Bahasa:** Go
**Lokasi:** `/home/void/lab/git/CLIProxyAPI/`
**Fokus:** Proxy server ringan untuk CLI tools, model registry dengan reference counting

### 3.1 Keunggulan Utama

#### A. Model Registry dengan Reference Counting (Fitur Paling Canggih)
- **Lokasi:** `internal/registry/model_registry.go` (1321 baris, 40.3 KB)
- **Reference counting:** Setiap model tahu berapa "client" yang bisa menyediakannya
  - Model hanya tersedia jika `Count > 0`
  - Quota exceeded per-client tracking dengan cooldown window (5 menit)
  - Client suspension per-model (temporary disable dengan reason)
- **ModelInfo sangat kaya:**
  - Input/Output token limits, context length, max completion tokens
  - Supported parameters, input/output modalities (TEXT, IMAGE, VIDEO, AUDIO)
  - Thinking support (min/max budget, dynamic allowed, levels)
  - User-defined flag (config-based vs auto-detected)
- **Provider-aware:** `InfoByProvider` — satu model ID bisa punya metadata berbeda per provider
- **Hook system:** `ModelRegistryHook` interface — observer pattern untuk integrasi eksternal
- **Thread-safe:** Full mutex protection dengan RW lock
- **Cached available models:** Per-handler type cache dengan expiration
- **Static + Dynamic:** Fallback ke static model definitions jika dynamic registry miss

**Relevansi untuk Hermes:** ⭐⭐⭐⭐⭐ Sangat Tinggi — registry pattern ini jauh lebih sophisticated dari apapun yang diusulkan di dokumen planning. Reference counting dan per-client quota tracking sangat bernilai.

#### B. Multi-Format Translator
- **Lokasi:** `internal/translator/` (8 subdirectory)
- Translator per provider:
  - Antigravity, Claude, Codex, Gemini, Gemini CLI, OpenAI
  - Common translator base
  - Translator factory pattern
- Setiap translator menangani konversi format request/response antara format OpenAI-compatible dan format native provider

**Relevansi untuk Hermes:** ⭐⭐⭐⭐ Tinggi — hermes sudah punya `ProviderProfile.prepare_messages()`, tapi translator CLIProxyAPI lebih systematic.

#### C. Cache Layer
- **Lokasi:** `internal/cache/`
- In-memory cache untuk model metadata dan responses
- Thread-safe dengan proper locking

**Relevansi untuk Hermes:** ⭐⭐⭐ Sedang — lebih sederhana dari OmniRoute semantic cache.

#### D. Auth System
- **Lokasi:** `internal/auth/`, `auths/`
- Multi-account authentication
- OAuth flow support
- Credential management

**Relevansi untuk Hermes:** ⭐⭐⭐ Sedang — pola multi-account berguna tapi implementasi Go tidak bisa di-port langsung.

#### E. Thinking/Reasoning Budget Normalization
- **Dalam ModelInfo:** `ThinkingSupport` struct
- Normalisasi budget reasoning antar provider (min/max/dynamic/levels)
- Contoh: Claude reasoning effort → Gemini thinking budget → OpenAI o1 reasoning_effort
- Levels-based vs token-based reasoning control

**Relevansi untuk Hermes:** ⭐⭐⭐⭐ Tinggi — hermes perlu menangani perbedaan cara setiap provider mengekspos reasoning/thinking.

---

## 4. LiteLLM

**Bahasa:** Python
**Lokasi:** `/home/void/lab/git/litellm/`
**Fokus:** Unified Python SDK untuk 100+ LLM provider + proxy server enterprise

### 4.1 Keunggulan Utama

#### A. Unified Completion API (Core Value)
- **Lokasi:** `litellm/main.py` (317 KB!) + `litellm/__init__.py` (89 KB)
- Satu fungsi `litellm.completion()` untuk semua provider
- Auto-detect provider dari model string: `"openai/gpt-4o"`, `"anthropic/claude-opus-4-6"`, `"ollama/llama3.2"`
- Drop incompatible params (`litellm.drop_params = True`)
- 100+ provider support built-in

**Relevansi untuk Hermes:** ⭐⭐⭐⭐⭐ Sangat Tinggi — ini satu-satunya repo Python. Bisa langsung dijadikan dependency `pip install litellm` tanpa reimplementasi apapun.

#### B. Router dengan 13+ Strategi (Paling Lengkap)
- **Lokasi:** `litellm/router.py` (492 KB!), `litellm/router_strategy/` (13 file)
- Strategi routing:

| Strategi | File | Ukuran |
|----------|------|--------|
| Base Strategy | `base_routing_strategy.py` | 10.4 KB |
| Budget Limiter | `budget_limiter.py` | 37.2 KB |
| Lowest Latency | `lowest_latency.py` | 24.6 KB |
| Lowest Cost | `lowest_cost.py` | 12.5 KB |
| Lowest TPM/RPM | `lowest_tpm_rpm.py` | 9.5 KB |
| Lowest TPM/RPM v2 | `lowest_tpm_rpm_v2.py` | 27.4 KB |
| Least Busy | `least_busy.py` | 9.6 KB |
| Simple Shuffle | `simple_shuffle.py` | 2.8 KB |
| Tag-based Routing | `tag_based_routing.py` | 10.8 KB |
| Adaptive Router | `adaptive_router/` | Directory |
| Auto Router | `auto_router/` | Directory |
| Complexity Router | `complexity_router/` | Directory |
| Quality Router | `quality_router/` | Directory |

- Redis-backed multi-instance sync (batched pipeline increments)
- Budget management per-deployment
- TPM/RPM tracking with periodic sync

**Relevansi untuk Hermes:** ⭐⭐⭐⭐ Tinggi — bisa dipakai via `litellm.Router()` langsung tanpa reimplementasi.

#### C. Cost Calculator (Model Price Database)
- **Lokasi:** `litellm/cost_calculator.py` (104 KB), `model_prices_and_context_window.json` (1.47 MB!)
- Database harga 1.47 MB — mencakup semua model dari semua provider
- Cost estimation per-request

**Relevansi untuk Hermes:** ⭐⭐⭐⭐⭐ Sangat Tinggi — `litellm.completion_cost()` memberikan estimasi biaya tanpa effort apapun.

#### D. Compression Module
- **Lokasi:** `litellm/compression/` (6 file)
- Message compression untuk mengurangi token count
- Content detection (auto-detect tipe konten)
- Message stubbing (replace konten panjang dengan summary)
- Scoring-based importance ranking

**Relevansi untuk Hermes:** ⭐⭐⭐ Sedang — hermes sudah punya `trajectory_compressor.py` (65.3 KB) yang cukup canggih.

#### E. BaseRoutingStrategy Pattern
- **Lokasi:** `litellm/router_strategy/base_routing_strategy.py` (262 baris)
- ABC untuk semua routing strategy
- Built-in Redis pipeline batching untuk multi-instance sync
- In-memory + Redis dual cache
- Periodic sync task (configurable interval)
- Atomic increment operations

**Relevansi untuk Hermes:** ⭐⭐⭐⭐ Tinggi — pattern yang bersih untuk membuat pluggable routing strategies.

---

## 5. Tabel Ringkasan Perbandingan

| Fitur | 9router | OmniRoute | CLIProxyAPI | LiteLLM |
|-------|---------|-----------|-------------|---------|
| **Bahasa** | JavaScript | TypeScript | Go | **Python** ✅ |
| **Bisa jadi dependency hermes?** | ❌ | ❌ | ❌ | **✅ pip install** |
| **Provider count** | ~7 | 177+ | ~6 | **100+** |
| **Routing strategies** | Basic tier | **11-faktor scoring** | Basic | 13+ strategi |
| **Circuit breaker** | ❌ | **✅ Mature** | Via registry | ✅ (via router) |
| **Semantic cache** | ❌ | **✅ Two-tier** | Basic | ✅ (via caching/) |
| **Usage/quota tracking** | **✅ Per-provider API** | ✅ (quota preflight) | ✅ (ref counting) | ✅ (cost calc) |
| **Guardrails** | ❌ | **✅ Full suite** | ❌ | ❌ |
| **MITM proxy** | **✅ Unik** | ❌ | ❌ | ❌ |
| **Format translator** | Basic | ✅ (stream transform) | **✅ Per-provider** | ✅ (built-in) |
| **Model registry** | ❌ | ✅ (via catalog) | **✅ Ref counting** | ✅ (model_list) |
| **Proxy/tunnel support** | **✅ Full** | ✅ (Cloudflare, ngrok, Tailscale) | ❌ | ❌ |
| **Cost tracking** | ✅ (per request) | ✅ (spend batch writer) | ❌ | **✅ 1.47MB price DB** |
| **Thinking/reasoning normalization** | ❌ | ❌ | **✅ Per-provider** | ❌ |
| **Multi-instance sync** | ❌ | ❌ | ❌ | **✅ Redis** |
| **SQLite storage** | **✅ Migration system** | ✅ | ❌ | ❌ |
| **Maturity (kode)** | ~50KB core | ~200KB core | ~60KB core | **~900KB core** |

---

## 6. Rekomendasi Fitur untuk Hermes

### Tier 1: Fitur yang HARUS diadopsi (Nilai tertinggi, effort terendah)

| Fitur | Sumber Utama | Alasan |
|-------|-------------|--------|
| **LiteLLM sebagai optional backend** | LiteLLM | Satu-satunya Python. `pip install litellm` → 100+ provider instan |
| **Usage/quota tracking SQLite** | 9router schema | Schema `usageHistory` + `usageDaily` bisa langsung diadaptasi |
| **Cost estimation** | LiteLLM `completion_cost()` | 1.47 MB price database gratis |
| **Model Registry metadata** | CLIProxyAPI `ModelInfo` | Thinking support, modalities, token limits |

### Tier 2: Fitur yang SEBAIKNYA diadopsi (Nilai tinggi, effort sedang)

| Fitur | Sumber Utama | Alasan |
|-------|-------------|--------|
| **Circuit breaker + cooldown** | OmniRoute resilience | Implementasi paling mature, per-auth-type |
| **Weighted scoring routing** | OmniRoute intelligent routing | 11-faktor → bisa disederhanakan jadi 5-6 faktor |
| **Composite tier fallback** | OmniRoute composite tiers | Anti-cycle validation, multi-level fallback |
| **Semantic cache (two-tier)** | OmniRoute semantic cache | In-memory + SQLite, temperature-aware |
| **Quota preflight** | OmniRoute quota preflight | Pre-check kuota sebelum kirim request |

### Tier 3: Fitur yang BOLEH diadopsi nanti (Nilai sedang, effort tinggi)

| Fitur | Sumber Utama | Alasan |
|-------|-------------|--------|
| **Guardrails (prompt injection + PII)** | OmniRoute guardrails | Severity-based, mode configurable |
| **Outbound proxy support** | 9router network | Untuk bypass region restrictions |
| **Thinking budget normalization** | CLIProxyAPI ThinkingSupport | Normalisasi antar provider |
| **Provider-specific usage API** | 9router usage/fetcher | GitHub Copilot quota, dll. |

### Tier 4: Fitur Opsional — Opt-in Extension (Default OFF, effort tinggi)

> **Catatan:** Proyek ini bersifat **open-source**. Fitur-fitur di bawah ini TIDAK ditolak, melainkan
> diimplementasikan sebagai **ekstensi modular (opt-in)** yang default disabled.
> Pengguna dapat mengaktifkan secara eksplisit via konfigurasi sesuai kebutuhan.
> Lihat detail di: [fitur-opsional-dan-arahan-implementasi.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/fitur-opsional-dan-arahan-implementasi.md)

| Fitur | Sumber Utama | Mengapa Opsional | Nilai bagi Power User |
|-------|-------------|------------------|----------------------|
| **MITM proxy** | 9router MITM manager | Butuh elevated privileges + sertifikat Root CA | Intercept traffic IDE (Copilot, Cursor) → route ke LLM lokal/murah |
| **Tunnel (Cloudflare/Tailscale)** | 9router + OmniRoute tunnel | Membuka akses jaringan lokal ke luar | Kolaborasi tim remote — berbagi endpoint router |
| **Web dashboard UI** | 9router + LiteLLM dashboard | Menambah dependensi frontend berat | Visualisasi metrik performa, usage, dan cost via browser |
| **Redis multi-instance sync** | LiteLLM router strategy | Butuh infrastruktur Redis external | Deploy Hermes sebagai shared gateway untuk tim/kantor |
| **OAuth device flow** | 9router + OmniRoute auth | Alur autentikasi interaktif yang kompleks | Login satu-klik untuk user yang prefer browser-based auth |

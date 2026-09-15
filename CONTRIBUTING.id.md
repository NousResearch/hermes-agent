# Berkontribusi ke Hermes Agent

Terima kasih telah berkontribusi ke Hermes Agent! Panduan ini mencakup hal-hal yang Anda perlukan: menyiapkan environment pengembangan, memahami arsitektur, menentukan apa yang perlu dibangun, dan membawa PR Anda hingga merge.

---

## Prioritas Kontribusi

Kami menghargai kontribusi dalam urutan berikut:

1. **Perbaikan bug** — crash, perilaku yang salah, kehilangan data. Selalu menjadi prioritas tertinggi.
2. **Kompatibilitas lintas platform** — macOS, berbagai distro Linux, dan WSL2 pada Windows. Kami ingin Hermes bekerja di mana saja.
3. **Hardening keamanan** — shell injection, prompt injection, path traversal, privilege escalation. Lihat [Pertimbangan Keamanan](#pertimbangan-keamanan).
4. **Performa dan robustness** — retry logic, error handling, graceful degradation.
5. **Skill baru** — tetapi hanya yang bermanfaat secara luas. Lihat [Apakah harus Skill atau Tool?](#apakah-harus-skill-atau-tool).
6. **Tool baru** — jarang diperlukan. Sebagian besar capability seharusnya berupa skill. Lihat di bawah.
7. **Dokumentasi** — perbaikan, klarifikasi, dan contoh baru.

---

## Sebelum Memulai: Cari Terlebih Dahulu

Pencarian singkat sebelum mulai membangun menghemat waktu Anda dan menjaga antrean PR tetap bersih — duplikasi umum terjadi di sini, jadi satu menit di awal sangat berguna.

- **Cari PR dan issue yang terbuka *maupun* sudah merged** untuk topik atau gejala error Anda — duplicate-check pada template PR baru berjalan saat review, setelah Anda telanjur mengerjakan perubahan:
  ```bash
  gh search issues --repo NousResearch/hermes-agent "<your terms>"
  gh search prs --repo NousResearch/hermes-agent --state all "<your terms>"
  ```
  Atau gunakan UI web: [issues](https://github.com/NousResearch/hermes-agent/issues?q=) · [PRs (all states)](https://github.com/NousResearch/hermes-agent/pulls?q=is%3Apr).
- **Issue tracker dapat tertinggal dari kode.** Banyak fitur yang diminta sebenarnya sudah diimplementasikan di dalam tree, jadi cari juga capability tersebut di source (`search_files`, atau grep editor Anda) sebelum mengusulkannya.
- **Jika PR terbuka sudah menangani masalah itu**, pertimbangkan untuk mereview atau memperbaiki PR tersebut alih-alih membuka duplikat yang bersaing.
- **Untuk pekerjaan yang lebih besar**, komentari issue untuk menandai bahwa Anda sedang mengerjakannya agar orang lain tidak memulai pekerjaan yang sama.

Terkait: #38284 mencakup analog pada sisi agen — Hermes sendiri memeriksa issue dan PR yang sudah ada sebelum melakukan self-troubleshooting mendalam. Bagian ini adalah pelengkap bagi kontributor manusia.

---

## Apakah harus Skill atau Tool?

Ini adalah pertanyaan yang paling umum bagi kontributor baru. Jawabannya hampir selalu **skill**.

### Jadikan Skill ketika:

- Capability dapat dinyatakan sebagai instruksi + perintah shell + tool yang sudah ada
- Capability membungkus CLI atau API eksternal yang dapat dipanggil agen melalui `terminal` atau `web_extract`
- Tidak membutuhkan integrasi Python khusus atau pengelolaan API key yang ditanamkan ke agent harness
- Contoh: pencarian arXiv, workflow git, pengelolaan Docker, pemrosesan PDF, email melalui tool CLI

### Jadikan Tool ketika:

- Memerlukan integrasi end-to-end dengan API key, auth flow, atau konfigurasi multi-komponen yang dikelola agent harness
- Membutuhkan logic pemrosesan khusus yang harus dieksekusi secara presisi setiap kali, bukan interpretasi LLM yang bersifat "best effort"
- Menangani binary data, streaming, atau real-time event yang tidak dapat melewati terminal
- Contoh: browser automation (pengelolaan session Browserbase), TTS (encoding audio + pengiriman platform), vision analysis (penanganan image base64)

### Apakah Skill harus dibundel?

Bundled skills (di `skills/`) dikirim bersama setiap instalasi Hermes. Skill tersebut harus **bermanfaat secara luas bagi sebagian besar pengguna**:

- Penanganan dokumen, riset web, workflow pengembangan umum, administrasi sistem
- Digunakan secara rutin oleh berbagai jenis pengguna

Jika skill Anda resmi dan berguna tetapi tidak dibutuhkan secara universal (misalnya integrasi layanan berbayar atau dependency yang berat), tempatkan di **`optional-skills/`** — skill tetap dikirim bersama repo tetapi tidak diaktifkan secara default. Pengguna dapat menemukannya melalui `hermes skills browse` (berlabel "official") dan memasangnya dengan `hermes skills install` (tanpa peringatan pihak ketiga, dengan trust bawaan).

Jika skill Anda bersifat khusus, kontribusi komunitas, atau niche, lebih sesuai ditempatkan di **Skills Hub** — unggah ke skills registry dan bagikan di [Nous Research Discord](https://discord.gg/NousResearch). Pengguna dapat memasangnya dengan `hermes skills install`.

---

## Memory Provider: Kirim sebagai Plugin Mandiri

**Kami tidak lagi menerima memory provider baru ke repo ini.** Kumpulan provider bawaan di `plugins/memory/` (honcho, mem0, supermemory, byterover, hindsight, holographic, openviking, retaindb) sudah ditutup. Jika ingin menambahkan backend memori baru, publikasikan sebagai **repo plugin mandiri** yang dipasang pengguna ke `~/.hermes/plugins/` (atau melalui pip entry point).

Plugin memori mandiri:

- Mengimplementasikan ABC `MemoryProvider` yang sama (`agent/memory_provider.py`) — `sync_turn`, `prefetch`, `shutdown`, dan opsional `post_setup(hermes_home, config)` untuk integrasi setup wizard
- Menggunakan sistem discovery yang sama — `discover_memory_providers()` mengambilnya dari direktori plugin user/project dan pip entry point
- Terintegrasi dengan `hermes memory setup` melalui `post_setup()` — tidak perlu menyentuh core code
- Dapat mendaftarkan subcommand CLI sendiri melalui `register_cli(subparser)` di file `cli.py`
- Mendapat lifecycle hook dan config plumbing yang sama dengan provider in-tree

PR yang menambahkan direktori baru di bawah `plugins/memory/` akan ditutup dengan arahan untuk memublikasikan provider sebagai repo tersendiri. Provider in-tree yang sudah ada tetap dipertahankan; perbaikan bug untuk provider tersebut tetap diterima.

Ini bukan persoalan standar kualitas — ini keputusan coupling dan maintenance. Memory provider adalah jenis plugin yang paling umum dan tidak semuanya seharusnya berada dalam tree ini.

---

## Integrasi Produk Pihak Ketiga: Kirim sebagai Plugin Mandiri

Aturan yang sama berlaku untuk **plugin apa pun yang mengintegrasikan produk atau proyek milik pihak lain** — backend observability/metrics, konektor SaaS vendor, dashboard analytics, integrasi layanan berbayar, dan integrasi pihak ketiga serupa. **Integrasi tersebut tidak masuk ke repo ini.**

Alasannya adalah beban maintenance, bukan kualitas. Setiap produk eksternal yang diserap ke core tree menjadi tanggung jawab kami untuk terus dijaga agar berfungsi terhadap codebase yang bergerak cepat, padahal backend tersebut tidak kami miliki dan tidak kami kendalikan. Hermes sering merilis dan core bergerak cepat; coupling produk pihak ketiga ke dalamnya menciptakan beban tanpa batas bagi maintainer.

Publikasikan sebagai **repo plugin mandiri**:

- Implementasikan ABC yang relevan dan gunakan jalur discovery plugin yang tersedia (`~/.hermes/plugins/`, `.hermes/plugins/` milik project, atau pip entry point) — lihat [Build a Hermes Plugin](https://hermes-agent.nousresearch.com/docs/guides/build-a-hermes-plugin)
- Daftarkan lifecycle hook (`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`), tool (`ctx.register_tool`), dan subcommand CLI (`ctx.register_cli_command`) melalui surface yang sudah tersedia — tidak perlu perubahan core
- Jika plugin Anda membutuhkan capability yang belum diekspos framework, itu adalah feature request untuk **memperluas generic plugin surface** (hook baru atau method `ctx`) — jangan pernah membuat special-case plugin Anda di core
- Promosikan di channel `#plugins-skills-and-skins` pada [Nous Research Discord](https://discord.gg/NousResearch) agar pengguna dapat menemukan dan memasangnya

Plugin produk pihak ketiga yang dibuat dengan baik dapat lolos automated review tetapi tetap ditutup karena alasan ini — itu adalah keputusan penempatan, bukan penilaian terhadap kode. PR yang menambahkan direktori semacam itu di bawah `plugins/` akan ditutup dengan arahan untuk memublikasikannya dalam repo khusus.

---

## Setup Pengembangan

### Prasyarat

| Requirement | Catatan |
|-------------|-------|
| **Git** | Dengan extension `git-lfs` terpasang |
| **Python 3.11–3.13** | uv akan memasangnya jika belum tersedia |
| **uv** | Package manager Python yang cepat ([install](https://docs.astral.sh/uv/)) |
| **Node.js 20+** | Opsional — diperlukan untuk browser tools dan WhatsApp bridge (sesuai engines pada root `package.json`) |

### Instal dengan installer standar

Bagi sebagian besar kontributor, bootstrap pengembangan terbaik adalah jalur yang sama
dengan pengguna: jalankan installer standar, lalu bekerja di dalam repository yang di-clone.
Installer membuat venv Hermes, memasang perintah `hermes`, menandai
metode instalasi untuk `hermes update`, dan meng-clone project git lengkap ke
`$HERMES_HOME/hermes-agent` (biasanya `~/.hermes/hermes-agent`). Ini membuat
environment pengembangan menggunakan layout yang sama dengan asumsi CLI, updater,
lazy dependency installer, gateway, dan dokumentasi.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"

# Add dev/test extras on top of the standard install.
uv pip install -e ".[all,dev]"

# Optional: docs site + workspace dependencies.
npm install
```

Setelah itu, buat branch dan jalankan test dari checkout tersebut:

```bash
git checkout -b fix/description
scripts/run_tests.sh
```

### Fallback clone manual

Gunakan ini hanya jika Anda memang tidak ingin memakai managed install layout milik Hermes
(misalnya clone sementara di dalam container atau CI). Jika menginstal
dengan cara ini, pastikan Anda menjalankan entrypoint `hermes` dari venv ini; menjalankan
system `python3 -m hermes_cli.main` dapat mengambil package Python sistem lain yang tidak terkait.

Buat venv **di luar** source tree yang di-clone. Venv yang berada di dalam direktori
yang dioperasikan agen dapat terhapus oleh perintah relative-path yang dijalankan agen
terhadap checkout-nya sendiri (`rm -rf venv`, `uv venv venv`, dan sebagainya),
yang secara diam-diam menghancurkan runtime aktif di tengah sesi. Menempatkannya di luar
tree berarti tidak ada relative path dari workspace yang akan mengarah ke venv.

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Create venv with Python 3.11, OUTSIDE the source tree
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
export VIRTUAL_ENV="$HOME/.hermes/venvs/hermes-dev"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Install with all extras (messaging, cron, CLI menus, dev tools)
uv pip install -e ".[all,dev]"

# Optional: workspace / docs dependencies
npm install
```

### Konfigurasi untuk pengembangan

```bash
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env

# Add at minimum an LLM provider key:
echo "OPENROUTER_API_KEY=***" >> ~/.hermes/.env
```

### Menjalankan

```bash
# The standard installer already put `hermes` on PATH.
hermes doctor
hermes chat -q "Hello"
```

Jika menggunakan fallback clone manual, jalankan `./hermes` dari checkout atau
buat symlink ke venv clone ini secara eksplisit:

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes
```

### Menjalankan test

```bash
# Preferred — matches CI (hermetic `env -i`, per-file subprocess isolation
# via run_tests_parallel.py, worker count auto-scaled); see AGENTS.md
scripts/run_tests.sh

# Alternative (activate the venv first). The wrapper is still recommended
# for parity with GitHub Actions before you open a PR:
pytest tests/ -v
```

---

## Struktur Project

```
hermes-agent/
├── run_agent.py              # AIAgent class — core conversation loop, tool dispatch, session persistence
├── cli.py                    # HermesCLI class — interactive TUI, prompt_toolkit integration
├── model_tools.py            # Tool orchestration (thin layer over tools/registry.py)
├── toolsets.py               # Tool groupings and presets (hermes-cli, hermes-telegram, etc.)
├── hermes_state.py           # SQLite session database with FTS5 full-text search, session titles
├── batch_runner.py           # Parallel batch processing for trajectory generation
│
├── agent/                    # Agent internals (extracted modules)
│   ├── prompt_builder.py         # System prompt assembly (identity, skills, context files, memory)
│   ├── context_compressor.py     # Auto-summarization when approaching context limits
│   ├── auxiliary_client.py       # Resolves auxiliary OpenAI clients (summarization, vision)
│   ├── display.py                # KawaiiSpinner, tool progress formatting
│   ├── model_metadata.py         # Model context lengths, token estimation
│   └── trajectory.py             # Trajectory saving helpers
│
├── hermes_cli/               # CLI command implementations
│   ├── main.py                   # Entry point, argument parsing, command dispatch
│   ├── config.py                 # Config management, migration, env var definitions
│   ├── setup.py                  # Interactive setup wizard
│   ├── auth.py                   # Provider resolution, OAuth, Nous Portal
│   ├── models.py                 # OpenRouter model selection lists
│   ├── banner.py                 # Welcome banner, ASCII art
│   ├── commands.py               # Central slash command registry (CommandDef), autocomplete, gateway helpers
│   ├── callbacks.py              # Interactive callbacks (clarify, sudo, approval)
│   ├── doctor.py                 # Diagnostics
│   ├── skills_hub.py             # Skills Hub CLI + /skills slash command
│   └── skin_engine.py            # Skin/theme engine — data-driven CLI visual customization
│
├── tools/                    # Tool implementations (self-registering)
│   ├── registry.py               # Central tool registry (schemas, handlers, dispatch)
│   ├── approval.py               # Dangerous command detection + per-session approval
│   ├── terminal_tool.py          # Terminal orchestration (sudo, env lifecycle, backends)
│   ├── file_operations.py        # read_file, write_file, search, patch, etc.
│   ├── web_tools.py              # web_search, web_extract (Parallel/Firecrawl + Gemini summarization)
│   ├── vision_tools.py           # Image analysis via multimodal models
│   ├── delegate_tool.py          # Subagent spawning and parallel task execution
│   ├── code_execution_tool.py    # Sandboxed Python with RPC tool access
│   ├── session_search_tool.py    # Search past conversations with FTS5 + anchored windows
│   ├── cronjob_tools.py          # Scheduled task management
│   ├── skill_tools.py            # Skill search, load, manage
│   └── environments/             # Terminal execution backends
│       ├── base.py                   # BaseEnvironment ABC
│       ├── local.py, docker.py, ssh.py, singularity.py, modal.py, daytona.py
│
├── gateway/                  # Messaging gateway
│   ├── run.py                    # GatewayRunner — platform lifecycle, message routing, cron
│   ├── config.py                 # Platform configuration resolution
│   ├── session.py                # Session store, context prompts, reset policies
│   └── platforms/                # Platform adapters
│       ├── telegram.py, discord_adapter.py, slack.py, whatsapp.py
│
├── scripts/                  # Installer and bridge scripts
│   ├── install.sh                # Linux/macOS installer
│   ├── install.ps1               # Windows PowerShell installer
│   └── whatsapp-bridge/          # Node.js WhatsApp bridge (Baileys)
│
├── skills/                   # Bundled skills (copied to ~/.hermes/skills/ on install)
├── optional-skills/          # Official optional skills (discoverable via hub, not activated by default)
├── tests/                    # Test suite
├── website/                  # Documentation site (hermes-agent.nousresearch.com)
│
├── cli-config.yaml.example   # Example configuration (copied to ~/.hermes/config.yaml)
└── AGENTS.md                 # Development guide for AI coding assistants
```

### Konfigurasi pengguna (disimpan di `~/.hermes/`)

| Path | Tujuan |
|------|---------|
| `~/.hermes/config.yaml` | Pengaturan (model, terminal, toolsets, compression, dan sebagainya) |
| `~/.hermes/.env` | API key dan secret |
| `~/.hermes/auth.json` | Credential OAuth (Nous Portal) |
| `~/.hermes/skills/` | Semua skill aktif (bundled + hub-installed + agent-created) |
| `~/.hermes/memories/` | Memori persisten (MEMORY.md, USER.md) |
| `~/.hermes/state.db` | Database session SQLite |
| `~/.hermes/sessions/` | Routing index gateway (`sessions.json`), breadcrumb request-dump, transcript gateway `*.jsonl`, dan (opsional) snapshot JSON per-session ketika `sessions.write_json_snapshots: true` diatur. Snapshot per-session nonaktif secara default; state.db adalah canonical. |
| `~/.hermes/cron/` | Data scheduled job |
| `~/.hermes/whatsapp/session/` | Credential WhatsApp bridge |

---

## Ikhtisar Arsitektur

### Core Loop

```
User message → AIAgent._run_agent_loop()
  ├── Build system prompt (prompt_builder.py)
  ├── Build API kwargs (model, messages, tools, reasoning config)
  ├── Call LLM (OpenAI-compatible API)
  ├── If tool_calls in response:
  │     ├── Execute each tool via registry dispatch
  │     ├── Add tool results to conversation
  │     └── Loop back to LLM call
  ├── If text response:
  │     ├── Persist session to DB
  │     └── Return final_response
  └── Context compression if approaching token limit
```

### Pola Desain Utama

- **Tool self-registering**: Setiap file tool memanggil `registry.register()` pada saat import. `model_tools.py` memicu discovery dengan mengimpor semua modul tool.
- **Pengelompokan toolset**: Tool dikelompokkan ke toolset (`web`, `terminal`, `file`, `browser`, dan sebagainya) yang dapat diaktifkan/dinonaktifkan per platform.
- **Persistensi session**: Semua percakapan disimpan di SQLite (`hermes_state.py`) dengan full-text search dan judul session unik. Snapshot JSON per-session di `~/.hermes/sessions/` telah digantikan oleh SQLite store dan nonaktif secara default; aktifkan kembali dengan `sessions.write_json_snapshots: true` jika Anda memiliki tooling eksternal yang mengonsumsi file JSON secara langsung.
- **Ephemeral injection**: System prompt dan prefill message diinjeksikan pada saat API call, tidak pernah disimpan ke database atau log.
- **Abstraksi provider**: Agen bekerja dengan API apa pun yang kompatibel OpenAI. Resolusi provider terjadi saat init (Nous Portal OAuth, OpenRouter API key, atau custom endpoint).
- **Provider routing**: Saat menggunakan OpenRouter, `provider_routing` di config.yaml mengontrol pemilihan provider (sort berdasarkan throughput/latency/price, allow/ignore provider tertentu, kebijakan data retention). Nilai ini diinjeksikan sebagai `extra_body.provider` pada API request.

---

## Gaya Kode

- **PEP 8** dengan pengecualian praktis (kami tidak memberlakukan line length secara ketat)
- **Komentar**: Hanya untuk menjelaskan intent, trade-off, atau API quirk yang tidak jelas. Jangan menarasikan apa yang dilakukan kode — `# increment counter` tidak menambah informasi
- **Error handling**: Tangkap exception yang spesifik. Log dengan `logger.warning()`/`logger.error()` — gunakan `exc_info=True` untuk error tak terduga agar stack trace muncul di log
- **Lintas platform**: Jangan pernah mengasumsikan Unix. Lihat [Kompatibilitas Lintas Platform](#kompatibilitas-lintas-platform)

---

## Menambahkan Tool Baru

Sebelum menulis tool, tanyakan: [apakah seharusnya ini skill?](#apakah-harus-skill-atau-tool)

Tool melakukan self-register pada registry pusat. Setiap file tool menempatkan schema, handler, dan registration secara berdekatan:

```python
"""my_tool — Brief description of what this tool does."""

import json
from tools.registry import registry


def my_tool(param1: str, param2: int = 10, **kwargs) -> str:
    """Handler. Returns a string result (often JSON)."""
    result = do_work(param1, param2)
    return json.dumps(result)


MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this tool does and when the agent should use it.",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "What param1 is"},
                "param2": {"type": "integer", "description": "What param2 is", "default": 10},
            },
            "required": ["param1"],
        },
    },
}


def _check_requirements() -> bool:
    """Return True if this tool's dependencies are available."""
    return True


registry.register(
    name="my_tool",
    toolset="my_toolset",
    schema=MY_TOOL_SCHEMA,
    handler=lambda args, **kw: my_tool(**args, **kw),
    check_fn=_check_requirements,
)
```

**Masukkan ke toolset (wajib):** Tool bawaan ditemukan secara otomatis: setiap
file `tools/*.py` yang mengandung pemanggilan top-level `registry.register(...)` akan
diimpor oleh `discover_builtin_tools()` di `tools/registry.py` saat `model_tools`
dimuat. Tidak ada daftar import manual di `model_tools.py` yang perlu dipelihara.

Anda tetap harus menambahkan nama tool ke daftar yang sesuai di `toolsets.py`
(misalnya `_HERMES_CORE_TOOLS` atau toolset khusus); jika tidak, tool akan
terdaftar tetapi tidak pernah diekspos kepada agen. Jika Anda memperkenalkan toolset baru,
tambahkan di `toolsets.py` dan hubungkan ke preset platform yang relevan.

Lihat `AGENTS.md` (bagian **Adding New Tools**) untuk jalur profile-aware dan
panduan plugin vs core.

---

## Menambahkan Skill

Bundled skills berada di `skills/` dan diorganisasikan menurut kategori. Skill resmi opsional menggunakan struktur yang sama di `optional-skills/`:

```
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md              # Required: main instructions
│       └── scripts/              # Optional: helper scripts
│           └── search_arxiv.py
├── productivity/
│   └── ocr-and-documents/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
└── ...
```

### Format SKILL.md

```markdown
---
name: my-skill
description: Brief description (shown in skill search results)
version: 1.0.0
author: Your Name
license: MIT
platforms: [macos, linux]          # Optional — restrict to specific OS platforms
                                   #   Valid: macos, linux, windows
                                   #   Omit to load on all platforms (default)
required_environment_variables:    # Optional — secure setup-on-load metadata
  - name: MY_API_KEY
    prompt: API key
    help: Where to get it
    required_for: full functionality
prerequisites:                     # Optional legacy runtime requirements
  env_vars: [MY_API_KEY]           #   Backward-compatible alias for required env vars
  commands: [curl, jq]             #   Advisory only; does not hide the skill
metadata:
  hermes:
    tags: [Category, Subcategory, Keywords]
    related_skills: [other-skill-name]
    fallback_for_toolsets: [web]       # Optional — show only when toolset is unavailable
    requires_toolsets: [terminal]      # Optional — show only when toolset is available
---

# Skill Title

Brief intro.

## When to Use
Trigger conditions — when should the agent load this skill?

## Prerequisites
Env vars, install steps, MCP setup, API key sourcing.

## How to Run
Canonical invocation through the `terminal` tool.

## Quick Reference
Table of common commands or API calls.

## Procedure
Step-by-step instructions the agent follows.

## Pitfalls
Known failure modes and how to handle them.

## Verification
How the agent confirms it worked.
```

### Skill khusus platform

Skill dapat mendeklarasikan platform OS yang didukung melalui field frontmatter `platforms`. Skill dengan field ini otomatis disembunyikan dari system prompt, `skills_list()`, dan slash command pada platform yang tidak kompatibel.

```yaml
platforms: [macos]            # macOS only (e.g., iMessage, Apple Reminders)
platforms: [macos, linux]     # macOS and Linux
platforms: [windows]          # Windows only
```

Jika field tidak dicantumkan atau kosong, skill dimuat pada semua platform (backward compatible). Lihat `skills/apple/` untuk contoh skill macOS-only.

### Aktivasi skill bersyarat

Skill dapat mendeklarasikan kondisi yang menentukan kapan ia muncul dalam system prompt berdasarkan tool dan toolset yang tersedia dalam session saat ini. Ini terutama digunakan untuk **fallback skills** — alternatif yang hanya perlu ditampilkan saat tool utama tidak tersedia.

Empat field didukung di bawah `metadata.hermes`:

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # Show ONLY when these toolsets are unavailable
    requires_toolsets: [terminal]     # Show ONLY when these toolsets are available
    fallback_for_tools: [web_search]  # Show ONLY when these specific tools are unavailable
    requires_tools: [terminal]        # Show ONLY when these specific tools are available
```

**Semantik:**
- `fallback_for_*`: Skill merupakan backup. Skill **disembunyikan** saat tool/toolset yang tercantum tersedia, dan **ditampilkan** saat tidak tersedia. Gunakan untuk alternatif gratis terhadap tool premium.
- `requires_*`: Skill membutuhkan tool tertentu untuk berfungsi. Skill **disembunyikan** saat tool/toolset yang tercantum tidak tersedia. Gunakan untuk skill yang bergantung pada capability tertentu (misalnya skill yang hanya masuk akal dengan akses terminal).
- Jika keduanya ditentukan, kedua kondisi harus terpenuhi agar skill muncul.
- Jika keduanya tidak ditentukan, skill selalu ditampilkan (backward compatible).

**Contoh:**

```yaml
# DuckDuckGo search — shown when Firecrawl (web toolset) is unavailable
metadata:
  hermes:
    fallback_for_toolsets: [web]

# Smart home skill — only useful when terminal is available
metadata:
  hermes:
    requires_toolsets: [terminal]

# Local browser fallback — shown when Browserbase is unavailable
metadata:
  hermes:
    fallback_for_toolsets: [browser]
```

Filtering terjadi saat prompt dibangun di `agent/prompt_builder.py`. Fungsi `build_skills_system_prompt()` menerima kumpulan tool dan toolset yang tersedia dari agen dan menggunakan `_skill_should_show()` untuk mengevaluasi kondisi setiap skill.

### Metadata setup skill

Skill dapat mendeklarasikan metadata secure setup-on-load melalui field frontmatter `required_environment_variables`. Nilai yang belum tersedia tidak menyembunyikan skill dari discovery; kondisi tersebut memicu secure prompt khusus CLI ketika skill benar-benar dimuat.

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

Pengguna dapat melewati setup dan tetap memuat skill. Hermes hanya mengekspos metadata (`stored_as`, `skipped`, `validated`) kepada model — tidak pernah nilai secret.

Legacy `prerequisites.env_vars` tetap didukung dan dinormalisasi ke representasi baru.

```yaml
prerequisites:
  env_vars: [TENOR_API_KEY]       # Legacy alias for required_environment_variables
  commands: [curl, jq]            # Advisory CLI checks
```

Session gateway dan messaging tidak pernah mengumpulkan secret in-band; session tersebut menginstruksikan pengguna untuk menjalankan `hermes setup` atau memperbarui `~/.hermes/.env` secara lokal.

**Kapan mendeklarasikan environment variable wajib:**
- Skill menggunakan API key atau token yang seharusnya dikumpulkan secara aman saat load
- Skill tetap dapat berguna jika pengguna melewati setup, tetapi mungkin mengalami graceful degradation

**Kapan mendeklarasikan prasyarat command:**
- Skill bergantung pada tool CLI yang mungkin belum terpasang (misalnya `himalaya`, `openhue`, `ddgs`)
- Perlakukan pemeriksaan command sebagai panduan, bukan penyembunyian saat discovery

Lihat `skills/gifs/gif-search/` dan `skills/email/himalaya/` untuk contoh.

### Standar penulisan skill (HARDLINE)

Setiap skill baru atau yang dimodernisasi — bundled, optional, maupun contributed — harus memenuhi standar ini sebelum merge. Reviewer akan menolak PR yang melanggarnya.

1. **`description` ≤ 60 karakter, satu kalimat, diakhiri titik.** Deskripsi panjang membebani UI daftar skill dan mengencerkan perhatian model saat banyak skill dimuat. Nyatakan capability, bukan implementasi. Jangan gunakan kata pemasaran ("powerful", "comprehensive", "seamless", "advanced"). Jangan mengulang nama skill. Verifikasi dengan:
   ```python
   import re, pathlib
   m = re.search(r'^description: (.*)$',
                 pathlib.Path('skills/<cat>/<name>/SKILL.md').read_text(),
                 re.MULTILINE)
   assert len(m.group(1)) <= 60, len(m.group(1))
   ```

   Baik: `Search arXiv papers by keyword, author, category, or ID.`
   Buruk: `A powerful and comprehensive skill that allows the agent to search arXiv for relevant academic papers using various criteria including keywords, authors, and categories.`

2. **Tool yang dirujuk dalam prose SKILL.md harus merupakan tool native Hermes atau server MCP yang secara eksplisit diharapkan skill.** Ketika skill membutuhkan capability, arahkan ke tool yang benar berdasarkan nama dalam backtick: `terminal`, `web_extract`, `web_search`, `read_file`, `write_file`, `patch`, `search_files`, `vision_analyze`, `browser_navigate`, `delegate_task`, `image_generate`, `text_to_speech`, `cronjob`, `memory`, `skill_view`, `todo`, `execute_code`.

   Jangan menyebut shell utility yang sudah dibungkus agen:

   | Jangan katakan | Katakan |
   |---|---|
   | `grep`, `rg` | `search_files` |
   | `cat`, `head`, `tail` | `read_file` |
   | `sed`, `awk` | `patch` |
   | `find`, `ls` | `search_files` (dengan `target='files'`) |
   | `curl` untuk ekstraksi konten | `web_extract` |
   | `echo > file`, `cat <<EOF` | `write_file` |

   Jika skill bergantung pada server MCP, sebutkan server MCP dan dokumentasikan setup-nya di `## Prerequisites`. CLI pihak ketiga (misalnya `ffmpeg`, `gh`, SDK tertentu) boleh dipanggil dari dalam file skrip, tetapi prose sebaiknya membingkai interaksi sebagai "invoke through the `terminal` tool", bukan sebagai sesi shell manual.

3. **Gating `platforms:` diaudit terhadap import skrip yang sebenarnya.** Skill yang menggunakan primitive khusus POSIX (`fcntl`, `termios`, `os.setsid`, `os.kill(pid, 0)` untuk liveness, `/proc`, hardcoded `/tmp` path, `signal.SIGKILL`, bash heredoc, `osascript`, `apt`, `systemctl`) harus mendeklarasikan platform yang didukung melalui frontmatter `platforms:`. Postur default adalah memperbaikinya agar cross-platform terlebih dahulu — `tempfile.gettempdir()`, `pathlib.Path`, `psutil.pid_exists()`, filtering tingkat Python alih-alih `grep`. Batasi ke platform yang lebih sempit hanya ketika dependency memang terikat platform (misalnya `osascript` khusus macOS, `/proc` khusus Linux).

4. **`author` memberi kredit kepada kontributor manusia terlebih dahulu.** Untuk kontribusi eksternal, nama asli kontributor + handle GitHub diletakkan pertama (`Jane Doe (jane-doe)`); "Hermes Agent" adalah kolaborator sekunder. Jika commit kontributor menampilkan "Hermes Agent" sebagai author karena mereka menggunakan Hermes untuk membuat draft skill, ganti dengan nama mereka sebenarnya — beri kredit kepada manusia, bukan tool.

5. **Body SKILL.md menggunakan urutan section modern.** Judul `# <Skill> Skill`, intro 2-3 kalimat yang menyatakan apa yang dilakukan dan tidak dilakukan, kemudian:
   - `## When to Use` — kondisi trigger
   - `## Prerequisites` — env vars, langkah instalasi, setup MCP, sumber API key
   - `## How to Run` — invocation canonical melalui tool `terminal`
   - `## Quick Reference` — referensi command/API yang datar
   - `## Procedure` — langkah bernomor dengan command yang dapat di-copy-paste
   - `## Pitfalls` — limit yang diketahui, rate limit, hal yang terlihat rusak tetapi sebenarnya tidak
   - `## Verification` — satu command yang membuktikan skill bekerja

   Target sekitar 200 baris untuk skill kompleks dan 100 baris untuk skill sederhana. Potong intro yang berlebihan, prose pemasaran, dan pengulangan penjelasan env var yang sudah didokumentasikan di `## Prerequisites`.

6. **Skrip masuk ke `scripts/`, referensi ke `references/`, template ke `templates/`.** Jangan mengharapkan model menulis parser, XML walker, atau logic non-trivial secara inline setiap kali dipanggil — kirim helper script. Referensikan script dari SKILL.md dengan path relatif terhadap direktori skill.

7. **Test berada di `tests/skills/test_<skill>_skill.py`** dan hanya menggunakan stdlib + pytest + `unittest.mock`. Tidak ada live network call. Jalankan melalui `scripts/run_tests.sh tests/skills/test_<skill>_skill.py -q`. Harus lolos pada hermetic CI env (tidak ada API key yang bocor). Gunakan `monkeypatch` dan `tmp_path` untuk dependency environment variable atau filesystem.

8. **Tambahan `.env.example` diisolasi dalam block yang dibatasi jelas.** Jangan menyentuh file di sekitarnya — versi `.env.example` dari kontributor biasanya stale dan perubahan di luar block milik skill akan dibuang saat salvage. Beri komentar pada semua value dengan `#` (ini dokumentasi, bukan live config).

### Panduan skill

- **Tidak ada dependency eksternal kecuali benar-benar diperlukan.** Utamakan stdlib Python, curl, dan tool Hermes yang sudah ada (`web_extract`, `terminal`, `read_file`).
- **Progressive disclosure.** Letakkan workflow paling umum terlebih dahulu. Edge case dan penggunaan advanced diletakkan di bagian bawah.
- **Sertakan helper script** untuk parsing XML/JSON atau logic kompleks — jangan berharap LLM menulis parser inline setiap saat.
- **Uji.** Jalankan `hermes --toolsets skills -q "Use the X skill to do Y"` dan verifikasi bahwa agen mengikuti instruksi dengan benar.

---

## Menambahkan Skin / Theme

Hermes menggunakan sistem skin data-driven — tidak perlu perubahan kode untuk menambahkan skin baru.

**Opsi A: Skin pengguna (file YAML)**

Buat `~/.hermes/skins/<name>.yaml`:

```yaml
name: mytheme
description: Short description of the theme

colors:
  banner_border: "#HEX"     # Panel border color
  banner_title: "#HEX"      # Panel title color
  banner_accent: "#HEX"     # Section header color
  banner_dim: "#HEX"        # Muted/dim text color
  banner_text: "#HEX"       # Body text color
  response_border: "#HEX"   # Response box border

spinner:
  waiting_faces: ["(⚔)", "(⛨)"]
  thinking_faces: ["(⚔)", "(⌁)"]
  thinking_verbs: ["forging", "plotting"]
  wings:                     # Optional left/right decorations
    - ["⟪⚔", "⚔⟫"]

branding:
  agent_name: "My Agent"
  welcome: "Welcome message"
  response_label: " ⚔ Agent "
  prompt_symbol: "⚔"

tool_prefix: "╎"             # Tool output line prefix
```

Semua field bersifat opsional — value yang tidak diberikan mewarisi skin default.

**Opsi B: Skin bawaan**

Tambahkan ke dict `_BUILTIN_SKINS` di `hermes_cli/skin_engine.py`. Gunakan schema yang sama seperti di atas tetapi sebagai dict Python. Built-in skin dikirim bersama package dan selalu tersedia.

**Mengaktifkan:**
- CLI: `/skin mytheme` atau atur `display.skin: mytheme` di config.yaml
- Config: `display: { skin: mytheme }`

Lihat `hermes_cli/skin_engine.py` untuk schema lengkap dan skin yang sudah ada sebagai contoh.

---

## Kompatibilitas Lintas Platform

Hermes berjalan pada Linux, macOS, dan Windows native (serta WSL2). Saat menulis kode
yang menyentuh OS, asumsikan *platform apa pun* dapat mencapai code path Anda.

> **Sebelum membuat PR:** jalankan `scripts/check-windows-footguns.py` untuk menangkap
> pola umum yang tidak aman di Windows dalam diff Anda. Pemeriksaan ini berbasis grep dan ringan;
> CI juga menjalankannya pada setiap PR.

### Aturan kritis

1. **Jangan pernah memanggil `os.kill(pid, 0)` untuk pemeriksaan liveness.** `os.kill(pid, 0)`
   adalah idiom POSIX standar untuk memeriksa "apakah PID ini hidup" — signal 0
   merupakan permission check no-op. **Pada Windows ini BUKAN no-op.** `os.kill`
   milik Python pada Windows memetakan `sig=0` ke `CTRL_C_EVENT` (keduanya bertabrakan pada
   nilai integer 0) dan merutekannya melalui `GenerateConsoleCtrlEvent(0, pid)`,
   yang menyiarkan Ctrl+C ke **seluruh console process group** yang berisi
   target PID. "Probe apakah hidup" secara diam-diam menjadi "bunuh target dan
   sering kali proses lain yang tidak terkait tetapi berbagi console." Lihat [bpo-14484](https://bugs.python.org/issue14484)
   (terbuka sejak 2012 — tidak akan diperbaiki karena alasan compat).

   **Direkomendasikan:** gunakan `psutil` (core dependency — selalu tersedia):

   ```python
   import psutil
   if psutil.pid_exists(pid):
       # process is alive — safe on every platform
       ...
   ```

   Jika secara khusus membutuhkan wrapper Hermes (wrapper ini memiliki stdlib fallback
   untuk import pada fase scaffold sebelum pip install selesai), gunakan
   `gateway.status._pid_exists(pid)`. Fungsi ini memanggil `psutil.pid_exists` terlebih dahulu
   dan fallback ke kombinasi `OpenProcess + WaitForSingleObject`
   yang dibuat manual pada Windows hanya jika psutil tidak tersedia.

   Audit grep untuk callsite baru: `rg "os\.kill\([^,]+,\s*0\s*\)"`. Setiap hit
   pada kode non-test secara presumptif merupakan bug silent-kill Windows.

2. **Gunakan `shutil.which()` sebelum menjalankan shell command — jangan mengasumsikan Windows memiliki
   tool yang dimiliki Linux.** `wmic` dihapus pada Windows 10 21H1 dan setelahnya. `ps`,
   `kill`, `grep`, `awk`, `fuser`, `lsof`, `pgrep`, dan sebagian besar tool CLI POSIX
   tidak tersedia di Windows. Uji ketersediaan dengan
   `shutil.which("tool")` dan fallback ke ekuivalen native Windows —
   biasanya PowerShell melalui `subprocess.run(["powershell", "-NoProfile",
   "-Command", ...])`.

   Untuk enumerasi proses: `Get-CimInstance Win32_Process` milik PowerShell adalah
   pengganti modern untuk `wmic process`. Lihat
   `hermes_cli/gateway.py::_scan_gateway_pids` untuk polanya.
   ```

3. **Encoding file.** Windows dapat menyimpan file `.env` dalam `cp1252`. Selalu
   tangani error encoding:
   ```python
   try:
       load_dotenv(env_path)
   except UnicodeDecodeError:
       load_dotenv(env_path, encoding="latin-1")
   ```
   File konfigurasi (`config.yaml`) dapat disimpan dengan UTF-8 BOM oleh Notepad dan
   editor GUI serupa — gunakan `encoding="utf-8-sig"` saat membaca file yang
   mungkin disentuh editor GUI Windows.

4. **Pengelolaan process.** `os.setsid()`, `os.killpg()`, `os.fork()`,
   `os.getuid()`, dan penanganan signal POSIX berbeda pada Windows. Guard dengan
   `platform.system()`, `sys.platform`, atau `hasattr(os, "setsid")`:
   ```python
   if platform.system() != "Windows":
       kwargs["preexec_fn"] = os.setsid
   else:
       kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
   ```

   **Direkomendasikan:** untuk membunuh process DAN child-nya (apa yang dilakukan `os.killpg`
   pada POSIX), gunakan `psutil` — berfungsi pada setiap platform:
   ```python
   import psutil
   try:
       parent = psutil.Process(pid)
       # Kill children first (leaf-up), then the parent.
       for child in parent.children(recursive=True):
           child.kill()
       parent.kill()
   except psutil.NoSuchProcess:
       pass
   ```

5. **Signal yang tidak tersedia di Windows: `SIGALRM`, `SIGCHLD`, `SIGHUP`,
   `SIGUSR1`, `SIGUSR2`, `SIGPIPE`, `SIGQUIT`, `SIGKILL`.** Module `signal`
   Python menimbulkan `AttributeError` saat import jika Anda mereferensikannya
   di Windows. Gunakan `getattr(signal, "SIGKILL", signal.SIGTERM)` atau
   letakkan seluruh block di belakang platform check. `loop.add_signal_handler`
   menimbulkan `NotImplementedError` pada Windows — selalu tangkap error tersebut.

6. **Path separator.** Gunakan `pathlib.Path` alih-alih string concatenation
   dengan `/`. Forward slash bekerja hampir di semua tempat di Windows, tetapi
   `subprocess.run(["cmd.exe", "/c", ...])` dan konteks shell lainnya dapat
   membutuhkan backslash — konversikan dengan `str(path)` pada boundary subprocess,
   bukan di dalam logic Python.

7. **Symlink membutuhkan privilege elevated pada Windows** (kecuali Developer Mode
   aktif). Test yang membuat symlink membutuhkan `@pytest.mark.skipif(sys.platform ==
   "win32", reason="Symlinks require elevated privileges on Windows")`.

8. **File mode POSIX (0o600, 0o644, dan sebagainya) TIDAK ditegakkan pada NTFS** secara
   default. Test yang melakukan assert terhadap `stat().st_mode & 0o777` harus di-skip pada
   Windows — konsepnya tidak setara. Gunakan ACL (`icacls`, `pywin32`)
   untuk perlindungan secret-file Windows bila diperlukan.

9. **Background daemon detached pada Windows membutuhkan `pythonw.exe`, BUKAN
    `python.exe`.** `python.exe` selalu mengalokasikan atau attach ke console,
    yang membuatnya rentan terhadap broadcast `CTRL_C_EVENT` dari sibling process.
    `pythonw.exe` adalah varian tanpa console. Kombinasikan dengan
    `CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
    CREATE_BREAKAWAY_FROM_JOB` di `subprocess.Popen(creationflags=...)`.
    Lihat `hermes_cli/gateway_windows.py::_spawn_detached` untuk implementasi referensi.

10. **`subprocess.Popen` dengan shim `.cmd` atau `.bat` membutuhkan `shutil.which`
    untuk resolve.** Meneruskan `"agent-browser"` ke `Popen` pada Windows menemukan
    shim POSIX tanpa extension di `node_modules/.bin/`, yang tidak dapat dieksekusi
    `CreateProcessW` — Anda akan mendapat `WinError 193 "not a valid
    Win32 application"`. Gunakan `shutil.which("agent-browser", path=local_bin)`
    yang menghormati PATHEXT dan memilih varian `.CMD` pada Windows.

11. **Jangan menggunakan shell shebang sebagai cara menjalankan Python.** `#!/usr/bin/env
    python` hanya bekerja ketika file dieksekusi melalui shell Unix.
    `subprocess.run(["./myscript.py"])` gagal di Windows sekalipun file
    memiliki shebang. Selalu invoke Python secara eksplisit:
    `[sys.executable, "myscript.py"]`.

12. **Perintah shell dalam installer.** Jika mengubah `scripts/install.sh`,
    buat perubahan ekuivalen pada `scripts/install.ps1`. Kedua script
    adalah contoh canonical bahwa "berfungsi di Linux tidak berarti berfungsi di
    Windows" dan telah beberapa kali drift — pertahankan keduanya lockstep.

13. **Known path yang dialihkan OneDrive di Windows:** Desktop,
    Documents, Pictures, Videos. Path "sebenarnya" ketika OneDrive Backup
    aktif adalah `%USERPROFILE%\OneDrive\Desktop` (dan seterusnya), BUKAN
    `%USERPROFILE%\Desktop` (yang tetap ada sebagai shell kosong). Resolve
    lokasi sebenarnya melalui `ctypes` + `SHGetKnownFolderPath` atau dengan membaca
    registry key `Shell Folders` — jangan pernah mengasumsikan `~/Desktop`.

14. **CRLF vs LF pada script yang dihasilkan.** Windows `cmd.exe` dan `schtasks`
    melakukan parsing per baris; line ending campuran atau LF-only dapat merusak file
    `.cmd` / `.bat` multi-baris. Gunakan `open(path, "w", encoding="utf-8",
    newline="\r\n")` — atau `open(path, "wb")` + explicit bytes — ketika
    menghasilkan script yang akan dieksekusi Windows.

15. **Dua skema quoting berbeda dalam satu command line.** `subprocess.run
    (["schtasks", "/TR", some_cmd])` → schtasks sendiri melakukan parsing `/TR`, DAN
    string `some_cmd` diparse ulang oleh `cmd.exe` ketika task berjalan.
    Parser berbeda, aturan escape berbeda. Gunakan dua helper quoting terpisah
    dan jangan pernah mencampurkannya. Lihat `hermes_cli/gateway_windows.py::
    _quote_cmd_script_arg` dan `_quote_schtasks_arg` sebagai pasangan referensi.

### Pengujian lintas platform

Test yang menguji perilaku spesifik platform harus berjalan pada platform target masing-masing.

```python
@pytest.mark.linux_only
@pytest.mark.macos_only
@pytest.mark.windows_only
```
Hindari monkeypatch `sys.platform` kecuali benar-benar diperlukan, tetapi jika melakukannya, patch juga `platform.system()` / `platform.release()` / `platform.mac_ver()`.
Symlink, permission 0o600, SIGALRM, os.setsid/fork semuanya khusus unix.

---

## Pertimbangan Keamanan

Hermes memiliki akses terminal. Keamanan penting.

### Perlindungan yang sudah ada

| Layer | Implementasi |
|-------|---------------|
| **Sudo password piping** | Menggunakan `shlex.quote()` untuk mencegah shell injection |
| **Dangerous command detection** | Pola regex di `tools/approval.py` dengan flow persetujuan pengguna |
| **Cron prompt injection** | Scanner di `tools/cronjob_tools.py` memblokir pola instruction-override |
| **Write deny list** | Protected path (`~/.ssh/authorized_keys`, `/etc/shadow`) di-resolve melalui `os.path.realpath()` untuk mencegah bypass symlink |
| **Skills guard** | Security scanner untuk skill yang dipasang dari hub (`tools/skills_guard.py`) |
| **Code execution sandbox** | Child process `execute_code` berjalan dengan API key dihapus dari environment |
| **Container hardening** | Docker: semua capability di-drop, tanpa privilege escalation, PID limit, tmpfs dengan batas ukuran |

### Saat berkontribusi pada kode sensitif keamanan

- **Selalu gunakan `shlex.quote()`** ketika menginterpolasi input pengguna ke shell command
- **Resolve symlink** dengan `os.path.realpath()` sebelum access-control check berbasis path
- **Jangan log secret.** API key, token, dan password tidak boleh muncul di output log
- **Tangkap exception secara luas** di sekitar eksekusi tool agar satu failure tidak membuat agent loop crash
- **Uji pada semua platform** jika perubahan menyentuh file path, process management, atau shell command

Jika PR Anda berdampak pada keamanan, nyatakan secara eksplisit dalam deskripsi.

### Kebijakan pinning dependency (supply chain hardening)

Setelah [kompromi supply chain litellm](https://github.com/BerriAI/litellm/issues/24512) pada Maret 2026 dan [kampanye worm Mini Shai-Hulud](https://socket.dev/blog/tanstack-npm-packages-compromised-mini-shai-hulud-supply-chain-attack) pada Mei 2026, semua dependency harus mengikuti aturan berikut:

| Tipe source | Perlakuan wajib | Alasan |
|---|---|---|
| **Package PyPI** | `>=floor,<next_major` | Versi PyPI immutable setelah dipublikasikan, tetapi versi baru dapat ditambahkan ke range Anda. Ceiling `<next_major` mencegah instalasi 1.x naik ke 2.0.0 berbahaya. |
| **Git URL** (atroposlib, tinker, yc-bench, Baileys) | Full commit SHA | Branch dan tag adalah mutable ref; SHA bersifat content-addressed. |
| **GitHub Actions** | Full commit SHA + version comment | Tag action adalah mutable ref (misalnya tj-actions/changed-files Maret 2025). Pin sebagai `uses: owner/action@<sha>  # vX.Y.Z` |
| **CI-only pip installs** | `==exact` | Build CI hermetic; churn dapat diterima. |

**Setiap dependency PyPI baru dalam PR harus memiliki upper bound `<next_major`.** PR yang menambahkan spec `>=X.Y.Z` tanpa batas atas akan ditolak reviewer. Workflow CI `supply-chain-audit.yml` juga menandai perubahan dependency manifest untuk review manual.

**Cara menentukan ceiling:**
- Jika package berada pada versi `1.x.y`, gunakan `<2`.
- Jika package berada pada versi `0.x.y` (pre-1.0), gunakan `<0.(current_minor + 2)` — misalnya jika saat ini `0.29.x`, gunakan `<0.32`. Ini memberi sekitar 2 minor version ruang sambil menjaga window cukup kecil sehingga versi hostile-takeover kecil kemungkinan masuk ke dalamnya.
- Pengecualian: package dengan API sangat stabil (misalnya `aiohttp-socks`) dapat menggunakan `<1` atas pertimbangan reviewer.

**Contoh:**
```toml
# ✅ Correct — post-1.0
"openai>=2.21.0,<3"
"pydantic>=2.12.5,<3"

# ✅ Correct — pre-1.0 (tight minor window)
"asyncpg>=0.29,<0.32"
"aiosqlite>=0.20,<0.23"
"hindsight-client>=0.4.22,<0.5"

# ❌ Rejected — no upper bound
"some-package>=1.2.3"

# ❌ Rejected — too tight (blocks legitimate patches)
"some-package==1.2.3"

# ❌ Rejected — too loose for pre-1.0 (allows 80 minor versions)
"some-package>=0.20,<1"
```

**PR referensi:** #2796 (penghapusan litellm), #2810 (pass upper bounds), #9801 (SHA pinning + supply-chain-audit CI).

---

## Proses Pull Request

### Penamaan branch

```
fix/description        # Bug fixes
feat/description       # New features
docs/description       # Documentation
test/description       # Tests
refactor/description   # Code restructuring
```

### Sebelum mengirim

1. **Jalankan test**: `scripts/run_tests.sh` (direkomendasikan; sama dengan CI) atau `pytest tests/ -v` dengan project venv aktif
2. **Uji manual**: Jalankan `hermes` dan exercise code path yang Anda ubah
3. **Periksa dampak lintas platform**: Jika menyentuh file I/O, process management, atau terminal handling, pertimbangkan macOS, Linux, dan WSL2
4. **Jaga PR tetap fokus**: Satu perubahan logis per PR. Jangan mencampur bug fix dengan refactor dan fitur baru.

### Deskripsi PR

Sertakan:
- **Apa** yang berubah dan **mengapa**
- **Cara mengujinya** (langkah reproduksi untuk bug, contoh penggunaan untuk fitur)
- **Platform yang diuji**
- Referensikan issue terkait

### Pesan commit

Kami menggunakan [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>
```

| Type | Digunakan untuk |
|------|---------|
| `fix` | Perbaikan bug |
| `feat` | Fitur baru |
| `docs` | Dokumentasi |
| `test` | Test |
| `refactor` | Restrukturisasi kode (tanpa perubahan perilaku) |
| `chore` | Build, CI, update dependency |

Scope: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`, dan sebagainya.

Contoh:
```
fix(cli): prevent crash in save_config_value when model is a string
feat(gateway): add WhatsApp multi-user session isolation
fix(security): prevent shell injection in sudo password piping
test(tools): add unit tests for file_operations
```

---

## Melaporkan Issue

- Gunakan [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
- Sertakan: OS, versi Python, versi Hermes (`hermes --version`), full error traceback
- Sertakan langkah reproduksi
- Periksa issue yang sudah ada sebelum membuat duplikat
- Untuk kerentanan keamanan, harap laporkan secara privat

---

## Komunitas

- **Discord**: [discord.gg/NousResearch](https://discord.gg/NousResearch) — untuk pertanyaan, showcase project, dan berbagi skill
- **GitHub Discussions**: Untuk proposal desain dan diskusi arsitektur
- **Skills Hub**: Unggah skill khusus ke registry dan bagikan dengan komunitas

---

## Lisensi

Dengan berkontribusi, Anda menyetujui bahwa kontribusi Anda akan dilisensikan berdasarkan [MIT License](LICENSE).
# Fitur Opsional (Opt-in Extension) dan Arahan Implementasi untuk Hermes Agent

> **Penyusun:** Claude Opus 4.6 (Thinking)  
> **Tanggal:** 29 Mei 2026  
> **Revisi:** v2 — reklasifikasi "tidak diadopsi" → "fitur opsional" sesuai prinsip open-source  
> **Konteks:** Fitur-fitur yang sebelumnya dianggap "ditolak" kini dirancang sebagai ekstensi modular (opt-in) karena proyek bersifat open-source dan transparansi kode menjamin auditabilitas.

---

## 1. Filosofi: Mengapa Opt-in, Bukan Ditolak

Hermes Agent adalah proyek **open-source**. Artinya:

1. **Kode terbuka → auditabilitas penuh.** Kekhawatiran "kepercayaan" (*trust*) terhadap fitur seperti MITM Proxy tidak relevan karena siapa saja bisa membaca, mengaudit, dan memodifikasi kode sumber.
2. **Pengguna beragam.** Sebagian menjalankan Hermes sebagai CLI personal, sebagian ingin menjadikannya gateway kantor, sebagian ingin menggunakannya untuk intercept traffic IDE. Melarang fitur tertentu adalah membatasi potensi adopsi.
3. **Plugin architecture = extensibility.** Fitur berat dipisahkan dari core, sehingga pengguna yang tidak membutuhkannya tidak terbebani.

Prinsip desain baru:

```
┌──────────────────────────────────────────────────────────┐
│                    HERMES CORE (ringan)                   │
│  Routing · Circuit Breaker · Telemetry · SQLite Cache    │
│  → Selalu aktif, zero-config, bebas dependensi berat     │
├──────────────────────────────────────────────────────────┤
│                EXTENSIONS (opt-in, modular)               │
│  MITM · Dashboard · Tunnel · Redis · OAuth Device Flow   │
│  → Default OFF, aktif via config, dependensi opsional    │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Daftar Lengkap Fitur Opsional

### 2.1. MITM Proxy Lokal

| Aspek | Detail |
|-------|--------|
| **Sumber Referensi** | 9router `src/mitm/manager.js` (852 baris), `src/mitm/server.js` (15.6 KB) |
| **Fungsi** | Intercept traffic HTTPS dari IDE (VS Code Copilot, Cursor, Kiro, dll.) → reroute ke Hermes router lokal |
| **Mengapa Opsional** | Membutuhkan: (1) elevated privileges (`sudo`) untuk install Root CA cert, (2) modifikasi `/etc/hosts` untuk DNS hijack, (3) penanganan lintas-platform (Debian, Arch, macOS, Windows). Fitur ini powerful tapi berdampak pada konfigurasi OS global pengguna. |
| **Nilai bagi Power User** | Mengubah Hermes menjadi "AI traffic controller" lokal — semua request dari IDE apapun melewati Hermes untuk routing cerdas, cost tracking, dan caching tanpa perlu mengubah konfigurasi setiap IDE satu per satu |
| **Dependensi Opsional** | `cryptography` (Python), `mitmproxy` atau implementasi custom TLS |

**Cara kerja (dari 9router):**
```
1. User jalankan: hermes extension enable mitm
2. Generate Root CA cert → prompt user untuk install ke system trust store
3. Hijack DNS (hosts file) untuk domain provider target
4. Spawn HTTPS listener di port 443
5. Intercept request → forward ke hermes router lokal (port 20128)
6. Router pilih provider + apply transformasi → response kembali ke IDE
```

---

### 2.2. Web Dashboard UI

| Aspek | Detail |
|-------|--------|
| **Sumber Referensi** | 9router (Next.js dashboard), LiteLLM (Admin UI) |
| **Fungsi** | Visualisasi real-time: usage per provider, cost breakdown, circuit breaker status, latency chart, cache hit rate |
| **Mengapa Opsional** | Bundle frontend (HTML/JS/CSS atau React) menambah ukuran instalasi dan RAM. Mayoritas pengguna CLI cukup dengan `hermes status` di terminal. |
| **Nilai bagi Power User** | Analisis tren penggunaan jangka panjang, perbandingan cost antar provider dalam grafik visual, monitoring health dashboard untuk tim |
| **Dependensi Opsional** | `flask` atau `fastapi` (server), aset static pre-built (tanpa Node.js runtime) |

**Strategi implementasi ringan:**
- Gunakan Python web framework ringan (Flask/FastAPI) untuk backend
- Aset frontend di-bundle sebagai static HTML/JS (pre-built, bukan runtime compile)
- Endpoint REST API yang melayani data dari SQLite `provider_usage.db`
- Aktivasi: `hermes dashboard` → buka browser di `http://127.0.0.1:8080`

---

### 2.3. Tunnel Support (Cloudflare / Tailscale)

| Aspek | Detail |
|-------|--------|
| **Sumber Referensi** | 9router (`src/lib/tunnel/`), OmniRoute (Cloudflare + Tailscale integration) |
| **Fungsi** | Ekspos endpoint router lokal Hermes ke internet agar bisa diakses dari mesin remote lain |
| **Mengapa Opsional** | Membuka port lokal ke internet = potensi eksposur keamanan jika tidak dikonfigurasi dengan benar. Hanya relevan untuk skenario kolaborasi tim. |
| **Nilai bagi Power User** | (1) Tim developer berbagi satu instance Hermes gateway, (2) Akses Hermes dari laptop/mesin berbeda via Tailscale mesh, (3) Demo atau testing dari remote |
| **Dependensi Opsional** | Binary `cloudflared` atau `tailscale` (external, bukan pip) |

**Pendekatan implementasi:**
- Hermes hanya menyediakan wrapper CLI, BUKAN bundle binary tunnel
- Deteksi apakah `cloudflared` / `tailscale` sudah terinstal di sistem
- Jika ya, spawn subprocess wrapper; jika tidak, tampilkan petunjuk instalasi
- Konfigurasi: `tunnel.provider: "cloudflare"` + `tunnel.enabled: true`

---

### 2.4. Redis Multi-instance Sync

| Aspek | Detail |
|-------|--------|
| **Sumber Referensi** | LiteLLM `router_strategy/base_routing_strategy.py` (Redis pipeline batching) |
| **Fungsi** | Sinkronisasi state routing (rate limit counter, cache, circuit breaker) antar beberapa instance Hermes |
| **Mengapa Opsional** | Membutuhkan server Redis berjalan. Untuk pengguna individu, SQLite lokal sudah lebih dari cukup dan zero-config. |
| **Nilai bagi Power User** | Deploy Hermes sebagai shared AI gateway kantor — beberapa developer berbagi pool kuota, rate limit terpusat, cache bersama |
| **Dependensi Opsional** | `redis` (pip install) |

**Arsitektur storage backend (Interface/Adapter):**
```python
# provider_gateway/storage/base.py
class StorageBackend(ABC):
    """Interface untuk semua storage backend."""
    
    @abstractmethod
    def increment(self, key: str, amount: int = 1) -> int: ...
    
    @abstractmethod
    def get(self, key: str) -> Optional[str]: ...
    
    @abstractmethod
    def set(self, key: str, value: str, ttl_ms: int = 0) -> None: ...

# provider_gateway/storage/sqlite_backend.py  ← DEFAULT
class SQLiteBackend(StorageBackend): ...

# provider_gateway/storage/redis_backend.py  ← OPT-IN
class RedisBackend(StorageBackend): ...
```

Pemilihan backend berdasarkan config:
```yaml
# cli-config.yaml
storage:
  backend: "sqlite"     # "sqlite" (default) | "redis"
  redis:
    host: "localhost"
    port: 6379
    db: 0
```

---

### 2.5. OAuth Device Flow

| Aspek | Detail |
|-------|--------|
| **Sumber Referensi** | 9router (GitHub OAuth), OmniRoute (multi-provider OAuth) |
| **Fungsi** | Autentikasi provider via browser — user mendapat kode, buka URL di browser, approve, dan token otomatis tersimpan |
| **Mengapa Opsional** | Alur OAuth memerlukan callback server lokal atau polling. Hermes sudah punya credential management via file config + API key yang lebih sederhana dan stabil. |
| **Nilai bagi Power User** | (1) Tidak perlu menyalin API key manual, (2) Token refresh otomatis, (3) Multi-account management untuk provider yang mendukung OAuth |
| **Dependensi Opsional** | `authlib` atau `oauthlib` (pip install) |

---

## 3. Arahan Implementasi untuk Coding Agent

### 3.1. Struktur Direktori Ekstensi

```
hermes_agent/
├── provider_gateway/           # Core gateway (sudah ada dari Fase 1-4)
│   ├── __init__.py
│   ├── config.py
│   ├── litellm_backend.py
│   ├── usage_tracker.py
│   ├── circuit_breaker.py
│   ├── router.py
│   ├── semantic_cache.py
│   ├── guardrails.py
│   │
│   └── storage/                # Storage backend abstraction
│       ├── __init__.py
│       ├── base.py             # ABC StorageBackend
│       ├── sqlite_backend.py   # Default (sudah ada)
│       └── redis_backend.py    # [OPT-IN] Redis adapter
│
├── extensions/                 # Semua ekstensi opsional
│   ├── __init__.py             # ExtensionRegistry + dynamic loader
│   │
│   ├── mitm/                   # [OPT-IN] MITM Proxy
│   │   ├── __init__.py
│   │   ├── cert_manager.py     # Generate & install Root CA
│   │   ├── dns_hijacker.py     # Manage /etc/hosts entries
│   │   ├── proxy_server.py     # HTTPS intercept server
│   │   └── README.md           # Dokumentasi standalone
│   │
│   ├── dashboard/              # [OPT-IN] Web Dashboard
│   │   ├── __init__.py
│   │   ├── server.py           # Flask/FastAPI server
│   │   ├── api.py              # REST endpoints untuk data
│   │   └── static/             # Pre-built HTML/JS/CSS
│   │       ├── index.html
│   │       └── app.js
│   │
│   ├── tunnel/                 # [OPT-IN] Cloudflare/Tailscale wrapper
│   │   ├── __init__.py
│   │   ├── cloudflare.py       # cloudflared subprocess wrapper
│   │   └── tailscale.py        # tailscale subprocess wrapper
│   │
│   └── oauth/                  # [OPT-IN] OAuth Device Flow
│       ├── __init__.py
│       ├── device_flow.py      # OAuth device code flow
│       └── token_store.py      # Persistent token management
│
└── pyproject.toml              # Extras dependencies
```

### 3.2. Konfigurasi `pyproject.toml` — Optional Extras

```toml
[project.optional-dependencies]
# Core gateway (sudah ada)
gateway = ["litellm>=1.40.0"]

# Ekstensi opsional — install per fitur
mitm = ["cryptography>=42.0", "mitmproxy>=10.0"]
dashboard = ["flask>=3.0"]
redis = ["redis>=5.0"]
oauth = ["authlib>=1.3"]

# Bundle semua ekstensi sekaligus
all-extensions = [
    "hermes-agent[mitm]",
    "hermes-agent[dashboard]",
    "hermes-agent[redis]",
    "hermes-agent[oauth]",
]
```

Instalasi oleh user:
```bash
# Core saja (default, tanpa ekstensi)
pip install hermes-agent

# Dengan gateway LiteLLM
pip install hermes-agent[gateway]

# Dengan MITM proxy
pip install hermes-agent[mitm]

# Semua fitur
pip install hermes-agent[all-extensions]
```

### 3.3. Extension Registry — Dynamic Loader

```python
"""
extensions/__init__.py — Registry dan dynamic loader untuk ekstensi opsional.

Prinsip:
1. TIDAK ADA import top-level ke modul ekstensi
2. Semua import dilakukan secara lazy saat extension.load() dipanggil
3. ImportError ditangkap dan dikonversi ke pesan informatif
4. Core TIDAK PERNAH bergantung pada extensions/
"""
from __future__ import annotations
import logging
from typing import Any, Optional

logger = logging.getLogger("hermes.extensions")

# Registry ekstensi yang tersedia
_EXTENSIONS: dict[str, dict] = {
    "mitm": {
        "module": "hermes_agent.extensions.mitm",
        "class": "MITMExtension",
        "pip_extra": "mitm",
        "requires": ["cryptography"],
        "description": "MITM Proxy — intercept traffic IDE untuk routing cerdas",
    },
    "dashboard": {
        "module": "hermes_agent.extensions.dashboard",
        "class": "DashboardExtension",
        "pip_extra": "dashboard",
        "requires": ["flask"],
        "description": "Web Dashboard — visualisasi metrik dan usage via browser",
    },
    "tunnel": {
        "module": "hermes_agent.extensions.tunnel",
        "class": "TunnelExtension",
        "pip_extra": None,  # Butuh binary eksternal, bukan pip
        "requires": [],
        "external_binaries": ["cloudflared", "tailscale"],
        "description": "Tunnel — ekspos router lokal via Cloudflare/Tailscale",
    },
    "redis": {
        "module": "hermes_agent.provider_gateway.storage.redis_backend",
        "class": "RedisBackend",
        "pip_extra": "redis",
        "requires": ["redis"],
        "description": "Redis Backend — sinkronisasi multi-instance",
    },
    "oauth": {
        "module": "hermes_agent.extensions.oauth",
        "class": "OAuthExtension",
        "pip_extra": "oauth",
        "requires": ["authlib"],
        "description": "OAuth Device Flow — autentikasi via browser",
    },
}


def load_extension(name: str, config: dict) -> Any:
    """
    Load ekstensi secara dinamis.
    
    Raises RuntimeError dengan pesan informatif jika dependensi tidak tersedia.
    """
    if name not in _EXTENSIONS:
        raise ValueError(f"Ekstensi '{name}' tidak dikenal. Tersedia: {list(_EXTENSIONS.keys())}")
    
    ext_info = _EXTENSIONS[name]
    
    # Cek dependensi Python
    missing = []
    for req in ext_info.get("requires", []):
        try:
            __import__(req)
        except ImportError:
            missing.append(req)
    
    if missing:
        pip_extra = ext_info.get("pip_extra", name)
        install_cmd = f"pip install hermes-agent[{pip_extra}]" if pip_extra else f"pip install {' '.join(missing)}"
        raise RuntimeError(
            f"Ekstensi '{name}' memerlukan dependensi: {', '.join(missing)}.\n"
            f"Install dengan: {install_cmd}"
        )
    
    # Cek binary eksternal (untuk tunnel)
    import shutil
    for binary in ext_info.get("external_binaries", []):
        if not shutil.which(binary):
            logger.warning("Binary '%s' tidak ditemukan di PATH. Beberapa fitur '%s' mungkin tidak tersedia.", binary, name)
    
    # Dynamic import
    import importlib
    module = importlib.import_module(ext_info["module"])
    cls = getattr(module, ext_info["class"])
    
    instance = cls(config)
    logger.info("Ekstensi '%s' berhasil dimuat: %s", name, ext_info["description"])
    return instance


def list_available() -> list[dict]:
    """Daftar semua ekstensi yang tersedia beserta status instalasinya."""
    result = []
    for name, info in _EXTENSIONS.items():
        installed = True
        for req in info.get("requires", []):
            try:
                __import__(req)
            except ImportError:
                installed = False
                break
        
        result.append({
            "name": name,
            "description": info["description"],
            "installed": installed,
            "pip_extra": info.get("pip_extra"),
        })
    return result
```

### 3.4. Konfigurasi Lengkap `cli-config.yaml`

Section baru untuk ekstensi (tambahkan di `cli-config.yaml.example`):

```yaml
# ─── Ekstensi Opsional (Semua Default OFF) ──────────────────────────────
# Aktifkan fitur yang dibutuhkan. Pastikan dependensi terinstal.
#
# extensions:
#   mitm:
#     enabled: false
#     port: 443                   # Port HTTPS listener
#     router_port: 20128          # Port router lokal Hermes
#     auto_install_cert: false    # true = install Root CA otomatis (butuh sudo)
#     targets:                    # Domain yang di-intercept
#       - api.openai.com
#       - api.anthropic.com
#       - generativelanguage.googleapis.com
#
#   dashboard:
#     enabled: false
#     host: "127.0.0.1"          # JANGAN ubah ke 0.0.0.0 kecuali paham risikonya
#     port: 8080
#
#   tunnel:
#     enabled: false
#     provider: "cloudflare"      # "cloudflare" | "tailscale"
#
#   oauth:
#     enabled: false
#     providers:                  # Provider yang mendukung OAuth
#       - github
#
#   redis:
#     enabled: false
#     host: "localhost"
#     port: 6379
#     db: 0
#     password: ""                # Kosongkan jika tanpa auth
```

### 3.5. CLI Commands untuk Ekstensi

Tambahkan subcommand baru di hermes CLI:

```python
# Contoh interface CLI (implementasi disesuaikan dengan pola cli.py yang ada)

# hermes extensions list — tampilkan semua ekstensi dan status
# Output:
#   EXTENSION   STATUS       DESCRIPTION
#   mitm        not installed  MITM Proxy — intercept traffic IDE
#   dashboard   installed      Web Dashboard — visualisasi metrik
#   tunnel      installed      Tunnel — ekspos via Cloudflare/Tailscale
#   redis       not installed  Redis Backend — sinkronisasi multi-instance
#   oauth       not installed  OAuth Device Flow — auth via browser

# hermes extensions info mitm — detail satu ekstensi

# hermes dashboard — shortcut untuk jalankan web dashboard
# hermes tunnel start — shortcut untuk start tunnel
```

### 3.6. Prinsip Keamanan untuk Ekstensi yang Butuh Elevated Privileges

Khusus untuk **MITM Proxy** dan operasi yang butuh `sudo`:

1. **JANGAN** jalankan `sudo` secara otomatis atau diam-diam
2. **TAMPILKAN** perintah yang akan dijalankan dan minta konfirmasi eksplisit:
   ```
   [hermes] Untuk mengaktifkan MITM proxy, perintah berikut perlu dijalankan 
            dengan hak akses root:
   
   1. sudo cp /home/user/.hermes/certs/hermes-ca.pem /usr/local/share/ca-certificates/
   2. sudo update-ca-certificates
   3. sudo tee -a /etc/hosts <<< "127.0.0.1 api.openai.com"
   
   Jalankan otomatis? [y/N]:
   ```
3. **SEDIAKAN** mode manual — user bisa menjalankan perintah tersebut sendiri
4. **SEDIAKAN** perintah rollback: `hermes extension disable mitm --cleanup`
5. **LOG** semua perubahan sistem yang dibuat oleh ekstensi

### 3.7. Urutan Implementasi Ekstensi (Prioritas)

| Prioritas | Ekstensi | Alasan |
|-----------|----------|--------|
| 1 | **Storage Backend Abstraction** | Fondasi — Redis adapter bergantung pada ini |
| 2 | **Extension Registry** | Infrastruktur — semua ekstensi bergantung pada loader ini |
| 3 | **Dashboard** | Paling berguna segera setelah telemetry stabil |
| 4 | **Redis Backend** | Untuk deployment shared gateway |
| 5 | **OAuth Device Flow** | Kenyamanan autentikasi |
| 6 | **Tunnel** | Wrapper sederhana, effort rendah |
| 7 | **MITM Proxy** | Paling kompleks, implementasi paling akhir |

---

## 4. Commit Order untuk Fase Ekstensi

```
Commit E-01: refactor(storage): extract StorageBackend ABC dari SQLite
             → provider_gateway/storage/base.py, sqlite_backend.py
             → Refactor semantic_cache.py + circuit_breaker.py untuk pakai StorageBackend

Commit E-02: feat(extensions): add extension registry + dynamic loader
             → extensions/__init__.py
             → tests/test_extensions/test_registry.py

Commit E-03: feat(extensions): add Redis storage backend (opt-in)
             → provider_gateway/storage/redis_backend.py
             → tests/test_extensions/test_redis_backend.py
             → Update pyproject.toml (extras: redis)

Commit E-04: feat(extensions): add web dashboard (opt-in)
             → extensions/dashboard/server.py, api.py, static/
             → tests/test_extensions/test_dashboard.py
             → Update pyproject.toml (extras: dashboard)

Commit E-05: feat(extensions): add OAuth device flow (opt-in)
             → extensions/oauth/device_flow.py, token_store.py
             → Update pyproject.toml (extras: oauth)

Commit E-06: feat(extensions): add tunnel wrapper (opt-in)
             → extensions/tunnel/cloudflare.py, tailscale.py
             → Deteksi binary external, bukan pip dependency

Commit E-07: feat(extensions): add MITM proxy (opt-in)
             → extensions/mitm/cert_manager.py, dns_hijacker.py, proxy_server.py
             → Update pyproject.toml (extras: mitm)
             → WAJIB: prompt sudo + rollback command

Commit E-08: feat(cli): add hermes extensions subcommand
             → list, info, enable, disable
             → Update cli-config.yaml.example

Commit E-09: docs: update README tentang extensions system
```

---

## 5. Checklist Kualitas untuk Setiap Ekstensi

Sebelum ekstensi dianggap selesai, pastikan:

- [ ] **Default OFF** — Hermes berjalan normal tanpa ekstensi aktif
- [ ] **Graceful degradation** — ImportError menghasilkan pesan informatif, bukan crash
- [ ] **README.md per ekstensi** — Dokumentasi standalone di folder ekstensi
- [ ] **Test isolasi** — Unit test yang berjalan tanpa dependensi opsional aktif
- [ ] **Rollback** — Ada mekanisme disable/cleanup yang bersih
- [ ] **Logging** — Semua aktivitas ekstensi di-log ke `hermes.extensions.*`
- [ ] **Keamanan** — Operasi privileged memerlukan konfirmasi eksplisit
- [ ] **Config documented** — Section config terdokumentasi di `cli-config.yaml.example`

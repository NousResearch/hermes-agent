# Arahan Coding Agent: Penambahan Fitur Multi-Provider ke Hermes Agent

> **Versi:** 2.0 (Berdasarkan riset source code aktual)
> **Tanggal:** 29 Mei 2026
> **Referensi:** [riset-fitur-repo-sumber.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/riset-fitur-repo-sumber.md)
> **Target:** hermes_agent di `/home/void/lab/git/hermes_agent/`

---

## PENTING: Baca Ini Sebelum Menulis Kode Apapun

### Arsitektur Provider Hermes yang HARUS Dipahami

Hermes menggunakan **pola deklaratif** untuk provider, BUKAN adapter pattern:

```python
# File: providers/base.py — ProviderProfile adalah DATACLASS, bukan ABC
@dataclass
class ProviderProfile:
    name: str                           # "openrouter", "anthropic"
    api_mode: str = "chat_completions"  # bukan method transport
    base_url: str = ""
    auth_type: str = "api_key"
    env_vars: tuple = ()
    fallback_models: tuple = ()
    default_headers: dict = field(default_factory=dict)
    # ... field deklaratif lainnya
    
    # Hooks (bukan transport!) — hanya transformasi pesan
    def prepare_messages(self, messages): return messages
    def build_extra_body(self, **ctx): return {}
    def fetch_models(self, **kwargs): ...
```

**Transport (panggilan API) dilakukan oleh `AIAgent` menggunakan `openai.OpenAI(base_url=...)`.** Provider profiles TIDAK punya method `complete()` atau `stream()`.

Provider didaftar melalui plugin system:
```
plugins/model-providers/
├── openrouter/          # 28 provider sudah terdaftar
├── anthropic/
├── deepseek/
├── ollama-cloud/
├── ... (28 total)
└── README.md
```

Registrasi: `providers/__init__.py` → lazy discovery → `register_provider(profile)`.

### File Besar yang JANGAN Dimodifikasi Tanpa Pemahaman Penuh

| File | Ukuran | Risiko |
|------|--------|--------|
| `cli.py` | 693 KB | JANGAN SENTUH kecuali command baru |
| `run_agent.py` | 203 KB | HATI-HATI — entry point agent loop |
| `hermes_state.py` | 142 KB | HATI-HATI — state management |
| `model_tools.py` | 41 KB | BACA PENUH sebelum modifikasi |
| `trajectory_compressor.py` | 65 KB | JANGAN SENTUH — sudah mature |

### Prinsip Desain

1. **Opt-in via config** — User tanpa config baru = hermes berjalan normal
2. **Extend, jangan replace** — Tambah field/method ke `ProviderProfile`, jangan buat ABC baru
3. **SQLite-first** — Hermes sudah pakai SQLite (FTS5, WAL), gunakan untuk semua persistensi
4. **Test sebelum integrasi** — Setiap modul harus testable secara isolasi
5. **Log di DEBUG, bukan INFO** — Jangan polusi output user
6. **Backward compatible** — Config lama tetap valid

---

## Fase 0: Persiapan & Audit (WAJIB sebelum coding)

### Tugas Audit

1. **Baca `providers/base.py`** — pahami `ProviderProfile` sepenuhnya
2. **Baca `providers/__init__.py`** — pahami registry dan plugin system
3. **Baca salah satu plugin** (contoh: `plugins/model-providers/openrouter/__init__.py`) — pahami cara registrasi
4. **Baca `model_tools.py`** — pahami bagaimana tool definitions bekerja
5. **Scan `run_agent.py`** — cari di mana `openai.OpenAI()` client dibuat dan dipanggil
6. **Scan `hermes_state.py`** — pahami SQLite schema yang sudah ada
7. **Baca `cli-config.yaml.example`** — pahami format config yang sudah ada
8. **List `tests/`** — pahami test infrastructure yang sudah ada

### Keputusan yang Harus Diambil

Sebelum mulai Fase 1, jawab pertanyaan ini:

- **Q1:** Di mana tepatnya dalam `run_agent.py` panggilan ke LLM API dibuat?
- **Q2:** Bagaimana `ProviderProfile` digunakan oleh `AIAgent`?
- **Q3:** SQLite schema apa saja yang sudah ada di `hermes_state.py`?
- **Q4:** Apakah `hermes_state.py` bisa ditambah tabel baru, atau harus file DB terpisah?

---

## Fase 1: LiteLLM Integration + Usage Tracking

### Alasan LiteLLM Duluan
LiteLLM adalah satu-satunya repo sumber yang **Python-native**. Dengan `pip install litellm`, hermes langsung dapat:
- 100+ provider tanpa kode adapter manual
- `litellm.completion_cost()` untuk estimasi biaya
- `litellm.model_list` untuk model discovery
- Drop-in replacement untuk `openai.OpenAI()` calls

### 1.1 Tambahkan LiteLLM sebagai Optional Dependency

Di `pyproject.toml`, tambahkan ke extras:
```toml
[project.optional-dependencies]
gateway = ["litellm>=1.40.0"]
```

> **JANGAN** tambahkan ke dependencies utama — litellm besar (~100MB+ dengan transitive deps).

### 1.2 Buat `provider_gateway/__init__.py`

Buat folder baru `provider_gateway/` di root hermes:

```
provider_gateway/
├── __init__.py          # Module marker + public API
├── litellm_backend.py   # LiteLLM wrapper (opt-in)
├── usage_tracker.py     # SQLite usage logging (dari 9router schema)
├── cost_estimator.py    # Wrapper litellm.completion_cost()
└── config.py            # Gateway config loader
```

### 1.3 `provider_gateway/config.py` — Gateway Configuration

```python
"""
Konfigurasi provider gateway.
Dimuat dari cli-config.yaml section 'provider_gateway'.

Contoh config:
    provider_gateway:
      enabled: true
      backend: litellm          # "litellm" | "native" (default openai SDK)
      default_model: openai/gpt-4o
      track_usage: true         # log usage ke SQLite
      track_cost: true          # estimasi cost
      
      # Routing (fase berikutnya)
      routing:
        strategy: lowest-cost   # lowest-cost | lowest-latency | round-robin
        fallback_models:
          - anthropic/claude-sonnet-4-6
          - ollama/llama3.2
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("hermes.gateway")


@dataclass
class GatewayConfig:
    """Konfigurasi provider gateway."""
    enabled: bool = False
    backend: str = "native"          # "litellm" | "native"
    default_model: str = ""
    track_usage: bool = True
    track_cost: bool = True
    
    # Routing config (fase 2)
    routing_strategy: str = "round-robin"
    fallback_models: list[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> "GatewayConfig":
        """Parse dari config dict."""
        if not data:
            return cls()
        return cls(
            enabled=data.get("enabled", False),
            backend=data.get("backend", "native"),
            default_model=data.get("default_model", ""),
            track_usage=data.get("track_usage", True),
            track_cost=data.get("track_cost", True),
            routing_strategy=data.get("routing", {}).get("strategy", "round-robin"),
            fallback_models=data.get("routing", {}).get("fallback_models", []),
        )


def load_gateway_config() -> GatewayConfig:
    """Load gateway config dari cli-config.yaml."""
    try:
        from hermes_cli.config import get_config
        config = get_config()
        gw_config = config.get("provider_gateway", {})
        return GatewayConfig.from_dict(gw_config)
    except Exception as e:
        logger.debug("Gateway config tidak tersedia: %s", e)
        return GatewayConfig()
```

### 1.4 `provider_gateway/litellm_backend.py` — LiteLLM Wrapper

```python
"""
LiteLLM backend — wrapper tipis di atas litellm.completion().
Aktif hanya jika litellm terinstall DAN gateway.backend = "litellm".

Fitur:
- 100+ provider via satu interface
- Cost tracking otomatis
- Drop incompatible params

Adaptasi dari: litellm/litellm/main.py
"""
from __future__ import annotations
import logging
from typing import Any, Optional

logger = logging.getLogger("hermes.gateway.litellm")

_AVAILABLE = False
try:
    import litellm
    litellm.drop_params = True      # Abaikan param yang tidak didukung
    litellm.set_verbose = False
    _AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """True jika litellm terinstall."""
    return _AVAILABLE


def complete(
    model: str,
    messages: list[dict],
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    **kwargs,
) -> Any:
    """
    Panggil LLM via LiteLLM.

    Format model sesuai LiteLLM:
      "openai/gpt-4o"
      "anthropic/claude-opus-4-6"
      "openrouter/anthropic/claude-opus-4-6"
      "ollama/llama3.2"
    """
    if not _AVAILABLE:
        raise ImportError("litellm tidak terinstall. Install: pip install litellm")

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if api_key:
        params["api_key"] = api_key
    if api_base:
        params["api_base"] = api_base
    params.update(kwargs)

    return litellm.completion(**params)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimasi biaya dalam USD. Return 0.0 jika tidak bisa."""
    if not _AVAILABLE:
        return 0.0
    try:
        return litellm.completion_cost(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
    except Exception:
        return 0.0


def list_models() -> list[str]:
    """Daftar semua model yang didukung LiteLLM."""
    if not _AVAILABLE:
        return []
    try:
        return list(litellm.model_list)
    except Exception:
        return []
```

### 1.5 `provider_gateway/usage_tracker.py` — SQLite Usage Logging

Schema diadaptasi dari **9router** `src/lib/db/schema.js`:

```python
"""
Usage tracker — log setiap request ke SQLite untuk analisis.
Schema diadaptasi dari 9router usageHistory + usageDaily.

Fitur:
- Log per-request: provider, model, tokens, cost, status, latency
- Agregasi harian otomatis
- Query helper untuk `hermes status`
"""
from __future__ import annotations
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("hermes.gateway.usage")

# Gunakan DB terpisah dari hermes_state untuk menghindari konflik
_DB_PATH: Optional[Path] = None
_conn: Optional[sqlite3.Connection] = None


def _get_db_path() -> Path:
    """Path ke usage database."""
    global _DB_PATH
    if _DB_PATH is None:
        try:
            from hermes_constants import get_hermes_home
            _DB_PATH = get_hermes_home() / "provider_usage.db"
        except ImportError:
            _DB_PATH = Path.home() / ".hermes" / "provider_usage.db"
    return _DB_PATH


def _get_conn() -> sqlite3.Connection:
    """Get/create SQLite connection (singleton, thread-local untuk safety)."""
    global _conn
    if _conn is None:
        db_path = _get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(db_path))
        _conn.execute("PRAGMA journal_mode = WAL")
        _conn.execute("PRAGMA synchronous = NORMAL")
        _conn.execute("PRAGMA busy_timeout = 5000")
        _init_tables(_conn)
    return _conn


def _init_tables(conn: sqlite3.Connection):
    """Inisialisasi tabel — idempotent."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS usage_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            endpoint TEXT DEFAULT 'chat/completions',
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0,
            latency_ms REAL DEFAULT 0,
            status TEXT DEFAULT 'success',
            error_message TEXT,
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_uh_ts 
            ON usage_history(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_uh_provider 
            ON usage_history(provider);
        CREATE INDEX IF NOT EXISTS idx_uh_model 
            ON usage_history(model);

        CREATE TABLE IF NOT EXISTS usage_daily (
            date_key TEXT PRIMARY KEY,
            total_requests INTEGER DEFAULT 0,
            total_prompt_tokens INTEGER DEFAULT 0,
            total_completion_tokens INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0,
            providers_json TEXT DEFAULT '{}',
            models_json TEXT DEFAULT '{}'
        );
    """)


@dataclass
class UsageRecord:
    """Record untuk satu request."""
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    status: str = "success"
    error_message: str = ""
    endpoint: str = "chat/completions"
    metadata: Optional[dict] = None


def log_usage(record: UsageRecord):
    """Log satu usage record ke database."""
    try:
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(record.metadata) if record.metadata else None
        
        conn.execute(
            """INSERT INTO usage_history 
               (timestamp, provider, model, endpoint, prompt_tokens, 
                completion_tokens, total_tokens, cost_usd, latency_ms,
                status, error_message, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, record.provider, record.model, record.endpoint,
             record.prompt_tokens, record.completion_tokens, 
             record.total_tokens, record.cost_usd, record.latency_ms,
             record.status, record.error_message or None, meta_json),
        )
        
        # Update daily aggregate
        date_key = now[:10]  # YYYY-MM-DD
        _update_daily_aggregate(conn, date_key, record)
        conn.commit()
        
    except Exception as e:
        logger.debug("Failed to log usage: %s", e)


def _update_daily_aggregate(conn: sqlite3.Connection, date_key: str, record: UsageRecord):
    """Update atau insert daily aggregate."""
    row = conn.execute(
        "SELECT providers_json, models_json FROM usage_daily WHERE date_key = ?",
        (date_key,)
    ).fetchone()
    
    if row:
        providers = json.loads(row[0] or "{}")
        models = json.loads(row[1] or "{}")
        providers[record.provider] = providers.get(record.provider, 0) + 1
        models[record.model] = models.get(record.model, 0) + 1
        
        conn.execute(
            """UPDATE usage_daily SET 
               total_requests = total_requests + 1,
               total_prompt_tokens = total_prompt_tokens + ?,
               total_completion_tokens = total_completion_tokens + ?,
               total_cost_usd = total_cost_usd + ?,
               providers_json = ?,
               models_json = ?
               WHERE date_key = ?""",
            (record.prompt_tokens, record.completion_tokens, record.cost_usd,
             json.dumps(providers), json.dumps(models), date_key),
        )
    else:
        conn.execute(
            """INSERT INTO usage_daily 
               (date_key, total_requests, total_prompt_tokens, 
                total_completion_tokens, total_cost_usd, 
                providers_json, models_json)
               VALUES (?, 1, ?, ?, ?, ?, ?)""",
            (date_key, record.prompt_tokens, record.completion_tokens, 
             record.cost_usd,
             json.dumps({record.provider: 1}),
             json.dumps({record.model: 1})),
        )


def get_today_summary() -> dict:
    """Ringkasan usage hari ini untuk `hermes status`."""
    try:
        conn = _get_conn()
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT * FROM usage_daily WHERE date_key = ?", (date_key,)
        ).fetchone()
        
        if not row:
            return {"date": date_key, "total_requests": 0, "total_cost_usd": 0.0}
        
        return {
            "date": row[0],
            "total_requests": row[1],
            "total_prompt_tokens": row[2],
            "total_completion_tokens": row[3],
            "total_cost_usd": round(row[4], 4),
            "providers": json.loads(row[5] or "{}"),
            "models": json.loads(row[6] or "{}"),
        }
    except Exception:
        return {"date": "", "total_requests": 0, "total_cost_usd": 0.0}


def get_usage_history(limit: int = 20) -> list[dict]:
    """Ambil N request terakhir."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM usage_history ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        
        return [
            {
                "id": r[0], "timestamp": r[1], "provider": r[2],
                "model": r[3], "endpoint": r[4],
                "prompt_tokens": r[5], "completion_tokens": r[6],
                "total_tokens": r[7], "cost_usd": r[8],
                "latency_ms": r[9], "status": r[10],
            }
            for r in rows
        ]
    except Exception:
        return []
```

### 1.6 Update `cli-config.yaml.example`

Tambahkan section baru di akhir file:

```yaml
# ─── Provider Gateway (Opsional) ─────────────────────────────────────────
# Aktifkan untuk mendapatkan multi-provider routing, usage tracking, dan cost estimation.
# Membutuhkan: pip install litellm (untuk backend litellm)
#
# provider_gateway:
#   enabled: true
#   backend: litellm              # "litellm" | "native"
#   track_usage: true             # Log usage ke SQLite
#   track_cost: true              # Estimasi cost per request
#   
#   routing:
#     strategy: round-robin       # round-robin | lowest-cost | lowest-latency
#     fallback_models:
#       - anthropic/claude-sonnet-4-6
#       - ollama/llama3.2
```

---

## Fase 2: Circuit Breaker + Routing Engine

### Sumber Inspirasi Utama
- **OmniRoute** `src/lib/resilience/settings.ts` — untuk arsitektur circuit breaker
- **OmniRoute** `src/lib/combos/intelligentRouting.ts` — untuk weighted scoring
- **LiteLLM** `litellm/router_strategy/base_routing_strategy.py` — untuk strategy pattern

### 2.1 `provider_gateway/circuit_breaker.py`

Arsitektur diadaptasi dari **OmniRoute resilience settings** dengan penyederhanaan:

```python
"""
Circuit breaker — mencegah request ke provider yang sedang bermasalah.
Diadaptasi dari OmniRoute src/lib/resilience/settings.ts.

States:
  CLOSED  → normal, forward semua request
  OPEN    → provider gagal, blokir request (auto-reset setelah timeout)
  HALF    → testing, izinkan 1 request untuk cek recovery

Per-provider tracking dengan konfigurasi berbeda per auth-type.
"""
from __future__ import annotations
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("hermes.gateway.breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class BreakerConfig:
    """Konfigurasi circuit breaker per provider."""
    failure_threshold: int = 5          # Gagal berapa kali sebelum OPEN
    reset_timeout_ms: int = 60_000      # Berapa lama OPEN sebelum coba HALF_OPEN
    base_cooldown_ms: int = 5_000       # Cooldown antar retry saat HALF_OPEN
    max_backoff_steps: int = 5          # Max exponential backoff level


@dataclass
class ProviderHealth:
    """Status kesehatan satu provider."""
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_requests: int = 0
    total_failures: int = 0
    latency_samples: list[float] = field(default_factory=list)
    backoff_level: int = 0

    @property
    def latency_p50(self) -> float:
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = len(sorted_samples) // 2
        return sorted_samples[idx]

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_failures / self.total_requests


class CircuitBreaker:
    """Circuit breaker multi-provider."""

    MAX_LATENCY_SAMPLES = 100

    def __init__(self, default_config: Optional[BreakerConfig] = None):
        self._config = default_config or BreakerConfig()
        self._providers: dict[str, ProviderHealth] = {}
        self._lock = threading.Lock()

    def is_available(self, provider: str) -> bool:
        """Cek apakah provider bisa menerima request."""
        with self._lock:
            health = self._providers.get(provider)
            if health is None:
                return True

            if health.state == CircuitState.CLOSED:
                return True

            if health.state == CircuitState.OPEN:
                elapsed = (time.time() - health.last_failure_time) * 1000
                cooldown = self._config.reset_timeout_ms * (2 ** health.backoff_level)
                if elapsed >= cooldown:
                    health.state = CircuitState.HALF_OPEN
                    logger.debug("Circuit %s: OPEN → HALF_OPEN", provider)
                    return True
                return False

            # HALF_OPEN: izinkan satu request
            return True

    def record_success(self, provider: str, latency_ms: float = 0.0):
        """Catat request sukses."""
        with self._lock:
            health = self._get_or_create(provider)
            health.total_requests += 1
            health.last_success_time = time.time()
            health.consecutive_failures = 0
            health.backoff_level = 0

            if latency_ms > 0:
                health.latency_samples.append(latency_ms)
                if len(health.latency_samples) > self.MAX_LATENCY_SAMPLES:
                    health.latency_samples = health.latency_samples[-self.MAX_LATENCY_SAMPLES:]

            if health.state != CircuitState.CLOSED:
                logger.debug("Circuit %s: %s → CLOSED", provider, health.state.value)
                health.state = CircuitState.CLOSED

    def record_failure(self, provider: str, error: Optional[Exception] = None):
        """Catat request gagal."""
        with self._lock:
            health = self._get_or_create(provider)
            health.total_requests += 1
            health.total_failures += 1
            health.consecutive_failures += 1
            health.last_failure_time = time.time()

            if health.state == CircuitState.HALF_OPEN:
                health.state = CircuitState.OPEN
                health.backoff_level = min(
                    health.backoff_level + 1,
                    self._config.max_backoff_steps,
                )
                logger.debug("Circuit %s: HALF_OPEN → OPEN (backoff=%d)",
                           provider, health.backoff_level)

            elif (health.state == CircuitState.CLOSED
                  and health.consecutive_failures >= self._config.failure_threshold):
                health.state = CircuitState.OPEN
                logger.debug("Circuit %s: CLOSED → OPEN (failures=%d)",
                           provider, health.consecutive_failures)

    def get_all_health(self) -> dict[str, ProviderHealth]:
        """Snapshot kesehatan semua provider."""
        with self._lock:
            return dict(self._providers)

    def _get_or_create(self, provider: str) -> ProviderHealth:
        if provider not in self._providers:
            self._providers[provider] = ProviderHealth()
        return self._providers[provider]
```

### 2.2 `provider_gateway/router.py` — Simplified Weighted Scoring

Disederhanakan dari **OmniRoute** 11-faktor menjadi **6-faktor** yang relevan untuk CLI agent:

```python
"""
Provider Router — pilih provider terbaik berdasarkan weighted scoring.
Disederhanakan dari OmniRoute intelligentRouting.ts (11 faktor → 6 faktor).

Faktor scoring:
  1. health    (0.30) — dari circuit breaker
  2. cost      (0.25) — inverse cost per token
  3. latency   (0.20) — inverse P50 latency
  4. quota     (0.15) — sisa kuota (jika diketahui)
  5. priority  (0.05) — user-defined priority
  6. stability (0.05) — error rate historis
"""
from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from .circuit_breaker import CircuitBreaker, CircuitState

logger = logging.getLogger("hermes.gateway.router")


@dataclass
class RoutingWeights:
    """Bobot scoring — total harus = 1.0."""
    health: float = 0.30
    cost: float = 0.25
    latency: float = 0.20
    quota: float = 0.15
    priority: float = 0.05
    stability: float = 0.05


@dataclass
class ProviderCandidate:
    """Kandidat provider untuk routing."""
    name: str
    model: str
    base_url: str = ""
    api_key: str = ""
    priority: int = 0           # 0 = highest
    cost_per_1m_tokens: float = 0.0
    quota_remaining_pct: float = 100.0  # 0-100
    
    # Calculated
    score: float = 0.0


@dataclass
class RoutingDecision:
    """Hasil keputusan routing."""
    provider: str
    model: str
    reason: str
    score: float = 0.0
    alternatives_count: int = 0


class ProviderRouter:
    """Router dengan weighted scoring."""

    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        weights: Optional[RoutingWeights] = None,
        exploration_rate: float = 0.05,  # 5% random exploration
    ):
        self.circuit_breaker = circuit_breaker
        self.weights = weights or RoutingWeights()
        self.exploration_rate = exploration_rate

    def select(
        self,
        candidates: list[ProviderCandidate],
        preferred_provider: Optional[str] = None,
    ) -> Optional[RoutingDecision]:
        """
        Pilih provider terbaik dari kandidat.
        
        Returns None jika tidak ada kandidat yang available.
        """
        if not candidates:
            return None

        # Filter by circuit breaker
        available = [
            c for c in candidates
            if self.circuit_breaker.is_available(c.name)
        ]

        if not available:
            logger.warning("Semua provider circuit-open. Fallback ke random.")
            available = candidates

        # Preferred provider override
        if preferred_provider:
            preferred = [c for c in available if c.name == preferred_provider]
            if preferred:
                c = preferred[0]
                return RoutingDecision(
                    provider=c.name,
                    model=c.model,
                    reason="user_preferred",
                    score=1.0,
                    alternatives_count=len(available) - 1,
                )

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate and len(available) > 1:
            c = random.choice(available)
            return RoutingDecision(
                provider=c.name,
                model=c.model,
                reason="exploration",
                score=0.0,
                alternatives_count=len(available) - 1,
            )

        # Score each candidate
        for c in available:
            c.score = self._compute_score(c)

        # Sort by score descending
        available.sort(key=lambda c: c.score, reverse=True)
        best = available[0]

        return RoutingDecision(
            provider=best.name,
            model=best.model,
            reason="weighted_scoring",
            score=best.score,
            alternatives_count=len(available) - 1,
        )

    def _compute_score(self, candidate: ProviderCandidate) -> float:
        """Hitung skor weighted untuk satu kandidat."""
        health = self.circuit_breaker.get_all_health()
        h = health.get(candidate.name)

        # Health score (0-1): 1.0 jika CLOSED, 0.5 jika HALF_OPEN, 0.0 jika OPEN
        if h is None or h.state == CircuitState.CLOSED:
            health_score = 1.0
        elif h.state == CircuitState.HALF_OPEN:
            health_score = 0.5
        else:
            health_score = 0.0

        # Cost score (0-1): inverse, normalize. 0 cost = 1.0
        cost_score = 1.0 / (1.0 + candidate.cost_per_1m_tokens / 10.0)

        # Latency score (0-1): inverse P50
        latency_p50 = h.latency_p50 if h else 0.0
        latency_score = 1.0 / (1.0 + latency_p50 / 1000.0)

        # Quota score (0-1): linear
        quota_score = candidate.quota_remaining_pct / 100.0

        # Priority score (0-1): lower priority value = higher score
        priority_score = 1.0 / (1.0 + candidate.priority)

        # Stability score (0-1): inverse error rate
        error_rate = h.error_rate if h else 0.0
        stability_score = 1.0 - error_rate

        w = self.weights
        return (
            w.health * health_score
            + w.cost * cost_score
            + w.latency * latency_score
            + w.quota * quota_score
            + w.priority * priority_score
            + w.stability * stability_score
        )
```

---

## Fase 3: Semantic Cache + hermes status

### Sumber Inspirasi: OmniRoute `src/lib/semanticCache.ts`

### 3.1 `provider_gateway/semantic_cache.py`

Diadaptasi dari **OmniRoute** — two-tier (in-memory LRU + SQLite persistent):

```python
"""
Semantic cache — hindari request duplikat ke LLM.
Diadaptasi dari OmniRoute src/lib/semanticCache.ts.

Cara kerja:
1. Hash(model + messages + temperature) → lookup
2. Hit → kembalikan response tersimpan
3. Miss → forward ke provider, simpan response

Hanya cache request dengan temperature=0 (deterministic).
Bypass: set HERMES_NO_CACHE=true di environment.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import threading
from collections import OrderedDict
from typing import Any, Optional

_MAX_SIZE = int(os.environ.get("HERMES_CACHE_MAX_SIZE", "200"))
_TTL_MS = int(os.environ.get("HERMES_CACHE_TTL_MS", "1800000"))  # 30 menit


class SemanticCache:
    """In-memory LRU cache + SQLite persistent."""

    def __init__(self, max_size: int = _MAX_SIZE, ttl_ms: int = _TTL_MS):
        self.max_size = max_size
        self.ttl_seconds = ttl_ms / 1000.0
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def make_key(self, model: str, messages: list[dict], temperature: float = 0) -> str:
        """Buat cache key deterministik."""
        # Normalisasi messages: hanya ambil role + content
        normalized = [
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
            for m in messages
        ]
        payload = json.dumps(
            {"model": model, "messages": normalized, "temperature": temperature},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def is_cacheable(self, temperature: float) -> bool:
        """Hanya cache request deterministik."""
        if os.environ.get("HERMES_NO_CACHE", "").lower() == "true":
            return False
        return temperature == 0

    def get(self, key: str) -> Optional[Any]:
        """Ambil dari cache. None jika miss/expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expires_at = self._cache[key]
            if time.time() > expires_at:
                del self._cache[key]
                self._misses += 1
                return None

            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, response: Any):
        """Simpan ke cache."""
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (response, time.time() + self.ttl_seconds)

    def get_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / total:.1%}" if total > 0 else "0.0%",
            "size": len(self._cache),
            "max_size": self.max_size,
        }

    def clear(self):
        with self._lock:
            self._cache.clear()
```

### 3.2 Command `hermes status` (tambahan)

Tambahkan ke CLI hermes (di `cli.py` atau `hermes_cli/`):

```python
def cmd_provider_status():
    """
    hermes status providers — tampilkan status semua provider.
    
    Output contoh:
        Provider Status (Today: 2026-05-29)
        ──────────────────────────────────────────────────────
        PROVIDER       CIRCUIT   P50     REQUESTS   COST
        openrouter     CLOSED    145ms   1,234      $2.45
        anthropic      CLOSED    89ms    456        $1.20
        ollama         CLOSED    12ms    789        $0.00
        ──────────────────────────────────────────────────────
        Total: 2,479 requests, $3.65 today
        Cache: 45 hits / 12 misses (78.9% hit rate)
    """
    try:
        from provider_gateway.circuit_breaker import CircuitBreaker
        from provider_gateway.usage_tracker import get_today_summary
        # ... render output
    except ImportError:
        print("Provider gateway belum diaktifkan.")
```

---

## Fase 4: Guardrails + Quota Preflight (Opsional)

### Sumber: OmniRoute `src/lib/guardrails/`

### 4.1 `provider_gateway/guardrails.py`

```python
"""
Guardrails — filter konten sebelum dikirim ke provider.
Diadaptasi dari OmniRoute src/lib/guardrails/promptInjection.ts.

Mode: block | warn | log (default: warn)
Semua guardrails default OFF — harus diaktifkan via config.
"""
# ... implementasi diadaptasi dari OmniRoute
# Lihat referensi: OmniRoute promptInjection.ts (260 baris)
# Lihat referensi: OmniRoute piiMasker.ts (6.4 KB)
```

### 4.2 Quota Preflight Check (dari OmniRoute)

```python
"""
Quota preflight — cek sisa kuota sebelum kirim request.
Diadaptasi dari OmniRoute src/lib/resilience/settings.ts QuotaPreflightSettings.

Konfigurasi:
  quota_preflight:
    default_threshold_percent: 2    # Stop saat sisa kuota <= 2%
    warn_threshold_percent: 20      # Warn saat sisa kuota <= 20%
"""
# ... implementasi berdasarkan pola OmniRoute
```

---

## Fase 5: Extension System — Fitur Opsional (Opt-in)

> **Referensi lengkap:** [fitur-opsional-dan-arahan-implementasi.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/fitur-opsional-dan-arahan-implementasi.md)
>
> Proyek ini bersifat **open-source**. Fitur-fitur tingkat lanjut TIDAK ditolak,
> melainkan diimplementasikan sebagai **ekstensi modular (opt-in)** yang default OFF.

### 5.1 Daftar Ekstensi

| Ekstensi | Deskripsi | Dependensi | Effort |
|----------|-----------|------------|--------|
| **Storage Backend Abstraction** | ABC + SQLite adapter + Redis adapter | `redis` (opsional) | Sedang |
| **Extension Registry** | Dynamic loader + graceful degradation | Tidak ada | Rendah |
| **Web Dashboard** | Visualisasi metrik via browser lokal | `flask` | Sedang |
| **OAuth Device Flow** | Autentikasi provider via browser | `authlib` | Sedang |
| **Tunnel Wrapper** | Ekspos router via Cloudflare/Tailscale | Binary eksternal | Rendah |
| **MITM Proxy** | Intercept traffic IDE untuk routing cerdas | `cryptography`, `mitmproxy` | Tinggi |

### 5.2 Prinsip Implementasi Ekstensi

1. **Default OFF** — `enabled: false` di semua config ekstensi
2. **Zero import di core** — `extensions/` TIDAK diimpor oleh `provider_gateway/` atau `core/`
3. **Dynamic loading** — `extensions/__init__.py` sebagai registry + lazy loader
4. **Graceful degradation** — `ImportError` → pesan informatif (bukan crash)
5. **Extras di pyproject.toml** — Setiap ekstensi punya pip extras sendiri

### 5.3 Struktur Direktori Ekstensi

```
extensions/                     # Semua ekstensi opsional
├── __init__.py                 # ExtensionRegistry + dynamic loader
├── mitm/                       # MITM Proxy
│   ├── cert_manager.py
│   ├── dns_hijacker.py
│   └── proxy_server.py
├── dashboard/                  # Web Dashboard
│   ├── server.py
│   ├── api.py
│   └── static/
├── tunnel/                     # Cloudflare/Tailscale wrapper
│   ├── cloudflare.py
│   └── tailscale.py
└── oauth/                      # OAuth Device Flow
    ├── device_flow.py
    └── token_store.py

provider_gateway/storage/       # Storage backend abstraction
├── base.py                     # ABC StorageBackend
├── sqlite_backend.py           # Default
└── redis_backend.py            # Opt-in Redis adapter
```

### 5.4 Config untuk Ekstensi (tambah di `cli-config.yaml.example`)

```yaml
# ─── Ekstensi Opsional (Semua Default OFF) ──────────────────────────────
# extensions:
#   mitm:
#     enabled: false
#     port: 443
#     auto_install_cert: false
#     targets: [api.openai.com, api.anthropic.com]
#
#   dashboard:
#     enabled: false
#     host: "127.0.0.1"
#     port: 8080
#
#   tunnel:
#     enabled: false
#     provider: "cloudflare"
#
#   oauth:
#     enabled: false
#
#   redis:
#     enabled: false
#     host: "localhost"
#     port: 6379
```

### 5.5 Keamanan Ekstensi dengan Elevated Privileges

Khusus MITM Proxy dan operasi yang butuh `sudo`:
- **JANGAN** jalankan `sudo` otomatis tanpa konfirmasi
- **TAMPILKAN** perintah yang akan dijalankan, minta persetujuan eksplisit
- **SEDIAKAN** mode manual + perintah rollback (`hermes extension disable mitm --cleanup`)

---

## Urutan Commit yang Disarankan

### Core Gateway (Fase 1-4)

```
Commit 01: chore(gateway): add provider_gateway module skeleton
           → provider_gateway/__init__.py, config.py
           → ZERO integration, hanya module setup

Commit 02: feat(gateway): add LiteLLM optional backend
           → provider_gateway/litellm_backend.py
           → Update pyproject.toml (extras)
           → tests/test_gateway/test_litellm_backend.py

Commit 03: feat(gateway): add SQLite usage tracker
           → provider_gateway/usage_tracker.py
           → tests/test_gateway/test_usage_tracker.py
           → Schema dari 9router, DB terpisah (provider_usage.db)

Commit 04: feat(gateway): add circuit breaker
           → provider_gateway/circuit_breaker.py
           → tests/test_gateway/test_circuit_breaker.py
           → Pola dari OmniRoute resilience

Commit 05: feat(gateway): add weighted scoring router
           → provider_gateway/router.py
           → tests/test_gateway/test_router.py
           → 6-faktor scoring dari OmniRoute

Commit 06: feat(gateway): add semantic cache
           → provider_gateway/semantic_cache.py
           → tests/test_gateway/test_semantic_cache.py
           → Two-tier dari OmniRoute

Commit 07: feat(config): add gateway section to cli-config
           → Update cli-config.yaml.example
           → Dokumentasi konfigurasi

Commit 08: feat(integration): wire gateway into agent loop
           → Modifikasi run_agent.py (opt-in, config-driven)
           → HATI-HATI: baca file penuh dulu!

Commit 09: feat(cli): add hermes status providers command
           → Tampilkan circuit breaker + usage summary

Commit 10: feat(guardrails): add prompt injection + PII guard
           → provider_gateway/guardrails.py (default OFF)
           → tests/test_gateway/test_guardrails.py

Commit 11: docs: update README + AGENTS.md
           → Dokumentasi fitur gateway baru
```

### Extension System (Fase 5)

```
Commit E-01: refactor(storage): extract StorageBackend ABC
             → provider_gateway/storage/base.py, sqlite_backend.py
             → Refactor existing code untuk pakai StorageBackend interface

Commit E-02: feat(extensions): add extension registry + dynamic loader
             → extensions/__init__.py
             → tests/test_extensions/test_registry.py

Commit E-03: feat(extensions): add Redis storage backend (opt-in)
             → provider_gateway/storage/redis_backend.py
             → Update pyproject.toml (extras: redis)

Commit E-04: feat(extensions): add web dashboard (opt-in)
             → extensions/dashboard/server.py, api.py, static/
             → Update pyproject.toml (extras: dashboard)

Commit E-05: feat(extensions): add OAuth device flow (opt-in)
             → extensions/oauth/device_flow.py, token_store.py
             → Update pyproject.toml (extras: oauth)

Commit E-06: feat(extensions): add tunnel wrapper (opt-in)
             → extensions/tunnel/cloudflare.py, tailscale.py

Commit E-07: feat(extensions): add MITM proxy (opt-in, paling kompleks)
             → extensions/mitm/cert_manager.py, dns_hijacker.py, proxy_server.py
             → WAJIB: prompt sudo + rollback command

Commit E-08: feat(cli): add hermes extensions subcommand
             → list, info, enable, disable

Commit E-09: docs: update README tentang extensions system
```

---

## Catatan Kritis untuk Coding Agent

### JANGAN

- ❌ Membuat `ProviderAdapter` ABC baru — gunakan `ProviderProfile` yang sudah ada
- ❌ Memodifikasi `providers/base.py` tanpa backward compatibility
- ❌ Menambah litellm ke dependencies utama (gunakan extras)
- ❌ Menulis ke `hermes_state.py` DB — buat DB terpisah (`provider_usage.db`)
- ❌ Memodifikasi `trajectory_compressor.py` — sudah mature
- ❌ Menghapus kode existing di `providers/` atau `model_tools.py`
- ❌ Logging di INFO level — gunakan DEBUG untuk semua log internal gateway
- ❌ Import modul ekstensi di top-level core — gunakan dynamic import
- ❌ Menjalankan `sudo` otomatis tanpa konfirmasi user (khusus MITM)

### LAKUKAN

- ✅ Baca file yang akan diubah **secara penuh** sebelum mengedit
- ✅ Tulis test untuk setiap modul baru **sebelum** integrasi
- ✅ Gunakan `from __future__ import annotations` di semua file baru
- ✅ Gunakan `logging` bukan `print` untuk output internal
- ✅ Pastikan hermes tetap berjalan normal tanpa config `provider_gateway`
- ✅ Mask API key di semua log output (`sk-xxx...xxx` → `sk-***`)
- ✅ Buat file baru yang berdiri sendiri dulu, uji isolasi, baru integrasikan
- ✅ Ekstensi default OFF — wajib `enabled: false` di semua config
- ✅ Graceful degradation — ImportError → pesan informatif, bukan crash
- ✅ Baca [fitur-opsional-dan-arahan-implementasi.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/fitur-opsional-dan-arahan-implementasi.md) untuk detail lengkap ekstensi

### Referensi Source Code yang Harus Dibaca

| File | Alasan |
|------|--------|
| `providers/base.py` (185 baris) | Pahami ProviderProfile |
| `providers/__init__.py` (192 baris) | Pahami registry system |
| `plugins/model-providers/openrouter/__init__.py` | Contoh registrasi provider |
| `model_tools.py` (924 baris) | Pahami tool system |
| `hermes_constants.py` | Path dan konstanta |
| `hermes_logging.py` | Pola logging yang benar |
| `cli-config.yaml.example` | Format config existing |
| [fitur-opsional-dan-arahan-implementasi.md](file:///home/void/lab/git/hermes_agent/.plan_improvement/fitur-opsional-dan-arahan-implementasi.md) | Detail lengkap arsitektur ekstensi opsional |

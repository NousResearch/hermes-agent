# HERMES AGENT — Panduan Peningkatan untuk Coding Agent

> **Dokumen ini ditujukan untuk coding agent** (Claude Code, Codex, Cursor, Cline, dll.)
> yang bertugas mengimplementasikan peningkatan pada repositori `hermes_agent`.
> Baca seluruh dokumen ini sebelum menulis satu baris kode pun.

---

## Daftar Isi

1. [Konteks & Tujuan](#1-konteks--tujuan)
2. [Pemahaman Arsitektur Saat Ini](#2-pemahaman-arsitektur-saat-ini)
3. [Prinsip Pengembangan](#3-prinsip-pengembangan)
4. [Peta Peningkatan (Feature Map)](#4-peta-peningkatan-feature-map)
5. [Implementasi P1 — Smart Provider Gateway](#5-implementasi-p1--smart-provider-gateway)
6. [Implementasi P2 — LiteLLM Unified Backend](#6-implementasi-p2--litellm-unified-backend)
7. [Implementasi P3 — Health Dashboard & Semantic Cache](#7-implementasi-p3--health-dashboard--semantic-cache)
8. [Implementasi P4 — Local Model & Proxy Support](#8-implementasi-p4--local-model--proxy-support)
9. [Implementasi P5 — Key Rotation & Guardrails](#9-implementasi-p5--key-rotation--guardrails)
10. [Standar Kode & Testing](#10-standar-kode--testing)
11. [Urutan Commit yang Disarankan](#11-urutan-commit-yang-disarankan)
12. [Referensi Sumber](#12-referensi-sumber)

---

## 1. Konteks & Tujuan

### Repositori target
`https://github.com/fajarkurnia0388/hermes_agent`
(fork dari `NousResearch/hermes-agent`)

### Repositori sumber inspirasi
| Repo | Fitur utama yang diadopsi |
|------|--------------------------|
| `fajarkurnia0388/9router` | RTK token compression, auto-fallback tier, format translation, quota tracking |
| `fajarkurnia0388/OmniRoute` | 177 provider, 14 routing strategy, circuit breaker, Caveman mode, proxy geo-bypass, guardrails |
| `fajarkurnia0388/CLIProxyAPI` | Go-based proxy reference untuk desain protocol layer |
| `fajarkurnia0388/litellm` | Python SDK unifikasi 100+ LLM, retry logic, cost tracking, streaming |

### Fokus utama
**Ketersediaan model AI dari banyak provider** — hermes harus bisa menggunakan 100+ model dari 50+ provider dengan zero-downtime melalui fallback otomatis, kompresi token real-time, dan monitoring kesehatan provider.

### Yang TIDAK boleh diubah
- Sistem skills (skills/ dan optional-skills/)
- Memory system (hermes_state.py, FTS5 session search)
- Cron scheduler (cron/)
- Messaging gateway untuk Telegram/Discord/WA/Signal (gateway/)
- ACP adapter (acp_adapter/, acp_registry/)
- MCP server (mcp_serve.py)
- Trajectory compression untuk training (trajectory_compressor.py)
- Antarmuka CLI/TUI yang sudah ada

---

## 2. Pemahaman Arsitektur Saat Ini

### Struktur direktori kunci
```
hermes_agent/
├── agent/                  # Core agent loop
├── providers/              # Adapter per provider (AKAN DIREFACTOR)
├── gateway/                # Messaging platforms (JANGAN DIUBAH)
├── tools/                  # Tool implementations
├── skills/                 # Procedural memory
├── optional-skills/        # Optional skill packs
├── optional-mcps/          # Optional MCP servers
├── acp_adapter/            # ACP protocol adapter
├── acp_registry/           # ACP service registry
├── cron/                   # Scheduled tasks
├── plugins/                # Plugin system
├── hermes_cli/             # CLI implementation
├── ui-tui/                 # TUI implementation
├── web/                    # Web UI
├── model_tools.py          # Model selection utilities (AKAN DIPERLUAS)
├── hermes_state.py         # State management
├── hermes_constants.py     # Constants
├── toolsets.py             # Tool configuration
├── run_agent.py            # Entry point
└── cli.py                  # CLI entry point
```

### Alur provider saat ini
```
User input
    → cli.py / run_agent.py
    → agent/  (agent loop)
    → model_tools.py  (model selection)
    → providers/<provider_name>/  (adapter)
    → API eksternal
```

### Provider yang sudah ada (providers/)
Periksa isi direktori `providers/` sebelum mulai. Provider yang kemungkinan sudah ada:
- Anthropic (Claude)
- OpenAI
- OpenRouter
- NovitaAI
- NVIDIA NIM
- Kimi/Moonshot
- MiniMax
- GLM / z.ai
- Xiaomi MiMo
- HuggingFace

---

## 3. Prinsip Pengembangan

### 3.1 Backward compatibility adalah hukum
Setiap kode baru WAJIB backward compatible. User yang sudah menggunakan `hermes model` atau `hermes setup` tidak boleh mengalami breaking change. Feature baru harus opt-in dengan flag atau config.

### 3.2 Python-first, tidak ada Node.js dependency baru
Hermes adalah proyek Python. Jangan menambahkan dependency Node.js. Semua implementasi dalam Python murni kecuali yang memang sudah Node.js (hermes_cli pakai TypeScript).

### 3.3 Satu modul, satu tanggung jawab
Buat modul terpisah untuk setiap fitur baru. Jangan menambahkan ratusan baris ke file yang sudah ada.

### 3.4 Config-driven, bukan hardcoded
Semua behavior baru harus dapat dikonfigurasi lewat `cli-config.yaml` atau environment variable. Default yang aman harus selalu ada.

### 3.5 Fail gracefully
Jika komponen baru gagal (cache tidak tersambung, proxy timeout), hermes harus jatuh kembali ke perilaku lama tanpa crash.

### 3.6 Log semua keputusan routing
Setiap keputusan routing provider (fallback, retry, kompresi) harus di-log ke `hermes_logging.py` dengan level DEBUG, agar user bisa mengaudit tanpa noise di output normal.

---

## 4. Peta Peningkatan (Feature Map)

```
P1 ── Smart Provider Gateway ──────────────── DAMPAK TINGGI, mulai dari sini
│     ├── RTK token compression (real-time)
│     ├── Caveman output compression (opsional)
│     ├── Auto-fallback 4 tier
│     ├── Format translation (OpenAI ↔ Claude ↔ Gemini)
│     ├── Multi-account round-robin
│     ├── Quota tracking per provider
│     └── Circuit breaker

P2 ── LiteLLM Unified Backend ─────────────── FONDASI PROVIDER
│     ├── LiteLLM sebagai backend pilihan (opt-in)
│     ├── Provider registry otomatis dari LiteLLM
│     ├── Cost tracking terintegrasi
│     └── Retry dengan exponential backoff

P3 ── Health Dashboard & Semantic Cache ────── OBSERVABILITY
│     ├── Provider health monitor (latency, error rate)
│     ├── Auto-disable provider yang down
│     ├── Semantic cache (Redis atau in-memory)
│     └── Dashboard status di CLI (`hermes status`)

P4 ── Local Model & Proxy ──────────────────── PRIVASI & AKSESIBILITAS
│     ├── Ollama integration
│     ├── llama.cpp / LM Studio support
│     ├── HTTP/SOCKS5 proxy per provider
│     └── TLS fingerprint stealth (opsional)

P5 ── Key Rotation & Guardrails ────────────── KEAMANAN & KUALITAS
      ├── API key pool & auto-rotation
      ├── PII detection sebelum kirim ke provider
      ├── Prompt injection guard
      └── Golden-set eval framework
```

---

## 5. Implementasi P1 — Smart Provider Gateway

### 5.1 Buat modul baru: `provider_gateway/`

```
hermes_agent/
└── provider_gateway/
    ├── __init__.py
    ├── gateway.py          # Entry point utama
    ├── router.py           # Logic routing & fallback
    ├── compression.py      # RTK + Caveman token compression
    ├── translator.py       # Format translation
    ├── quota_tracker.py    # Quota & usage tracking
    ├── circuit_breaker.py  # Circuit breaker per provider
    ├── combo.py            # Combo (rantai fallback) management
    └── models.py           # Data models (Pydantic)
```

### 5.2 `provider_gateway/models.py`

```python
"""Data models untuk provider gateway."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class ProviderTier(str, Enum):
    SUBSCRIPTION = "subscription"   # Claude Pro, Codex Plus — sudah bayar
    API_KEY = "api_key"             # OpenRouter, Anthropic direct, dll.
    CHEAP = "cheap"                 # GLM $0.5/1M, MiniMax $0.2/1M
    FREE = "free"                   # Kiro AI, OpenCode Free, Vertex credits


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal, request lewat
    OPEN = "open"           # Provider down, request diblokir
    HALF_OPEN = "half_open" # Sedang probe apakah provider sudah pulih


@dataclass
class ProviderConfig:
    """Konfigurasi satu provider."""
    name: str                           # e.g. "openrouter", "kiro", "glm"
    tier: ProviderTier
    base_url: str
    api_key: Optional[str] = None
    api_keys: list[str] = field(default_factory=list)   # Pool untuk rotation
    models: list[str] = field(default_factory=list)
    max_tokens_per_request: Optional[int] = None
    rate_limit_rpm: Optional[int] = None
    proxy_url: Optional[str] = None
    weight: float = 1.0                # Untuk weighted round-robin
    enabled: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class ComboEntry:
    """Satu model dalam sebuah combo/rantai fallback."""
    provider: str
    model: str
    priority: int = 0                  # Lebih kecil = lebih diutamakan
    compression: str = "none"          # "none" | "rtk" | "caveman" | "stacked"


@dataclass
class RoutingDecision:
    """Hasil keputusan routing, untuk logging dan audit."""
    selected_provider: str
    selected_model: str
    reason: str                        # "primary" | "fallback_tier2" | "circuit_open" dll.
    compression_applied: str
    tokens_before: int = 0
    tokens_after: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProviderHealth:
    """State kesehatan satu provider."""
    provider: str
    circuit_state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure: float = 0.0
    last_success: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    error_rate_1min: float = 0.0
    consecutive_failures: int = 0
```

### 5.3 `provider_gateway/compression.py`

Implementasi RTK token compression untuk output tool (`git diff`, `grep`, `ls`, dll.) dan Caveman mode untuk output respons.

```python
"""
RTK + Caveman token compression.

RTK: kompresi tool_result content (output shell commands).
Caveman: kompresi input prompt untuk respons lebih ringkas.
Stacked: RTK dulu, lalu Caveman — hemat 78-95% pada sesi tool-heavy.
"""
from __future__ import annotations
import re
from typing import Any


# ── RTK Filters ──────────────────────────────────────────────────────────────

def _compress_git_diff(text: str) -> str:
    """Kompres output git diff: hapus context lines yang tidak berubah."""
    lines = text.splitlines(keepends=True)
    out = []
    skip_context = 0
    for line in lines:
        if line.startswith(("---", "+++", "@@", "diff ", "index ")):
            out.append(line)
            skip_context = 0
        elif line.startswith(("+", "-")):
            out.append(line)
            skip_context = 0
        elif skip_context < 2:
            out.append(line)
            skip_context += 1
        # else: skip context lines yang berulang
    return "".join(out)


def _compress_ls_output(text: str) -> str:
    """Kompres output ls/find: hapus permission bits, simpan nama file."""
    lines = text.strip().splitlines()
    compressed = []
    for line in lines:
        # Hapus kolom permission, owner, size, date — simpan path
        parts = line.split()
        if len(parts) >= 9 and parts[0].startswith(("-", "d", "l")):
            compressed.append(parts[-1])
        else:
            compressed.append(line)
    return "\n".join(compressed)


def _compress_grep_output(text: str) -> str:
    """Kompres output grep: hapus duplikat, batasi per file."""
    seen: set[str] = set()
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            out.append(line)
    return "\n".join(out[:200])  # cap 200 baris


def _smart_truncate(text: str, max_chars: int = 8000) -> str:
    """Truncate cerdas: simpan awal dan akhir, sisipkan marker di tengah."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n[... {len(text) - max_chars} karakter dipotong oleh RTK ...]\n\n"
        + text[-half:]
    )


def _detect_rtk_filter(text: str) -> str:
    """Deteksi tipe konten dari 1KB pertama untuk pilih filter."""
    sample = text[:1024]
    if "diff --git" in sample or sample.startswith("--- a/"):
        return "git_diff"
    if re.search(r"^[-d][rwx-]{9}", sample, re.MULTILINE):
        return "ls"
    if re.search(r"^\S+:\d+:", sample, re.MULTILINE):
        return "grep"
    if len(text) > 6000:
        return "truncate"
    return "none"


def apply_rtk(tool_result_content: str) -> tuple[str, bool]:
    """
    Terapkan RTK compression pada tool_result content.

    Returns:
        (compressed_text, was_modified)
    """
    if not tool_result_content or len(tool_result_content) < 500:
        return tool_result_content, False

    filter_type = _detect_rtk_filter(tool_result_content)
    original_len = len(tool_result_content)

    try:
        if filter_type == "git_diff":
            result = _compress_git_diff(tool_result_content)
        elif filter_type == "ls":
            result = _compress_ls_output(tool_result_content)
        elif filter_type == "grep":
            result = _compress_grep_output(tool_result_content)
        elif filter_type == "truncate":
            result = _smart_truncate(tool_result_content)
        else:
            return tool_result_content, False

        # Jika hasil kompresi malah lebih besar, kembalikan asli
        if len(result) >= original_len:
            return tool_result_content, False

        return result, True

    except Exception:
        # Jangan pernah crash request hanya karena kompresi gagal
        return tool_result_content, False


# ── Caveman Mode ──────────────────────────────────────────────────────────────

_CAVEMAN_INJECTION = (
    "\n\n[SYSTEM INSTRUCTION — RESPOND CONCISELY]\n"
    "Reply short. No filler. Technical terms OK. "
    "Skip pleasantries. Substance only.\n"
)


def apply_caveman(messages: list[dict]) -> list[dict]:
    """
    Injeksikan Caveman instruction ke system prompt.
    Hemat ~65% output tokens. Hanya untuk sesi coding/teknis.
    """
    import copy
    msgs = copy.deepcopy(messages)
    # Cari system message yang sudah ada
    for msg in msgs:
        if msg.get("role") == "system":
            msg["content"] = msg["content"] + _CAVEMAN_INJECTION
            return msgs
    # Kalau tidak ada, prepend
    msgs.insert(0, {"role": "system", "content": _CAVEMAN_INJECTION.strip()})
    return msgs


# ── Compress Request ──────────────────────────────────────────────────────────

def compress_request(
    messages: list[dict],
    compression_mode: str = "rtk",  # "none" | "rtk" | "caveman" | "stacked"
) -> tuple[list[dict], dict]:
    """
    Entry point utama. Kompres request sebelum dikirim ke provider.

    Returns:
        (compressed_messages, stats_dict)
    """
    import copy, json

    stats = {
        "mode": compression_mode,
        "rtk_applied": 0,
        "tokens_saved_estimate": 0,
    }

    if compression_mode == "none":
        return messages, stats

    msgs = copy.deepcopy(messages)

    # RTK: kompres tool_result
    if compression_mode in ("rtk", "stacked"):
        for msg in msgs:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        original = block.get("content", "")
                        if isinstance(original, str):
                            compressed, modified = apply_rtk(original)
                            if modified:
                                saved = len(original) - len(compressed)
                                stats["rtk_applied"] += 1
                                stats["tokens_saved_estimate"] += saved // 4
                                block["content"] = compressed

    # Caveman: injeksi instruction
    if compression_mode in ("caveman", "stacked"):
        msgs = apply_caveman(msgs)

    return msgs, stats
```

### 5.4 `provider_gateway/circuit_breaker.py`

```python
"""
Circuit breaker untuk setiap provider.

States:
  CLOSED   → request normal
  OPEN     → provider down, skip selama cooldown
  HALF_OPEN → kirim satu probe request untuk cek recovery
"""
from __future__ import annotations
import time
import threading
from .models import CircuitState, ProviderHealth


class CircuitBreaker:
    """
    Thread-safe circuit breaker.

    Konfigurasi default:
    - failure_threshold: 5 kegagalan berturut → OPEN
    - recovery_timeout: 60 detik sebelum coba HALF_OPEN
    - success_threshold: 2 sukses berturut dari HALF_OPEN → CLOSED
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self._lock = threading.Lock()
        self._providers: dict[str, ProviderHealth] = {}

    def _get_health(self, provider: str) -> ProviderHealth:
        if provider not in self._providers:
            self._providers[provider] = ProviderHealth(provider=provider)
        return self._providers[provider]

    def is_available(self, provider: str) -> bool:
        """Apakah provider boleh menerima request saat ini?"""
        with self._lock:
            health = self._get_health(provider)

            if health.circuit_state == CircuitState.CLOSED:
                return True

            if health.circuit_state == CircuitState.OPEN:
                # Cek apakah sudah waktunya probe
                elapsed = time.time() - health.last_failure
                if elapsed >= self.recovery_timeout:
                    health.circuit_state = CircuitState.HALF_OPEN
                    return True
                return False

            # HALF_OPEN: biarkan satu request
            return True

    def record_success(self, provider: str, latency_ms: float = 0.0):
        """Catat request sukses."""
        with self._lock:
            health = self._get_health(provider)
            health.last_success = time.time()
            health.consecutive_failures = 0

            if health.circuit_state == CircuitState.HALF_OPEN:
                # Perlu beberapa sukses untuk close kembali
                health.failure_count = max(0, health.failure_count - 1)
                if health.failure_count == 0:
                    health.circuit_state = CircuitState.CLOSED

            # Update latency dengan exponential moving average
            if latency_ms > 0:
                alpha = 0.1
                health.latency_p50 = (
                    alpha * latency_ms + (1 - alpha) * health.latency_p50
                    if health.latency_p50 > 0 else latency_ms
                )

    def record_failure(self, provider: str, error: Exception = None):
        """Catat request gagal. Mungkin trip circuit."""
        with self._lock:
            health = self._get_health(provider)
            health.failure_count += 1
            health.consecutive_failures += 1
            health.last_failure = time.time()

            if (
                health.circuit_state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
                and health.consecutive_failures >= self.failure_threshold
            ):
                health.circuit_state = CircuitState.OPEN

    def get_all_health(self) -> dict[str, ProviderHealth]:
        """Snapshot kesehatan semua provider."""
        with self._lock:
            return dict(self._providers)

    def reset(self, provider: str):
        """Reset manual circuit breaker (untuk testing atau admin override)."""
        with self._lock:
            if provider in self._providers:
                self._providers[provider] = ProviderHealth(provider=provider)
```

### 5.5 `provider_gateway/quota_tracker.py`

```python
"""
Tracking quota dan usage per provider.
Disimpan ke file JSON agar persist antar sesi.
"""
from __future__ import annotations
import json
import os
import time
import threading
from pathlib import Path
from typing import Optional


class QuotaTracker:
    """
    Track token usage dan estimated cost per provider.
    Reset otomatis berdasarkan window (5h, daily, weekly, monthly).
    """

    RESET_WINDOWS = {
        "5h": 5 * 3600,
        "daily": 86400,
        "weekly": 7 * 86400,
        "monthly": 30 * 86400,
    }

    # Cost perkiraan per 1M input tokens (USD)
    COST_PER_1M_INPUT = {
        "anthropic": 3.0,
        "openai": 5.0,
        "openrouter": 3.0,
        "glm": 0.6,
        "minimax": 0.2,
        "kiro": 0.0,        # Free
        "opencode": 0.0,    # Free
        "groq": 0.05,
        "deepseek": 0.14,
    }

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = str(Path.home() / ".hermes" / "quota")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.data_dir / "quota.json"
        self._lock = threading.Lock()
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text())
            except Exception:
                pass
        return {}

    def _save(self):
        """Simpan ke disk (dipanggil dalam lock)."""
        try:
            self._file.write_text(json.dumps(self._data, indent=2))
        except Exception:
            pass

    def _get_provider_data(self, provider: str) -> dict:
        if provider not in self._data:
            self._data[provider] = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_requests": 0,
                "total_cost_usd": 0.0,
                "last_reset": time.time(),
                "window_tokens": 0,
                "window_start": time.time(),
            }
        return self._data[provider]

    def record_usage(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
    ):
        """Catat penggunaan setelah request selesai."""
        with self._lock:
            d = self._get_provider_data(provider)
            d["total_input_tokens"] += input_tokens
            d["total_output_tokens"] += output_tokens
            d["total_requests"] += 1
            d["window_tokens"] += input_tokens + output_tokens

            # Estimasi cost
            cost_rate = self.COST_PER_1M_INPUT.get(provider.lower(), 1.0)
            d["total_cost_usd"] += (input_tokens / 1_000_000) * cost_rate

            self._save()

    def get_summary(self, provider: str) -> dict:
        """Ambil summary usage untuk satu provider."""
        with self._lock:
            d = self._get_provider_data(provider)
            return {
                "provider": provider,
                "total_requests": d["total_requests"],
                "total_input_tokens": d["total_input_tokens"],
                "total_output_tokens": d["total_output_tokens"],
                "total_cost_usd": round(d["total_cost_usd"], 4),
            }

    def get_all_summaries(self) -> list[dict]:
        """Summary semua provider, diurutkan berdasarkan total request."""
        with self._lock:
            summaries = []
            for provider in self._data:
                summaries.append(self.get_summary(provider))
            return sorted(summaries, key=lambda x: x["total_requests"], reverse=True)
```

### 5.6 `provider_gateway/router.py`

```python
"""
Smart router: pilih provider terbaik berdasarkan tier, health, dan quota.
Ini adalah otak dari provider gateway.
"""
from __future__ import annotations
import time
import logging
from typing import Optional

from .models import ProviderConfig, ComboEntry, RoutingDecision, ProviderTier
from .circuit_breaker import CircuitBreaker
from .quota_tracker import QuotaTracker
from .compression import compress_request

logger = logging.getLogger("hermes.gateway.router")


class ProviderRouter:
    """
    Router utama yang memilih provider untuk setiap request.

    Fallback tiers:
      1. SUBSCRIPTION (sudah bayar, prioritas tertinggi)
      2. API_KEY (bayar per-token, biaya sedang)
      3. CHEAP (provider murah: GLM, MiniMax, dll.)
      4. FREE (Kiro AI, OpenCode Free, dll.)
    """

    def __init__(
        self,
        providers: list[ProviderConfig],
        circuit_breaker: Optional[CircuitBreaker] = None,
        quota_tracker: Optional[QuotaTracker] = None,
    ):
        self.providers = {p.name: p for p in providers}
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.quota_tracker = quota_tracker or QuotaTracker()
        self._rr_index: dict[str, int] = {}  # Round-robin index per provider

    def _get_providers_by_tier(self, tier: ProviderTier) -> list[ProviderConfig]:
        """Ambil semua provider yang aktif dan tier-nya cocok."""
        return [
            p for p in self.providers.values()
            if p.tier == tier
            and p.enabled
            and self.circuit_breaker.is_available(p.name)
        ]

    def _select_from_pool(self, pool: list[ProviderConfig]) -> Optional[ProviderConfig]:
        """Round-robin selection dari pool provider yang tersedia."""
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0]

        # Gunakan round-robin berdasarkan nama gabungan pool
        key = "_".join(sorted(p.name for p in pool))
        idx = self._rr_index.get(key, 0) % len(pool)
        self._rr_index[key] = idx + 1
        return pool[idx]

    def select_provider(
        self,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        fallback: bool = True,
    ) -> Optional[tuple[ProviderConfig, str, str]]:
        """
        Pilih provider dan model terbaik.

        Returns:
            (provider_config, model_name, reason) atau None jika semua gagal.
        """
        # Coba provider pilihan user dulu
        if preferred_provider and preferred_provider in self.providers:
            p = self.providers[preferred_provider]
            if p.enabled and self.circuit_breaker.is_available(p.name):
                model = preferred_model or (p.models[0] if p.models else "default")
                return p, model, "user_preferred"

        if not fallback:
            return None

        # Fallback berdasarkan tier
        tier_order = [
            ProviderTier.SUBSCRIPTION,
            ProviderTier.API_KEY,
            ProviderTier.CHEAP,
            ProviderTier.FREE,
        ]

        for tier in tier_order:
            pool = self._get_providers_by_tier(tier)
            selected = self._select_from_pool(pool)
            if selected:
                model = preferred_model or (
                    selected.models[0] if selected.models else "default"
                )
                return selected, model, f"fallback_{tier.value}"

        logger.error("Semua provider tidak tersedia. Tidak ada yang bisa dipilih.")
        return None

    def route_request(
        self,
        messages: list[dict],
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        compression_mode: str = "rtk",
        max_retries: int = 3,
    ) -> tuple[list[dict], RoutingDecision]:
        """
        Route request ke provider terpilih, dengan kompresi dan fallback.

        Returns:
            (compressed_messages, routing_decision)
        """
        # Hitung ukuran sebelum kompresi
        original_size = sum(
            len(str(m.get("content", ""))) for m in messages
        )

        # Terapkan kompresi
        compressed_msgs, compression_stats = compress_request(
            messages, compression_mode
        )

        compressed_size = sum(
            len(str(m.get("content", ""))) for m in compressed_msgs
        )

        # Pilih provider
        result = self.select_provider(preferred_provider, preferred_model)
        if result is None:
            # Fallback darurat: kembalikan pesan asli tanpa routing
            logger.warning("Tidak ada provider tersedia. Mengembalikan pesan asli.")
            return messages, RoutingDecision(
                selected_provider="none",
                selected_model="none",
                reason="no_provider_available",
                compression_applied="none",
            )

        provider, model, reason = result

        decision = RoutingDecision(
            selected_provider=provider.name,
            selected_model=model,
            reason=reason,
            compression_applied=compression_mode,
            tokens_before=original_size // 4,      # Estimasi kasar
            tokens_after=compressed_size // 4,
        )

        logger.debug(
            f"Routing ke {provider.name}/{model} "
            f"(reason={reason}, "
            f"compression={compression_mode}, "
            f"saved≈{decision.tokens_before - decision.tokens_after} tokens)"
        )

        return compressed_msgs, decision
```

### 5.7 `provider_gateway/gateway.py`

```python
"""
Entry point utama Provider Gateway.
Ini yang dipanggil dari agent loop hermes.
"""
from __future__ import annotations
import logging
import time
from typing import Optional, Any, Generator

from .models import ProviderConfig, ProviderTier
from .router import ProviderRouter
from .circuit_breaker import CircuitBreaker
from .quota_tracker import QuotaTracker

logger = logging.getLogger("hermes.gateway")


class ProviderGateway:
    """
    Facade utama yang dipakai oleh agent loop.

    Cara pakai:
        gateway = ProviderGateway.from_config(config_dict)
        response = gateway.complete(messages, model="openrouter/claude-opus-4-6")
    """

    def __init__(
        self,
        providers: list[ProviderConfig],
        default_compression: str = "rtk",
        max_retries: int = 3,
    ):
        self.circuit_breaker = CircuitBreaker()
        self.quota_tracker = QuotaTracker()
        self.router = ProviderRouter(
            providers=providers,
            circuit_breaker=self.circuit_breaker,
            quota_tracker=self.quota_tracker,
        )
        self.default_compression = default_compression
        self.max_retries = max_retries

    @classmethod
    def from_config(cls, config: dict) -> "ProviderGateway":
        """
        Buat gateway dari config dict (dari cli-config.yaml).

        Contoh config:
            gateway:
              default_compression: rtk
              max_retries: 3
              providers:
                - name: openrouter
                  tier: api_key
                  base_url: https://openrouter.ai/api/v1
                  api_key: sk-xxx
                  models: [anthropic/claude-opus-4-6]
                - name: kiro
                  tier: free
                  base_url: https://kiro.ai/api/v1
        """
        providers = []
        for p_conf in config.get("providers", []):
            providers.append(ProviderConfig(
                name=p_conf["name"],
                tier=ProviderTier(p_conf.get("tier", "api_key")),
                base_url=p_conf["base_url"],
                api_key=p_conf.get("api_key"),
                api_keys=p_conf.get("api_keys", []),
                models=p_conf.get("models", []),
                rate_limit_rpm=p_conf.get("rate_limit_rpm"),
                proxy_url=p_conf.get("proxy_url"),
                weight=p_conf.get("weight", 1.0),
                enabled=p_conf.get("enabled", True),
            ))

        return cls(
            providers=providers,
            default_compression=config.get("default_compression", "rtk"),
            max_retries=config.get("max_retries", 3),
        )

    def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        compression: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Kirim request ke provider terpilih, dengan fallback otomatis.

        model format: "provider/model_name" atau hanya "model_name"
        """
        # Parse model string
        preferred_provider, preferred_model = None, None
        if model and "/" in model:
            parts = model.split("/", 1)
            preferred_provider, preferred_model = parts[0], parts[1]
        elif model:
            preferred_model = model

        compression_mode = compression or self.default_compression

        # Route request
        compressed_msgs, decision = self.router.route_request(
            messages=messages,
            preferred_provider=preferred_provider,
            preferred_model=preferred_model,
            compression_mode=compression_mode,
        )

        if decision.selected_provider == "none":
            raise RuntimeError("Tidak ada provider yang tersedia.")

        provider_config = self.router.providers.get(decision.selected_provider)
        if not provider_config:
            raise RuntimeError(f"Provider config tidak ditemukan: {decision.selected_provider}")

        # Eksekusi request dengan retry
        last_error = None
        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                response = self._call_provider(
                    provider=provider_config,
                    model=decision.selected_model,
                    messages=compressed_msgs,
                    stream=stream,
                    **kwargs,
                )

                latency_ms = (time.time() - start_time) * 1000
                self.circuit_breaker.record_success(
                    provider_config.name, latency_ms
                )

                # Catat usage jika ada info token di response
                if hasattr(response, "usage") and response.usage:
                    self.quota_tracker.record_usage(
                        provider=provider_config.name,
                        input_tokens=response.usage.get("prompt_tokens", 0),
                        output_tokens=response.usage.get("completion_tokens", 0),
                        model=decision.selected_model,
                    )

                return response

            except Exception as e:
                last_error = e
                self.circuit_breaker.record_failure(provider_config.name, e)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} gagal "
                    f"({provider_config.name}): {e}"
                )

                # Fallback ke provider lain jika ada
                if attempt < self.max_retries - 1:
                    result = self.router.select_provider(fallback=True)
                    if result:
                        provider_config, decision.selected_model, decision.reason = result

        raise RuntimeError(
            f"Semua {self.max_retries} attempt gagal. Error terakhir: {last_error}"
        )

    def _call_provider(
        self,
        provider: ProviderConfig,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Panggil provider API.
        Ini menggunakan interface standar OpenAI-compatible.
        Jika provider membutuhkan format lain, translate dulu di translator.py.
        """
        # Import di sini untuk menghindari circular import
        # Gunakan openai library yang sudah pasti ada di hermes
        import openai

        # Pilih API key dari pool (untuk rotation)
        api_key = provider.api_key
        if provider.api_keys:
            # Sederhana: ambil yang pertama, rotasi bisa ditambahkan nanti
            api_key = provider.api_keys[0]

        client = openai.OpenAI(
            base_url=provider.base_url,
            api_key=api_key or "none",
        )

        return client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )

    def get_status(self) -> dict:
        """Status semua provider untuk `hermes status`."""
        health = self.circuit_breaker.get_all_health()
        usage = self.quota_tracker.get_all_summaries()
        usage_map = {u["provider"]: u for u in usage}

        statuses = []
        for name, h in health.items():
            statuses.append({
                "provider": name,
                "circuit": h.circuit_state.value,
                "latency_p50_ms": round(h.latency_p50, 1),
                "consecutive_failures": h.consecutive_failures,
                "total_requests": usage_map.get(name, {}).get("total_requests", 0),
                "total_cost_usd": usage_map.get(name, {}).get("total_cost_usd", 0),
            })

        return {"providers": statuses}
```

### 5.8 Integrasi ke `model_tools.py`

Tambahkan fungsi berikut ke `model_tools.py` yang sudah ada. **Jangan hapus fungsi yang sudah ada — hanya tambahkan.**

```python
# Tambahkan di AKHIR model_tools.py

_gateway_instance: Optional["ProviderGateway"] = None


def get_provider_gateway(config: dict = None) -> "ProviderGateway":
    """
    Singleton factory untuk ProviderGateway.
    Dipanggil dari agent loop ketika mode gateway diaktifkan.
    """
    global _gateway_instance
    if _gateway_instance is None:
        from provider_gateway.gateway import ProviderGateway
        cfg = config or _load_gateway_config()
        _gateway_instance = ProviderGateway.from_config(cfg)
    return _gateway_instance


def _load_gateway_config() -> dict:
    """
    Load konfigurasi gateway dari cli-config.yaml atau .env.
    Kembalikan dict kosong jika tidak ada (gateway disabled).
    """
    import os
    from pathlib import Path
    import yaml

    config_file = Path.home() / ".hermes" / "cli-config.yaml"
    if config_file.exists():
        try:
            with open(config_file) as f:
                full_config = yaml.safe_load(f) or {}
            return full_config.get("provider_gateway", {})
        except Exception:
            pass
    return {}
```

### 5.9 Tambahkan ke `cli-config.yaml.example`

```yaml
# Provider Gateway — Smart routing dengan auto-fallback dan token compression
provider_gateway:
  enabled: true
  default_compression: rtk        # "none" | "rtk" | "caveman" | "stacked"
  max_retries: 3

  providers:
    # Tier FREE — selalu aktif sebagai fallback terakhir
    - name: kiro
      tier: free
      base_url: https://kiro.ai/api/v1
      api_key: ""
      models:
        - claude-sonnet-4.5
        - glm-5

    # Tier API_KEY — provider utama
    - name: openrouter
      tier: api_key
      base_url: https://openrouter.ai/api/v1
      api_key: "${OPENROUTER_API_KEY}"
      models:
        - anthropic/claude-opus-4-6
        - google/gemini-2.5-pro

    # Tier CHEAP — backup hemat
    - name: glm
      tier: cheap
      base_url: https://open.bigmodel.cn/api/paas/v4
      api_key: "${GLM_API_KEY}"
      models:
        - glm-5.1
        - glm-4.7

    # Tier SUBSCRIPTION — kalau punya Claude Pro
    - name: anthropic
      tier: subscription
      base_url: https://api.anthropic.com/v1
      api_key: "${ANTHROPIC_API_KEY}"
      models:
        - claude-opus-4-6
        - claude-sonnet-4-6
```

---

## 6. Implementasi P2 — LiteLLM Unified Backend

### 6.1 Tambahkan dependency

Di `pyproject.toml`, tambahkan ke dependencies:
```toml
litellm = ">=1.40.0"
```

### 6.2 Buat `provider_gateway/litellm_backend.py`

```python
"""
LiteLLM backend sebagai alternatif untuk _call_provider().
Aktifkan dengan: USE_LITELLM_BACKEND=true di .env

Keuntungan:
- 100+ provider tanpa kode adapter manual
- Retry dan fallback bawaan
- Cost tracking terintegrasi
- Format standar untuk semua provider
"""
from __future__ import annotations
import os
import logging
from typing import Any, Optional

logger = logging.getLogger("hermes.gateway.litellm")

_LITELLM_AVAILABLE = False
try:
    import litellm
    litellm.drop_params = True          # Abaikan parameter yang tidak didukung
    litellm.set_verbose = False
    _LITELLM_AVAILABLE = True
except ImportError:
    logger.warning("litellm tidak terinstall. Jalankan: pip install litellm")


def is_available() -> bool:
    return _LITELLM_AVAILABLE


def complete(
    model: str,
    messages: list[dict],
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    max_retries: int = 2,
    stream: bool = False,
    **kwargs,
) -> Any:
    """
    Panggil LLM via LiteLLM.

    model format sesuai LiteLLM:
      "openai/gpt-4o"
      "anthropic/claude-opus-4-6"
      "openrouter/anthropic/claude-opus-4-6"
      "ollama/llama3.2"
    """
    if not _LITELLM_AVAILABLE:
        raise ImportError("litellm tidak terinstall.")

    params = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "num_retries": max_retries,
    }

    if api_key:
        params["api_key"] = api_key
    if api_base:
        params["api_base"] = api_base

    params.update(kwargs)

    response = litellm.completion(**params)
    return response


def get_supported_models() -> list[str]:
    """Daftar semua model yang didukung LiteLLM."""
    if not _LITELLM_AVAILABLE:
        return []
    try:
        return list(litellm.model_list)
    except Exception:
        return []


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimasi biaya dalam USD menggunakan LiteLLM cost database."""
    if not _LITELLM_AVAILABLE:
        return 0.0
    try:
        cost = litellm.completion_cost(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return cost
    except Exception:
        return 0.0
```

### 6.3 Update `provider_gateway/gateway.py`

Tambahkan method berikut ke class `ProviderGateway`:

```python
def _call_provider_litellm(
    self,
    provider: ProviderConfig,
    model: str,
    messages: list[dict],
    stream: bool = False,
    **kwargs,
) -> Any:
    """
    Alternatif _call_provider menggunakan LiteLLM.
    Aktif jika USE_LITELLM_BACKEND=true di environment.
    """
    from .litellm_backend import complete as litellm_complete, is_available

    if not is_available():
        return self._call_provider(provider, model, messages, stream, **kwargs)

    # Format model untuk LiteLLM: "provider/model"
    litellm_model = f"{provider.name}/{model}"

    return litellm_complete(
        model=litellm_model,
        messages=messages,
        api_key=provider.api_key,
        api_base=provider.base_url,
        stream=stream,
        **kwargs,
    )
```

---

## 7. Implementasi P3 — Health Dashboard & Semantic Cache

### 7.1 Buat `provider_gateway/semantic_cache.py`

```python
"""
Semantic cache untuk menghindari request duplikat ke LLM.
Backend: in-memory (default) atau Redis (jika tersedia).

Cara kerja:
1. Hash prompt + model → lookup cache
2. Cache hit → kembalikan respons tersimpan
3. Cache miss → forward ke provider, simpan respons
"""
from __future__ import annotations
import hashlib
import json
import time
import threading
from typing import Optional, Any
from collections import OrderedDict


class SemanticCache:
    """
    In-memory LRU cache untuk LLM responses.

    Untuk production dengan Redis:
        cache = SemanticCache(backend="redis", redis_url="redis://localhost:6379")
    """

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: float = 3600.0,   # 1 jam default
        backend: str = "memory",
        redis_url: Optional[str] = None,
    ):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.backend = backend
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

        if backend == "redis" and redis_url:
            self._init_redis(redis_url)

    def _init_redis(self, redis_url: str):
        """Inisialisasi Redis backend jika tersedia."""
        try:
            import redis
            self._redis = redis.from_url(redis_url)
            self.backend = "redis"
        except ImportError:
            import logging
            logging.getLogger("hermes.cache").warning(
                "redis tidak terinstall, fallback ke in-memory cache. "
                "Install dengan: pip install redis"
            )
            self.backend = "memory"

    def _make_key(self, messages: list[dict], model: str) -> str:
        """Buat cache key dari messages dan model."""
        content = json.dumps(
            {"model": model, "messages": messages},
            sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, messages: list[dict], model: str) -> Optional[Any]:
        """Ambil dari cache. Return None jika miss atau expired."""
        key = self._make_key(messages, model)

        if self.backend == "redis" and hasattr(self, "_redis"):
            try:
                raw = self._redis.get(f"hermes:cache:{key}")
                if raw:
                    self._hits += 1
                    return json.loads(raw)
            except Exception:
                pass
            self._misses += 1
            return None

        # In-memory
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expires_at = self._cache[key]
            if time.time() > expires_at:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, messages: list[dict], model: str, response: Any):
        """Simpan ke cache."""
        key = self._make_key(messages, model)

        # Serialisasi response
        try:
            if hasattr(response, "model_dump"):
                value = response.model_dump()
            elif hasattr(response, "dict"):
                value = response.dict()
            else:
                value = response
        except Exception:
            return  # Jangan crash jika response tidak bisa diserialisasi

        if self.backend == "redis" and hasattr(self, "_redis"):
            try:
                self._redis.setex(
                    f"hermes:cache:{key}",
                    int(self.ttl),
                    json.dumps(value)
                )
            except Exception:
                pass
            return

        # In-memory
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Hapus yang paling lama

            self._cache[key] = (value, time.time() + self.ttl)
            self._cache.move_to_end(key)

    def get_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "size": len(self._cache) if self.backend == "memory" else "redis",
            "backend": self.backend,
        }

    def clear(self):
        """Kosongkan cache."""
        with self._lock:
            self._cache.clear()
```

### 7.2 Tambahkan command `hermes status`

Tambahkan ke `hermes_cli/` atau `cli.py` (sesuaikan dengan pola CLI yang sudah ada):

```python
def cmd_status(args):
    """
    hermes status — tampilkan status semua provider.

    Output:
        PROVIDER     CIRCUIT   LATENCY   REQUESTS   COST
        openrouter   CLOSED    145ms     1,234      $2.45
        kiro         CLOSED    89ms      456        $0.00
        glm          OPEN      —         23         $0.01
    """
    from model_tools import get_provider_gateway

    gateway = get_provider_gateway()
    status = gateway.get_status()

    print("\nProvider Status:")
    print(f"{'PROVIDER':<16} {'CIRCUIT':<10} {'LATENCY':<10} {'REQUESTS':<10} {'COST'}")
    print("─" * 65)

    for p in status["providers"]:
        circuit = p["circuit"].upper()
        latency = f"{p['latency_p50_ms']:.0f}ms" if p["latency_p50_ms"] > 0 else "—"
        cost = f"${p['total_cost_usd']:.2f}"
        print(
            f"{p['provider']:<16} {circuit:<10} {latency:<10} "
            f"{p['total_requests']:<10} {cost}"
        )

    cache_stats = gateway.cache.get_stats() if hasattr(gateway, "cache") else {}
    if cache_stats:
        print(f"\nCache: {cache_stats['hits']} hits / "
              f"{cache_stats['misses']} misses "
              f"({cache_stats['hit_rate']:.1%} hit rate)")
```

---

## 8. Implementasi P4 — Local Model & Proxy Support

### 8.1 Tambahkan Ollama provider

Buat `providers/ollama/` dengan struktur:
```
providers/ollama/
├── __init__.py
└── provider.py
```

```python
# providers/ollama/provider.py
"""
Ollama local model provider.
Pastikan Ollama berjalan: ollama serve

Model tersedia: ollama list
Pull model baru: ollama pull llama3.2
"""
from __future__ import annotations
import requests
import json
from typing import Iterator, Any


OLLAMA_BASE_URL = "http://localhost:11434"


def list_local_models() -> list[str]:
    """Daftar model yang sudah di-pull ke Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.ok:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


def is_running() -> bool:
    """Cek apakah Ollama server aktif."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return resp.ok
    except Exception:
        return False


def complete(
    model: str,
    messages: list[dict],
    stream: bool = False,
    base_url: str = OLLAMA_BASE_URL,
    **kwargs,
) -> Any:
    """
    Panggil Ollama dengan OpenAI-compatible endpoint.
    Ollama mendukung /v1/chat/completions sejak versi 0.1.24+
    """
    import openai

    client = openai.OpenAI(
        base_url=f"{base_url}/v1",
        api_key="ollama",   # Ollama tidak butuh API key
    )

    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        **kwargs,
    )
```

### 8.2 Tambahkan proxy support di `provider_gateway/gateway.py`

Update method `_call_provider` untuk mendukung proxy:

```python
def _call_provider(
    self,
    provider: ProviderConfig,
    model: str,
    messages: list[dict],
    stream: bool = False,
    **kwargs,
) -> Any:
    """Panggil provider API dengan dukungan proxy."""
    import openai
    import httpx

    api_key = provider.api_key
    if provider.api_keys:
        # Ambil key dari pool — round robin sederhana
        idx = hash(f"{provider.name}{int(time.time() // 60)}") % len(provider.api_keys)
        api_key = provider.api_keys[idx]

    # Setup HTTP client dengan proxy jika dikonfigurasi
    http_client = None
    if provider.proxy_url:
        http_client = httpx.Client(
            proxy=provider.proxy_url,
            timeout=httpx.Timeout(60.0),
        )

    client_kwargs = {
        "base_url": provider.base_url,
        "api_key": api_key or "none",
    }
    if http_client:
        client_kwargs["http_client"] = http_client

    client = openai.OpenAI(**client_kwargs)

    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        **kwargs,
    )
```

### 8.3 Konfigurasi proxy di `cli-config.yaml.example`

```yaml
provider_gateway:
  providers:
    - name: anthropic
      tier: api_key
      base_url: https://api.anthropic.com/v1
      api_key: "${ANTHROPIC_API_KEY}"
      # Proxy untuk akses dari region yang dibatasi
      proxy_url: "socks5://127.0.0.1:1080"

    - name: ollama
      tier: free
      base_url: http://localhost:11434
      api_key: ""
      models:
        - llama3.2
        - qwen2.5-coder:7b
        - deepseek-r1:8b
```

---

## 9. Implementasi P5 — Key Rotation & Guardrails

### 9.1 Buat `provider_gateway/guardrails.py`

```python
"""
Guardrails: filter konten sebelum dikirim ke provider.
Mencegah kebocoran data sensitif dan prompt injection.
"""
from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger("hermes.guardrails")


# ── PII Detection ─────────────────────────────────────────────────────────────

_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_id": re.compile(r"\b08[0-9]{8,11}\b"),        # Format nomor HP Indonesia
    "nik": re.compile(r"\b[0-9]{16}\b"),                  # NIK KTP Indonesia
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "api_key_generic": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
}


def detect_pii(text: str) -> list[str]:
    """
    Deteksi PII dalam teks.
    Return list tipe PII yang ditemukan.
    """
    found = []
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            found.append(pii_type)
    return found


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Redact PII dari teks. Return (redacted_text, list_of_redacted_types).
    """
    redacted_types = []
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            text = pattern.sub(f"[REDACTED:{pii_type.upper()}]", text)
            redacted_types.append(pii_type)
    return text, redacted_types


# ── Prompt Injection Detection ────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?different", re.IGNORECASE),
    re.compile(r"disregard\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+(have\s+no|don't\s+have)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+(are|were)|an?)\s+(?!assistant)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s*mode", re.IGNORECASE),   # "Do Anything Now"
]


def detect_injection(text: str) -> bool:
    """Return True jika ada indikasi prompt injection."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ── Main Guardrail Pipeline ───────────────────────────────────────────────────

class GuardrailResult:
    def __init__(self):
        self.blocked: bool = False
        self.block_reason: Optional[str] = None
        self.pii_detected: list[str] = []
        self.pii_redacted: bool = False
        self.injection_detected: bool = False
        self.modified_messages: Optional[list[dict]] = None


def run_guardrails(
    messages: list[dict],
    block_injection: bool = True,
    redact_pii: bool = False,          # Default OFF — hanya log
    block_on_pii: bool = False,        # Default OFF — terlalu agresif
) -> GuardrailResult:
    """
    Jalankan guardrail pipeline pada messages sebelum dikirim ke provider.

    Gunakan secara defensif: log dulu, block hanya jika yakin.
    """
    import copy
    result = GuardrailResult()
    msgs = copy.deepcopy(messages)

    for msg in msgs:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue

        # Cek injection
        if detect_injection(content):
            result.injection_detected = True
            if block_injection:
                result.blocked = True
                result.block_reason = "prompt_injection_detected"
                logger.warning(
                    f"Prompt injection terdeteksi di pesan role={msg.get('role')}. "
                    "Request diblokir."
                )
                return result

        # Cek PII
        pii_types = detect_pii(content)
        if pii_types:
            result.pii_detected.extend(pii_types)
            logger.info(f"PII terdeteksi ({', '.join(pii_types)}) di pesan. "
                       "Pertimbangkan redact_pii=True.")

            if redact_pii:
                msg["content"], _ = redact_pii(content)  # type: ignore[assignment]
                result.pii_redacted = True

            if block_on_pii:
                result.blocked = True
                result.block_reason = f"pii_detected:{','.join(pii_types)}"
                return result

    result.modified_messages = msgs
    return result
```

### 9.2 Integrasi ke gateway

Tambahkan ke `ProviderGateway.complete()`:

```python
# Tambahkan sebelum pemanggilan router.route_request()
if self.guardrails_enabled:
    from .guardrails import run_guardrails
    guard_result = run_guardrails(
        messages=messages,
        block_injection=self.block_injection,
        redact_pii=self.redact_pii,
    )
    if guard_result.blocked:
        raise PermissionError(
            f"Request diblokir oleh guardrail: {guard_result.block_reason}"
        )
    if guard_result.modified_messages:
        messages = guard_result.modified_messages
```

---

## 10. Standar Kode & Testing

### 10.1 Gaya kode

Semua kode baru harus mengikuti standar yang sudah ada di hermes:
- Python 3.11+
- Type hints di semua function signature
- Docstring singkat di setiap class dan method publik
- `from __future__ import annotations` di semua file baru
- Gunakan `logging` bukan `print` untuk output internal

### 10.2 Test yang wajib ditulis

Buat direktori `tests/test_gateway/` dengan minimal:

```
tests/test_gateway/
├── __init__.py
├── test_compression.py       # Unit test untuk RTK dan Caveman
├── test_circuit_breaker.py   # Test state transitions
├── test_router.py            # Test fallback logic
├── test_quota_tracker.py     # Test usage tracking
├── test_guardrails.py        # Test PII dan injection detection
└── test_semantic_cache.py    # Test cache hit/miss/eviction
```

Contoh test minimal untuk compression:

```python
# tests/test_gateway/test_compression.py
import pytest
from provider_gateway.compression import apply_rtk, compress_request


def test_rtk_git_diff_reduces_size():
    diff = "diff --git a/main.py b/main.py\n"
    diff += "--- a/main.py\n+++ b/main.py\n@@ -1,5 +1,5 @@\n"
    diff += " unchanged\n unchanged\n unchanged\n"
    diff += "-old line\n+new line\n"
    diff += " more context\n more context\n more context\n" * 50

    compressed, modified = apply_rtk(diff)
    assert modified is True
    assert len(compressed) < len(diff)


def test_rtk_small_content_unchanged():
    small = "just a short message"
    compressed, modified = apply_rtk(small)
    assert modified is False
    assert compressed == small


def test_rtk_never_crashes():
    """RTK tidak boleh crash bahkan pada input aneh."""
    weird_inputs = [
        "",
        "\x00\x01\x02",
        "a" * 100_000,
        None,   # type: ignore
    ]
    for inp in weird_inputs:
        try:
            apply_rtk(inp or "")
        except Exception as e:
            pytest.fail(f"apply_rtk crash pada input '{inp}': {e}")


def test_compress_request_stacked():
    messages = [
        {"role": "system", "content": "Kamu adalah asisten."},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "content": "diff --git a/x.py b/x.py\n" + "context\n" * 100
                }
            ],
        },
    ]
    compressed, stats = compress_request(messages, "stacked")
    assert stats["rtk_applied"] > 0
    assert stats["tokens_saved_estimate"] > 0
```

### 10.3 Cara jalankan test

```bash
# Jalankan semua test gateway
python -m pytest tests/test_gateway/ -v

# Jalankan dengan coverage
python -m pytest tests/test_gateway/ --cov=provider_gateway --cov-report=term-missing

# Test cepat tanpa dependency eksternal
python -m pytest tests/test_gateway/test_compression.py tests/test_gateway/test_circuit_breaker.py -v
```

---

## 11. Urutan Commit yang Disarankan

Implementasikan dalam urutan ini untuk meminimalkan risiko:

```
Commit 1: feat(gateway): add provider_gateway module skeleton
  - provider_gateway/__init__.py
  - provider_gateway/models.py
  - Tests untuk models

Commit 2: feat(gateway): implement RTK + Caveman compression
  - provider_gateway/compression.py
  - tests/test_gateway/test_compression.py
  - TIDAK ada integrasi ke agent loop dulu

Commit 3: feat(gateway): implement circuit breaker
  - provider_gateway/circuit_breaker.py
  - tests/test_gateway/test_circuit_breaker.py

Commit 4: feat(gateway): implement quota tracker
  - provider_gateway/quota_tracker.py
  - tests/test_gateway/test_quota_tracker.py

Commit 5: feat(gateway): implement provider router
  - provider_gateway/router.py
  - tests/test_gateway/test_router.py

Commit 6: feat(gateway): implement gateway facade
  - provider_gateway/gateway.py
  - Integrasi ke model_tools.py (opt-in via config)
  - Update cli-config.yaml.example

Commit 7: feat(cache): implement semantic cache
  - provider_gateway/semantic_cache.py
  - tests/test_gateway/test_semantic_cache.py

Commit 8: feat(providers): add Ollama provider
  - providers/ollama/
  - Dokumentasi setup Ollama

Commit 9: feat(guardrails): add PII and injection detection
  - provider_gateway/guardrails.py
  - tests/test_gateway/test_guardrails.py
  - Default OFF untuk tidak breaking existing behavior

Commit 10: feat(cli): add hermes status command
  - Tampilkan status semua provider
  - Tampilkan cache stats
  - Tampilkan quota usage

Commit 11: feat(litellm): add LiteLLM backend (opt-in)
  - provider_gateway/litellm_backend.py
  - Dokumentasi cara aktifkan

Commit 12: docs: update README dengan fitur gateway baru
  - Tambahkan seksi Provider Gateway di README.md
  - Update cli-config.yaml.example
```

---

## 12. Referensi Sumber

### Repo yang dipelajari

| Repo | File kunci yang dipelajari | Fitur yang diadaptasi |
|------|---------------------------|----------------------|
| `decolua/9router` | `src/rtk.js`, `src/router.js`, `src/combo.js` | RTK compression pipeline, tier fallback, format translation |
| `diegosouzapw/OmniRoute` | `src/lib/router/`, `src/lib/compression/`, `src/lib/circuit-breaker/` | Circuit breaker, Caveman mode, 14 routing strategies, guardrails |
| `router-for-me/CLIProxyAPI` | Core Go proxy logic | Arsitektur proxy layer, token rotation |
| `BerriAI/litellm` | `litellm/main.py`, `litellm/utils.py` | Unified completion interface, cost tracking, model list |

### Dokumentasi pendukung

- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) — daftar 100+ provider yang didukung
- [Ollama API docs](https://github.com/ollama/ollama/blob/main/docs/api.md) — OpenAI-compatible endpoints
- [OpenAI Python SDK](https://github.com/openai/openai-python) — base_url override untuk provider lain
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html) — referensi pola

### File hermes yang perlu dipahami sebelum mulai

Baca file-file ini sebelum menulis kode:

1. `hermes_constants.py` — konstanta global, termasuk path direktori hermes
2. `hermes_state.py` — cara hermes menyimpan state
3. `hermes_logging.py` — cara logging yang benar di hermes
4. `model_tools.py` — cara model dipilih saat ini
5. `agent/` — agent loop, pahami di mana request ke LLM dibuat
6. `providers/` — lihat salah satu provider yang sudah ada sebagai contoh pola
7. `cli-config.yaml.example` — format config yang sudah ada

---

## Catatan Akhir untuk Agent

- **Selalu baca file yang akan diubah secara penuh** sebelum mengedit. Jangan asumsikan isi berdasarkan nama file.
- **Jalankan test sebelum commit** — minimal `python -m pytest tests/test_gateway/ -v`
- **Jika ragu antara dua pendekatan**, pilih yang lebih sederhana dan tambahkan komentar mengapa.
- **Gateway adalah opt-in** — kalau user tidak punya `provider_gateway.enabled: true` di config, hermes harus berjalan persis seperti sebelumnya.
- **Jangan menghapus kode yang sudah ada** di `providers/` atau `model_tools.py` — hanya tambahkan.
- **Log semua keputusan routing** di level DEBUG, bukan INFO, untuk menghindari noise.
- **Test dengan skenario offline** — pastikan hermes tetap berjalan normal kalau semua provider gateway tidak tersambung.
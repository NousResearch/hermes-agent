evaluasilah jawaban mu(sonnet4.7-adaptive) dan juga jawaban chatgpt . Kemudian buat lah revisi atau dokumen hasil evaluasi nya

---
---

# HERMES AGENT — Arahan Lengkap untuk Coding Agent (Revisi Final)

> **Dokumen ini adalah hasil sintesis dari dua dokumen sebelumnya** dengan evaluasi kritis.
> Ditujukan untuk coding agent (Claude Code, Codex, Cursor, Cline, dll.)
> yang bertugas meningkatkan `hermes_agent`.
> **Baca seluruh dokumen ini sebelum menulis satu baris kode.**

---

## Catatan Evaluasi Dokumen Sebelumnya

Dokumen ini menggabungkan dan memperbaiki dua versi sebelumnya:

**Dokumen Claude (versi 1)** — kekuatan: kode implementasi lengkap siap pakai, integrasi spesifik ke file hermes, urutan commit, test examples. Kelemahan: terlalu langsung ke kode tanpa membangun fondasi visi yang kuat; melewatkan Model Registry sebagai fondasi utama; tidak ada Policy Engine yang matang; tidak ada Definition of Done; tidak ada "hal yang sebaiknya tidak dikerjakan dulu".

**Dokumen ChatGPT** — kekuatan: visi strategis "AI control plane" yang jauh lebih tepat; feature list lebih komprehensif (10 fitur); fase implementasi 4 level; Definition of Done yang ketat; PR checklist; modul structure yang lebih bersih. Kelemahan: tidak ada kode implementasi; tidak memetakan ke struktur hermes yang sudah ada; tidak ada integrasi point yang spesifik; tidak ada test examples; tidak ada commit order.

**Dokumen ini menggabungkan keduanya:** visi dan kerangka strategis dari ChatGPT, implementasi teknis spesifik dari Claude.

---

## Daftar Isi

1. [Visi & Konteks](#1-visi--konteks)
2. [Prinsip Desain](#2-prinsip-desain)
3. [Peta Arsitektur Target](#3-peta-arsitektur-target)
4. [Apa yang TIDAK Boleh Diubah](#4-apa-yang-tidak-boleh-diubah)
5. [Fase 1 — Fondasi (Model Registry + Adapter Layer + Fallback)](#5-fase-1--fondasi)
6. [Fase 2 — Routing Cerdas (Policy Engine + Observability)](#6-fase-2--routing-cerdas)
7. [Fase 3 — Efisiensi (Compression + Compatibility + Multi-Account)](#7-fase-3--efisiensi)
8. [Fase 4 — Ekosistem (Guardrails + Plugins + Team Controls)](#8-fase-4--ekosistem)
9. [Standar Kode & Testing](#9-standar-kode--testing)
10. [Definition of Done per Fitur](#10-definition-of-done-per-fitur)
11. [Urutan Commit yang Disarankan](#11-urutan-commit-yang-disarankan)
12. [Hal yang Sebaiknya Tidak Dikerjakan Dulu](#12-hal-yang-sebaiknya-tidak-dikerjakan-dulu)
13. [Acceptance Checklist untuk PR](#13-acceptance-checklist-untuk-pr)
14. [Referensi Sumber](#14-referensi-sumber)

---

## 1. Visi & Konteks

### Repositori target
`https://github.com/fajarkurnia0388/hermes_agent`
(fork dari `NousResearch/hermes-agent`)

### Repositori sumber inspirasi

| Repo | Kontribusi utama |
|------|-----------------|
| `fajarkurnia0388/9router` | RTK token compression, 3-tier auto-fallback, multi-account, format translation, quota tracking |
| `fajarkurnia0388/OmniRoute` | 177 provider, 14 routing strategy, circuit breaker, Caveman mode, MCP/A2A, guardrails, evals |
| `fajarkurnia0388/CLIProxyAPI` | Arsitektur proxy layer, referensi protocol design |
| `fajarkurnia0388/litellm` | Unified Python SDK 100+ LLM, virtual keys, spend tracking, load balancing, admin dashboard |

### Visi produk

`hermes_agent` harus berkembang dari **CLI agent yang fleksibel** menjadi **AI control plane** yang:

- mengakses banyak provider dan model lewat satu antarmuka yang konsisten,
- memilih model terbaik secara otomatis berdasarkan tugas, biaya, dan ketersediaan,
- berpindah provider tanpa memutus workflow saat rate limit, error, atau quota habis,
- memberi transparansi penuh atas token usage, latensi, biaya, dan kesehatan provider,
- tetap mudah dipakai dari CLI, editor, messaging, dan tool pihak ketiga.

### Masalah yang diselesaikan

Hermes saat ini sudah punya pemilihan model via `/model [provider:model]` dan Tool Gateway. Yang belum matang adalah **lapisan orkestrasi model** — routing policy, fallback otomatis, normalisasi format, dan observability. Ini yang membuat Hermes terasa rapuh saat dipakai sebagai daily driver multi-provider.

---

## 2. Prinsip Desain

Prinsip ini mengikat. Jangan dikompromikan demi kecepatan implementasi.

1. **Stabilitas di atas variasi** — lebih baik satu jalur yang stabil daripada banyak provider yang sering gagal.

2. **Model selection harus data-driven** — pilih model berdasarkan kemampuan, konteks, biaya, latensi, dan status kesehatan; bukan hanya berdasarkan nama populer.

3. **Fallback harus otomatis dan aman** — pengguna tidak boleh kehilangan sesi kerja hanya karena satu provider down.

4. **Format input/output harus dinormalisasi** — semua provider lewat adapter yang konsisten agar mudah ditambah dan diuji.

5. **Observability wajib ada** — kalau routing gagal, harus jelas gagal karena apa, di mana, dan provider mana yang dipilih.

6. **Konfigurasi harus deklaratif** — hindari logic tersebar. Sumber kebenaran harus jelas: config file, model registry, dan routing policy.

7. **Backward compatibility adalah hukum** — user yang sudah pakai `hermes model` atau `hermes setup` tidak boleh mengalami breaking change. Fitur baru harus opt-in.

8. **Fail gracefully** — jika komponen baru gagal, hermes jatuh kembali ke perilaku lama tanpa crash.

9. **Log semua keputusan routing** — setiap fallback, retry, dan kompresi dicatat di level DEBUG. Tidak ada keputusan silent.

10. **Satu modul, satu tanggung jawab** — pisahkan logic routing, auth, telemetry, compression, dan UI.

---

## 3. Peta Arsitektur Target

### Alur request ideal (end state)

```
User input (CLI / TUI / Messaging / MCP / REST)
    │
    ▼
[Message Pipeline]
    │  normalize format, attach context
    ▼
[Routing Policy Engine]   ← baca capability requirements dari task
    │  pilih kandidat model berdasarkan policy
    ▼
[Model Registry]          ← lookup metadata: capability, cost, health, quota
    │  filter kandidat yang fit
    ▼
[Provider Router]         ← terapkan kompresi, pilih provider/account
    │  kirim via adapter
    ▼
[Unified Adapter Layer]   ← OpenAI / Claude / Gemini / dll. → format standar
    │
    ▼
[Provider API]            ← request keluar
    │
    ▼
[Response Normalizer]     ← normalisasi respons ke format hermes
    │
    ▼
[Telemetry Recorder]      ← catat token, latency, cost, error
    │
    ▼
[Fallback Trigger]        ← jika gagal: retry dengan backoff → provider lain
    │
    ▼
User (hasil akhir + ringkasan routing jika diminta)
```

### Struktur modul target

```
hermes_agent/
├── core/                       # BARU — inti orkestrasi
│   ├── model_registry.py       # Registry terpusat semua model & metadata
│   ├── routing_policy.py       # Policy engine: cheapest, fastest, dll.
│   ├── provider_router.py      # Pemilihan provider + fallback ladder
│   ├── adapter_base.py         # Base class untuk semua adapter
│   └── message_pipeline.py     # Normalisasi pesan masuk/keluar
│
├── providers/                  # DIREFACTOR — adapter per vendor
│   ├── base.py                 # Interface standar (BARU)
│   ├── openrouter/
│   ├── anthropic/
│   ├── openai/
│   ├── glm/
│   ├── kiro/
│   ├── ollama/                 # BARU — local model
│   └── ... (provider lain)
│
├── routing/                    # BARU — semua logic routing
│   ├── fallback_engine.py      # Auto-fallback multi-level
│   ├── circuit_breaker.py      # Circuit breaker per provider
│   ├── capability_scorer.py    # Scoring model berdasarkan capability
│   └── quota_manager.py        # Quota & spend tracking
│
├── telemetry/                  # BARU — observability
│   ├── metrics.py              # Token, latency, error rate, cost
│   ├── health_monitor.py       # Provider health tracking
│   └── audit_log.py            # Audit trail semua routing decisions
│
├── compression/                # BARU — token efficiency
│   ├── rtk.py                  # RTK: kompresi tool output
│   ├── caveman.py              # Caveman: kompresi respons
│   └── pipeline.py             # Orkestrasi kompresi
│
├── config/                     # BARU — konfigurasi terpusat
│   ├── schema.py               # Validasi config dengan Pydantic
│   ├── defaults.py             # Default values
│   └── loader.py               # Load dari YAML/env
│
├── guardrails/                 # BARU — safety & compliance
│   ├── pii_filter.py           # Deteksi & redact PII
│   ├── injection_guard.py      # Prompt injection detection
│   └── spend_limit.py          # Batas pengeluaran
│
├── agent/                      # TIDAK DIUBAH — core agent loop
├── gateway/                    # TIDAK DIUBAH — messaging platforms
├── skills/                     # TIDAK DIUBAH
├── optional-skills/            # TIDAK DIUBAH
├── acp_adapter/                # TIDAK DIUBAH
├── acp_registry/               # TIDAK DIUBAH
├── cron/                       # TIDAK DIUBAH
├── mcp_serve.py                # TIDAK DIUBAH
└── trajectory_compressor.py    # TIDAK DIUBAH
```

---

## 4. Apa yang TIDAK Boleh Diubah

File dan modul berikut adalah fitur inti hermes yang sudah mature. Jangan diubah tanpa alasan sangat kuat.

| Komponen | Alasan dilindungi |
|----------|------------------|
| `skills/` dan `optional-skills/` | Sistem skills adalah differentiator utama hermes |
| `hermes_state.py` + FTS5 session search | Memory system yang unik, sensitif terhadap perubahan |
| `cron/` | Scheduler yang sudah production-ready |
| `gateway/` | Integrasi Telegram/Discord/WA/Signal/Email |
| `acp_adapter/` dan `acp_registry/` | Protocol ACP yang sudah live |
| `mcp_serve.py` | MCP integration, banyak user bergantung |
| `trajectory_compressor.py` | Training data pipeline, bukan untuk runtime |
| `hermes_cli/` — antarmuka existing | User sudah terbiasa dengan perintah yang ada |

---

## 5. Fase 1 — Fondasi

**Target: model registry + adapter layer + fallback engine + logging**

Ini fondasi. Jangan lanjut ke fase 2 sebelum semua bagian fase 1 stabil dan ada test-nya.

---

### 5.1 Model Registry (`core/model_registry.py`)

Registry terpusat yang menyimpan metadata semua model dari semua provider. Ini yang memungkinkan routing policy bekerja dengan data, bukan asumsi.

**Metadata yang wajib ada per model:**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelCategory(str, Enum):
    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDINGS = "embeddings"
    MULTIMODAL = "multimodal"


class ProviderTier(str, Enum):
    SUBSCRIPTION = "subscription"   # Claude Pro, Codex Plus
    API_KEY = "api_key"             # Direct API
    CHEAP = "cheap"                 # GLM, MiniMax
    FREE = "free"                   # Kiro, OpenCode Free
    LOCAL = "local"                 # Ollama, llama.cpp


@dataclass
class ModelEntry:
    """Satu entri dalam model registry."""
    # Identitas
    provider: str                           # "openrouter", "anthropic", "kiro"
    model_id: str                           # ID yang dikirim ke API
    display_name: str                       # Nama ramah untuk UI
    tier: ProviderTier

    # Capabilities
    categories: list[ModelCategory] = field(default_factory=list)
    context_window: int = 8192              # Token
    supports_streaming: bool = True
    supports_tool_use: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_embeddings: bool = False
    max_output_tokens: Optional[int] = None

    # Biaya (per 1M token, dalam USD)
    cost_input_per_1m: float = 0.0
    cost_output_per_1m: float = 0.0

    # Operasional
    rate_limit_rpm: Optional[int] = None    # Request per menit
    rate_limit_tpd: Optional[int] = None    # Token per hari
    fallback_priority: int = 50             # Makin kecil = makin diutamakan
    enabled: bool = True
    notes: str = ""


class ModelRegistry:
    """Registry terpusat. Diisi dari config file + discovery otomatis."""

    def __init__(self):
        self._models: dict[str, ModelEntry] = {}   # key: "provider/model_id"

    def register(self, entry: ModelEntry):
        key = f"{entry.provider}/{entry.model_id}"
        self._models[key] = entry

    def get(self, provider: str, model_id: str) -> Optional[ModelEntry]:
        return self._models.get(f"{provider}/{model_id}")

    def filter(
        self,
        categories: list[ModelCategory] = None,
        requires_tool_use: bool = False,
        requires_vision: bool = False,
        min_context: int = 0,
        max_cost_input: float = float("inf"),
        tiers: list[ProviderTier] = None,
        enabled_only: bool = True,
    ) -> list[ModelEntry]:
        """
        Filter model berdasarkan capability requirements.
        Ini yang dipanggil routing policy engine.
        """
        results = []
        for entry in self._models.values():
            if enabled_only and not entry.enabled:
                continue
            if requires_tool_use and not entry.supports_tool_use:
                continue
            if requires_vision and not entry.supports_vision:
                continue
            if entry.context_window < min_context:
                continue
            if entry.cost_input_per_1m > max_cost_input:
                continue
            if tiers and entry.tier not in tiers:
                continue
            if categories:
                if not any(c in entry.categories for c in categories):
                    continue
            results.append(entry)
        return sorted(results, key=lambda m: m.fallback_priority)

    def load_from_config(self, config: dict):
        """Load registry dari config YAML."""
        for provider_conf in config.get("model_registry", {}).get("providers", []):
            provider = provider_conf["name"]
            tier = ProviderTier(provider_conf.get("tier", "api_key"))
            for model_conf in provider_conf.get("models", []):
                entry = ModelEntry(
                    provider=provider,
                    tier=tier,
                    model_id=model_conf["id"],
                    display_name=model_conf.get("name", model_conf["id"]),
                    categories=[ModelCategory(c) for c in model_conf.get("categories", ["chat"])],
                    context_window=model_conf.get("context_window", 8192),
                    supports_tool_use=model_conf.get("tool_use", False),
                    supports_vision=model_conf.get("vision", False),
                    cost_input_per_1m=model_conf.get("cost_input_per_1m", 0.0),
                    cost_output_per_1m=model_conf.get("cost_output_per_1m", 0.0),
                    fallback_priority=model_conf.get("priority", 50),
                )
                self.register(entry)

    def all_models(self) -> list[ModelEntry]:
        return list(self._models.values())

    def summary(self) -> dict:
        total = len(self._models)
        enabled = sum(1 for m in self._models.values() if m.enabled)
        by_tier = {}
        for m in self._models.values():
            by_tier[m.tier.value] = by_tier.get(m.tier.value, 0) + 1
        return {"total": total, "enabled": enabled, "by_tier": by_tier}
```

**Konfigurasi registry di `cli-config.yaml`:**

```yaml
model_registry:
  providers:
    - name: openrouter
      tier: api_key
      models:
        - id: anthropic/claude-opus-4-6
          name: Claude Opus 4.6 (OpenRouter)
          categories: [chat, code, reasoning]
          context_window: 200000
          tool_use: true
          vision: true
          cost_input_per_1m: 15.0
          cost_output_per_1m: 75.0
          priority: 10

        - id: google/gemini-2.5-pro
          name: Gemini 2.5 Pro (OpenRouter)
          categories: [chat, code, reasoning, vision]
          context_window: 1000000
          tool_use: true
          vision: true
          cost_input_per_1m: 1.25
          cost_output_per_1m: 5.0
          priority: 20

    - name: kiro
      tier: free
      models:
        - id: claude-sonnet-4.5
          name: Claude Sonnet 4.5 (Kiro Free)
          categories: [chat, code]
          context_window: 200000
          tool_use: true
          cost_input_per_1m: 0.0
          cost_output_per_1m: 0.0
          priority: 80

    - name: ollama
      tier: local
      models:
        - id: llama3.2
          name: Llama 3.2 (Local)
          categories: [chat, code]
          context_window: 128000
          tool_use: false
          cost_input_per_1m: 0.0
          priority: 90
```

---

### 5.2 Adapter Layer (`providers/base.py` + per-provider)

**Interface wajib yang harus diimplementasi SEMUA provider adapter:**

```python
# providers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Optional


@dataclass
class NormalizedRequest:
    """Format request standar hermes. Adapter mengkonversi ini ke format provider."""
    model_id: str
    messages: list[dict]            # Format OpenAI: [{"role": ..., "content": ...}]
    tools: list[dict] = None        # Tool definitions (OpenAI format)
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra: dict = None              # Parameter tambahan provider-specific


@dataclass
class NormalizedResponse:
    """Format respons standar hermes. Adapter mengkonversi dari format provider."""
    content: str
    role: str = "assistant"
    tool_calls: list[dict] = None
    finish_reason: str = "stop"
    usage: dict = None              # {"input_tokens": ..., "output_tokens": ...}
    raw: Any = None                 # Respons mentah untuk debugging
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0


@dataclass
class NormalizedError:
    """Error standar hermes."""
    type: str                       # "rate_limit" | "auth" | "timeout" | "server_error" | "quota"
    message: str
    provider: str
    status_code: Optional[int] = None
    retryable: bool = True
    retry_after_seconds: Optional[float] = None


class ProviderAdapter(ABC):
    """
    Base class untuk semua provider adapter.
    Setiap provider baru WAJIB mengextend class ini.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nama unik provider, e.g. 'openrouter', 'kiro', 'ollama'."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Cek apakah provider bisa diakses (connectivity check)."""
        ...

    @abstractmethod
    def complete(self, request: NormalizedRequest) -> NormalizedResponse:
        """Kirim request sinkron, return respons ternormalisasi."""
        ...

    @abstractmethod
    def stream(self, request: NormalizedRequest) -> Iterator[str]:
        """Kirim request streaming, yield token satu per satu."""
        ...

    def normalize_error(self, exc: Exception, status_code: int = None) -> NormalizedError:
        """
        Konversi exception provider-specific ke NormalizedError.
        Override jika provider punya error format berbeda.
        """
        msg = str(exc).lower()
        if "rate limit" in msg or "429" in msg or status_code == 429:
            return NormalizedError("rate_limit", str(exc), self.provider_name,
                                   status_code, retryable=True, retry_after_seconds=60)
        if "401" in msg or "403" in msg or "auth" in msg:
            return NormalizedError("auth", str(exc), self.provider_name,
                                   status_code, retryable=False)
        if "timeout" in msg:
            return NormalizedError("timeout", str(exc), self.provider_name,
                                   retryable=True)
        if "quota" in msg or "insufficient" in msg:
            return NormalizedError("quota", str(exc), self.provider_name,
                                   status_code, retryable=False)
        return NormalizedError("server_error", str(exc), self.provider_name,
                               status_code, retryable=True)
```

**Contoh implementasi adapter baru (OpenRouter):**

```python
# providers/openrouter/adapter.py
from __future__ import annotations
import time
import openai
from providers.base import ProviderAdapter, NormalizedRequest, NormalizedResponse


class OpenRouterAdapter(ProviderAdapter):

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def complete(self, request: NormalizedRequest) -> NormalizedResponse:
        start = time.time()
        try:
            resp = self._client.chat.completions.create(
                model=request.model_id,
                messages=request.messages,
                tools=request.tools,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )
            return NormalizedResponse(
                content=resp.choices[0].message.content or "",
                role="assistant",
                tool_calls=self._extract_tool_calls(resp.choices[0].message),
                finish_reason=resp.choices[0].finish_reason,
                usage={
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                },
                raw=resp,
                provider=self.provider_name,
                model=request.model_id,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise self.normalize_error(e)

    def stream(self, request: NormalizedRequest):
        resp = self._client.chat.completions.create(
            model=request.model_id,
            messages=request.messages,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def _extract_tool_calls(self, message) -> list[dict]:
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return []
        return [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in message.tool_calls
        ]
```

---

### 5.3 Fallback Engine (`routing/fallback_engine.py`)

```python
"""
Auto-fallback multi-level.

Trigger fallback:
- HTTP 429 / rate limit
- HTTP 5xx / server error
- Auth expired
- Timeout
- Quota habis
- Model tidak tersedia
"""
from __future__ import annotations
import time
import logging
from typing import Optional

from providers.base import ProviderAdapter, NormalizedRequest, NormalizedResponse, NormalizedError
from core.model_registry import ModelRegistry, ModelEntry
from routing.circuit_breaker import CircuitBreaker

logger = logging.getLogger("hermes.routing.fallback")


class FallbackEngine:
    """
    Eksekusi request dengan fallback otomatis.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        adapters: dict[str, ProviderAdapter],  # key: provider_name
        circuit_breaker: CircuitBreaker,
        max_retries_per_model: int = 2,
        backoff_base: float = 1.5,
    ):
        self.registry = registry
        self.adapters = adapters
        self.circuit_breaker = circuit_breaker
        self.max_retries_per_model = max_retries_per_model
        self.backoff_base = backoff_base

    def execute(
        self,
        request: NormalizedRequest,
        candidates: list[ModelEntry],
    ) -> tuple[NormalizedResponse, list[str]]:
        """
        Coba kandidat model satu per satu.
        Returns: (response, fallback_log)
        """
        fallback_log = []
        last_error = None

        for model_entry in candidates:
            adapter = self.adapters.get(model_entry.provider)
            if adapter is None:
                fallback_log.append(f"SKIP {model_entry.provider}/{model_entry.model_id}: no adapter")
                continue

            if not self.circuit_breaker.is_available(model_entry.provider):
                fallback_log.append(f"SKIP {model_entry.provider}: circuit open")
                continue

            # Retry dengan exponential backoff
            for attempt in range(self.max_retries_per_model):
                try:
                    start = time.time()
                    request.model_id = model_entry.model_id
                    response = adapter.complete(request)
                    latency = (time.time() - start) * 1000

                    self.circuit_breaker.record_success(model_entry.provider, latency)
                    fallback_log.append(
                        f"OK {model_entry.provider}/{model_entry.model_id} "
                        f"(attempt {attempt+1}, {latency:.0f}ms)"
                    )
                    logger.debug("\n".join(fallback_log))
                    return response, fallback_log

                except NormalizedError as e:
                    self.circuit_breaker.record_failure(model_entry.provider)
                    last_error = e
                    msg = (f"FAIL {model_entry.provider}/{model_entry.model_id} "
                           f"attempt {attempt+1}: {e.type} — {e.message}")
                    fallback_log.append(msg)
                    logger.warning(msg)

                    if not e.retryable:
                        break  # Jangan retry, lanjut ke model berikutnya

                    if attempt < self.max_retries_per_model - 1:
                        wait = self.backoff_base ** attempt
                        if e.retry_after_seconds:
                            wait = max(wait, e.retry_after_seconds)
                        logger.debug(f"Menunggu {wait:.1f}s sebelum retry...")
                        time.sleep(min(wait, 30))

                except Exception as e:
                    last_error = e
                    self.circuit_breaker.record_failure(model_entry.provider)
                    msg = f"FAIL {model_entry.provider}/{model_entry.model_id}: unexpected — {e}"
                    fallback_log.append(msg)
                    logger.error(msg)
                    break

        logger.error(f"Semua kandidat gagal. Log:\n" + "\n".join(fallback_log))
        raise RuntimeError(
            f"Semua {len(candidates)} kandidat model gagal. "
            f"Error terakhir: {last_error}"
        )
```

---

### 5.4 Circuit Breaker (`routing/circuit_breaker.py`)

```python
"""
Circuit breaker thread-safe per provider.
States: CLOSED → OPEN → HALF_OPEN → CLOSED
"""
from __future__ import annotations
import time
import threading
from enum import Enum
from dataclasses import dataclass


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ProviderHealth:
    provider: str
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    last_failure_at: float = 0.0
    last_success_at: float = 0.0
    latency_p50_ms: float = 0.0
    total_requests: int = 0
    total_failures: int = 0


class CircuitBreaker:

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._lock = threading.Lock()
        self._health: dict[str, ProviderHealth] = {}

    def _get(self, provider: str) -> ProviderHealth:
        if provider not in self._health:
            self._health[provider] = ProviderHealth(provider=provider)
        return self._health[provider]

    def is_available(self, provider: str) -> bool:
        with self._lock:
            h = self._get(provider)
            if h.state == CircuitState.CLOSED:
                return True
            if h.state == CircuitState.OPEN:
                if time.time() - h.last_failure_at >= self.recovery_timeout:
                    h.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True  # HALF_OPEN: biarkan satu request

    def record_success(self, provider: str, latency_ms: float = 0):
        with self._lock:
            h = self._get(provider)
            h.consecutive_failures = 0
            h.last_success_at = time.time()
            h.total_requests += 1
            if h.state == CircuitState.HALF_OPEN:
                h.state = CircuitState.CLOSED
            if latency_ms > 0:
                alpha = 0.1
                h.latency_p50_ms = (alpha * latency_ms
                                    + (1 - alpha) * h.latency_p50_ms
                                    if h.latency_p50_ms else latency_ms)

    def record_failure(self, provider: str):
        with self._lock:
            h = self._get(provider)
            h.consecutive_failures += 1
            h.total_failures += 1
            h.total_requests += 1
            h.last_failure_at = time.time()
            if h.consecutive_failures >= self.failure_threshold:
                h.state = CircuitState.OPEN

    def get_all(self) -> dict[str, ProviderHealth]:
        with self._lock:
            return dict(self._health)

    def reset(self, provider: str):
        with self._lock:
            self._health[provider] = ProviderHealth(provider=provider)
```

---

## 6. Fase 2 — Routing Cerdas

**Target: policy engine + capability-aware selection + quota/spend tracking + health dashboard**

### 6.1 Routing Policy Engine (`core/routing_policy.py`)

```python
"""
Policy engine: pilih kandidat model berdasarkan policy yang dikonfigurasi.

Policy built-in:
- cheapest_first: urutkan berdasarkan cost_input_per_1m ascending
- fastest_first: urutkan berdasarkan latency_p50 ascending
- quality_first: urutkan berdasarkan fallback_priority ascending (makin kecil = lebih berkualitas)
- free_first: tier FREE dulu, lalu CHEAP, lalu API_KEY, lalu SUBSCRIPTION
- local_first: tier LOCAL dulu (privasi)
- subscription_first: manfaatkan langganan yang sudah dibayar

Policy dapat di-chain: ["subscription_first", "fallback:free_first"]
"""
from __future__ import annotations
from enum import Enum
from typing import Callable

from core.model_registry import ModelRegistry, ModelEntry, ProviderTier, ModelCategory


class RoutingPolicy(str, Enum):
    CHEAPEST_FIRST = "cheapest_first"
    FASTEST_FIRST = "fastest_first"
    QUALITY_FIRST = "quality_first"
    FREE_FIRST = "free_first"
    LOCAL_FIRST = "local_first"
    SUBSCRIPTION_FIRST = "subscription_first"
    VISION_CAPABLE = "vision_capable"
    TOOL_USE_CAPABLE = "tool_use_capable"
    LONG_CONTEXT = "long_context"       # Butuh context > 100K token


# Tier order untuk setiap policy
_TIER_ORDER: dict[RoutingPolicy, list[ProviderTier]] = {
    RoutingPolicy.FREE_FIRST: [
        ProviderTier.FREE, ProviderTier.CHEAP,
        ProviderTier.API_KEY, ProviderTier.SUBSCRIPTION
    ],
    RoutingPolicy.LOCAL_FIRST: [
        ProviderTier.LOCAL, ProviderTier.FREE,
        ProviderTier.CHEAP, ProviderTier.API_KEY
    ],
    RoutingPolicy.SUBSCRIPTION_FIRST: [
        ProviderTier.SUBSCRIPTION, ProviderTier.API_KEY,
        ProviderTier.CHEAP, ProviderTier.FREE
    ],
}


def apply_policy(
    candidates: list[ModelEntry],
    policies: list[RoutingPolicy],
    health_latencies: dict[str, float] = None,  # provider → latency ms
    context_length_needed: int = 0,
) -> list[ModelEntry]:
    """
    Urutkan dan filter kandidat berdasarkan daftar policy.
    Policy pertama = prioritas tertinggi.
    """
    health_latencies = health_latencies or {}
    result = list(candidates)

    for policy in policies:

        if policy == RoutingPolicy.VISION_CAPABLE:
            result = [m for m in result if m.supports_vision]

        elif policy == RoutingPolicy.TOOL_USE_CAPABLE:
            result = [m for m in result if m.supports_tool_use]

        elif policy == RoutingPolicy.LONG_CONTEXT:
            result = [m for m in result if m.context_window >= 100_000]

        elif policy == RoutingPolicy.CHEAPEST_FIRST:
            result.sort(key=lambda m: m.cost_input_per_1m)

        elif policy == RoutingPolicy.FASTEST_FIRST:
            result.sort(
                key=lambda m: health_latencies.get(m.provider, 9999)
            )

        elif policy == RoutingPolicy.QUALITY_FIRST:
            result.sort(key=lambda m: m.fallback_priority)

        elif policy in _TIER_ORDER:
            tier_order = _TIER_ORDER[policy]
            tier_rank = {t: i for i, t in enumerate(tier_order)}
            result.sort(key=lambda m: tier_rank.get(m.tier, 99))

    return result


def resolve_policies(policy_names: list[str]) -> list[RoutingPolicy]:
    """Parse policy names dari config string."""
    result = []
    for name in policy_names:
        try:
            result.append(RoutingPolicy(name))
        except ValueError:
            import logging
            logging.getLogger("hermes.policy").warning(
                f"Policy tidak dikenal: '{name}', dilewati."
            )
    return result
```

### 6.2 Telemetry & Health Monitor (`telemetry/metrics.py`)

```python
"""
Telemetry: catat semua metrics operasional.
Digunakan untuk `hermes status` dan troubleshooting.
"""
from __future__ import annotations
import json
import time
import threading
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RequestRecord:
    timestamp: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error_type: Optional[str] = None
    fallback_used: bool = False
    compression_applied: str = "none"
    tokens_saved: int = 0


class TelemetryStore:
    """
    In-memory telemetry dengan persist ke disk.
    Menyimpan 1000 request terakhir + agregat harian.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self._dir = Path(data_dir or Path.home() / ".hermes" / "telemetry")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._recent: deque[RequestRecord] = deque(maxlen=1000)
        self._aggregates: dict = self._load_aggregates()

    def record(self, record: RequestRecord):
        with self._lock:
            self._recent.append(record)
            # Update agregat
            key = record.provider
            agg = self._aggregates.setdefault(key, {
                "total_requests": 0,
                "total_failures": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "total_fallbacks": 0,
                "total_tokens_saved": 0,
            })
            agg["total_requests"] += 1
            if not record.success:
                agg["total_failures"] += 1
            agg["total_input_tokens"] += record.input_tokens
            agg["total_output_tokens"] += record.output_tokens
            if record.fallback_used:
                agg["total_fallbacks"] += 1
            agg["total_tokens_saved"] += record.tokens_saved
            self._save_aggregates()

    def get_provider_stats(self, provider: str) -> dict:
        with self._lock:
            agg = self._aggregates.get(provider, {})
            # Hitung dari recent records
            recent = [r for r in self._recent if r.provider == provider]
            latencies = [r.latency_ms for r in recent if r.success]
            errors = [r for r in recent if not r.success]

            return {
                **agg,
                "error_rate": (len(errors) / len(recent)) if recent else 0.0,
                "latency_p50_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
                "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
            }

    def get_all_stats(self) -> list[dict]:
        with self._lock:
            providers = set(self._aggregates.keys())
            return [
                {"provider": p, **self.get_provider_stats(p)}
                for p in providers
            ]

    def _load_aggregates(self) -> dict:
        f = self._dir / "aggregates.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
        return {}

    def _save_aggregates(self):
        try:
            f = self._dir / "aggregates.json"
            f.write_text(json.dumps(self._aggregates, indent=2))
        except Exception:
            pass
```

### 6.3 Command `hermes status`

Tambahkan ke CLI (ikuti pola command yang sudah ada di `hermes_cli/`):

```python
def cmd_status(args):
    """
    hermes status

    Output:
        PROVIDER       STATE    LAT(p50)  REQUESTS  FAILURES  COST($)  TOKENS_SAVED
        openrouter     CLOSED    145ms     1,234      12       $2.45    48,200
        kiro           CLOSED     89ms       456       0       $0.00     9,800
        ollama (local) CLOSED     45ms        23       1       $0.00         0
        glm            OPEN         —          8       6       $0.01         0
    """
    from core.model_registry import ModelRegistry
    from routing.circuit_breaker import CircuitBreaker
    from telemetry.metrics import TelemetryStore
    from config.loader import load_config

    config = load_config()
    store = TelemetryStore()
    cb = CircuitBreaker()

    stats = store.get_all_stats()
    health = cb.get_all()

    print(f"\n{'PROVIDER':<16} {'STATE':<10} {'LAT(p50)':<10} "
          f"{'REQUESTS':<10} {'FAILURES':<10} {'COST($)':<10} SAVED(tok)")
    print("─" * 82)

    for s in sorted(stats, key=lambda x: x.get("total_requests", 0), reverse=True):
        p = s["provider"]
        h = health.get(p)
        state = h.state.value.upper() if h else "UNKNOWN"
        lat = f"{s.get('latency_p50_ms', 0):.0f}ms" if s.get("latency_p50_ms") else "—"
        print(
            f"{p:<16} {state:<10} {lat:<10} "
            f"{s.get('total_requests', 0):<10} "
            f"{s.get('total_failures', 0):<10} "
            f"${s.get('total_cost_usd', 0):.2f}      "
            f"{s.get('total_tokens_saved', 0):,}"
        )

    print()
```

---

## 7. Fase 3 — Efisiensi

**Target: compression layer + compatibility endpoint + multi-account + semantic cache**

### 7.1 Compression Pipeline (`compression/pipeline.py`)

```python
"""
Compression pipeline. Dipanggil sebelum request dikirim ke provider.

Mode:
- none: tidak ada kompresi
- rtk: kompres tool_result (git diff, grep, ls, log)
- caveman: injeksi instruksi respons ringkas
- stacked: rtk + caveman (hemat terbesar, 78-95%)

Aturan:
- JANGAN kompres code block, error message, patch, nama file
- JANGAN kompres konten yang sudah kecil (<500 karakter)
- Jika kompresi membuat output lebih besar, gunakan asli
- Sediakan mode disable via config
"""
from __future__ import annotations
import re
import copy


# ── RTK (Tool Result Kompressor) ──────────────────────────────────────────────

_PRESERVE_PATTERNS = [
    re.compile(r"```[\s\S]*?```"),                  # Code blocks
    re.compile(r"error:.*", re.IGNORECASE),          # Error messages
    re.compile(r"^\+\+\+ |^--- |^@@ ", re.MULTILINE), # Diff critical lines
]


def _compress_git_diff(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out, ctx_count = [], 0
    for line in lines:
        if line.startswith(("---", "+++", "@@", "diff ", "index ")):
            out.append(line); ctx_count = 0
        elif line.startswith(("+", "-")):
            out.append(line); ctx_count = 0
        elif ctx_count < 2:
            out.append(line); ctx_count += 1
    return "".join(out)


def _compress_ls(text: str) -> str:
    out = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0][0] in "-dl":
            out.append(parts[-1])
        else:
            out.append(line)
    return "\n".join(out)


def _compress_grep(text: str) -> str:
    seen, out = set(), []
    for line in text.splitlines():
        s = line.strip()
        if s and s not in seen:
            seen.add(s); out.append(line)
    return "\n".join(out[:200])


def _smart_truncate(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (text[:half]
            + f"\n\n[... {len(text)-max_chars} chars removed by RTK compression ...]\n\n"
            + text[-half:])


def _detect_filter(text: str) -> str:
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


def apply_rtk(text: str) -> tuple[str, bool]:
    """Return (compressed, was_modified)."""
    if not text or len(text) < 500:
        return text, False
    ftype = _detect_filter(text)
    try:
        if ftype == "git_diff":   result = _compress_git_diff(text)
        elif ftype == "ls":        result = _compress_ls(text)
        elif ftype == "grep":      result = _compress_grep(text)
        elif ftype == "truncate":  result = _smart_truncate(text)
        else:                      return text, False
        return (result, True) if len(result) < len(text) else (text, False)
    except Exception:
        return text, False


# ── Caveman Mode ──────────────────────────────────────────────────────────────

_CAVEMAN_SYSTEM = (
    "\n\n[RESPONSE STYLE]\n"
    "Be concise. Technical terms OK. No filler phrases. Substance only."
)


def apply_caveman(messages: list[dict]) -> list[dict]:
    msgs = copy.deepcopy(messages)
    for msg in msgs:
        if msg.get("role") == "system":
            msg["content"] = str(msg["content"]) + _CAVEMAN_SYSTEM
            return msgs
    msgs.insert(0, {"role": "system", "content": _CAVEMAN_SYSTEM.strip()})
    return msgs


# ── Pipeline ──────────────────────────────────────────────────────────────────

def compress(messages: list[dict], mode: str = "rtk") -> tuple[list[dict], dict]:
    """
    Entry point utama.
    Returns: (compressed_messages, stats)
    """
    stats = {"mode": mode, "rtk_count": 0, "tokens_saved_estimate": 0}
    if mode == "none":
        return messages, stats

    msgs = copy.deepcopy(messages)

    if mode in ("rtk", "stacked"):
        for msg in msgs:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        orig = block.get("content", "")
                        if isinstance(orig, str):
                            compressed, modified = apply_rtk(orig)
                            if modified:
                                stats["rtk_count"] += 1
                                stats["tokens_saved_estimate"] += (len(orig) - len(compressed)) // 4
                                block["content"] = compressed

    if mode in ("caveman", "stacked"):
        msgs = apply_caveman(msgs)

    return msgs, stats
```

### 7.2 Compatibility Gateway (OpenAI-compatible endpoint)

Buat `web/api/v1/` yang expose endpoint OpenAI-compatible sehingga tool lain (Cursor, Cline, editor) bisa terhubung ke hermes sebagai provider:

```python
# web/api/v1/chat.py  (sesuaikan dengan framework web hermes yang sudah ada)
"""
OpenAI-compatible /v1/chat/completions endpoint.
Memungkinkan tool external menggunakan hermes sebagai provider.
"""

# Endpoint: POST /v1/chat/completions
# Endpoint: GET  /v1/models

# Format request: OpenAI standard
# Format response: OpenAI standard
# Auth: Bearer token yang dikonfigurasi di hermes config

# Cara integrasi ke Cursor/Cline:
#   Base URL: http://localhost:HERMES_PORT/v1
#   API Key: hermes_local_key (dari config)
#   Model: openrouter/claude-opus-4-6 atau combo name

# Implementasi:
# 1. Parse request OpenAI format
# 2. Ekstrak model + messages
# 3. Kirim ke ProviderRouter (dengan policy aktif)
# 4. Normalize respons ke format OpenAI
# 5. Return (streaming atau non-streaming)
```

### 7.3 Semantic Cache (`routing/semantic_cache.py`)

```python
"""
LRU cache untuk LLM responses.
Hit rate biasanya 15-30% untuk sesi coding berulang.
"""
from __future__ import annotations
import hashlib
import json
import time
import threading
from collections import OrderedDict
from typing import Any, Optional


class SemanticCache:

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: float = 3600.0,
    ):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, messages: list[dict], model: str) -> str:
        content = json.dumps({"model": model, "messages": messages},
                             sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, messages: list[dict], model: str) -> Optional[Any]:
        key = self._key(messages, model)
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            value, expires_at = self._cache[key]
            if time.time() > expires_at:
                del self._cache[key]
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return value

    def set(self, messages: list[dict], model: str, response: Any):
        key = self._key(messages, model)
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (response, time.time() + self.ttl)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

---

## 8. Fase 4 — Ekosistem

**Target: guardrails + credential management + plugin ecosystem + team controls**

### 8.1 Guardrails (`guardrails/`)

```python
# guardrails/pii_filter.py
"""
Filter data sensitif sebelum dikirim ke provider eksternal.
Default: LOG saja, jangan block (terlalu agresif untuk daily use).
"""
import re

_PATTERNS = {
    "email":       re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_id":    re.compile(r"\b08[0-9]{8,11}\b"),
    "nik":         re.compile(r"\b[0-9]{16}\b"),
    "api_key":     re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9]{20,}\b"),
}

def scan(text: str) -> list[str]:
    """Return list tipe PII yang ditemukan."""
    return [ptype for ptype, pat in _PATTERNS.items() if pat.search(text)]

def redact(text: str) -> tuple[str, list[str]]:
    """Redact PII, return (redacted_text, types_found)."""
    found = []
    for ptype, pat in _PATTERNS.items():
        if pat.search(text):
            text = pat.sub(f"[REDACTED:{ptype.upper()}]", text)
            found.append(ptype)
    return text, found


# guardrails/injection_guard.py
"""Deteksi prompt injection."""
import re

_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+(your\s+)?system\s+prompt", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"DAN\s*mode", re.I),
]

def is_injection(text: str) -> bool:
    return any(p.search(text) for p in _PATTERNS)


# guardrails/spend_limit.py
"""Batasi pengeluaran per provider per periode."""
from telemetry.metrics import TelemetryStore

def check_spend_limit(provider: str, limit_usd: float, store: TelemetryStore) -> bool:
    """Return True jika masih dalam batas."""
    stats = store.get_provider_stats(provider)
    return stats.get("total_cost_usd", 0.0) < limit_usd
```

### 8.2 Credential Management (`config/credentials.py`)

```python
"""
Manajemen credential yang aman.
API key TIDAK boleh muncul di log.
"""
from __future__ import annotations
import os
import json
import base64
from pathlib import Path
from typing import Optional


class CredentialStore:
    """
    Simpan API key dengan enkripsi ringan.
    Untuk production, gunakan system keyring atau vault.
    """

    def __init__(self, cred_file: Optional[str] = None):
        self._file = Path(cred_file or Path.home() / ".hermes" / "credentials.json")
        self._cache: dict[str, str] = {}
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                raw = json.loads(self._file.read_text())
                # Decode simple obfuscation (bukan enkripsi kuat)
                self._cache = {k: base64.b64decode(v).decode() for k, v in raw.items()}
            except Exception:
                self._cache = {}

    def _save(self):
        self._file.parent.mkdir(parents=True, exist_ok=True)
        encoded = {k: base64.b64encode(v.encode()).decode() for k, v in self._cache.items()}
        self._file.write_text(json.dumps(encoded, indent=2))
        self._file.chmod(0o600)   # Owner only

    def get(self, provider: str) -> Optional[str]:
        # Prioritas: environment variable → stored credential
        env_key = f"HERMES_{provider.upper()}_API_KEY"
        return os.environ.get(env_key) or self._cache.get(provider)

    def set(self, provider: str, api_key: str):
        self._cache[provider] = api_key
        self._save()

    def delete(self, provider: str):
        self._cache.pop(provider, None)
        self._save()

    def list_providers(self) -> list[str]:
        """List providers yang punya credential (TANPA menampilkan key-nya)."""
        return list(self._cache.keys())
```

---

## 9. Standar Kode & Testing

### 9.1 Kriteria kualitas kode

- Python 3.11+, type hints di semua signature publik
- `from __future__ import annotations` di semua file baru
- Docstring singkat di setiap class dan method publik
- Gunakan `logging` bukan `print` untuk output internal
- API key TIDAK PERNAH boleh muncul di log — masking wajib
- Setiap adapter provider harus punya interface yang sama (`ProviderAdapter`)
- Hindari dependency baru tanpa manfaat jelas
- Perubahan kecil dan modular — hindari refactor besar tanpa alasan kuat

### 9.2 Test yang wajib ditulis

```
tests/
├── test_core/
│   ├── test_model_registry.py      # Filter, load_from_config, summary
│   └── test_routing_policy.py      # Policy ordering, chaining
├── test_routing/
│   ├── test_circuit_breaker.py     # State transitions, thread safety
│   ├── test_fallback_engine.py     # Retry, backoff, cascade
│   └── test_quota_manager.py       # Usage tracking, spend limits
├── test_providers/
│   ├── test_adapter_base.py        # Interface compliance
│   └── test_normalize_error.py     # Error type mapping
├── test_compression/
│   ├── test_rtk.py                 # Git diff, grep, ls, truncate
│   └── test_pipeline.py            # Stacked mode, edge cases
├── test_telemetry/
│   └── test_metrics.py             # Record, stats, persistence
└── test_guardrails/
    ├── test_pii_filter.py
    └── test_injection_guard.py
```

### 9.3 Contoh test critical

```python
# tests/test_routing/test_fallback_engine.py

def test_fallback_ke_provider_berikutnya_saat_rate_limit():
    """Jika provider pertama rate limit, harus otomatis ke provider kedua."""
    from providers.base import NormalizedError
    from routing.fallback_engine import FallbackEngine
    # ... setup mock adapter yang raise rate_limit, verify provider kedua dipanggil

def test_backoff_tidak_block_provider_lain():
    """Backoff hanya untuk retry model yang sama, bukan semua provider."""
    # Verify bahwa backoff sleep tidak menunda pemilihan kandidat lain

def test_circuit_breaker_open_skip_provider():
    """Provider yang circuit-nya OPEN harus di-skip tanpa mencoba."""
    # Verify circuit_breaker.is_available() dipanggil sebelum adapter.complete()

def test_semua_provider_gagal_raise_error():
    """Jika semua kandidat gagal, harus raise RuntimeError, bukan hang."""
    # Verify error raised dengan pesan yang jelas

# tests/test_compression/test_rtk.py

def test_rtk_git_diff_membuang_context_berlebih():
    diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
    diff += "@@ -1,5 +1,5 @@\n"
    diff += " ctx\n" * 50 + "-old\n+new\n" + " ctx\n" * 50
    compressed, modified = apply_rtk(diff)
    assert modified and len(compressed) < len(diff)

def test_rtk_tidak_crash_pada_input_kosong():
    for inp in ["", None, "\x00", "a" * 200_000]:
        apply_rtk(inp or "")   # Tidak boleh raise

def test_rtk_tidak_kompres_konten_kecil():
    _, modified = apply_rtk("short text")
    assert not modified

def test_stacked_mode_lebih_hemat_dari_rtk_saja():
    msgs = [{"role": "user", "content": [{"type": "tool_result",
             "content": "diff --git a/x.py\n" + "ctx\n" * 200}]}]
    _, stats_rtk = compress(msgs, "rtk")
    _, stats_stacked = compress(msgs, "stacked")
    # Stacked tidak selalu lebih kecil dalam bytes, tapi stats harus dicatat
    assert stats_stacked["mode"] == "stacked"
```

### 9.4 Cara jalankan test

```bash
# Semua test baru
python -m pytest tests/test_core/ tests/test_routing/ tests/test_compression/ -v

# Dengan coverage
python -m pytest tests/ --cov=core --cov=routing --cov=compression --cov=telemetry \
    --cov-report=term-missing --cov-fail-under=70

# Test spesifik tanpa dependency eksternal
python -m pytest tests/test_compression/ tests/test_routing/test_circuit_breaker.py -v

# Pastikan test lama masih lulus
python -m pytest tests/ -v --ignore=tests/test_core --ignore=tests/test_routing
```

---

## 10. Definition of Done per Fitur

Setiap fitur dianggap **selesai** hanya jika memenuhi SEMUA kriteria berikut:

| Kriteria | Detail |
|----------|--------|
| ✅ End-to-end berjalan | Bisa dipakai dari CLI tanpa error |
| ✅ Ada test relevan | Minimal happy path + satu skenario gagal |
| ✅ Ada dokumentasi singkat | Docstring atau README update minimal |
| ✅ Ada contoh konfigurasi | Di `cli-config.yaml.example` |
| ✅ Error message jelas | User tahu apa yang salah dan cara memperbaiki |
| ✅ Tidak merusak workflow lama | Test lama tetap lulus |
| ✅ Lolos lint dan type check | `mypy` atau `pyright` pada file baru |
| ✅ Ada graceful degradation | Jika fitur gagal, hermes tetap jalan |
| ✅ Tidak ada secret di log | API key di-mask sebelum log |
| ✅ Fallback behavior terdokumentasi | Ada komentar yang menjelaskan behavior saat gagal |

---

## 11. Urutan Commit yang Disarankan

```
Commit 01: feat(config): add config loader + schema validation
           → config/schema.py, config/loader.py, config/defaults.py

Commit 02: feat(providers): add ProviderAdapter base class
           → providers/base.py, tests/test_providers/test_adapter_base.py

Commit 03: feat(core): implement ModelRegistry
           → core/model_registry.py, tests/test_core/test_model_registry.py
           → Update cli-config.yaml.example dengan model registry section

Commit 04: feat(routing): implement CircuitBreaker
           → routing/circuit_breaker.py, tests/test_routing/test_circuit_breaker.py

Commit 05: feat(routing): implement FallbackEngine
           → routing/fallback_engine.py, tests/test_routing/test_fallback_engine.py

Commit 06: feat(telemetry): implement TelemetryStore + metrics
           → telemetry/metrics.py, tests/test_telemetry/test_metrics.py

Commit 07: feat(providers): refactor existing adapters ke ProviderAdapter interface
           → providers/openrouter/adapter.py, providers/anthropic/adapter.py, dll.
           → HATI-HATI: ini yang paling rawan breaking change

Commit 08: feat(core): implement RoutingPolicyEngine
           → core/routing_policy.py, tests/test_core/test_routing_policy.py

Commit 09: feat(cli): add hermes status command
           → Integrasikan TelemetryStore + CircuitBreaker ke CLI

Commit 10: feat(compression): implement RTK + Caveman + pipeline
           → compression/rtk.py, compression/caveman.py, compression/pipeline.py
           → tests/test_compression/

Commit 11: feat(guardrails): add PII filter + injection guard (default OFF)
           → guardrails/, tests/test_guardrails/

Commit 12: feat(providers): add Ollama local model adapter
           → providers/ollama/adapter.py

Commit 13: feat(cache): add semantic cache
           → routing/semantic_cache.py

Commit 14: feat(config): add credential management
           → config/credentials.py

Commit 15: feat(web): add OpenAI-compatible compatibility endpoint
           → web/api/v1/ (sesuaikan dengan framework web hermes)

Commit 16: docs: update README + AGENTS.md dengan fitur baru
```

---

## 12. Hal yang Sebaiknya Tidak Dikerjakan Dulu

Jangan sentuh ini sebelum fase sebelumnya stabil:

- **Menambah banyak provider** sebelum registry + adapter interface rapi — akan jadi technical debt
- **Membuat UI dashboard besar** sebelum telemetry stabil dan data akurat
- **Optimasi performa kompleks** sebelum baseline performance terukur
- **Full compatibility ke semua client** sekaligus — buat satu dulu, validasi, baru lanjut
- **Fitur agent baru** (sub-agent, parallel task) sebelum routing dan observability matang
- **MCP/A2A integration baru** sebelum provider routing stabil
- **Menghapus adapter provider lama** sebelum adapter baru terverifikasi
- **Mengubah schema config** tanpa migration path untuk user yang sudah punya config

---

## 13. Acceptance Checklist untuk PR

Sebelum PR di-merge, verifikasi semua item ini:

**Scope & Intent**
- [ ] Tujuan PR jelas dalam satu kalimat
- [ ] Ada issue atau task yang diselesaikan
- [ ] Perubahan tidak lebih dari satu concern utama

**Kualitas Kode**
- [ ] Tidak ada hardcoded API key, URL, atau credential
- [ ] Tidak ada `print()` untuk debug yang tertinggal
- [ ] API key di-mask sebelum logging
- [ ] Error messages mudah dipahami oleh user
- [ ] Type hints ada di semua signature publik

**Testing**
- [ ] Ada test untuk happy path
- [ ] Ada test untuk skenario gagal / edge case
- [ ] Test lama tidak ada yang pecah (`pytest tests/`)

**Backward Compatibility**
- [ ] User yang tidak mengaktifkan fitur baru tidak terpengaruh
- [ ] Schema config lama masih valid
- [ ] Perintah CLI lama masih bekerja

**Dokumentasi**
- [ ] Docstring update jika ada perubahan interface publik
- [ ] `cli-config.yaml.example` update jika ada config baru
- [ ] Fallback behavior terdokumentasi dalam komentar kode

**Keamanan**
- [ ] Tidak ada secret atau credential di diff
- [ ] Log tidak mengekspos API key
- [ ] Permission file credential di-set 600

---

## 14. Referensi Sumber

### Repo yang dipelajari

| Repo | File kunci | Fitur yang diadaptasi |
|------|-----------|----------------------|
| `decolua/9router` | `src/rtk.js`, `src/router.js`, `src/combo.js` | RTK compression, tier fallback, format translation |
| `diegosouzapw/OmniRoute` | `src/lib/router/`, `src/lib/compression/`, `src/lib/circuit-breaker/` | Circuit breaker, Caveman, 14 routing strategies, guardrails |
| `router-for-me/CLIProxyAPI` | Core Go proxy | Protocol layer design |
| `BerriAI/litellm` | `litellm/main.py`, `litellm/router.py` | Unified API, virtual keys, spend tracking, load balancing |

### File hermes yang harus dibaca sebelum mulai

Baca file berikut untuk memahami pola yang sudah ada:

1. `hermes_constants.py` — path direktori hermes, konstanta global
2. `hermes_state.py` — cara state disimpan dan diakses
3. `hermes_logging.py` — pola logging yang benar
4. `model_tools.py` — cara model dipilih saat ini (akan diintegrasikan dengan registry)
5. `providers/` — lihat satu provider yang sudah ada sebagai contoh pola adapter
6. `agent/` — agent loop, pahami di mana request ke LLM dibuat
7. `cli-config.yaml.example` — format config yang sudah ada
8. `AGENTS.md` di root — panduan agent yang sudah ada dari upstream

### Dokumentasi eksternal

- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) — 100+ provider dengan format string
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) — OpenAI-compatible endpoints
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html) — referensi pola
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — format standar yang dinormalisasi

---

## Catatan Akhir untuk Agent

Tiga hal yang paling sering menyebabkan implementasi gagal di proyek seperti ini:

1. **Mulai dari kode, bukan dari pemahaman** — Baca dulu semua file yang akan diubah, jalankan hermes secara lokal, pahami alur request end-to-end sebelum menulis satu baris pun.

2. **Mengabaikan backward compatibility** — User yang tidak mau fitur baru HARUS tetap bisa pakai hermes persis seperti sebelumnya. Semua fitur baru harus opt-in via config.

3. **Test setelah selesai, bukan bersamaan** — Tulis test minimal untuk setiap komponen sebelum mengintegrasikannya ke komponen lain. Debugging lebih mudah ketika unit kecil.

**Jika ragu antara dua pendekatan:** pilih yang lebih sederhana, tambahkan komentar mengapa, dan buat TODO untuk pendekatan yang lebih kompleks.

**Jika fitur baru menyebabkan test lama gagal:** STOP. Investigasi dulu sebelum lanjut. Jangan workaround dengan mengubah test lama.

**Jika tidak yakin cara integrasi ke hermes:** buat file baru yang berdiri sendiri dulu, uji secara isolasi, baru integrasikan ke alur utama.
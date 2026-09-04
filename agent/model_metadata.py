"""Model metadata, context lengths, and token estimation utilities.

Pure utility functions with no AIAgent dependency. Used by ContextCompressor
and run_agent.py for pre-flight context checks.
"""

import base64
import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

import yaml

if TYPE_CHECKING:  # pragma: no cover — runtime import is lazy (see below)
    import requests

from utils import atomic_json_write, atomic_yaml_write, base_url_host_matches, base_url_hostname

from hermes_constants import OPENROUTER_MODELS_URL
from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS

logger = logging.getLogger(__name__)

# ``requests`` costs ~27 ms of the `import cli` waterfall, so it is resolved lazily:
# ``_ensure_requests()`` at runtime, PEP 562 ``__getattr__`` for ``patch("agent.model_metadata.requests.get")``.


def _ensure_requests():
    if "requests" not in globals():
        import requests as _requests
        globals()["requests"] = _requests
    return globals()["requests"]


def __getattr__(name: str):
    if name == "requests":
        return _ensure_requests()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_requests_verify(base_url: str = "") -> bool | str:
    """SSL ``verify`` for ``requests`` probes; mirrors ``agent.ssl_verify.resolve_httpx_verify``.
    Priority: per-provider ``ssl_verify: false`` -> per-provider ``ssl_ca_cert`` (else probes log
    spurious CERTIFICATE_VERIFY_FAILED while the httpx chat path succeeds) -> CA env vars -> certifi."""
    if base_url:
        try:
            from hermes_cli.config import get_custom_provider_tls_settings
            tls = get_custom_provider_tls_settings(base_url)
            if tls.get("ssl_verify") is False:
                return False
            ca = tls.get("ssl_ca_cert")
            if isinstance(ca, str) and ca and os.path.isfile(ca):
                return ca
        except Exception:
            pass  # fall through to env vars — never break a probe on config lookup
    for env_var in ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        val = os.getenv(env_var)
        if val and os.path.isfile(val):
            return val
    return True


# Snapshot for callers inspecting this constant; prefix routing queries the registry live.
try:
    from providers import list_providers as _list_providers
except Exception:
    def _list_providers():
        return []

_PROVIDER_PREFIXES: frozenset[str] = frozenset(
    value.lower()
    for profile in _list_providers()
    for value in (profile.name, *profile.aliases)
)
_OLLAMA_TAG_PATTERN = re.compile(r"^(\d+\.?\d*b|latest|stable|q\d|fp?\d|instruct|chat|coder|vision|text)", re.IGNORECASE)
# Tailscale CGNAT (RFC 6598): `ipaddress.is_private` excludes it, yet Ollama
# reached over Tailscale must count as local (timeout auto-bumps).
_TAILSCALE_CGNAT = ipaddress.IPv4Network("100.64.0.0/10")


def _strip_provider_prefix(model: str) -> str:
    """Strip a registry-known provider prefix: ``"local:m"`` -> ``"m"``. Ollama ``model:tag``
    ids are preserved even when the model half is a provider name (``qwen:0.5b``)."""
    if ":" not in model or model.startswith("http"):
        return model
    prefix, suffix = model.split(":", 1)
    try:
        from providers import get_provider_profile
        is_provider = get_provider_profile(prefix.strip().lower()) is not None
    except Exception:
        is_provider = False
    if is_provider and not _OLLAMA_TAG_PATTERN.match(suffix.strip()):
        return suffix
    return model


_model_metadata_cache: Dict[str, Dict[str, Any]] = {}
_model_metadata_cache_time: float = 0
_MODEL_CACHE_TTL = 3600
_endpoint_model_metadata_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
_endpoint_model_metadata_cache_time: Dict[str, float] = {}
_ENDPOINT_MODEL_CACHE_TTL = 300
# Server-type verdicts (server_type, monotonic_ts): positive ones live an hour so a
# server swap on the same port is re-detected; None gets the short TTL so a
# transient failure recovers in minutes without re-running the waterfall each turn.
_ENDPOINT_PROBE_TTL_SECONDS = 3600.0
# A failed probe verdict (server_type is None — no known endpoint answered) is cached for a much shorter
# window: the in-memory entry exists only to keep one image-bearing turn from re-running the 5-request
# waterfall on every subsequent turn (#89863 — a keyed remote endpoint answered 401 to each leg and the None
# verdict was never cached, so every turn re-probed). Short TTL keeps a transient failure (server starting
# up, key being fixed) recoverable within minutes instead of pinning "undetected" for an hour.
_ENDPOINT_PROBE_FAILURE_TTL_SECONDS = 300.0
_endpoint_probe_path_cache: Dict[str, tuple] = {}
# Routable-but-dead endpoints (corp LAN off-VPN) blackhole TCP: once ANY probe paid
# a full connect timeout, later probes short-circuit for a while.
_ENDPOINT_BLACKHOLE_TTL_SECONDS = 30.0
_endpoint_blackhole_cache: Dict[str, float] = {}  # host:port -> monotonic ts


def _parse_base_url(base_url: str, scheme: str = "http"):
    """``urlparse`` of the normalized URL (``scheme`` prepended when absent); None when empty."""
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return None
    return urlparse(normalized if "://" in normalized else f"{scheme}://{normalized}")


def _endpoint_host_key(base_url: str) -> Optional[str]:
    """``host:port`` key (None without a host) so every probe path for one server shares an entry."""
    try:
        parsed = _parse_base_url(base_url)
        if parsed is None or not parsed.hostname:
            return None
        return f"{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}"
    except Exception:
        return None


def _note_endpoint_blackholed(base_url: str) -> None:
    """Record that a probe to ``base_url`` timed out during TCP connect."""
    key = _endpoint_host_key(base_url)
    if key is not None:
        _endpoint_blackhole_cache[key] = time.monotonic()
        logger.debug("Endpoint %s timed out connecting — skipping further probes for %.0fs", key, _ENDPOINT_BLACKHOLE_TTL_SECONDS)


def _endpoint_blackholed(base_url: str) -> bool:
    """True if a recent probe to ``base_url`` timed out during TCP connect (cache lookup only)."""
    if _ENDPOINT_BLACKHOLE_TTL_SECONDS <= 0:
        return False
    key = _endpoint_host_key(base_url)
    seen = _endpoint_blackhole_cache.get(key) if key is not None else None
    if seen is None:
        return False
    if (time.monotonic() - seen) >= _ENDPOINT_BLACKHOLE_TTL_SECONDS:
        del _endpoint_blackhole_cache[key]
        return False
    return True


def _note_if_connect_timeout(exc: BaseException, base_url: str) -> None:
    """Blackhole ``base_url`` when ``exc`` is a connect-phase timeout."""
    if _is_connect_timeout(exc):
        _note_endpoint_blackholed(base_url)


def _is_connect_timeout(exc: BaseException) -> bool:
    """True for connect-phase timeouts raised by httpx or requests. Read timeouts are
    excluded: the server accepted the connection, the opposite of a blackhole."""
    try:
        import httpx
        from requests.exceptions import ConnectTimeout
        return isinstance(exc, (httpx.ConnectTimeout, ConnectTimeout))
    except Exception:
        return False


# Disk L2 for local-endpoint probes so back-to-back CLI cold starts skip the waterfall.
# Only SUCCESSFUL probes persist (a down server must not pin a negative verdict).
_LOCAL_PROBE_DISK_TTL_SECONDS = 300.0


def _cache_file(name: str) -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / name


def _load_json_dict(path: Path) -> Dict[str, Any]:
    """JSON object at ``path``, or {} when missing/invalid."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _ttl_memo_get(path: Path, key: str, ttl: float, *, ts_key: str, value_key: str) -> Optional[Any]:
    """Fresh (< ``ttl`` seconds) ``value_key`` of the ``key`` entry in a TTL memo file, else None."""
    entry = _load_json_dict(path).get(key)
    if not isinstance(entry, dict):
        return None
    try:
        if (time.time() - float(entry[ts_key])) >= ttl:
            return None
        return entry[value_key]
    except Exception:
        return None


def _ttl_memo_put(path: Path, key: str, value: Any, ttl: float, *, ts_key: str, value_key: str, what: str, ts_first: bool = False) -> None:
    """Write ``key`` into a TTL memo file, dropping expired siblings. Best-effort."""
    try:
        now = time.time()
        data = {k: v for k, v in _load_json_dict(path).items() if isinstance(v, dict) and (now - float(v.get(ts_key, 0))) < ttl}
        data[key] = {ts_key: now, value_key: value} if ts_first else {value_key: value, ts_key: now}
        atomic_json_write(path, data, indent=0, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save %s: %s", what, e)


def _local_probe_disk_cache_path() -> Path:
    return _cache_file("local_endpoint_probes.json")


def _local_probe_disk_get(kind: str, key: str) -> Optional[Any]:
    """Return a fresh cached value for ``kind:key``, else None."""
    return _ttl_memo_get(_local_probe_disk_cache_path(), f"{kind}:{key}", _LOCAL_PROBE_DISK_TTL_SECONDS, ts_key="ts", value_key="value")


def _local_probe_disk_put(kind: str, key: str, value: Any) -> None:
    """Persist a successful probe result. Best-effort; prunes stale entries."""
    _ttl_memo_put(_local_probe_disk_cache_path(), f"{kind}:{key}", value, _LOCAL_PROBE_DISK_TTL_SECONDS, ts_key="ts", value_key="value", what="local probe disk cache")


def _get_model_metadata_cache_path() -> Path:
    """Path to the OpenRouter model metadata disk cache."""
    return _cache_file("openrouter_model_metadata.json")


def _model_metadata_disk_cache_age_seconds() -> Optional[float]:
    """Disk-cache age in seconds, or None if freshness is unknown."""
    try:
        age = time.time() - _get_model_metadata_cache_path().stat().st_mtime
        return age if age >= 0 else None
    except Exception:
        return None


def _load_model_metadata_disk_cache() -> Dict[str, Dict[str, Any]]:
    """Processed OpenRouter metadata cache from disk ({} on any failure)."""
    try:
        with _get_model_metadata_cache_path().open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(key): value for key, value in data.items() if isinstance(value, dict)} if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug("Failed to load OpenRouter model metadata disk cache: %s", e)
        return {}


def _save_model_metadata_disk_cache(data: Dict[str, Dict[str, Any]]) -> None:
    try:
        atomic_json_write(_get_model_metadata_cache_path(), data, indent=0, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save OpenRouter model metadata disk cache: %s", e)


def _get_endpoint_metadata_cache_path() -> Path:
    """On-disk memo of remote ``/models`` probes (see ``_endpoint_disk_cache_get``)."""
    return _cache_file("endpoint_model_metadata.json")


def _endpoint_disk_cache_get(normalized: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fresh cross-process memo of a remote ``/models`` probe (same TTL as in-memory): one-shot
    runs (``hermes -q``, cron) start cold and Nous bypasses the persistent context cache, so
    without this every launch paid the live probe. Local endpoints are never memoized."""
    models = _ttl_memo_get(_get_endpoint_metadata_cache_path(), normalized, _ENDPOINT_MODEL_CACHE_TTL, ts_key="at", value_key="models")
    return models if isinstance(models, dict) else None


def _endpoint_disk_cache_put(normalized: str, cache: Dict[str, Dict[str, Any]]) -> None:
    """Memoize a successful remote ``/models`` probe; expired siblings are dropped."""
    _ttl_memo_put(
        _get_endpoint_metadata_cache_path(), normalized, cache, _ENDPOINT_MODEL_CACHE_TTL,
        ts_key="at", value_key="models", what="endpoint model metadata disk cache", ts_first=True)


# Descending probe tiers for unknown models; tier[0] is also the default fallback.
CONTEXT_PROBE_TIERS = [256_000, 128_000, 64_000, 32_000, 16_000, 8_000]
DEFAULT_FALLBACK_CONTEXT = CONTEXT_PROBE_TIERS[0]
_FALLBACK_WARNED: set = set()  # the fallback is never cached, so dedupe its warning per (model, base_url)


def _warn_context_length_fallback(model: str, base_url: str) -> None:
    """Warn once per model+endpoint that detection failed and the hard default is used,
    so small-context models (8K, 32K) don't silently get 256K and fail at the API."""
    key = (model, base_url or "")
    if key not in _FALLBACK_WARNED:
        _FALLBACK_WARNED.add(key)
        logger.warning(
            "Could not determine context length for model %r (base_url=%s) — falling back to %s tokens. "
            "Set model.context_length in config.yaml to override.",
            model, base_url or "default", f"{DEFAULT_FALLBACK_CONTEXT:,}",
        )


# Sessions, model switches and cron jobs reject models below this (too little working memory).
MINIMUM_CONTEXT_LENGTH = 64_000
# In-process (model, base_url) -> (result, monotonic_ts) memo for local probes: one
# startup resolves the same model several times (banner, /model, compressor). Never persisted.
_LOCAL_CTX_PROBE_TTL_SECONDS = 30.0
_LOCAL_CTX_PROBE_CACHE: Dict[tuple, tuple] = {}
# Family-pattern fallbacks, used only when provider-aware sources all miss.
# Lookups are longest-key-first substring matches, so dict order is cosmetic
# and a specific key must be STRICTLY longer than its catch-all.
DEFAULT_CONTEXT_LENGTHS = {
    # Anthropic — bare ids only (prefixed ids resolve via OpenRouter/models.dev
    # and would collide: "anthropic/claude-sonnet-4" ⊂ "anthropic/claude-sonnet-4.6").
    "claude-fable-5": 1000000, "claude-fable": 1000000, "claude-opus-5": 1000000, "claude-sonnet-5": 1000000,
    "claude-opus-4-8": 1000000, "claude-opus-4.8": 1000000, "claude-opus-4-7": 1000000, "claude-opus-4.7": 1000000,
    "claude-opus-4-6": 1000000, "claude-sonnet-4-6": 1000000, "claude-opus-4.6": 1000000, "claude-sonnet-4.6": 1000000,
    "claude": 200000,  # catch-all for older Claude models
    # OpenAI — direct-API windows (Codex OAuth caps gpt-5.4+/5.5/5.6 at 272K, resolved by
    # its own branch). 5.4-nano/-mini are 400k, not 1.05M; gpt-5.3-codex-spark is
    # Codex-OAuth-only and listed so "gpt-5" (400k) doesn't win.
    "gpt-5.6-luna": 1050000, "gpt-5.6-terra": 1050000, "gpt-5.6-sol": 1050000, "gpt-5.5": 1050000,
    "gpt-5.4-nano": 400000, "gpt-5.4-mini": 400000, "gpt-5.4": 1050000,
    "gpt-5.3-codex-spark": 128000, "gpt-5.1-chat": 128000, "gpt-5": 400000,
    "gpt-4.1": 1047576, "gpt-4": 128000,
    # Google / Gemma ("gemma4" is Ollama-style naming, e.g. gemma4:31b-cloud)
    "gemini": 1048576,
    "gemma-4": 256000, "gemma4": 256000, "gemma-4-31b": 256000, "gemma-3": 131072, "gemma": 8192,
    # DeepSeek — V4 family is 1M; deepseek-chat/-reasoner alias v4-flash modes.
    # https://api-docs.deepseek.com/zh-cn/quick_start/pricing
    "deepseek-v4-pro": 1_000_000, "deepseek-v4-flash": 1_000_000, "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000, "deepseek": 128000,
    # Meta; Muse Spark family (1.1/1.2/1.3, -contributor(-free), meta/ prefixed) is 1M per OpenRouter,
    # models.dev and api.commandcode.ai /models — keep the "muse-spark" prefix (bare "muse" would match
    # muse-image/muse-voice). Thinking Machines inkling (covers inkling-small and :free/:batch variants)
    "llama": 131072, "muse-spark-1.3": 1_048_576, "muse-spark": 1_048_576, "inkling": 1_048_576,
    # Qwen — https://help.aliyun.com/zh/model-studio/developer-reference/ (3.8-max/flash
    # 1M verified on OpenRouter & Nous portal 2026-08; qwen3-max = 256K Coding Plan snapshot)
    "qwen3.8-max": 1_000_000, "qwen3.8-flash": 1_000_000, "qwen3.6-plus": 1048576, "qwen3.7-plus": 1048576,
    "qwen3-coder-plus": 1000000, "qwen3-coder": 262144, "qwen3-max": 262144, "qwen": 131072,
    # MiniMax — M3 is 1M; M2.x is 204,800. https://platform.minimax.io/docs/api-reference/text-chat-openai
    "minimax-m3": 1000000, "minimax": 204800,
    # GLM — 5.2/5.3 are 1M (5.2 verified empirically at 789K on api.z.ai); older GLM ~202K.
    # The OpenRouter :free variant is capped; the longer key wins.
    "glm-5.2": 1_048_576, "glm-5.2:free": 256_000, "glm-5.3": 1_048_576, "glm": 202752,
    # xAI — /v1/models returns no context_length, so these prevent probe-down on api.x.ai
    # custom providers (docs.x.ai). grok-composer(-2.5-fast, Grok Build CLI) is OAuth-only:
    # 200k usable (the /v1/responses ~262144 input+output budget is a separate limit).
    # grok-build-latest aliases grok-4.5; grok-4-fast / grok-4.20 also match their
    # -(non-)reasoning and -multi-agent variants; "grok" is the catch-all.
    "grok-composer": 200000, "grok-build-latest": 500000, "grok-build": 256000, "grok-code-fast": 256000,
    "grok-2-vision": 8192, "grok-4-fast": 2000000, "grok-4.20": 2000000,
    "grok-4.6": 500000, "grok-4.5": 500000, "grok-4.3": 1000000, "grok-4": 256000,
    "grok-3": 131072, "grok-2": 131072, "grok": 131072,
    # Kimi — K3 is 1 Mi (matches the endpoint-scoped override); older Kimi 256K.
    "kimi-k3": 1_048_576, "kimi": 262144,
    # Upstage Solar — /v1/models returns no context_length; dated variants resolve via prefix.
    "solar-open2": 262144, "solar-pro3": 131072, "solar-pro2": 65536, "solar-mini": 32768,
    # Tencent Hunyuan (262144 = 256 × 1024, aligned with OpenRouter live metadata)
    "hy4-preview": 1_048_576, "hy3-preview": 262144, "hy3": 262144,
    # "Ox Alpha" stealth model (OpenCode Zen / OpenRouter slugs); NVIDIA Nemotron (128K
    # except 3.5 Lightning); Poolside Laguna 2.1 (:free / -free slugs); Arcee; OpenRouter.
    "x-preview-f": 1_048_576, "ox-alpha": 1_048_576,
    "nemotron-3.5-lightning": 1_000_000, "nemotron": 131072,
    "laguna-s-2.1": 262144, "laguna-xs-2.1": 262144, "trinity": 262144, "elephant": 262144,
    # Hugging Face Inference Providers — model IDs use org/name format
    "Qwen/Qwen3.5-397B-A17B": 131072, "Qwen/Qwen3.5-35B-A3B": 131072, "deepseek-ai/DeepSeek-V3.2": 65536,
    "moonshotai/Kimi-K2.5": 262144, "moonshotai/Kimi-K2.6": 262144, "moonshotai/Kimi-K2-Thinking": 262144,
    "MiniMaxAI/MiniMax-M2.5": 204800, "XiaomiMiMo/MiMo-V2-Flash": 262144,
    "mimo-v2-pro": 1048576, "mimo-v2.5-pro": 1048576, "mimo-v2.5": 1048576, "mimo-v2-omni": 262144, "mimo-v2-flash": 262144,
    "zai-org/GLM-5": 202752,
}
# xAI Grok models that ACCEPT `reasoning.effort` (verified live against
# /v1/responses). Unlisted Grok models still reason natively but 400 on the
# parameter, so callers must send no `reasoning` key rather than a default `medium`.
# grok-4.5/4.6 accept low/medium/high (default high) but REJECT "none", unlike grok-4.3.
_GROK_EFFORT_CAPABLE_PREFIXES = ("grok-3-mini", "grok-4.20-multi-agent", "grok-4.3", "grok-4.5", "grok-4.6")


def grok_supports_reasoning_effort(model: str) -> bool:
    """Allowlist check (aggregator prefixes like ``x-ai/`` stripped); unknown Grok models get no effort dial."""
    name = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bool(name) and any(name.startswith(prefix) for prefix in _GROK_EFFORT_CAPABLE_PREFIXES)


def is_grok_46_family(model: str) -> bool:
    """Whether *model* is a Grok 4.6 family identifier."""
    name = (model or "").strip().lower().replace("_", "-").rsplit("/", 1)[-1]
    return name == "grok-4.6" or name.startswith("grok-4.6-")


_CONTEXT_LENGTH_KEYS = (
    "context_length", "context_window", "context_size", "max_context_length", "max_position_embeddings",
    "max_model_len", "max_input_tokens", "max_sequence_length", "max_seq_len", "n_ctx_train", "n_ctx", "ctx_size",
)
_MAX_COMPLETION_KEYS = ("max_completion_tokens", "max_output_tokens", "max_tokens")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
# Docker / Podman / Lima DNS names that resolve to the host machine
_CONTAINER_LOCAL_SUFFIXES = (".docker.internal", ".containers.internal", ".lima.internal")


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _auth_headers(api_key: str = "") -> Dict[str, str]:
    token = str(api_key or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _is_custom_endpoint(base_url: str) -> bool:
    normalized = _normalize_base_url(base_url)
    return bool(normalized) and not base_url_host_matches(normalized, "openrouter.ai")


# Host substring -> provider. ".githubcopilot.com" covers api.enterprise./api.business.
# hosts; models.inference.ai.azure.com (GitHub Models free tier, ~8K per-request cap)
# is mapped so a targeted hint fires instead of the custom-endpoint path.
_URL_TO_PROVIDER: Dict[str, str] = {
    "api.openai.com": "openai", "chatgpt.com": "openai", "api.anthropic.com": "anthropic",
    "api.z.ai": "zai", "open.bigmodel.cn": "zai",
    "api.moonshot.ai": "kimi-coding", "api.moonshot.cn": "kimi-coding-cn", "api.kimi.com": "kimi-coding",
    "api.stepfun.ai": "stepfun", "api.stepfun.com": "stepfun", "api.arcee.ai": "arcee", "api.minimax": "minimax",
    "dashscope.aliyuncs.com": "alibaba", "dashscope-intl.aliyuncs.com": "alibaba", "portal.qwen.ai": "qwen-oauth",
    "openrouter.ai": "openrouter", "generativelanguage.googleapis.com": "gemini",
    "inference-api.nousresearch.com": "nous", "api.deepseek.com": "deepseek",
    "api.githubcopilot.com": "copilot", ".githubcopilot.com": "copilot", "models.github.ai": "copilot",
    "models.inference.ai.azure.com": "copilot",
    "api.fireworks.ai": "fireworks", "opencode.ai": "opencode-go", "api.x.ai": "xai",
    "integrate.api.nvidia.com": "nvidia", "api.xiaomimimo.com": "xiaomi", "xiaomimimo.com": "xiaomi",
    "api.gmi-serving.com": "gmi", "api.novita.ai": "novita",
    "tokenhub.tencentmaas.com": "tencent-tokenhub", "api.lkeap.cloud.tencent.com": "tencent-tokenplan",
    "ollama.com": "ollama-cloud",
}

# Auto-extend with provider-profile hostnames not already mapped.
try:
    for _pp in _list_providers():
        _host = _pp.get_hostname()
        if _host and _host not in _URL_TO_PROVIDER:
            _URL_TO_PROVIDER[_host] = _pp.name
except Exception:
    pass


def _infer_provider_from_url(base_url: str) -> Optional[str]:
    """models.dev provider name for a base URL (custom endpoints need no explicit provider)."""
    parsed = _parse_base_url(base_url, "https")
    if parsed is None:
        return None
    host = parsed.netloc.lower() or parsed.path.lower()
    for url_part, provider in _URL_TO_PROVIDER.items():
        if url_part in host:
            return provider
    return None


def _is_known_provider_base_url(base_url: str) -> bool:
    return _infer_provider_from_url(base_url) is not None


def _lmstudio_server_root(base_url: str) -> str:
    """LM Studio server root for native ``/api/v1`` endpoints."""
    root = _normalize_base_url(base_url)
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            return root[: -len(suffix)].rstrip("/")
    return root


def _server_root(base_url: str) -> str:
    """Probe root for a local server: IPv4-resolved, ``/v1`` suffix stripped."""
    server_url = _localhost_to_ipv4(base_url.rstrip("/"))
    return server_url[:-3] if server_url.endswith("/v1") else server_url


def _longest_key_match(table: Dict[str, int], model_lower: str) -> Optional[Tuple[str, int]]:
    """First ``(key, value)`` whose key is a substring of ``model_lower``, longest key first so
    specific entries (``gpt-5.4-mini``) beat their family catch-all (``gpt-5``); ties keep table order."""
    for key, value in sorted(table.items(), key=lambda x: len(x[0]), reverse=True):
        if key in model_lower:
            return key, value
    return None


def _ollama_show_context(data: Dict[str, Any], *, gguf_first: bool, minimum: Optional[int] = None) -> Optional[int]:
    """Context length from an Ollama ``/api/show`` payload. ``parameters.num_ctx`` is the RUNTIME
    window (Modelfile override), ``model_info.*.context_length`` the GGUF training max. Local users
    control num_ctx (prefer it); hosted operators may cap it arbitrarily (``gguf_first``)."""
    def _num_ctx():
        for line in data.get("parameters", "").split("\n"):
            parts = line.strip().split()
            if "num_ctx" in line and len(parts) >= 2:
                try:
                    yield int(parts[-1])
                except ValueError:
                    continue
    def _gguf():
        for key, value in data.get("model_info", {}).items():
            if "context_length" in key and isinstance(value, (int, float)):
                yield int(value)
    for reader in ((_gguf, _num_ctx) if gguf_first else (_num_ctx, _gguf)):
        for ctx in reader():
            if minimum is None or ctx >= minimum:
                return ctx
    return None


# (host, canonical paths, model ids, context) — see _endpoint_scoped_context_length.
_ENDPOINT_SCOPED_CONTEXT = (
    ("api.kimi.com", {"/coding", "/coding/v1"}, {"k3", "kimi-k3", "kimi-k3-cot"}, 1_048_576),
    ("integrate.api.nvidia.com", {"/v1"}, {"deepseek-ai/deepseek-v4-pro"}, 262_144),
)


def _endpoint_scoped_context_length(model: str, base_url: str) -> Optional[int]:
    """Context confirmed for one provider endpoint only (see _ENDPOINT_SCOPED_CONTEXT): Kimi Coding
    serves K3 at 1 Mi only on the canonical ``api.kimi.com/coding`` host (legacy Moonshot keys do
    not); NVIDIA NIM serves deepseek-v4-pro at 262,144 while DeepSeek's native endpoint is 1M."""
    try:
        parsed = urlparse(_normalize_base_url(base_url))
        port = parsed.port
    except ValueError:
        return None
    # Only canonical https://host[:443]/path with no credentials/query/fragment.
    if parsed.scheme.lower() != "https" or port not in (None, 443) or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        return None
    host, path, model_key = (parsed.hostname or "").lower(), parsed.path.rstrip("/"), model.strip().lower()
    return next((ctx for scoped_host, paths, models, ctx in _ENDPOINT_SCOPED_CONTEXT if host == scoped_host and path in paths and model_key in models), None)


def _skip_persistent_context_cache(base_url: str, provider: str) -> bool:
    """Providers whose disk context cache must not short-circuit probing: LM Studio (loaded
    context is transient), Codex OAuth (entitlement-specific window; a persisted fallback would suppress revalidation)."""
    return (provider or "").strip().lower() in {"lmstudio", "openai-codex"}


def _save_unless_skipped(model: str, base_url: str, ctx: int, provider: str) -> None:
    """Persist ``ctx`` unless the provider opts out of the disk cache."""
    if not _skip_persistent_context_cache(base_url, provider):
        save_context_length(model, base_url, ctx)


def _maybe_cache_local_context_length(model: str, base_url: str, length: int) -> None:
    """Persist a probed local window only at/above MINIMUM_CONTEXT_LENGTH: sub-minimum windows are
    still returned so agent_init can reject them, but must not be blessed into the disk cache."""
    if length >= MINIMUM_CONTEXT_LENGTH:
        save_context_length(model, base_url, length)


def _probe_local_context_length(model: str, base_url: str, api_key: str, provider: str) -> Optional[int]:
    """Live local probe; persists a positive result unless the provider opts out of the disk cache."""
    local_ctx = _query_local_context_length(model, base_url, api_key=api_key)
    if not (local_ctx and local_ctx > 0):
        return None
    if not _skip_persistent_context_cache(base_url, provider):
        _maybe_cache_local_context_length(model, base_url, local_ctx)
    return local_ctx


def _reconcile_local_cached_context_length(model: str, base_url: str, cached: int, api_key: str = "") -> int:
    """*cached* unless a live local probe reports a different limit (operators restart
    vLLM/Ollama with a new --max-model-len / num_ctx under the same id). A failed
    probe keeps the disk entry; sub-minimum live windows invalidate but are not persisted."""
    live_ctx = _query_local_context_length(model, base_url, api_key=api_key)
    if not (live_ctx and live_ctx > 0 and live_ctx != cached):
        return cached
    if live_ctx < MINIMUM_CONTEXT_LENGTH:
        logger.info("Live local probe for %s@%s reports %s (< minimum %s); invalidating stale cache — agent init should reject", model, base_url, f"{live_ctx:,}", f"{MINIMUM_CONTEXT_LENGTH:,}")
    else:
        logger.info("Reconciling stale local cache entry %s@%s: %s -> %s (live probe)", model, base_url, f"{cached:,}", f"{live_ctx:,}")
    _invalidate_cached_context_length(model, base_url)
    if live_ctx >= MINIMUM_CONTEXT_LENGTH:
        _maybe_cache_local_context_length(model, base_url, live_ctx)
    return live_ctx


def is_local_endpoint(base_url: str) -> bool:
    """True for loopback, container-internal DNS, unqualified hosts, RFC-1918,
    link-local and Tailscale CGNAT (so a trusted Ollama box over Tailscale gets
    the same timeout auto-bumps as localhost)."""
    try:
        parsed = _parse_base_url(base_url)
        host = parsed.hostname or "" if parsed is not None else None
    except Exception:
        return False
    if host is None:
        return False
    # Unqualified hostnames (no dots) are local by definition — Docker Compose service names, /etc/hosts entries, mDNS.
    if host in _LOCAL_HOSTS or host.endswith(_CONTAINER_LOCAL_SUFFIXES) or (host and "." not in host):
        return True
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local or (isinstance(addr, ipaddress.IPv4Address) and addr in _TAILSCALE_CGNAT):
            return True
    except ValueError:
        pass
    # Dotted quad that ipaddress rejected but still looks like a private range
    # (e.g. 172.26.x.x for WSL) or Tailscale CGNAT (100.64.x.x–100.127.x.x).
    parts = host.split(".")
    if len(parts) != 4:
        return False
    try:
        first, second = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return first == 10 or (first == 172 and 16 <= second <= 31) or (first == 192 and second == 168) or (first == 100 and 64 <= second <= 127)


def _localhost_to_ipv4(url: str) -> str:
    """``localhost`` HOST -> ``127.0.0.1`` (Windows dual-stack resolves ::1 first and pays a ~2s IPv6
    connect timeout on IPv4-only servers). Scheme-anchored so ``?upstream=http://localhost`` is untouched."""
    if not url or not isinstance(url, str):
        return url  # non-string values (test doubles, lazy config) pass through
    return re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", url, count=1)


def detect_local_server_type(base_url: str, api_key: str = "") -> Optional[str]:
    """Probe known endpoints: "ollama", "lm-studio", "vllm", "llamacpp", or None (TTL-cached)."""
    import httpx
    # IPv4-resolve BEFORE deriving server/LM Studio URLs and the cache lookup, so localhost and 127.0.0.1 share a cache entry.
    normalized = _localhost_to_ipv4(_normalize_base_url(base_url))
    server_url = _server_root(normalized)
    lmstudio_url = _lmstudio_server_root(normalized)
    cached = _endpoint_probe_path_cache.get(server_url)
    if cached is not None and (time.monotonic() - cached[1]) < (_ENDPOINT_PROBE_TTL_SECONDS if cached[0] is not None else _ENDPOINT_PROBE_FAILURE_TTL_SECONDS):
        return cached[0]
    # Blackholed host: skip the waterfall. Deliberately NOT written to the
    # hour-long verdict cache, which would pin "undetected" after it comes back.
    if _endpoint_blackholed(server_url):
        return None
    disk_hit = _local_probe_disk_get("server_type", server_url)
    if isinstance(disk_hit, str):
        _endpoint_probe_path_cache[server_url] = (disk_hit, time.monotonic())
        return disk_hit
    # Most specific first: (name, paths tried until one answers 200, body check). LM Studio answers /api/tags
    # with {"error": ...} and 200, so Ollama's body must carry "models"; older llama.cpp builds have no /v1 prefix.
    waterfall = (
        ("lm-studio", (f"{lmstudio_url}/api/v1/models",), lambda r: True),
        ("ollama", (f"{server_url}/api/tags",), lambda r: "models" in r.json()),
        ("llamacpp", (f"{server_url}/v1/props", f"{server_url}/props"), lambda r: "default_generation_settings" in r.text),
        ("vllm", (f"{server_url}/version",), lambda r: "version" in r.json()),
    )
    result: Optional[str] = None
    try:
        with httpx.Client(timeout=2.0, headers=_auth_headers(api_key)) as client:
            for name, urls, check in waterfall:
                try:
                    for url in urls:
                        r = client.get(url)
                        if r.status_code == 200:
                            break
                    if r.status_code == 200 and check(r):
                        result = name
                        break
                except Exception as exc:
                    # A connect timeout condemns the host: skip the remaining legs.
                    if _is_connect_timeout(exc):
                        _note_endpoint_blackholed(server_url)
                        raise
    except Exception:
        pass
    # Negative verdict in memory only (never on disk — failures are often transient).
    _endpoint_probe_path_cache[server_url] = (result, time.monotonic())
    if result is not None:
        _local_probe_disk_put("server_type", server_url, result)
    return result


# Cache the negative verdict in memory only (never on disk — a failure is often transient: server starting,
# key being fixed) so the very next turn does not re-run the whole waterfall against an endpoint that just
# answered nothing (#89863).
def _iter_nested_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_nested_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


def _coerce_reasonable_int(value: Any, minimum: int = 1024, maximum: int = 10_000_000) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        result = int(value.strip().replace(",", "") if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return None
    return result if minimum <= result <= maximum else None


def _extract_first_int(payload: Dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    keyset = {key.lower() for key in keys}
    for mapping in _iter_nested_dicts(payload):
        for key, value in mapping.items():
            if str(key).lower() in keyset:
                coerced = _coerce_reasonable_int(value)
                if coerced is not None:
                    return coerced
    return None


def _extract_flat_context_length(payload: Dict[str, Any]) -> Optional[int]:
    """Top-level-only context WINDOW read (no nested walk, so an unrelated nested section can't leak
    a same-named key). ``max_tokens`` is NOT a window key: on OpenAI-compatible passthroughs it is the max OUTPUT."""
    return next((c for c in map(_coerce_reasonable_int, (payload.get(k) for k in _CONTEXT_LENGTH_KEYS)) if c is not None), None)


def _context_length_from_model_payload(payload: Dict[str, Any]) -> Optional[int]:
    """Context window from a ``/v1/models`` object: window keys first, ``max_tokens`` last (Anthropic
    payloads carry ``max_input_tokens`` = 1M window AND ``max_tokens`` = 128k OUTPUT cap)."""
    if not isinstance(payload, dict):
        return None
    ctx = _extract_flat_context_length(payload)
    if ctx is not None:
        return ctx
    raw = payload.get("max_tokens")
    return int(raw) if isinstance(raw, (int, float)) and int(raw) > 0 else None


def _extract_pricing(payload: Dict[str, Any]) -> Dict[str, Any]:
    def _per_token(source: Dict[str, Any], fields: Dict[str, str], scale) -> Dict[str, Any]:
        # Provider $/MTok (or Novita's 1/10_000-$ per M) -> per-token strings, the same path usage_pricing uses for OpenRouter.
        return {target: str(scale(float(source[key]))) for target, key in fields.items() if source.get(key) is not None}
    novita_fields = {"prompt": "input_token_price_per_m", "completion": "output_token_price_per_m"}
    if any(payload.get(k) is not None for k in novita_fields.values()):
        return _per_token(payload, novita_fields, lambda v: v / 10_000 / 1_000_000)
    # DeepInfra ships pricing under ``metadata.pricing`` in $/MTok.
    metadata = payload.get("metadata")
    deepinfra_pricing = metadata.get("pricing") if isinstance(metadata, dict) else None
    deepinfra_fields = {"prompt": "input_tokens", "completion": "output_tokens", "cache_read": "cache_read_tokens"}
    if isinstance(deepinfra_pricing, dict) and any(k in deepinfra_pricing for k in deepinfra_fields.values()):
        return _per_token(deepinfra_pricing, deepinfra_fields, lambda v: v / 1_000_000)
    alias_map = {
        "prompt": ("prompt", "input", "input_cost_per_token", "prompt_token_cost"),
        "completion": ("completion", "output", "output_cost_per_token", "completion_token_cost"),
        "request": ("request", "request_cost"),
        "cache_read": ("cache_read", "cached_prompt", "input_cache_read", "cache_read_cost_per_token"),
        "cache_write": ("cache_write", "cache_creation", "input_cache_write", "cache_write_cost_per_token"),
    }
    for mapping in _iter_nested_dicts(payload):
        normalized = {str(key).lower(): value for key, value in mapping.items()}
        pricing: Dict[str, Any] = {}
        for target, aliases in alias_map.items():
            for alias in aliases:
                if alias in normalized and normalized[alias] not in {None, ""}:
                    pricing[target] = normalized[alias]
                    break
        if pricing:
            return pricing
    return {}


def _add_model_aliases(cache: Dict[str, Dict[str, Any]], model_id: str, entry: Dict[str, Any]) -> None:
    cache[model_id] = entry
    if "/" in model_id:
        cache.setdefault(model_id.split("/", 1)[1], entry)


def fetch_model_metadata(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch model metadata from OpenRouter (cached for 1 hour)."""
    global _model_metadata_cache, _model_metadata_cache_time
    if not force_refresh:
        if _model_metadata_cache and (time.time() - _model_metadata_cache_time) < _MODEL_CACHE_TTL:
            return _model_metadata_cache
        disk_age = _model_metadata_disk_cache_age_seconds()
        if disk_age is not None and disk_age < _MODEL_CACHE_TTL:
            disk_cache = _load_model_metadata_disk_cache()
            if disk_cache:
                _model_metadata_cache = disk_cache
                _model_metadata_cache_time = time.time() - disk_age
                return _model_metadata_cache
    try:
        _ensure_requests()
        # (connect, read) tuple: a flat timeout lets urllib3 block per retry stage through proxies that 403 CONNECT.
        # See #46620.
        response = requests.get(OPENROUTER_MODELS_URL, timeout=(5, 10), verify=_resolve_requests_verify())
        response.raise_for_status()
        cache = {}
        for model in response.json().get("data", []):
            model_id = model.get("id", "")
            entry = {
                "context_length": model.get("context_length", 128000),
                "max_completion_tokens": model.get("top_provider", {}).get("max_completion_tokens", 4096),
                "name": model.get("name", model_id), "pricing": model.get("pricing", {}),
            }
            canonical = model.get("canonical_slug", "")
            for alias in ((model_id, canonical) if canonical and canonical != model_id else (model_id,)):
                _add_model_aliases(cache, alias, entry)
        _model_metadata_cache = cache
        _model_metadata_cache_time = time.time()
        _save_model_metadata_disk_cache(cache)
        logger.debug("Fetched metadata for %s models from OpenRouter", len(cache))
        return cache
    except Exception as e:
        logger.warning("Failed to fetch model metadata from OpenRouter: %s", e)
        if _model_metadata_cache:
            return _model_metadata_cache
        disk_cache = _load_model_metadata_disk_cache()
        if disk_cache:
            _model_metadata_cache = disk_cache
            disk_age = _model_metadata_disk_cache_age_seconds()
            stale_by = min(disk_age, _MODEL_CACHE_TTL) if disk_age is not None else _MODEL_CACHE_TTL - 1
            _model_metadata_cache_time = time.time() - stale_by
            return _model_metadata_cache
        return {}


def _endpoint_model_entry(model: Dict[str, Any], model_id: str, context_length: Optional[int]) -> Dict[str, Any]:
    """Cache entry for one ``/models`` item; optional keys are set only when known."""
    optional = (("context_length", context_length), ("max_completion_tokens", _extract_first_int(model, _MAX_COMPLETION_KEYS)), ("pricing", _extract_pricing(model) or None))
    return {"name": model.get("name", model_id), **{k: v for k, v in optional if v is not None}}


def _lmstudio_loaded_context(model: Dict[str, Any]) -> Optional[int]:
    """Context of the first loaded LM Studio instance (the runtime value), else None."""
    for inst in model.get("loaded_instances", []) or []:
        cfg = inst.get("config", {}) if isinstance(inst, dict) else None
        ctx = cfg.get("context_length") if isinstance(cfg, dict) else None
        if isinstance(ctx, int) and ctx > 0:
            return ctx
    return None


def _lmstudio_native_models(normalized: str, headers: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """LM Studio ``/api/v1/models`` → cache; context comes from the first loaded instance."""
    response = requests.get(_lmstudio_server_root(normalized).rstrip("/") + "/api/v1/models", headers=headers, timeout=(5, 10), verify=_resolve_requests_verify(normalized))
    response.raise_for_status()
    cache: Dict[str, Dict[str, Any]] = {}
    for model in response.json().get("models", []):
        model_id = (model.get("key") or model.get("id")) if isinstance(model, dict) else None
        if not model_id:
            continue
        entry = _endpoint_model_entry(model, model_id, _lmstudio_loaded_context(model))
        _add_model_aliases(cache, model_id, entry)
        alt_id = model.get("id")
        if isinstance(alt_id, str) and alt_id and alt_id != model_id:
            _add_model_aliases(cache, alt_id, entry)
    return cache


def _apply_llamacpp_props(cache: Dict[str, Dict[str, Any]], request_candidate: str, headers: Dict[str, str], verify) -> None:
    """Overwrite ``context_length`` with llama.cpp's allocated ``n_ctx`` from /props (``/v1/props``, then
    ``/props`` for older builds). In router mode the bare endpoint 400s, so each LOADED child is read
    via ``/props?model=``; unloaded children are skipped — probing could autoload them."""
    base = request_candidate.rstrip("/").replace("/v1", "")
    def _props(params=None):
        resp = requests.get(base + "/v1/props", params=params, headers=headers, timeout=5, verify=verify)
        if not resp.ok:
            resp = requests.get(base + "/props", params=params, headers=headers, timeout=5, verify=verify)
        return resp
    def _n_ctx(props: Dict[str, Any]) -> Any:
        return (props.get("default_generation_settings") or {}).get("n_ctx")
    props_resp = _props()
    if props_resp.ok:
        props = props_resp.json()
        n_ctx, model_alias = _n_ctx(props), props.get("model_alias", "")
        if n_ctx and model_alias and model_alias in cache:
            cache[model_alias]["context_length"] = n_ctx
        return
    native = requests.get(base + "/models", headers=headers, timeout=5, verify=verify)
    if not native.ok:
        return
    for child in (native.json() or {}).get("data", [])[:16]:
        child_id = child.get("id") if isinstance(child, dict) else None
        if not child_id or child_id not in cache or (child.get("status") or {}).get("value") not in ("loaded", "ready"):
            continue
        pr = _props({"model": child_id})
        child_ctx = _n_ctx(pr.json()) if pr.ok else None
        if child_ctx:
            cache[child_id]["context_length"] = child_ctx


def _remember_endpoint_models(normalized: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    _endpoint_model_metadata_cache[normalized] = cache
    _endpoint_model_metadata_cache_time[normalized] = time.time()
    return cache


def _parse_models_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    for model in payload.get("data", []):
        model_id = model.get("id") if isinstance(model, dict) else None
        if model_id:
            _add_model_aliases(cache, model_id, _endpoint_model_entry(model, model_id, _extract_first_int(model, _CONTEXT_LENGTH_KEYS)))
    return cache


def fetch_endpoint_model_metadata(base_url: str, api_key: str = "", force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Model metadata from an OpenAI-compatible ``/models`` endpoint (cached per base URL)."""
    normalized = _normalize_base_url(base_url)
    if not normalized or base_url_host_matches(normalized, "openrouter.ai"):
        return {}
    _ensure_requests()
    local = is_local_endpoint(normalized)
    if not force_refresh:
        cached = _endpoint_model_metadata_cache.get(normalized)
        if cached is not None and (time.time() - _endpoint_model_metadata_cache_time.get(normalized, 0)) < _ENDPOINT_MODEL_CACHE_TTL:
            return cached
        memo = _endpoint_disk_cache_get(normalized) if not local else None
        if memo is not None:
            return _remember_endpoint_models(normalized, memo)
    # Blackholed: return empty WITHOUT caching so it is retried once the entry expires.
    if _endpoint_blackholed(normalized):
        return {}
    alternate = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized + "/v1"
    candidates = [normalized] + ([alternate] if alternate != normalized else [])
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    verify = _resolve_requests_verify(normalized)
    last_error: Optional[Exception] = None
    if local:
        try:
            if detect_local_server_type(normalized, api_key=api_key) == "lm-studio":
                return _remember_endpoint_models(normalized, _lmstudio_native_models(normalized, headers))
        except Exception as exc:
            last_error = exc
            _note_if_connect_timeout(exc, normalized)
    for candidate in candidates:
        # A connect timeout condemns the host, not the path.
        if _endpoint_blackholed(normalized):
            break
        # Cache keys stay unrewritten; only the outbound target is IPv4-resolved.
        request_candidate = _localhost_to_ipv4(candidate)
        url = request_candidate.rstrip("/") + "/models"
        response = None
        try:
            response = requests.get(url, headers=headers, timeout=(5, 10), verify=verify, stream=True)
            if response.status_code in (401, 403):
                logger.debug("Model metadata probe received HTTP %s from %s; stopping candidate probing", response.status_code, url)
                break
            response.raise_for_status()
            payload = response.json()
            cache = _parse_models_payload(payload)
            if any(m.get("owned_by") == "llamacpp" for m in payload.get("data", []) if isinstance(m, dict)):
                with contextlib.suppress(Exception):
                    _apply_llamacpp_props(cache, request_candidate, headers, verify)
            if cache and not local:
                _endpoint_disk_cache_put(normalized, cache)
            return _remember_endpoint_models(normalized, cache)
        except Exception as exc:
            last_error = exc
            _note_if_connect_timeout(exc, normalized)
        finally:
            if response is not None:
                response.close()
    if last_error:
        logger.debug("Failed to fetch model metadata from %s/models: %s", normalized, last_error)
    return _remember_endpoint_models(normalized, {})


def _resolve_endpoint_context_length(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Resolve context length from an endpoint's live ``/models`` metadata."""
    endpoint_metadata = fetch_endpoint_model_metadata(base_url, api_key=api_key)
    matched = endpoint_metadata.get(model)
    if not matched and len(endpoint_metadata) == 1:
        matched = next(iter(endpoint_metadata.values()))
    elif not matched and model:  # substring match; "" would match EVERY key and poison the window
        matched = next((entry for key, entry in endpoint_metadata.items() if model in key or key in model), None)
    context_length = matched.get("context_length") if matched else None
    return context_length if isinstance(context_length, int) else None


def _get_context_cache_path() -> Path:
    """Path to the persistent context length cache file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "context_length_cache.yaml"


def _load_context_cache() -> Dict[str, int]:
    """Load the model+provider -> context_length cache from disk."""
    path = _get_context_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("context_lengths") or {}
    except Exception as e:
        logger.debug("Failed to load context length cache: %s", e)
        return {}


def _write_context_cache(cache: Dict[str, int]) -> None:
    """Atomic write (a truncating write killed mid-dump leaves a partial file that
    _load_context_cache() swallows as {}, wiping EVERY cached length). Raises on failure."""
    atomic_yaml_write(_get_context_cache_path(), {"context_lengths": cache})


def _context_cache_key(model: str, base_url: str) -> str:
    """Canonical ``model@base_url`` key; trailing slashes stripped so ``/v1`` and ``/v1/`` share one entry."""
    return f"{model}@{(base_url or '').rstrip('/')}"


def save_context_length(model: str, base_url: str, length: int) -> None:
    """Persist a discovered context length under ``model@base_url`` (same model, different providers, different limits)."""
    # 0/negative is always a bug and would make get_model_context_length() return 0 (`0 is not None`).
    if length <= 0:
        logger.warning("Refusing to cache non-positive context length %s -> %s tokens", f"{model}@{base_url}", length)
        return
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    if cache.get(key) == length:
        return  # already stored
    cache[key] = length
    try:
        _write_context_cache(cache)
        logger.info("Cached context length %s -> %s tokens", key, f"{length:,}")
    except Exception as e:
        logger.debug("Failed to save context length cache: %s", e)


def get_cached_context_length(model: str, base_url: str) -> Optional[int]:
    """Look up a previously discovered context length for model+provider."""
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    # Legacy rows may carry a trailing slash, so probe the canonical key, the literal form and the slashed canonical form.
    return next((hit for hit in map(cache.get, (key, f"{model}@{base_url}", f"{key}/")) if hit is not None), None)


def _invalidate_cached_context_length(model: str, base_url: str) -> None:
    """Drop a stale cache entry so it gets re-resolved on the next lookup."""
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    # Also drop the in-memory TTL probe entries, or the next resolution inside the TTL window reuses the stale value.
    bare, stripped = _strip_provider_prefix(model), (base_url or "").rstrip("/")
    _LOCAL_CTX_PROBE_CACHE.pop((bare, stripped), None)
    _LOCAL_CTX_PROBE_CACHE.pop(("ollama_show", bare, stripped), None)
    # Every key shape get_cached_context_length consults.
    stale_keys = {key, f"{model}@{base_url}", f"{key}/"}
    if not any(k in cache for k in stale_keys):
        return
    for k in stale_keys:
        cache.pop(k, None)
    try:
        _write_context_cache(cache)
    except Exception as e:
        logger.debug("Failed to invalidate context length cache entry %s: %s", key, e)


def get_next_probe_tier(current_length: int) -> Optional[int]:
    """Next lower probe tier, or None if already at minimum."""
    return next((tier for tier in CONTEXT_PROBE_TIERS if tier < current_length), None)


def parse_context_limit_from_error(error_msg: str) -> Optional[int]:
    """Context limit quoted in a provider error ("maximum context length is 32768 tokens"), if any."""
    error_lower = error_msg.lower()
    patterns = (
        r'max_model_len\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM: "max_model_len 32768", "=32768", ": 32768", "(32768)", "is 32768"
        r'maximum model length\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM alt: "maximum model length 131072", "... is 131072"
        r'(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})',
        r'context\s*(?:length|size|window)\s*(?:is|of|:)?\s*(\d{4,})',
        r'(\d{4,})\s*(?:token)?\s*(?:context|limit)',
        r'>\s*(\d{4,})\s*(?:max|limit|token)',  # "250000 tokens > 200000 maximum"
        r'(\d{4,})\s*(?:max(?:imum)?)\b',  # "200000 maximum"
        # Gemini: "input token count is 32825 but model only supports up to
        # 32768" — anchor on the phrase so the input count isn't captured.
        r'supports?\s+(?:only\s+)?up\s+to\s+(\d{4,})',
    )
    for match in filter(None, (re.search(pattern, error_lower) for pattern in patterns)):
        limit = int(match.group(1))
        if 1024 <= limit <= 10_000_000:  # sanity: must be a plausible window
            return limit
    return None


def get_context_length_from_provider_error(error_msg: str, current_context_length: int) -> Optional[int]:
    """Provider-reported limit LOWER than the current window, else None. Overflow recovery must
    not invent a window: when the provider only says the input is too long, callers keep the
    configured length and compress rather than stepping down guessed probe tiers."""
    parsed_limit = parse_context_limit_from_error(error_msg)
    return parsed_limit if parsed_limit is not None and parsed_limit < current_context_length else None


def parse_available_output_tokens_from_error(error_msg: str) -> Optional[int]:
    """Available OUTPUT tokens from a "max_tokens too large" error, or None. Distinct from "prompt
    too long" (-> compress): here input + requested_output > window, so the fix is a smaller
    max_tokens for this call and context_length must NOT be touched."""
    error_lower = error_msg.lower()
    if not _any_phrase_group(error_lower, _PARSEABLE_OUTPUT_CAP_SIGNALS):
        return None
    # Direct cap figures, most specific first: "exceeds model's maximum output tokens (65536)", "Range of
    # max_tokens should be [1, 65536]" (upper bound is the cap), Anthropic "= available_tokens: 10000", last "= N".
    for pattern in (
        r'exceeds model(?:\'s)? maximum output tokens\s*\(?\s*(\d+)\s*\)?',
        r'range of max_tokens should be\s*\[\s*\d+\s*,\s*(\d+)\s*\]',
        r'available_tokens[:\s]+(\d+)',
        r'available\s+tokens[:\s]+(\d+)',
        r'=\s*(\d+)\s*$',
    ):
        match = re.search(pattern, error_lower)
        if match and int(match.group(1)) >= 1:
            return int(match.group(1))
    # OpenRouter/Nous: "maximum context length is N … (A of text input, B of tool input, C in the output)" -> ctx - A - B.
    _m_ctx = re.search(r'maximum context length is (\d+)', error_lower)
    _m_parts = re.search(r'\((\d+)\s+of text input,\s*(\d+)\s+of tool input,\s*(\d+)\s+in the output\)', error_lower)
    if _m_ctx and _m_parts:
        _available = int(_m_ctx.group(1)) - int(_m_parts.group(1)) - int(_m_parts.group(2))
        if _available >= 1:
            return _available
    # LM Studio / llama.cpp: window in tokens, prompt in CHARACTERS; ~3 chars/token over-reserves the input.
    _m_ctx_tok = re.search(r'maximum context length is (\d+)\s*token', error_lower)
    _m_chars = re.search(r'prompt contains (\d+)\s*character', error_lower)
    if _m_ctx_tok and _m_chars:
        _available = int(_m_ctx_tok.group(1)) - (int(_m_chars.group(1)) + 2) // 3
        if _available >= 1:
            return _available
    # vLLM: window and prompt both in TOKENS; available = window - input (None when the input alone
    # overflows -> compress). When max_tokens is the BINDING constraint vLLM reports "at least N input
    # tokens" with N == window + 1 - requested_output, so window - N == requested_output - 1 and each
    # retry walks the cap down by the safety margin without ever fitting: halve the cap instead.
    _m_vllm_input = re.search(r'prompt contains (?:at least )?(\d+)\s*input tokens', error_lower)
    if _m_ctx_tok and _m_vllm_input:
        _available = int(_m_ctx_tok.group(1)) - int(_m_vllm_input.group(1))
        _m_requested_out = re.search(r'requested (\d+)\s*output tokens', error_lower)
        if 'at least' in error_lower and _m_requested_out:
            _requested_out = int(_m_requested_out.group(1))
            if _available >= _requested_out - 1:
                # The budget is derived from the constraint, not measured.
                return max(1, _requested_out // 2)
        if _available >= 1:
            return _available
    return None


# Each entry is a phrase group; the group matches when ALL phrases are present.
# DashScope, Anthropic, OpenRouter/Nous, LM Studio/llama.cpp, generic "should be <= N", OpenAI-compat relays.
_OUTPUT_CAP_SIGNALS = (
    ("range of max_tokens should be",), ("available_tokens",), ("available tokens",),
    ("in the output", "maximum context length"), ("requested", "output tokens"),
    ("should be",), ("less than or equal",), ("must be",), ("exceeds model", "maximum output tokens"),
)
_INPUT_OVERFLOW_SIGNALS = (
    "prompt is too long", "prompt too long", "input is too long", "input token",
    "prompt length", "prompt contains", "reduce the length",
)
# Narrower than _OUTPUT_CAP_SIGNALS: only phrasings we can extract a number from.
# "requested N output tokens" means the OUTPUT cap is the problem (the input fits) —
# reduce max_tokens, don't compress. DashScope's bounded range upper bound IS the
# real max-output cap ("Range of max_tokens should be [1, 65536]").
_PARSEABLE_OUTPUT_CAP_SIGNALS = (
    ("max_tokens", "available_tokens"), ("max_tokens", "available tokens"),
    ("in the output", "maximum context length"),
    ("maximum context length", "requested", "output tokens"),
    ("range of max_tokens should be",), ("exceeds model", "maximum output tokens"),
)


def _any_phrase_group(text: str, groups: tuple) -> bool:
    return any(all(p in text for p in group) for group in groups)


def is_output_cap_error(error_msg: str) -> bool:
    """Yes/no sibling of :func:`parse_available_output_tokens_from_error` for unparseable wordings. An
    output-cap 400 misclassified as context overflow death-loops the compressor (same max_tokens, same
    rejection). Signal: talks about max_tokens as a cap/range/limit and NOT about an oversized input."""
    error_lower = error_msg.lower()
    # An error that ALSO describes an oversized INPUT is a genuine overflow — compression can fix it.
    return (
        any(p in error_lower for p in ("max_tokens", "max_output_tokens", "max_completion_tokens"))
        and _any_phrase_group(error_lower, _OUTPUT_CAP_SIGNALS)
        and not any(p in error_lower for p in _INPUT_OVERFLOW_SIGNALS)
    )


def _model_id_matches(candidate_id: str, lookup_model: str) -> bool:
    """Exact match, or ``publisher/slug`` (LM Studio native ids) whose slug equals the configured name."""
    return candidate_id == lookup_model or ("/" in candidate_id and candidate_id.rsplit("/", 1)[1] == lookup_model)


def _ollama_show(server_url: str, api_key: str, bare_model: str, timeout: float = 3.0, *, note_blackhole: bool = False) -> Optional[Dict[str, Any]]:
    """Ollama ``/api/show`` JSON for ``bare_model``, or None on any failure (``note_blackhole``: connect timeouts condemn the host)."""
    import httpx
    try:
        with httpx.Client(timeout=timeout, headers=_auth_headers(api_key)) as client:
            resp = client.post(f"{server_url}/api/show", json={"name": bare_model})
            return resp.json() if resp.status_code == 200 else None
    except Exception as exc:
        if note_blackhole:
            _note_if_connect_timeout(exc, server_url)
        return None


def _is_ollama_server(base_url: str, api_key: str) -> bool:
    try:
        # Forward the API key: a remote API-keyed endpoint answers the probe waterfall with 401s without it,
        # and an unauthorized probe can never produce a positive verdict (#89863).
        return detect_local_server_type(base_url, api_key=api_key) == "ollama"
    except Exception:
        return False


def query_ollama_num_ctx(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Ollama ``/api/show`` context (Modelfile num_ctx, else GGUF max); the value to send as ``num_ctx``."""
    bare_model, server_url = _strip_provider_prefix(model), _server_root(base_url)
    if not _is_ollama_server(base_url, api_key):
        return None
    _disk_key = f"{server_url}|{bare_model}"
    disk_hit = _local_probe_disk_get("ollama_num_ctx", _disk_key)
    if isinstance(disk_hit, int) and disk_hit > 0:
        return disk_hit
    data = _ollama_show(server_url, api_key, bare_model)
    ctx = _ollama_show_context(data, gguf_first=False) if data is not None else None
    if ctx is not None:
        _local_probe_disk_put("ollama_num_ctx", _disk_key, ctx)
    return ctx


def query_ollama_supports_vision(model: str, base_url: str, api_key: str = "") -> Optional[bool]:
    """True/False when Ollama ``/api/show`` reports vision support (``capabilities`` on 0.6.0+, else
    ``model_info.*.vision.block_count``); None when unreachable, not Ollama, or model unknown."""
    bare_model = _strip_provider_prefix(model)
    if not bare_model or not base_url or not _is_ollama_server(base_url, api_key):
        return None
    data = _ollama_show(_server_root(base_url), api_key, bare_model)
    if data is None:
        return None
    caps = data.get("capabilities")
    if isinstance(caps, list) and caps:
        return any(str(cap).lower() == "vision" for cap in caps)
    model_info = data.get("model_info")
    if isinstance(model_info, dict) and any("vision.block_count" in str(key).lower() for key in model_info):
        return True
    return None


def _memo_local_probe(cache_key: tuple, probe: Callable[[], Optional[int]]) -> Optional[int]:
    """Short-TTL, positive-only memo of a local probe (see _LOCAL_CTX_PROBE_CACHE): a failure during
    a startup race must not suppress the retry once the server is up, so only truthy results memoize."""
    now = time.monotonic()
    cached = _LOCAL_CTX_PROBE_CACHE.get(cache_key)
    if cached is not None and (now - cached[1]) < _LOCAL_CTX_PROBE_TTL_SECONDS:
        return cached[0]
    result = probe()
    if result:
        _LOCAL_CTX_PROBE_CACHE[cache_key] = (result, now)
    return result


def _query_ollama_api_show(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Provider-agnostic Ollama ``/api/show`` probe (any hostname; non-Ollama servers 404 fast).
    GGUF-first (hosted users can't set num_ctx) — the reverse of query_ollama_num_ctx(), hence the namespaced memo key."""
    return _memo_local_probe(("ollama_show", _strip_provider_prefix(model), base_url.rstrip("/")), lambda: _query_ollama_api_show_uncached(model, base_url, api_key=api_key))


def _query_ollama_api_show_uncached(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Uncached body of ``_query_ollama_api_show`` — one POST to ``/api/show``."""
    server_url = _server_root(base_url)
    if _endpoint_blackholed(server_url):
        return None
    data = _ollama_show(server_url, api_key, model, timeout=5.0, note_blackhole=True)
    # Hosted Ollama: the GGUF max is authoritative (the operator may have capped num_ctx arbitrarily).
    return _ollama_show_context(data, gguf_first=True, minimum=1024) if data is not None else None


def _model_name_suggests_kimi(model: str) -> bool:
    """Kimi family (``kimi-*``, ``moonshotai/*``) — guard against stale 32K underreports."""
    lower = model.lower()
    return lower.startswith("kimi") or "moonshot" in lower


def _model_name_suggests_minimax_m3(model: str) -> bool:
    """MiniMax M3 on any surface — models.dev underreport guard and agent_runtime_helpers cache-control gating."""
    return "minimax-m3" in model.lower()


# Catalog keys added AFTER the model was reachable via a shorter catch-all (or the
# 256K fallback): older builds persisted that smaller value and the step-1 cache hit
# would pin it forever. Only list keys whose catalog value is STRICTLY ABOVE every
# shorter matching key and the 256K fallback — the threshold is inferred from them.
_PRE_CATALOG_STALE_KEYS = frozenset({
    "minimax-m3",  # 1M; "minimax" catch-all persisted 204,800
    "muse-spark-1.3", "muse-spark",  # 1M; pre-entry builds fell through to the 256K fallback
    "grok-4.3", "grok-4.6",  # 1M / 500K; "grok-4" catch-all persisted 256,000
    "grok-4-fast", "grok-4.20",  # 2M; fell through to the 256K fallback
    "qwen3.6-plus",  # 1M; "qwen" catch-all persisted 131,072
})


def _stale_pre_catalog_cache_entry(model: str, cached: int) -> bool:
    """True when a persisted window is a pre-catalog leftover: the model resolves (longest-key-first) to a
    _PRE_CATALOG_STALE_KEYS key and the cached value is <= the largest shorter matching catch-all (or 256K)."""
    model_lower = model.lower()
    matches = [(key, value) for key, value in DEFAULT_CONTEXT_LENGTHS.items() if key in model_lower]
    if not matches:
        return False
    specific_key, specific_value = max(matches, key=lambda kv: len(kv[0]))
    if specific_key not in _PRE_CATALOG_STALE_KEYS or cached >= specific_value:
        return False
    shorter_values = [v for k, v in matches if len(k) < len(specific_key)]
    return cached <= max(shorter_values, default=DEFAULT_FALLBACK_CONTEXT)


def _model_name_suggests_minimax(model: str) -> bool:
    """MiniMax family (``minimax*``, ``minimaxai/*``) — guard against stale 32K underreports (real: 204.8K)."""
    lower = model.lower()
    return lower.startswith("minimax") or "minimaxai/" in lower


def _model_name_suggests_stale_32k_underreport(model: str) -> bool:
    """Model families known to be wrongly underreported as 32K."""
    return _model_name_suggests_kimi(model) or _model_name_suggests_minimax(model)


def _query_local_context_length(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Local-server context probe, short-TTL cached (see _LOCAL_CTX_PROBE_CACHE)."""
    return _memo_local_probe((_strip_provider_prefix(model), base_url.rstrip("/")), lambda: _query_local_context_length_uncached(model, base_url, api_key=api_key))


def _positive_int(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) and value else None


def _lmstudio_context(client, lmstudio_url: str, model: str) -> Optional[int]:
    """LM Studio native /api/v1/models (the OpenAI-compat list omits context);
    loaded-instance config is the runtime value."""
    resp = client.get(f"{lmstudio_url}/api/v1/models")
    if resp.status_code != 200:
        return None
    for m in resp.json().get("models", []):
        if _model_id_matches(m.get("key", ""), model) or _model_id_matches(m.get("id", ""), model):
            return next((ctx for ctx in (_positive_int(inst.get("config", {}).get("context_length")) for inst in m.get("loaded_instances", [])) if ctx is not None), None)
    return None


def _llamacpp_context(client, server_url: str, model: str) -> Optional[int]:
    """llama.cpp /props: the RUNTIME n_ctx, answered by the router even for a not-yet-loaded model
    (while /v1/models has meta=null), so a lazily-loaded model doesn't fall to a family catch-all."""
    import httpx
    for props_path in (f"/props?model={model}", "/props"):
        try:
            resp = client.get(f"{server_url}{props_path}")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            continue
        n_ctx = _positive_int((resp.json().get("default_generation_settings") or {}).get("n_ctx"))
        if n_ctx is not None:
            return n_ctx
    return None


def _openai_models_list_context(client, server_url: str, model: str) -> Optional[int]:
    """/v1/models list: match by id, else the sole model on single-model servers (llama.cpp reports a GGUF path as id)."""
    resp = client.get(f"{server_url}/v1/models")
    if resp.status_code != 200:
        return None
    models_list = resp.json().get("data", [])
    matched = next((m for m in models_list if isinstance(m, dict) and _model_id_matches(m.get("id", ""), model)), None)
    if matched is None and len(models_list) == 1:
        matched = models_list[0]
    if matched is None:
        return None
    # Runtime n_ctx (llama.cpp nests it under meta) beats n_ctx_train, which can exceed what the server allocates.
    sources = [s for s in (matched, matched.get("meta") or {}) if isinstance(s, dict)]
    for reader in (lambda src: _positive_int(src.get("n_ctx")), _context_length_from_model_payload):
        for source in sources:
            ctx = reader(source)
            if ctx is not None:
                return ctx
    return None


def _query_local_context_length_uncached(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Query a local server for the model's context length."""
    import httpx
    model = _strip_provider_prefix(model)
    server_url = _server_root(base_url)
    lmstudio_url = _localhost_to_ipv4(_lmstudio_server_root(base_url))
    if _endpoint_blackholed(server_url):
        return None
    try:
        server_type = detect_local_server_type(base_url, api_key=api_key)
    except Exception:
        server_type = None
    def _ollama_ctx(client) -> Optional[int]:
        # Ollama: num_ctx (runtime window) before the GGUF training max, or conversations
        # grow past what Ollama allocated and silently truncate. Matches query_ollama_num_ctx().
        resp = client.post(f"{server_url}/api/show", json={"name": model})
        return _ollama_show_context(resp.json(), gguf_first=False) if resp.status_code == 200 else None
    def _model_detail_ctx(client) -> Optional[int]:
        # LM Studio / vLLM / llama.cpp / Anthropic-compat proxies: /v1/models/{model}
        resp = client.get(f"{server_url}/v1/models/{model}")
        return _context_length_from_model_payload(resp.json()) if resp.status_code == 200 else None
    typed = {
        "ollama": _ollama_ctx,
        "lm-studio": lambda client: _lmstudio_context(client, lmstudio_url, model),
        "llamacpp": lambda client: _llamacpp_context(client, server_url, model),
    }.get(server_type)
    probes = ([typed] if typed else []) + [_model_detail_ctx, lambda client: _openai_models_list_context(client, server_url, model)]
    try:
        with httpx.Client(timeout=3.0, headers=_auth_headers(api_key)) as client:
            return next((ctx for ctx in (probe(client) for probe in probes) if ctx is not None), None)
    except Exception as exc:
        _note_if_connect_timeout(exc, server_url)
    return None


def _normalize_model_version(model: str) -> str:
    """Dots -> dashes so Nous ids (claude-opus-4-6) compare with OpenRouter's (claude-opus-4.6)."""
    return model.replace(".", "-")


def _query_anthropic_context_length(model: str, base_url: str, api_key: str) -> Optional[int]:
    """Anthropic /v1/models max_input_tokens; OAuth tokens (sk-ant-oat*) 401 and are skipped."""
    if not api_key or api_key.startswith("sk-ant-oat"):
        return None
    try:
        base = base_url.rstrip("/").removesuffix("/v1")
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        _ensure_requests()
        resp = requests.get(f"{base}/v1/models?limit=1000", headers=headers, timeout=(5, 10), verify=_resolve_requests_verify(base_url))
        if resp.status_code != 200:
            return None
        for m in resp.json().get("data", []):
            ctx = m.get("max_input_tokens") if m.get("id") == model else None
            if isinstance(ctx, int) and ctx > 0:
                return ctx
    except Exception as e:
        logger.debug("Anthropic /v1/models query failed: %s", e)
    return None


# Codex OAuth `context_window` values (what Codex enforces — lower than the direct API for the same
# slugs). Fallback when the live probe fails; longest-key-first. gpt-5.3-codex-spark is listed so "gpt-5.3-codex" doesn't win.
_CODEX_OAUTH_CONTEXT_FALLBACK: Dict[str, int] = {
    "gpt-5.1-codex-max": 272_000, "gpt-5.1-codex-mini": 272_000, "gpt-5.3-codex": 272_000,
    "gpt-5.3-codex-spark": 128_000, "gpt-5.2-codex": 272_000, "gpt-5.4-mini": 272_000,
    "gpt-5.6-sol": 272_000, "gpt-5.6-terra": 272_000, "gpt-5.6-luna": 272_000, "gpt-daybreak-blue-latest": 272_000,
    "gpt-5.5": 272_000, "gpt-5.4": 272_000, "gpt-5.2": 272_000, "gpt-5": 272_000,
}
# Codex OAuth advertises 272K for these families but ACCEPTS ~900K+ (verified live; gpt-5.5 and
# gpt-5.4-mini genuinely reject >272K). 900K keeps ≥11K margin. OPT-IN ONLY via explicit ``-900k``
# picker variants (a 900K default burned subscription usage); the suffix is stripped before the wire.
# The bump fires ONLY when the resolved value is exactly the stale 272,000. ``gpt-5.6`` is a FAMILY
# PREFIX (``-pro`` slugs aren't routable on Codex); ``gpt-5.4`` is EXACT because gpt-5.4-mini enforces 272K.
_CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_PREFIXES: Dict[str, int] = {"gpt-5.6": 900_000}  # sol / terra / luna
_CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_EXACT: Dict[str, int] = {"gpt-5.4": 900_000, "gpt-daybreak-blue-latest": 900_000}
_CODEX_OAUTH_STALE_ADVERTISED_CTX = 272_000  # the only advertised value the bump may override
CODEX_CONTEXT_VARIANT_SUFFIX = "-900k"  # picker-only opt-in suffix; never sent on the wire
# The ONLY bases eligible for ``-900k``: routable, live-verified. No family prefixing (it would synthesize
# dead ``-pro`` variants); dated snapshots of the 5.6 bases are allowed. gpt-daybreak-blue-latest is a verified Sol alias.
_CODEX_900K_SNAPSHOT_BASES = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
_CODEX_900K_ELIGIBLE_BASES = frozenset({*_CODEX_900K_SNAPSHOT_BASES, "gpt-5.4", "gpt-daybreak-blue-latest"})
_CODEX_900K_SNAPSHOT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _bare_codex_slug(model: Optional[str]) -> str:
    """Lowercased slug without ``vendor/`` (display/auxiliary callers pass ``openai/gpt-5.6-sol-900k``).

    Display/auxiliary callers pass ids like ``openai/gpt-5.6-sol-900k``; the main-agent path normalizes the
    namespace away earlier, but this resolver must accept both shapes (#92797 review).
    """
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def is_codex_900k_base(model: Optional[str]) -> bool:
    """Single source of truth for ``-900k`` eligibility (picker, resolution, /model validation, wire stripping)."""
    slug = _bare_codex_slug(model)
    if not slug or slug.endswith(CODEX_CONTEXT_VARIANT_SUFFIX):
        return False
    # Dated snapshots of the routable 5.6 bases (gpt-5.6-sol-2026-07-09) also qualify.
    return slug in _CODEX_900K_ELIGIBLE_BASES or any(
        slug.startswith(base + "-") and _CODEX_900K_SNAPSHOT_RE.match(slug[len(base) + 1:]) for base in _CODEX_900K_SNAPSHOT_BASES
    )


def _codex_variant_base(model: Optional[str]) -> Optional[str]:
    """Lowercased eligible base of a VALID ``-900k`` variant, else None (``gpt-5.5-900k`` is an invalid alias)."""
    slug = _bare_codex_slug(model)
    base = slug[: -len(CODEX_CONTEXT_VARIANT_SUFFIX)] if slug.endswith(CODEX_CONTEXT_VARIANT_SUFFIX) else None
    return base if base is not None and is_codex_900k_base(base) else None


def is_codex_context_variant(model: Optional[str]) -> bool:
    """Suffix AND eligible base — ``gpt-5.5-900k`` is an invalid alias, not a variant."""
    return _codex_variant_base(model) is not None


def strip_codex_context_variant_suffix(model: Optional[str]) -> str:
    """Wire-safe slug with a VALID ``-900k`` suffix removed (vendor prefix kept); an ineligible alias is
    returned unchanged so it fails honestly at the API instead of running as another model."""
    raw = (model or "").strip()
    return raw[: -len(CODEX_CONTEXT_VARIANT_SUFFIX)] if _codex_variant_base(raw) is not None else raw


def has_codex_context_variant(model_bare: str) -> bool:
    """Picker-side alias of :func:`is_codex_900k_base`."""
    return is_codex_900k_base(model_bare)


def _verified_codex_ctx_for_slug(model_bare: str) -> Optional[int]:
    """Live-verified cap for a VALID ``-900k`` variant only; base slugs and ineligible aliases -> None."""
    base = _codex_variant_base(model_bare)
    if base is None:
        return None
    exact = _CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_EXACT.get(base)
    if exact is not None:
        return exact
    return next((ctx for key, ctx in _CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_PREFIXES.items() if base == key or base.startswith((key + "-", key + "."))), None)


_codex_oauth_context_cache: Dict[str, Tuple[Dict[str, int], float]] = {}
_CODEX_OAUTH_CONTEXT_CACHE_TTL = 3600  # 1 hour


def _codex_oauth_token_fingerprint(access_token: str) -> str:
    """Non-secret cache key for a Codex OAuth access token."""
    return hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:16]


def _extract_chatgpt_account_id(access_token: str) -> Optional[str]:
    """``chatgpt_account_id`` from the Codex OAuth JWT, or None on any parse error. Without the
    ``ChatGPT-Account-Id`` header /backend-api/codex/models returns ``{"models":[]}`` (HTTP 200)
    and the probe silently falls back. Mirrors auxiliary_client.py."""
    try:
        payload_b64 = access_token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
        acct_id = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id") if isinstance(claims, dict) else None
        return acct_id if isinstance(acct_id, str) and acct_id else None
    except Exception:
        return None


def _fetch_codex_oauth_context_lengths_with_source(access_token: str) -> Tuple[Dict[str, int], bool]:
    """Codex catalogue ``{slug: context_window}`` plus whether it came from HTTP. Cached per token
    fingerprint (windows vary by entitlement). An in-process hit reports False: not a fresh
    provider confirmation, must not drive persistent writes."""
    now = time.time()
    cache_key = _codex_oauth_token_fingerprint(access_token)
    cached = _codex_oauth_context_cache.get(cache_key)
    if cached is not None and now - cached[1] < _CODEX_OAUTH_CONTEXT_CACHE_TTL:
        return cached[0], False
    headers = {"Authorization": f"Bearer {access_token}"}
    acct_id = _extract_chatgpt_account_id(access_token)
    if acct_id:
        headers["ChatGPT-Account-Id"] = acct_id
    try:
        _ensure_requests()
        resp = requests.get("https://chatgpt.com/backend-api/codex/models?client_version=1.0.0", headers=headers, timeout=(5, 10), verify=_resolve_requests_verify())
        if resp.status_code != 200:
            logger.debug("Codex /models probe returned HTTP %s; falling back to hardcoded defaults", resp.status_code)
            return {}, False
        data = resp.json()
    except Exception as exc:
        logger.debug("Codex /models probe failed: %s", exc)
        return {}, False
    result: Dict[str, int] = {}
    for item in data.get("models", []) if isinstance(data, dict) else []:
        slug, ctx = (item.get("slug"), item.get("context_window")) if isinstance(item, dict) else (None, None)
        if isinstance(slug, str) and isinstance(ctx, int) and ctx > 0:
            result[slug.strip()] = ctx
    if result:
        _codex_oauth_context_cache[cache_key] = (result, now)
    return result, True


def _resolve_codex_oauth_context_length_with_source(model: str, access_token: str = "") -> Tuple[Optional[int], str]:
    """``(context_length, source)`` for a Codex OAuth slug. source: "live" (fresh authenticated probe —
    the only one eligible for persistent writes), "memory" (same-token in-process hit), "fallback"
    (static table), or "" when unresolved."""
    model_bare = _strip_provider_prefix(model).strip()
    if not model_bare:
        return None, ""
    def _apply_verified_bump(ctx: int, source: str) -> Tuple[int, str]:
        """Lift an EXACT stale 272K advertisement to the verified cap for opted-in ``-900k`` variants only."""
        bumped = _verified_codex_ctx_for_slug(model_bare)
        if bumped is not None and ctx == _CODEX_OAUTH_STALE_ADVERTISED_CTX:
            logger.debug("Codex OAuth context for %s: advertised %d raised to live-verified %d", model_bare, ctx, bumped)
            return bumped, source
        return ctx, source
    # The Codex catalog only knows the base slug (no -900k, no vendor/).
    # ``-900k`` variants are Hermes picker aliases — the Codex catalog only knows the base slug, so resolve
    # against the stripped id. Also drop any ``vendor/`` namespace (``openai/gpt-5.6-sol-900k``): the
    # main-agent path normalizes it away before reaching here, but display/auxiliary callers pass it through
    # (#92797 review).
    lookup_bare = _bare_codex_slug(strip_codex_context_variant_suffix(model_bare))
    if access_token:
        live, fresh_probe = _fetch_codex_oauth_context_lengths_with_source(access_token)
        # Exact slug, then case-insensitive in case casing drifts.
        hit = live.get(lookup_bare)
        if hit is None:
            hit = next((ctx for slug, ctx in live.items() if slug.lower() == lookup_bare.lower()), None)
        if hit is not None:
            return _apply_verified_bump(hit, "live" if fresh_probe else "memory")
    hit = _longest_key_match(_CODEX_OAUTH_CONTEXT_FALLBACK, lookup_bare.lower())
    return _apply_verified_bump(hit[1], "fallback") if hit else (None, "")


def _resolve_nous_context_length(model: str, base_url: str = "", api_key: str = "") -> Tuple[Optional[int], str]:
    """``(context_length, source)`` for a Nous Portal model: portal /v1/models is authoritative
    ("portal"). Fallback matches OR's prefixed ids against the bare Nous id with dot/dash
    normalisation ("openrouter" — callers must NOT persist it, or a portal blip freezes the wrong value)."""
    if base_url:
        portal_ctx = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
        if portal_ctx is not None:
            return portal_ctx, "portal"
    metadata = fetch_model_metadata()
    def _safe_ctx(or_id: str, entry: dict) -> Optional[int]:
        """Context length minus the known stale 32K underreports (same guard as step 6)."""
        ctx = entry.get("context_length")
        if ctx is not None and ctx <= 32768 and _model_name_suggests_stale_32k_underreport(or_id):
            logger.info("Rejecting OpenRouter metadata context=%s for %r (known 32K underreport, Nous path); falling through to hardcoded defaults", ctx, or_id)
            return None
        return ctx
    model_lower, normalized = model.lower(), _normalize_model_version(model).lower()
    def _pairs(or_id: str):
        bare = or_id.split("/", 1)[1] if "/" in or_id else or_id
        return ((bare.lower(), model_lower), (_normalize_model_version(bare).lower(), normalized))
    def _exact(or_id: str) -> bool:
        return any(candidate == query for candidate, query in _pairs(or_id))
    def _prefix(or_id: str) -> bool:
        return any(candidate.startswith(query) and (len(candidate) == len(query) or candidate[len(query)] in "-:.") for candidate, query in _pairs(or_id))
    # Direct id, then exact bare-id match (with dot/dash normalisation), then prefix match on a
    # separator boundary — separate passes so any exact hit beats every prefix hit.
    for matcher in (lambda or_id: or_id == model, _exact, _prefix):
        for or_id, entry in metadata.items():
            ctx = _safe_ctx(or_id, entry) if matcher(or_id) else None
            if ctx is not None:
                return ctx, "openrouter"
    return None, ""


def _validate_cached_context_length(model: str, base_url: str, cached: int, is_bedrock_context: bool, *, api_key: str = "") -> Optional[int]:
    """Step 1 of get_model_context_length: accept, repair, or drop a persisted entry. Returns the
    value to use, or None to fall through to live resolution. Order matters: a value must be
    rejected as bogus before any provider-specific handling."""
    # Drop rules: (predicate, log level, message, shown value). 0/negative is always a bug (`0 is not
    # None` would hand the compressor a zero window); Kimi/MiniMax are underreported as 32K by stale
    # third-party metadata; pre-catalog leftovers persisted a shorter catch-all (see _PRE_CATALOG_STALE_KEYS).
    drop_rules = (
        (cached <= 0, logger.warning, "Dropping non-positive cache entry %s@%s -> %s; re-resolving", cached),
        (cached <= 32768 and _model_name_suggests_stale_32k_underreport(model), logger.info,
         "Dropping stale cached context entry %s@%s -> %s (known 32K underreport); re-resolving via hardcoded defaults", f"{cached:,}"),
        (_stale_pre_catalog_cache_entry(model, cached), logger.info,
         "Dropping stale pre-catalog cache entry %s@%s -> %s; re-resolving via hardcoded defaults", f"{cached:,}"),
    )
    for hit, log, msg, shown in drop_rules:
        if hit:
            log(msg, model, base_url, shown)
            _invalidate_cached_context_length(model, base_url)
            return None
    # Nous Portal: /v1/models is authoritative. Bypass (don't drop) the cache so step
    # 5b reconciles OR-seeded entries without touching disk when the portal is down.
    if _infer_provider_from_url(base_url) == "nous":
        logger.debug("Bypassing persistent cache for %s@%s (Nous portal authoritative)", model, base_url)
        return None
    if is_bedrock_context:
        # Bedrock: the static table is a FLOOR — probe-derived entries may legitimately exceed it.
        try:
            from agent.bedrock_adapter import get_bedrock_context_length
            bedrock_ctx = get_bedrock_context_length(model)
        except ImportError:
            return cached
        if cached < bedrock_ctx:
            logger.info("Dropping stale Bedrock cache entry %s@%s -> %s; using static Bedrock table value %s", model, base_url, f"{cached:,}", f"{bedrock_ctx:,}")
            _invalidate_cached_context_length(model, base_url)
            return bedrock_ctx
        return cached
    # For local endpoints, run the probe that respects configured Modelfile context values first.
    # _query_local_context_length prefers num_ctx from Modelfile, while _query_ollama_api_show returns the
    # GGUF training max first which can be larger and would create a false-safe window for compression
    # (#63122). Non-local endpoints preserve the existing GGUF-first behavior.
    if is_local_endpoint(base_url):
        return _reconcile_local_cached_context_length(model, base_url, cached, api_key=api_key)
    return cached


def _resolve_bedrock_context_length(model: str, base_url: str) -> Optional[int]:
    """Step 1b: Bedrock static table + one cached live probe (Bedrock exposes no context window via
    metadata APIs); None when boto3 is absent. Cached per model under base_url, else a synthetic
    bedrock:// key so display/offline paths share it."""
    try:
        from agent.bedrock_adapter import get_bedrock_context_length, resolve_bedrock_region
    except ImportError:
        return None  # boto3 not installed — fall through to generic resolution
    cache_key_url = base_url or "bedrock://"
    cached = get_cached_context_length(model, cache_key_url)
    if cached is not None:
        return cached
    # Region from the base_url host first, then the standard AWS chain. An empty region disables probing (table only).
    _m = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url) if base_url else None
    region = _m.group(1) if _m else ""
    if not region:
        with contextlib.suppress(Exception):
            region = resolve_bedrock_region()
    ctx = get_bedrock_context_length(model, region=region, probe=bool(region))
    # Only persist probe-derived values (region present); a pure table fallback must not poison the cache.
    if ctx and region:
        save_context_length(model, cache_key_url, ctx)
    return ctx


def _resolve_custom_endpoint_context_length(model: str, base_url: str, api_key: str, provider: str) -> int:
    """Steps 2-3 for a truly custom endpoint: /models, local probes, Ollama /api/show, catalog, default."""
    context_length = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
    if context_length is not None:
        return context_length
    # Local endpoints: the num_ctx-aware probe first — _query_ollama_api_show is GGUF-first, which
    # can be larger and create a false-safe compression window.
    local_ctx = _probe_local_context_length(model, base_url, api_key, provider) if is_local_endpoint(base_url) else None
    if local_ctx:
        return local_ctx
    # 2b. Ollama native /api/show (GGUF-first for non-local). Non-Ollama servers 404/405 quickly.
    ctx = _query_ollama_api_show(model, base_url, api_key=api_key)
    if ctx is not None:
        _save_unless_skipped(model, base_url, ctx, provider)
        return ctx
    # 3. Probe-down fallback after endpoint-specific detection failed
    logger.info(
        "Could not detect context length for model %r at %s — defaulting to %s tokens (probe-down). "
        "Set model.context_length in config.yaml to override.",
        model, base_url, f"{DEFAULT_FALLBACK_CONTEXT:,}",
    )
    # 3b. Hardcoded catalog as a last resort: a proxied Anthropic gateway fails the probes above
    # but its model name still matches DEFAULT_CONTEXT_LENGTHS.
    hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
    if hit:
        logger.info("Using hardcoded context length %s for model %r (custom endpoint, catalog match on %r)", f"{hit[1]:,}", model, hit[0])
        return hit[1]
    # Same silent-256K bug class as the step-9 fallback — warn here too.
    _warn_context_length_fallback(model, base_url)
    return DEFAULT_FALLBACK_CONTEXT


def _resolve_moa_context_length(model: str, custom_providers: list | None) -> Optional[int]:
    """Step 0a: MoA virtual provider — ``model`` is a preset name, so every probe would miss. Resolve
    the aggregator's real provider+model (references are advisory). None on any failure."""
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config
        from hermes_cli.moa_config import resolve_moa_preset
        from hermes_cli.runtime_provider import resolve_runtime_provider
        config = load_config()
        if custom_providers is None:
            custom_providers = get_compatible_custom_providers(config)
        agg = resolve_moa_preset(config.get("moa") or {}, model).get("aggregator") or {}
        agg_provider = str(agg.get("provider") or "").strip()
        agg_model = str(agg.get("model") or "").strip()
        if agg_model and agg_provider and agg_provider.lower() != "moa":
            rt = resolve_runtime_provider(requested=agg_provider, target_model=agg_model)
            return get_model_context_length(
                agg_model, base_url=rt.get("base_url", "") or "", api_key=rt.get("api_key", "") or "",
                provider=rt.get("provider") or agg_provider, custom_providers=custom_providers,
            )
    except Exception:
        logger.debug("MoA aggregator context-length resolution failed", exc_info=True)
    return None


def _config_override_context_length(model: str, base_url: str, provider: str, custom_providers: list | None) -> Optional[int]:
    """Steps 0b-0c: config-only overrides (never touch the network). 0b: EXPLICIT model_overrides
    only — fill-gap _default entries apply inside lookup_models_dev_context once the catalog has
    missed, so a _default can never preempt custom_providers or live probes. 0c: custom_providers."""
    # This is the supported self-unblock path for models with wrong context in models.dev (#84482) and for
    # custom/local models (#8731).
    if provider and model:
        with contextlib.suppress(Exception):  # fall through to other resolution paths
            from agent.models_dev import _override_context_window
            mo_ctx = _override_context_window(provider, model)
            if mo_ctx is not None and mo_ctx > 0:
                return mo_ctx
    # 0c. custom_providers per-model override — check before any probe. This closes the gap where /model
    # switch and display paths used to fall back to 128K despite the user having a per-model context_length
    # set. See #15779.
    if custom_providers and base_url and model:
        with contextlib.suppress(Exception):  # fall through to probing
            from hermes_cli.config import get_custom_provider_context_length
            cp_ctx = get_custom_provider_context_length(model=model, base_url=base_url, custom_providers=custom_providers)
            if cp_ctx:
                return cp_ctx
    return None


def _resolve_provider_aware_context_length(model: str, base_url: str, api_key: str, provider: str, effective_provider: str) -> Optional[int]:
    """Step 5: provider-specific sources, tried in order; None when all miss."""
    # 5a. Copilot live /models — account-specific models (claude-opus-4.6-1m) absent from
    # models.dev, and the provider-enforced limit for the rest.
    if effective_provider in {"copilot", "copilot-acp", "github-copilot"}:
        with contextlib.suppress(Exception):  # fall through to models.dev
            from hermes_cli.models import get_copilot_model_context
            ctx = get_copilot_model_context(model, api_key=api_key)
            if ctx:
                return ctx
    # 5b/5c. Nous portal and Codex OAuth (lower limits than the direct API for the same slug; its
    # own /models is authoritative). Persist ONLY the authoritative source ("portal" / "live"): an
    # OR-fallback or static-table value cached on a blip would be frozen in by step 1 forever.
    sourced = {
        "nous": lambda: _resolve_nous_context_length(model, base_url=base_url or "", api_key=api_key or "") + ("portal",),
        "openai-codex": lambda: _resolve_codex_oauth_context_length_with_source(model, access_token=api_key or "") + ("live",),
    }.get(effective_provider)
    if sourced is not None:
        ctx, source, persist_on = sourced()
        if ctx:
            if base_url and source == persist_on:
                save_context_length(model, base_url, ctx)
            return ctx
    if effective_provider in {"gmi", "commandcode", "commandcode-anthropic"} and base_url:
        # GMI and CommandCode expose authoritative context_length via /models (e.g. muse-spark 1M) but are
        # not in models.dev, and as known providers they skip step 2's probe — else they fell to 256K.
        ctx = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
        if ctx is not None:
            return ctx
    # 5e. Ollama native /api/show for any base_url that is not a known non-Ollama provider (there
    # the POST always 404s and cost ~300ms on the first turn).
    if base_url:
        inferred = _infer_provider_from_url(base_url)
        ctx = _query_ollama_api_show(model, base_url, api_key=api_key) if inferred is None or "ollama" in inferred else None
        if ctx is not None:
            _save_unless_skipped(model, base_url, ctx, provider)
            return ctx
    # 5f. OpenRouter live /models — authoritative for OR-routed models, so it must win over models.dev
    # and the family catch-all (a brand-new slug would otherwise fall to the generic "claude": 200K).
    if effective_provider == "openrouter":
        or_ctx = (fetch_model_metadata().get(model) or {}).get("context_length")
        # Guard against the known OpenRouter Kimi-family 32k underreport.
        if isinstance(or_ctx, int) and or_ctx > 0 and not (or_ctx == 32768 and _model_name_suggests_kimi(model)):
            return or_ctx
    if effective_provider:
        from agent.models_dev import lookup_models_dev_context
        ctx = lookup_models_dev_context(effective_provider, model)
        if ctx:
            # MiniMax M3: models.dev reports 512K but actual context is 1M — prefer the hardcoded catalog.
            catalog = DEFAULT_CONTEXT_LENGTHS.get("minimax-m3") if _model_name_suggests_minimax_m3(model) else None
            if catalog and ctx < catalog:
                logger.info("Rejecting models.dev context=%s for %r (MiniMax-M3 underreport); using hardcoded default %s", ctx, model, f"{catalog:,}")
                ctx = catalog
            return ctx
    return None


def get_model_context_length(
    model: str, base_url: str = "", api_key: str = "", config_context_length: int | None = None,
    provider: str = "", custom_providers: list | None = None,
) -> int:
    """Context length for a model. Resolution order: 0 config override / MoA aggregator /
    model_overrides / custom_providers / endpoint-scoped; 1 persistent cache (Nous, LM
    Studio, Codex OAuth bypass it) and Bedrock; 2-3 custom endpoints (/models, local
    probe, Ollama); 4 Anthropic /v1/models (API keys only); 5 provider-aware (Copilot,
    Nous, Codex OAuth, GMI, Ollama, OpenRouter live, models.dev); 6 OpenRouter for
    unknown providers; 7 local server; 8 hardcoded defaults; 9 256K fallback."""
    # 0. Explicit config override — user knows best
    if isinstance(config_context_length, int) and config_context_length > 0:
        return config_context_length
    if (provider or "").strip().lower() == "moa":
        ctx = _resolve_moa_context_length(model, custom_providers)
        if ctx is not None:
            return ctx
    ctx = _config_override_context_length(model, base_url, provider, custom_providers)
    if ctx is not None:
        return ctx
    # Malformed URLs (unmatched IPv6 bracket) make urllib.parse raise; treat them as unknown so
    # the inference layer reports the configuration error itself.
    if base_url:
        try:
            _ = urlparse(_normalize_base_url(base_url)).port
        except ValueError:
            base_url = ""
    # A blank model id would fuzzy-match an arbitrary catalog entry (`"" in key` is vacuously
    # true) and persist it under a junk "@<base_url>" cache key.
    if not str(model or "").strip():
        logger.info("No model id provided for context length resolution — defaulting to %s tokens.", f"{DEFAULT_FALLBACK_CONTEXT:,}")
        return DEFAULT_FALLBACK_CONTEXT
    model = _strip_provider_prefix(model)  # "local:x" -> "x"; Ollama "model:tag" colons preserved
    # Endpoint-scoped metadata goes AHEAD of the persistent cache so a value learned on a
    # multiplexed provider's other endpoint cannot override it.
    endpoint_context = _endpoint_scoped_context_length(model, base_url)
    if endpoint_context is not None:
        return endpoint_context
    is_bedrock_context = provider == "bedrock" or (
        base_url and base_url_hostname(base_url).startswith("bedrock-runtime.") and base_url_host_matches(base_url, "amazonaws.com")
    )
    # 1. Persistent cache (LM Studio / Codex OAuth excluded — see _skip_persistent_context_cache).
    cached = get_cached_context_length(model, base_url) if base_url and not _skip_persistent_context_cache(base_url, provider) else None
    validated = _validate_cached_context_length(model, base_url, cached, is_bedrock_context, api_key=api_key) if cached is not None else None
    if validated is not None:
        return validated
    # 1b. AWS Bedrock. Must run BEFORE the custom-endpoint step: bedrock-runtime.* is not in
    # _URL_TO_PROVIDER and would fail the /models probe into the default.
    ctx = _resolve_bedrock_context_length(model, base_url) if is_bedrock_context else None
    if ctx is not None:
        return ctx
    if provider == "novita" or (base_url and base_url_host_matches(base_url, "api.novita.ai")):
        ctx = _resolve_endpoint_context_length(model, base_url or "https://api.novita.ai/openai/v1", api_key=api_key)
        if ctx is not None:
            if base_url:
                save_context_length(model, base_url, ctx)
            return ctx
    # 2. Live /models for truly custom endpoints. Known providers skip this: their /models may
    # report a provider-imposed limit (Copilot: 128k) rather than the window.
    if _is_custom_endpoint(base_url) and not _is_known_provider_base_url(base_url):
        return _resolve_custom_endpoint_context_length(model, base_url, api_key, provider)
    # 4. Anthropic /v1/models API (only for regular API keys, not OAuth)
    if provider == "anthropic" or (base_url and base_url_hostname(base_url) == "api.anthropic.com"):
        ctx = _query_anthropic_context_length(model, base_url or "https://api.anthropic.com", api_key)
        if ctx:
            return ctx
    # 5. Provider-aware lookups — before the generic OR cache, since the same model has
    # different limits per provider. Generic providers are inferred from the URL.
    effective_provider = provider
    if base_url and (not effective_provider or effective_provider in {"openrouter", "custom"}):
        effective_provider = _infer_provider_from_url(base_url) or effective_provider
    ctx = _resolve_provider_aware_context_length(model, base_url, api_key, provider, effective_provider)
    if ctx is not None:
        return ctx
    # 6. OpenRouter metadata, provider-unaware fallback — only when the provider is unknown (OR
    # data is community-maintained); 32K underreport guard.
    if not effective_provider:
        metadata = fetch_model_metadata()
        if model in metadata:
            or_ctx = metadata[model].get("context_length", DEFAULT_FALLBACK_CONTEXT)
            if or_ctx == 32768 and _model_name_suggests_stale_32k_underreport(model):
                logger.info("Rejecting OpenRouter metadata context=%s for %r (known 32K underreport); falling through to hardcoded defaults", or_ctx, model)
            else:
                return or_ctx
    # 7. Local server before hardcoded defaults — ``Hermes-3-Llama-3.1-70B`` matches ``llama``
    # (131072) even when vLLM runs at a lower ``--max-model-len``.
    local_ctx = _probe_local_context_length(model, base_url, api_key, provider) if base_url and is_local_endpoint(base_url) else None
    if local_ctx:
        return local_ctx
    # 8. Hardcoded defaults: `key in model` only — the reverse would let "claude-sonnet-4" match "claude-sonnet-4-6" and return 1M.
    hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
    if hit:
        return hit[1]
    # 9. Default fallback — warn (deduped) so small-context models don't silently get 256K.
    _warn_context_length_fallback(model, base_url)
    return DEFAULT_FALLBACK_CONTEXT


async def get_model_context_length_async(model: str, base_url: str = "", api_key: str = "", config_context_length: int | None = None, provider: str = "", custom_providers: list | None = None) -> int:
    """get_model_context_length on a worker thread (its blocking HTTP would stall the event loop)."""
    import asyncio
    return await asyncio.to_thread(
        get_model_context_length, model, base_url=base_url, api_key=api_key,
        config_context_length=config_context_length, provider=provider, custom_providers=custom_providers)


# CJK/Hangul/Kana codepoints (~1 token each), counted in one C-level regex pass: Hangul
# Jamo (+Ext-A), CJK radicals/ideographs (+compat), Hangul syllables, fullwidth/halfwidth.
_CJK_DENSE_RE = re.compile("[\u1100-\u11ff\u2e80-\u9fff\ua960-\ua97f\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")


def _is_cjk_token_dense_char(ch: str) -> bool:
    return _CJK_DENSE_RE.fullmatch(ch) is not None


def estimate_tokens_rough(text: str) -> int:
    """Rough token estimate: CJK/Hangul/Kana codepoints ~1 token each; everything else ceil(UTF-8 bytes/4).
    Ceiling keeps short texts from estimating 0. Runs on every preflight walk, so all-ASCII stays O(1).

    Byte-counting (not chars) is the corrective for non-CJK, non-ASCII text: Cyrillic/Greek/Arabic are 2
    bytes/char so count ~chars/2, matching real BPE cost (~2-3 chars/token) where chars/4 under-counted
    ~2x and let sessions ride the provider ceiling below the compaction threshold. Calibrated vs
    cl100k/o200k/Qwen2.5 (estimate/real): Russian 0.67->1.24, Arabic 0.53->0.96, Hindi 0.34->0.90,
    Greek 0.37->0.68; accented Latin barely moves (French 1.02->1.03). errors="replace": lone surrogates
    (routine in tool output; see message_sanitization) must not turn an estimate into a raise."""
    if not text:
        return 0
    text = str(text)
    if text.isascii():  # flag check on CPython; ASCII cannot contain token-dense CJK
        return (len(text) + 3) // 4
    stripped = _CJK_DENSE_RE.sub("", text)
    dense = len(text) - len(stripped)
    return dense + ((len(stripped.encode("utf-8", "replace")) + 3) // 4)


def estimate_messages_tokens_rough(messages: List[Dict[str, Any]], *, charge_stale_thinking: bool = True) -> int:
    """Rough token estimate for a message list (pre-flight only). Images cost a flat ~1500 tokens
    each rather than their base64 length. ``charge_stale_thinking=False`` mirrors the tail-budget
    walk (``context_compressor._estimate_msg_budget_tokens``): on non-echo routes stale reasoning
    rides the wire only for the NEWEST assistant turn, so excluding it keeps the compaction TRIGGER
    in the same size class as the walk — otherwise reasoning-heavy sessions fire preflight forever."""
    _IMAGE_TOKEN_COST = 1500
    if not charge_stale_thinking:
        messages = _strip_stale_thinking_for_estimate(messages)
    return sum(_estimate_message_tokens_cached(msg, _IMAGE_TOKEN_COST) for msg in messages)


# Thinking-text keys replayed for at most the newest assistant turn on non-echo routes — must stay
# in lockstep with ``context_compressor._NEWEST_TURN_ONLY_BUDGET_KEYS``.
_STALE_THINKING_ESTIMATE_KEYS = ("reasoning", "reasoning_content")


def _strip_stale_thinking_for_estimate(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy of ``messages`` with stale thinking keys removed (newest kept). Shallow stripped copies
    share the original value objects, so the per-message memo still hits for the stripped shape."""
    def _is_assistant(m: Any) -> bool:
        return isinstance(m, dict) and m.get("role") == "assistant"
    newest = next((i for i in range(len(messages) - 1, -1, -1) if _is_assistant(messages[i])), -1)
    return [
        {k: v for k, v in m.items() if k not in _STALE_THINKING_ESTIMATE_KEYS}
        if i != newest and _is_assistant(m) and any(m.get(k) for k in _STALE_THINKING_ESTIMATE_KEYS) else m
        for i, m in enumerate(messages)
    ]


# Per-message token-estimate memo keyed by an exact value fingerprint: strings by ``id()`` AND
# pinned (strong ref in the entry, so the id can't be reused and immutability makes id-equality
# value-equality); numbers/bools/None by value; dicts/lists structurally in key order (``str(shadow)``
# depends on it); any other type aborts the memo. api_messages shallow-copies dicts but shares the strings.
# ``estimate_messages_tokens_rough`` is called on the full history every loop iteration (conversation_loop
# preflight), repeatedly during compaction telemetry, and inside an O(n^2) shrink loop in moa_loop. The
# per-message helpers are pure functions of the message's value, so a memo keyed on a fingerprint that
# uniquely determines the value is exactly equivalent. Fingerprint design (soundness argument): While the
# entry lives, that id cannot be reused by another object, so id-equality implies object-equality — strings
# are immutable, so value-equality too (no #50372-style aliasing). Equal fingerprints therefore imply
# deep-equal messages built from identical immutable leaves ⇒ identical ``str(shadow)`` bytes ⇒ identical
# estimate. Because the api_messages build shallow-copies history dicts each iteration, the copies share the
# same content strings — so unchanged history messages hit the memo even though the outer dicts are fresh
# objects every turn.
_MSG_TOKENS_CACHE: Dict[Any, Tuple[list, int]] = {}
_MSG_TOKENS_CACHE_MAX = 4096


def _msg_fingerprint(value: Any, pins: list) -> Any:
    if value is None or value is True or value is False:
        return value
    t = type(value)
    if t is str:
        pins.append(value)
        return ("s", id(value))
    if t is int or t is float:
        return ("n", t.__name__, value)
    if t is dict:
        return ("d", tuple((_msg_fingerprint(k, pins), _msg_fingerprint(v, pins)) for k, v in value.items()))
    if t is list or t is tuple:
        return ("l" if t is list else "t", tuple(_msg_fingerprint(v, pins) for v in value))
    raise ValueError("unfingerprintable message value")


def _estimate_message_tokens_cached(msg: Any, image_cost: int) -> int:
    def _compute() -> int:
        return _estimate_message_tokens_without_images(msg) + _count_image_tokens(msg, image_cost)
    try:
        pins: list = []
        key = _msg_fingerprint(msg, pins)
        hash(key)
    except Exception:
        return _compute()
    cached = _MSG_TOKENS_CACHE.get(key)
    if cached is not None:
        return cached[1]
    tokens = _compute()
    _MSG_TOKENS_CACHE[key] = (pins, tokens)
    while len(_MSG_TOKENS_CACHE) > _MSG_TOKENS_CACHE_MAX:
        try:
            _MSG_TOKENS_CACHE.pop(next(iter(_MSG_TOKENS_CACHE)))
        except (StopIteration, KeyError, RuntimeError):
            break
    return tokens


def _count_parts(parts: Any, types: set) -> int:
    return sum(1 for part in parts if isinstance(part, dict) and part.get("type") in types) if isinstance(parts, list) else 0


def _count_image_tokens(msg: Dict[str, Any], cost_per_image: int) -> int:
    """Count image-like content parts in a message; return their token cost."""
    if not isinstance(msg, dict):
        return 0
    content = msg.get("content")
    count = _count_parts(content, {"image", "image_url", "input_image"})
    count += _count_parts(msg.get("_anthropic_content_blocks"), {"image"})
    # Multimodal tool results that haven't been converted yet.
    if isinstance(content, dict) and content.get("_multimodal"):
        count += _count_parts(content.get("content"), {"image", "image_url"})
    return count * cost_per_image


def _wire_message_shadow(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Shadow of a message holding only what the provider actually receives.
    * ``api_content`` SUBSTITUTES ``content`` (mirrors ``turn_context.substitute_api_content`` exactly):
      only a non-empty STRING sidecar on a user/assistant row displaces content; substituting any
      other shape would UNDERcount — the dangerous direction.
    * Base64 images become a placeholder; ``_count_image_tokens`` charges them flat.
    * ``reasoning`` never ships as-is (request builds pop it after optionally promoting it into
      ``reasoning_content``); counting both inflated estimates up to +53%."""
    sidecar = msg.get("api_content")
    sidecar_wins = isinstance(sidecar, str) and bool(sidecar) and msg.get("role") in ("user", "assistant")
    _rc = msg.get("reasoning_content")
    drop_reasoning_dup = isinstance(_rc, str) and bool(_rc.strip())
    shadow: Dict[str, Any] = {}
    for k, v in msg.items():
        if k in ("_anthropic_content_blocks", "reasoning_details") or k in PERSISTENCE_ONLY_MESSAGE_FIELDS or (k == "reasoning" and drop_reasoning_dup):
            continue
        if k == "api_content":
            if sidecar_wins:
                shadow["content"] = v
        elif k == "content" and sidecar_wins:
            continue
        elif k == "content" and isinstance(v, list):
            shadow[k] = [
                {"type": part.get("type"), "image": "[stripped]"}
                if isinstance(part, dict) and part.get("type") in {"image", "image_url", "input_image"} else part
                for part in v
            ]
        elif k == "content" and isinstance(v, dict) and v.get("_multimodal"):
            shadow[k] = v.get("text_summary", "")
        else:
            shadow[k] = v
    return shadow


def _estimate_message_tokens_without_images(msg: Dict[str, Any]) -> int:
    """Token estimate for a message shadow with image payloads stripped."""
    return estimate_tokens_rough(str(_wire_message_shadow(msg) if isinstance(msg, dict) else msg))


def estimate_request_tokens_rough(
    messages: List[Dict[str, Any]], *, system_prompt: str = "", tools: Optional[List[Dict[str, Any]]] = None, charge_stale_thinking: bool = True,
) -> int:
    """Rough token estimate for a full request: system prompt + messages + tool schemas (50+ tools
    add 20-30K on their own). ``charge_stale_thinking`` is forwarded — pass False when the route
    provably strips stale thinking (``message_sanitization.stale_thinking_reaches_wire``)."""
    total = estimate_tokens_rough(system_prompt) if system_prompt else 0
    if messages:
        # Positional call: test seams and plugin engines monkeypatch estimate_messages_tokens_rough with (messages)-only signatures.
        total += estimate_messages_tokens_rough(messages) if charge_stale_thinking else estimate_messages_tokens_rough(messages, charge_stale_thinking=False)
    if tools:
        total += _estimate_tools_tokens_rough(tools)
    return total


# Usage-anchored accounting: ``usage.prompt_tokens`` is EXACT for everything sent on that request, so
# anchoring shrinks chars/4 estimation to the messages appended since. Fields: prompt_tokens /
# completion_tokens (provider usage at capture); base_count (len(messages) at capture — the reply is
# not yet appended and is covered by completion_tokens, so the delta walk skips it at index base_count);
# base_last_id / base_last_role (identity of the last message; compaction/splices replace it -> full estimation).


def capture_usage_anchor(prompt_tokens: Any, completion_tokens: Any, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build a usage anchor from provider-reported usage, or None."""
    try:
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
    except (TypeError, ValueError):
        return None
    if pt <= 0 or not isinstance(messages, list):
        return None  # no usable usage (some endpoints omit it) — caller keeps its anchor
    last = messages[-1] if messages else None
    return {
        "prompt_tokens": pt,
        "completion_tokens": max(0, ct),
        "base_count": len(messages),
        "base_last_id": id(last) if last is not None else None,
        "base_last_role": last.get("role") if isinstance(last, dict) else None,
    }


def anchored_context_tokens(messages: List[Dict[str, Any]], anchor: Optional[Dict[str, Any]], *, charge_stale_thinking: bool = True) -> Optional[int]:
    """Anchored prompt+completion tokens plus a rough estimate of ONLY the messages appended since;
    None when the anchor is missing or stale. The anchored response's own reply is skipped (already
    in completion_tokens). ``charge_stale_thinking`` is forwarded to the delta estimate."""
    if not isinstance(anchor, dict) or not isinstance(messages, list):
        return None
    base_count = anchor.get("base_count") or 0
    if base_count <= 0 or len(messages) < base_count:
        return None
    base_msg = messages[base_count - 1]
    base_role = base_msg.get("role") if isinstance(base_msg, dict) else None
    if id(base_msg) != anchor.get("base_last_id") or base_role != anchor.get("base_last_role"):
        return None
    total = int(anchor["prompt_tokens"]) + int(anchor.get("completion_tokens") or 0)
    delta = messages[base_count:]
    if delta and isinstance(delta[0], dict) and delta[0].get("role") == "assistant":
        delta = delta[1:]
    if delta:
        total += estimate_messages_tokens_rough(delta, charge_stale_thinking=charge_stale_thinking)
    return total


# Keyed by ``id(tools)``; bounded, oldest-first eviction. Repeated ``str(tools)`` on
# large schemas stalls GUI event loops under GIL pressure.
_TOOLS_TOKENS_CACHE: dict[int, Tuple[int, str, str, int]] = {}
_TOOLS_TOKENS_CACHE_MAX = 256


def _tool_name_for_cache(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    name = fn.get("name") if isinstance(fn, dict) else None
    name = name if isinstance(name, str) else tool.get("name")
    return name if isinstance(name, str) else ""


def _estimate_tools_tokens_rough(tools: List[Dict[str, Any]]) -> int:
    if not tools:
        return 0
    key = id(tools)
    signature = (len(tools), _tool_name_for_cache(tools[0]), _tool_name_for_cache(tools[-1]))
    cached = _TOOLS_TOKENS_CACHE.get(key)
    if cached is not None and cached[:3] == signature:
        return cached[3]
    # Sum the major schema fields (descriptions + parameters dominate).
    total_chars = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        src = fn if isinstance(fn, dict) else tool
        params = src.get("parameters") or {}
        total_chars += sum(len(v) for v in (src.get("name") or "", src.get("description") or "") if isinstance(v, str))
        try:  # JSON is closer to wire size than repr()
            total_chars += len(json.dumps(params, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            total_chars += len(str(params))
    tokens = (total_chars + 3) // 4
    if len(_TOOLS_TOKENS_CACHE) >= _TOOLS_TOKENS_CACHE_MAX:
        _TOOLS_TOKENS_CACHE.pop(next(iter(_TOOLS_TOKENS_CACHE)), None)
    _TOOLS_TOKENS_CACHE[key] = (*signature, tokens)
    return tokens

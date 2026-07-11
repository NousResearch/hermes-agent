"""Model metadata, context lengths, and token estimation utilities.

Pure utility functions with no AIAgent dependency. Used by ContextCompressor
and run_agent.py for pre-flight context checks.
"""

import ipaddress
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import yaml

from utils import atomic_json_write, base_url_host_matches, base_url_hostname

from hermes_constants import OPENROUTER_MODELS_URL

logger = logging.getLogger(__name__)


def _resolve_requests_verify() -> bool | str:
    """Resolve SSL verify setting for `requests` calls from env vars.

    The `requests` library only honours REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE
    by default. Hermes also honours HERMES_CA_BUNDLE (its own convention)
    and SSL_CERT_FILE (used by the stdlib `ssl` module and by httpx), so
    that a single env var can cover both `requests` and `httpx` callsites
    inside the same process.

    Returns either a filesystem path to a CA bundle, or True to defer to
    the requests default (certifi).
    """
    for env_var in ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        val = os.getenv(env_var)
        if val and os.path.isfile(val):
            return val
    return True

# Provider names that can appear as a "provider:" prefix before a model ID.
# Only these are stripped — Ollama-style "model:tag" colons (e.g. "qwen3.5:27b")
# are preserved so the full model name reaches cache lookups and server queries.
_PROVIDER_PREFIXES: frozenset[str] = frozenset({
    "openrouter", "nous", "openai-codex", "copilot", "copilot-acp",
    "gemini", "ollama-cloud", "zai", "kimi-coding", "kimi-coding-cn", "stepfun", "minimax", "minimax-oauth", "minimax-cn", "anthropic", "deepseek",
    "opencode-zen", "opencode-go", "kilocode", "alibaba", "novita",
    "qwen-oauth",
    "xiaomi",
    "arcee",
    "gmi",
    "tencent-tokenhub",
    "custom", "local",
    # Common aliases
    "google", "google-gemini", "google-ai-studio",
    "glm", "z-ai", "z.ai", "zhipu", "github", "github-copilot",
    "github-models", "kimi", "moonshot", "kimi-cn", "moonshot-cn", "claude", "deep-seek",
    "ollama",
    "stepfun", "opencode", "zen", "go", "kilo", "dashscope", "aliyun", "qwen",
    "mimo", "xiaomi-mimo",
    "tencent", "tokenhub", "tencent-cloud", "tencentmaas",
    "arcee-ai", "arceeai",
    "gmi-cloud", "gmicloud",
    "xai", "x-ai", "x.ai", "grok",
    "nvidia", "nim", "nvidia-nim", "nemotron",
    "qwen-portal", "novita-ai", "novitaai",
})


_OLLAMA_TAG_PATTERN = re.compile(
    r"^(\d+\.?\d*b|latest|stable|q\d|fp?\d|instruct|chat|coder|vision|text)",
    re.IGNORECASE,
)


# Tailscale's CGNAT range (RFC 6598). `ipaddress.is_private` excludes this
# block, so without an explicit check Ollama reached over Tailscale (e.g.
# `http://100.77.243.5:11434`) wouldn't be treated as local and its stream
# read / stale timeouts wouldn't get auto-bumped. Built once at import time.
_TAILSCALE_CGNAT = ipaddress.IPv4Network("100.64.0.0/10")


def _strip_provider_prefix(model: str) -> str:
    """Strip a recognised provider prefix from a model string.

    ``"local:my-model"`` → ``"my-model"``
    ``"qwen3.5:27b"``   → ``"qwen3.5:27b"``  (unchanged — not a provider prefix)
    ``"qwen:0.5b"``     → ``"qwen:0.5b"``    (unchanged — Ollama model:tag)
    ``"deepseek:latest"``→ ``"deepseek:latest"``(unchanged — Ollama model:tag)
    """
    if ":" not in model or model.startswith("http"):
        return model
    prefix, suffix = model.split(":", 1)
    prefix_lower = prefix.strip().lower()
    if prefix_lower in _PROVIDER_PREFIXES:
        # Don't strip if suffix looks like an Ollama tag (e.g. "7b", "latest", "q4_0")
        if _OLLAMA_TAG_PATTERN.match(suffix.strip()):
            return model
        return suffix
    return model

_model_metadata_cache: Dict[str, Dict[str, Any]] = {}
_model_metadata_cache_time: float = 0
_novita_metadata_cache: Dict[str, Dict[str, Any]] = {}
_novita_metadata_cache_time: float = 0
_MODEL_CACHE_TTL = 3600
_endpoint_model_metadata_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
_endpoint_model_metadata_cache_time: Dict[str, float] = {}
_ENDPOINT_MODEL_CACHE_TTL = 300
# Bounded-lifetime cache: after the first successful probe we remember the
# server type so subsequent refreshes skip the full waterfall (no more 404
# spam every 5 minutes on non-matching endpoints like /api/v1/models on vllm).
# Entries expire after _ENDPOINT_PROBE_TTL_SECONDS so a server swap on the
# same port (stop Ollama, start LM Studio) is eventually re-detected instead
# of being pinned to the stale type for the whole process lifetime.
# Values are (server_type, monotonic_timestamp).
_ENDPOINT_PROBE_TTL_SECONDS = 3600.0
_endpoint_probe_path_cache: Dict[str, tuple] = {}


def _get_model_metadata_cache_path() -> Path:
    """Return path to the OpenRouter model metadata disk cache."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "openrouter_model_metadata.json"


def _model_metadata_disk_cache_age_seconds() -> Optional[float]:
    """Return disk-cache age in seconds, or None if freshness is unknown."""
    try:
        cache_path = _get_model_metadata_cache_path()
        if not cache_path.exists():
            return None
        age = time.time() - cache_path.stat().st_mtime
        if age < 0:
            return None
        return age
    except Exception:
        return None


def _load_model_metadata_disk_cache() -> Dict[str, Dict[str, Any]]:
    """Load processed OpenRouter metadata cache from disk."""
    try:
        cache_path = _get_model_metadata_cache_path()
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {
            str(key): value
            for key, value in data.items()
            if isinstance(value, dict)
        }
    except Exception as e:
        logger.debug("Failed to load OpenRouter model metadata disk cache: %s", e)
        return {}


def _save_model_metadata_disk_cache(data: Dict[str, Dict[str, Any]]) -> None:
    """Save processed OpenRouter metadata cache to disk atomically."""
    try:
        atomic_json_write(
            _get_model_metadata_cache_path(),
            data,
            indent=0,
            separators=(",", ":"),
        )
    except Exception as e:
        logger.debug("Failed to save OpenRouter model metadata disk cache: %s", e)

# Descending tiers for context length probing when the model is unknown.
# We start at 256K (covers GPT-5.x, many current large-context models) and
# step down on context-length errors until one works.  Tier[0] is also the
# default fallback when no detection method succeeds.
CONTEXT_PROBE_TIERS = [
    256_000,
    128_000,
    64_000,
    32_000,
    16_000,
    8_000,
]

# Default context length when no detection method succeeds.
DEFAULT_FALLBACK_CONTEXT = CONTEXT_PROBE_TIERS[0]

# Minimum context length required to run Hermes Agent.  Models with fewer
# tokens cannot maintain enough working memory for tool-calling workflows.
# Sessions, model switches, and cron jobs should reject models below this.
MINIMUM_CONTEXT_LENGTH = 64_000

# Short-lived in-process cache for local-server context probes. Bounds the
# probe rate when the new local-endpoint live-probe paths (reconcile-on-hit +
# pre-defaults step 7) resolve the same model several times during one startup
# (banner, /model switch, compressor update_model). Keyed by (model, base_url);
# values are (result, monotonic_timestamp). Not persisted to disk — cross-
# restart freshness is handled by the reconcile logic re-probing after expiry.
_LOCAL_CTX_PROBE_TTL_SECONDS = 30.0
_LOCAL_CTX_PROBE_CACHE: Dict[tuple, tuple] = {}

# Thin fallback defaults — only broad model family patterns.
# These fire only when provider is unknown AND models.dev/OpenRouter/Anthropic
# all miss. Replaced the previous 80+ entry dict.
# For provider-specific context lengths, models.dev is the primary source.
DEFAULT_CONTEXT_LENGTHS = {
    # Anthropic Claude 4.6 (1M context) — bare IDs only to avoid
    # fuzzy-match collisions (e.g. "anthropic/claude-sonnet-4" is a
    # substring of "anthropic/claude-sonnet-4.6").
    # OpenRouter-prefixed models resolve via OpenRouter live API or models.dev.
    "claude-fable-5": 1000000,
    "claude-fable": 1000000,
    "claude-opus-4-8": 1000000,
    "claude-opus-4.8": 1000000,
    "claude-opus-4-7": 1000000,
    "claude-opus-4.7": 1000000,
    "claude-opus-4-6": 1000000,
    "claude-sonnet-4-6": 1000000,
    "claude-opus-4.6": 1000000,
    "claude-sonnet-4.6": 1000000,
    # Catch-all for older Claude models (must sort after specific entries)
    "claude": 200000,
    # OpenAI — GPT-5 family (most have 400k; specific overrides first)
    # Source: https://developers.openai.com/api/docs/models
    # GPT-5.5 (launched Apr 23 2026) is 1.05M on the direct OpenAI API and
    # ChatGPT Codex OAuth caps it at 272K; both paths resolve via their own
    # provider-aware branches (_resolve_codex_oauth_context_length + models.dev).
    # This hardcoded value is only reached when every probe misses.
    # GPT-5.6 series (Sol/Terra/Luna, GA 2026-07-09) — 1.05M on the direct
    # OpenAI API (same as gpt-5.5). Codex OAuth caps these at 272K.
    # (Lookups length-sort keys at match time, so dict order is cosmetic.)
    "gpt-5.6-luna": 1050000,
    "gpt-5.6-terra": 1050000,
    "gpt-5.6-sol": 1050000,
    "gpt-5.5": 1050000,
    "gpt-5.4-nano": 400000,           # 400k (not 1.05M like full 5.4)
    "gpt-5.4-mini": 400000,           # 400k (not 1.05M like full 5.4)
    "gpt-5.4": 1050000,               # GPT-5.4, GPT-5.4 Pro (1.05M context)
    # gpt-5.3-codex-spark is Codex-OAuth-only (ChatGPT Pro entitlement) and
    # uses a smaller 128k window than other gpt-5.x slugs. Listed here as
    # a defensive override so the longest-substring fallback doesn't match
    # the generic "gpt-5" entry below (400k) and report the wrong limit if
    # Spark's context ever needs to be resolved through this path. Real
    # usage flows through _CODEX_OAUTH_CONTEXT_FALLBACK at line ~1113.
    "gpt-5.3-codex-spark": 128000,
    "gpt-5.1-chat": 128000,           # Chat variant has 128k context
    "gpt-5": 400000,                  # GPT-5.x base, mini, codex variants (400k)
    "gpt-4.1": 1047576,
    "gpt-4": 128000,
    # Google
    "gemini": 1048576,
    # Gemma (open models served via AI Studio)
    "gemma-4": 256000,  # Gemma 4 family
    "gemma4": 256000,  # Ollama-style naming (e.g. gemma4:31b-cloud)
    "gemma-4-31b": 256000,
    "gemma-3": 131072,
    "gemma": 8192,  # fallback for older gemma models
    # DeepSeek — V4 family ships with a 1M context window. The legacy
    # aliases ``deepseek-chat`` / ``deepseek-reasoner`` are server-side
    # mapped to the non-thinking / thinking modes of ``deepseek-v4-flash``
    # and inherit the same 1M window. The ``deepseek`` substring entry
    # below remains as a 128K fallback for older / unknown DeepSeek model
    # ids (e.g. via custom endpoints).
    # https://api-docs.deepseek.com/zh-cn/quick_start/pricing
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek": 128000,
    # Meta
    "llama": 131072,
    # Qwen — specific model families before the catch-all.
    # Official docs: https://help.aliyun.com/zh/model-studio/developer-reference/
    "qwen3.6-plus": 1048576,      # 1M context (DashScope/Alibaba & OpenRouter)
    "qwen3-coder-plus": 1000000,  # 1M context
    "qwen3-coder": 262144,        # 256K context
    "qwen": 131072,
    # MiniMax — M3 is 1M context (max output 512K); M2.x series is 204,800.
    # Keys use substring matching (longest-first), so "minimax-m3" wins over
    # the generic "minimax" catch-all for the M3 slug on every surface
    # (native MiniMax-M3, OpenRouter/Nous minimax/minimax-m3).
    # https://platform.minimax.io/docs/api-reference/text-chat-openai
    "minimax-m3": 1000000,
    "minimax": 204800,
    # GLM — GLM-5.2 ships with a 1M context window (verified empirically:
    # needle-in-a-haystack retrieval at 789K prompt tokens succeeded with
    # zero errors on api.z.ai/api/coding/paas/v4).  Older GLM models
    # (5, 5.1, 5-turbo) are ~202K.  Longest-key-first substring matching
    # ensures "glm-5.2" resolves to 1M while older variants still hit the
    # generic 202K fallback.
    "glm-5.2": 1_048_576,
    "glm": 202752,
    # xAI Grok — xAI /v1/models does not return context_length metadata,
    # so these hardcoded fallbacks prevent Hermes from probing-down to
    # the default 128k when the user points at https://api.x.ai/v1
    # via a custom provider. Values sourced from models.dev (2026-04).
    # Keys use substring matching (longest-first), so e.g. "grok-4.20"
    # matches "grok-4.20-0309-reasoning" / "-non-reasoning" / "-multi-agent-0309".
    # OAuth-only slug; absent from GET /v1/models. xAI publishes a 200k
    # usable context window for Composer 2.5 on Grok Build (SuperGrok /
    # Premium+); /v1/responses additionally enforces a ~262144 input+output
    # budget, but the usable context (what we track here) is 200k.
    "grok-composer": 200000,    # grok-composer-2.5-fast (Grok Build CLI)
    "grok-build-latest": 500000,  # alias of grok-4.5 (early access)
    "grok-build": 256000,       # grok-build-0.1
    "grok-code-fast": 256000,   # grok-code-fast-1
    "grok-2-vision": 8192,      # grok-2-vision, -1212, -latest
    "grok-4-fast": 2000000,     # grok-4-fast-(non-)reasoning, also matches -reasoning
    "grok-4.20": 2000000,       # grok-4.20-0309-(non-)reasoning, -multi-agent-0309
    "grok-4.5": 500000,         # grok-4.5, grok-4.5-latest — 500K context per docs.x.ai
    "grok-4.3": 1000000,        # grok-4.3, grok-4.3-latest — 1M context per docs.x.ai
    "grok-4": 256000,           # grok-4, grok-4-0709
    "grok-3": 131072,           # grok-3, grok-3-mini, grok-3-fast, grok-3-mini-fast
    "grok-2": 131072,           # grok-2, grok-2-1212, grok-2-latest
    "grok": 131072,             # catch-all (grok-beta, unknown grok-*)
    # Kimi
    "kimi": 262144,
    # Tencent — Hy3 Preview (Hunyuan) with 256K context window.
    # OpenRouter live metadata reports 262144 (256 × 1024); align the
    # static fallback so cache and offline both agree (issue #22268).
    "hy3-preview": 262144,
    # Tencent — Hy3 (GA successor to Hy3 Preview), same 256K window.
    "hy3": 262144,
    # Nemotron — NVIDIA's open-weights series (128K context across all sizes)
    "nemotron": 131072,
    # Arcee
    "trinity": 262144,
    # OpenRouter
    "elephant": 262144,
    # Hugging Face Inference Providers — model IDs use org/name format
    "Qwen/Qwen3.5-397B-A17B": 131072,
    "Qwen/Qwen3.5-35B-A3B": 131072,
    "deepseek-ai/DeepSeek-V3.2": 65536,
    "moonshotai/Kimi-K2.5": 262144,
    "moonshotai/Kimi-K2.6": 262144,
    "moonshotai/Kimi-K2-Thinking": 262144,
    "MiniMaxAI/MiniMax-M2.5": 204800,
    "XiaomiMiMo/MiMo-V2-Flash": 262144,
    "mimo-v2-pro": 1048576,
    "mimo-v2.5-pro": 1048576,
    "mimo-v2.5": 1048576,
    "mimo-v2-omni": 262144,
    "mimo-v2-flash": 262144,
    "zai-org/GLM-5": 202752,
}

# xAI Grok models that ACCEPT the `reasoning.effort` parameter on
# api.x.ai. Verified live against /v1/responses 2026-05-10:
#
#   ACCEPTS effort:  grok-3-mini, grok-3-mini-fast, grok-4.20-multi-agent-0309,
#                    grok-4.3
#   REJECTS effort:  grok-3, grok-4, grok-4-0709, grok-4-fast-(non-)reasoning,
#                    grok-4-1-fast-(non-)reasoning, grok-4.20-0309-(non-)reasoning,
#                    grok-code-fast-1
#
# REJECTS-side models still reason natively — they just don't expose an
# effort dial — so callers should send no `reasoning` key at all rather
# than a default `medium` (which 400s with "Model X does not support
# parameter reasoningEffort").
_GROK_EFFORT_CAPABLE_PREFIXES = (
    "grok-3-mini",
    "grok-4.20-multi-agent",
    "grok-4.3",
    # grok-4.5: verified live against /v1/responses 2026-07-08 — accepts
    # effort low/medium/high (default: high when omitted) but REJECTS
    # "none" ("This model does not support `reasoning_effort` value `none`"),
    # unlike grok-4.3. models.dev agrees: effort values [low, medium, high].
    "grok-4.5",
)


def grok_supports_reasoning_effort(model: str) -> bool:
    """Return True when an xAI Grok model accepts ``reasoning.effort``.

    Allowlist by substring (matches both bare ``grok-3-mini`` and
    aggregator-prefixed ``x-ai/grok-3-mini``). Conservative by design:
    if a future Grok model isn't listed, we send no effort dial rather
    than 400.
    """
    name = (model or "").strip().lower()
    if not name:
        return False
    # Strip common aggregator prefixes (x-ai/, openrouter/x-ai/, xai/, ...)
    for sep in ("/",):
        if sep in name:
            name = name.rsplit(sep, 1)[-1]
    return any(name.startswith(prefix) for prefix in _GROK_EFFORT_CAPABLE_PREFIXES)


_CONTEXT_LENGTH_KEYS = (
    "context_length",
    "context_window",
    "context_size",
    "max_context_length",
    "max_position_embeddings",
    "max_model_len",
    "max_input_tokens",
    "max_sequence_length",
    "max_seq_len",
    "n_ctx_train",
    "n_ctx",
    "ctx_size",
)

_MAX_COMPLETION_KEYS = (
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
)

# Local server hostnames / address patterns
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
# Docker / Podman / Lima DNS names that resolve to the host machine
_CONTAINER_LOCAL_SUFFIXES = (
    ".docker.internal",
    ".containers.internal",
    ".lima.internal",
)


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _auth_headers(api_key: str = "") -> Dict[str, str]:
    token = str(api_key or "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _is_openrouter_base_url(base_url: str) -> bool:
    return base_url_host_matches(base_url, "openrouter.ai")


def _is_custom_endpoint(base_url: str) -> bool:
    normalized = _normalize_base_url(base_url)
    return bool(normalized) and not _is_openrouter_base_url(normalized)


_URL_TO_PROVIDER: Dict[str, str] = {
    "api.openai.com": "openai",
    "chatgpt.com": "openai",
    "api.anthropic.com": "anthropic",
    "api.z.ai": "zai",
    "open.bigmodel.cn": "zai",
    "api.moonshot.ai": "kimi-coding",
    "api.moonshot.cn": "kimi-coding-cn",
    "api.kimi.com": "kimi-coding",
    "api.stepfun.ai": "stepfun",
    "api.stepfun.com": "stepfun",
    "api.arcee.ai": "arcee",
    "api.minimax": "minimax",
    "dashscope.aliyuncs.com": "alibaba",
    "dashscope-intl.aliyuncs.com": "alibaba",
    "portal.qwen.ai": "qwen-oauth",
    "openrouter.ai": "openrouter",
    "generativelanguage.googleapis.com": "gemini",
    "inference-api.nousresearch.com": "nous",
    "api.deepseek.com": "deepseek",
    "api.githubcopilot.com": "copilot",
    # Enterprise Copilot endpoints look like api.enterprise.githubcopilot.com,
    # api.business.githubcopilot.com, etc.  Match the suffix so context-window
    # resolution works for enterprise accounts too.
    ".githubcopilot.com": "copilot",
    "models.github.ai": "copilot",
    # GitHub Models free tier (Azure-hosted prototyping endpoint) — same
    # canonical provider as the Copilot API.  Hard per-request token cap
    # (often 8K) makes it unusable for Hermes' system prompt, but mapping
    # it here lets us recognize the endpoint and emit a targeted hint
    # instead of falling through the unknown-custom-endpoint path.
    "models.inference.ai.azure.com": "copilot",
    "api.fireworks.ai": "fireworks",
    "opencode.ai": "opencode-go",
    "api.x.ai": "xai",
    "integrate.api.nvidia.com": "nvidia",
    "api.xiaomimimo.com": "xiaomi",
    "xiaomimimo.com": "xiaomi",
    "api.gmi-serving.com": "gmi",
    "api.novita.ai": "novita",
    "tokenhub.tencentmaas.com": "tencent-tokenhub",
    "ollama.com": "ollama-cloud",
}

# Auto-extend with hostnames derived from provider profiles.
# Any provider with a base_url not already in the map gets added automatically.
try:
    from providers import list_providers as _list_providers
    for _pp in _list_providers():
        _host = _pp.get_hostname()
        if _host and _host not in _URL_TO_PROVIDER:
            _URL_TO_PROVIDER[_host] = _pp.name
except Exception:
    pass


def _infer_provider_from_url(base_url: str) -> Optional[str]:
    """Infer the models.dev provider name from a base URL.

    This allows context length resolution via models.dev for custom endpoints
    like DashScope (Alibaba), Z.AI, Kimi, etc. without requiring the user to
    explicitly set the provider name in config.
    """
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return None
    parsed = urlparse(normalized if "://" in normalized else f"https://{normalized}")
    host = parsed.netloc.lower() or parsed.path.lower()
    for url_part, provider in _URL_TO_PROVIDER.items():
        if url_part in host:
            return provider
    return None


def _lmstudio_server_root(base_url: str) -> str:
    """Return the LM Studio server root for native ``/api/v1`` endpoints."""
    root = _normalize_base_url(base_url).rstrip("/")
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    return root


def _is_known_provider_base_url(base_url: str) -> bool:
    return _infer_provider_from_url(base_url) is not None


def _skip_persistent_context_cache(base_url: str, provider: str) -> bool:
    """Return True when the on-disk context cache must not short-circuit probing.

    LM Studio excludes caching because loaded context is transient — the user
    can reload the model with a different context_length at any time.
   """
    return provider == "lmstudio"


def _maybe_cache_local_context_length(
    model: str,
    base_url: str,
    length: int,
) -> None:
    """Persist a locally probed context length only when it meets Hermes minimum.

    Sub-minimum live windows (e.g. vLLM ``--max-model-len 32768``) are still
    returned to callers so ``agent_init`` can fail with the existing
    minimum-context guidance — they must not be normalized into the on-disk cache
    as if they were valid operating limits.
    """
    if length >= MINIMUM_CONTEXT_LENGTH:
        save_context_length(model, base_url, length)


def _reconcile_local_cached_context_length(
    model: str,
    base_url: str,
    cached: int,
    api_key: str = "",
) -> int:
    """Return *cached* unless a live local probe reports a different limit.

    vLLM/Ollama operators can restart with a new ``--max-model-len`` / ``num_ctx``
    without changing the model id.  When the server is reachable, prefer its
    reported window over a stale disk entry; when the probe fails (offline tests,
    network blip), keep the cached value.

    Live probes below :data:`MINIMUM_CONTEXT_LENGTH` invalidate stale cache
    entries but are not persisted — startup should reject them, not bless a
    sub-64K window as config.
    """
    live_ctx = _query_local_context_length(model, base_url, api_key=api_key)
    if live_ctx and live_ctx > 0 and live_ctx != cached:
        if live_ctx < MINIMUM_CONTEXT_LENGTH:
            logger.info(
                "Live local probe for %s@%s reports %s (< minimum %s); "
                "invalidating stale cache — agent init should reject",
                model, base_url, f"{live_ctx:,}", f"{MINIMUM_CONTEXT_LENGTH:,}",
            )
            _invalidate_cached_context_length(model, base_url)
            return live_ctx
        logger.info(
            "Reconciling stale local cache entry %s@%s: %s -> %s (live probe)",
            model, base_url, f"{cached:,}", f"{live_ctx:,}",
        )
        _invalidate_cached_context_length(model, base_url)
        _maybe_cache_local_context_length(model, base_url, live_ctx)
        return live_ctx
    return cached


def is_local_endpoint(base_url: str) -> bool:
    """Return True if base_url points to a local machine.

    Recognises loopback (``localhost``, ``127.0.0.0/8``, ``::1``),
    container-internal DNS names (``host.docker.internal`` et al.),
    RFC-1918 private ranges (``10/8``, ``172.16/12``, ``192.168/16``),
    link-local, and Tailscale CGNAT (``100.64.0.0/10``). Tailscale CGNAT
    is included so remote-but-trusted Ollama boxes reached over a
    Tailscale mesh get the same timeout auto-bumps as localhost Ollama.
    """
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return False
    url = normalized if "://" in normalized else f"http://{normalized}"
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return False
    if host in _LOCAL_HOSTS:
        return True
    # Docker / Podman / Lima internal DNS names (e.g. host.docker.internal)
    if any(host.endswith(suffix) for suffix in _CONTAINER_LOCAL_SUFFIXES):
        return True
    # Unqualified hostnames (no dots) are local by definition — Docker
    # Compose service names, /etc/hosts entries, or mDNS names.
    if host and "." not in host:
        return True
    # RFC-1918 private ranges, link-local, and Tailscale CGNAT
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return True
        if isinstance(addr, ipaddress.IPv4Address) and addr in _TAILSCALE_CGNAT:
            return True
    except ValueError:
        pass
    # Bare IP that looks like a private range (e.g. 172.26.x.x for WSL)
    # or Tailscale CGNAT (100.64.x.x–100.127.x.x).
    parts = host.split(".")
    if len(parts) == 4:
        try:
            first, second = int(parts[0]), int(parts[1])
            if first == 10:
                return True
            if first == 172 and 16 <= second <= 31:
                return True
            if first == 192 and second == 168:
                return True
            if first == 100 and 64 <= second <= 127:
                return True
        except ValueError:
            pass
    return False


def _localhost_to_ipv4(url: str) -> str:
    """Rewrite a ``localhost`` HOST to ``127.0.0.1`` in a probe URL.

    On Windows dual-stack machines, httpx resolves ``localhost`` to ``::1``
    first and pays a ~2s IPv6 connect timeout before falling back to IPv4
    when the local server only listens on IPv4 (LM Studio, Ollama defaults).
    Probing the IPv4 loopback directly skips that penalty.

    Only the URL's own host component is rewritten (anchored at the scheme),
    so a non-localhost URL whose path or query merely embeds the substring
    ``http://localhost...`` (e.g. ``?upstream=http://localhost:11434``)
    passes through untouched.
    """
    if not url:
        return url
    return re.sub(
        r"^(https?://)localhost(?=[:/]|$)",
        r"\g<1>127.0.0.1",
        url,
        count=1,
    )


def detect_local_server_type(base_url: str, api_key: str = "") -> Optional[str]:
    """Detect which local server is running at base_url by probing known endpoints.

    Returns one of: "ollama", "lm-studio", "vllm", "llamacpp", or None.

    The result is cached for the lifetime of the process so that repeated
    calls (e.g. every 5-minute metadata refresh) never re-run the waterfall
    and never spray 404s at endpoints the server does not expose.
    """
    import httpx

    normalized = _normalize_base_url(base_url)

    # Resolve localhost to IPv4 to avoid 2s IPv6 timeout on Windows dual-stack.
    # Applied to ``normalized`` before deriving server/LM Studio URLs AND
    # before the cache lookup, so localhost and 127.0.0.1 share a cache entry.
    normalized = _localhost_to_ipv4(normalized)

    server_url = normalized
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]
    lmstudio_url = _lmstudio_server_root(normalized)

    cached = _endpoint_probe_path_cache.get(server_url)
    if cached is not None and (time.monotonic() - cached[1]) < _ENDPOINT_PROBE_TTL_SECONDS:
        return cached[0]

    headers = _auth_headers(api_key)

    result: Optional[str] = None
    try:
        with httpx.Client(timeout=2.0, headers=headers) as client:
            # LM Studio exposes /api/v1/models — check first (most specific)
            try:
                r = client.get(f"{lmstudio_url}/api/v1/models")
                if r.status_code == 200:
                    result = "lm-studio"
            except Exception:
                pass
            if result is None:
                # Ollama exposes /api/tags and responds with {"models": [...]}
                # LM Studio returns {"error": "Unexpected endpoint"} with status 200
                # on this path, so we must verify the response contains "models".
                try:
                    r = client.get(f"{server_url}/api/tags")
                    if r.status_code == 200:
                        try:
                            data = r.json()
                            if "models" in data:
                                result = "ollama"
                        except Exception:
                            pass
                except Exception:
                    pass
            if result is None:
                # llama.cpp exposes /v1/props (older builds used /props without the /v1 prefix)
                try:
                    r = client.get(f"{server_url}/v1/props")
                    if r.status_code != 200:
                        r = client.get(f"{server_url}/props")  # fallback for older builds
                    if r.status_code == 200 and "default_generation_settings" in r.text:
                        result = "llamacpp"
                except Exception:
                    pass
            if result is None:
                # vLLM: /version
                try:
                    r = client.get(f"{server_url}/version")
                    if r.status_code == 200:
                        data = r.json()
                        if "version" in data:
                            result = "vllm"
                except Exception:
                    pass
    except Exception:
        pass

    if result is not None:
        _endpoint_probe_path_cache[server_url] = (result, time.monotonic())
    return result


def _iter_nested_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_nested_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


def _coerce_reasonable_int(value: Any, minimum: int = 1024, maximum: int = 10_000_000) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        result = int(value)
    except (TypeError, ValueError):
        return None
    if minimum <= result <= maximum:
        return result
    return None


def _extract_first_int(payload: Dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    keyset = {key.lower() for key in keys}
    for mapping in _iter_nested_dicts(payload):
        for key, value in mapping.items():
            if str(key).lower() not in keyset:
                continue
            coerced = _coerce_reasonable_int(value)
            if coerced is not None:
                return coerced
    return None


def _extract_context_length(payload: Dict[str, Any]) -> Optional[int]:
    return _extract_first_int(payload, _CONTEXT_LENGTH_KEYS)


def _extract_max_completion_tokens(payload: Dict[str, Any]) -> Optional[int]:
    return _extract_first_int(payload, _MAX_COMPLETION_KEYS)


def _extract_pricing(payload: Dict[str, Any]) -> Dict[str, Any]:
    novita_input = payload.get("input_token_price_per_m")
    novita_output = payload.get("output_token_price_per_m")
    if novita_input is not None or novita_output is not None:
        pricing: Dict[str, Any] = {}
        if novita_input is not None:
            pricing["prompt"] = str(float(novita_input) / 10_000 / 1_000_000)
        if novita_output is not None:
            pricing["completion"] = str(float(novita_output) / 10_000 / 1_000_000)
        return pricing

    alias_map = {
        "prompt": ("prompt", "input", "input_cost_per_token", "prompt_token_cost"),
        "completion": ("completion", "output", "output_cost_per_token", "completion_token_cost"),
        "request": ("request", "request_cost"),
        "cache_read": ("cache_read", "cached_prompt", "input_cache_read", "cache_read_cost_per_token"),
        "cache_write": ("cache_write", "cache_creation", "input_cache_write", "cache_write_cost_per_token"),
    }
    for mapping in _iter_nested_dicts(payload):
        normalized = {str(key).lower(): value for key, value in mapping.items()}
        if not any(any(alias in normalized for alias in aliases) for aliases in alias_map.values()):
            continue
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
        bare_model = model_id.split("/", 1)[1]
        cache.setdefault(bare_model, entry)


def fetch_model_metadata(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch model metadata from OpenRouter (cached for 1 hour)."""
    global _model_metadata_cache, _model_metadata_cache_time

    if not force_refresh and _model_metadata_cache and (time.time() - _model_metadata_cache_time) < _MODEL_CACHE_TTL:
        return _model_metadata_cache

    if not force_refresh:
        disk_age = _model_metadata_disk_cache_age_seconds()
        if disk_age is not None and disk_age < _MODEL_CACHE_TTL:
            disk_cache = _load_model_metadata_disk_cache()
            if disk_cache:
                _model_metadata_cache = disk_cache
                _model_metadata_cache_time = time.time() - disk_age
                return _model_metadata_cache

    try:
        # Tuple (connect, read) — flat timeout=10 means urllib3 can block 10s per
        # retry stage through proxies that 403 CONNECT, ballooning to minutes
        # (#46620). 5s connect / 10s read fails fast on unreachable hosts.
        response = requests.get(OPENROUTER_MODELS_URL, timeout=(5, 10), verify=_resolve_requests_verify())
        response.raise_for_status()
        data = response.json()

        cache = {}
        for model in data.get("data", []):
            model_id = model.get("id", "")
            entry = {
                "context_length": model.get("context_length", 128000),
                "max_completion_tokens": model.get("top_provider", {}).get("max_completion_tokens", 4096),
                "name": model.get("name", model_id),
                "pricing": model.get("pricing", {}),
            }
            _add_model_aliases(cache, model_id, entry)
            canonical = model.get("canonical_slug", "")
            if canonical and canonical != model_id:
                _add_model_aliases(cache, canonical, entry)

        _model_metadata_cache = cache
        _model_metadata_cache_time = time.time()
        _save_model_metadata_disk_cache(cache)
        logger.debug("Fetched metadata for %s models from OpenRouter", len(cache))
        return cache

    except Exception as e:
        logger.warning(f"Failed to fetch model metadata from OpenRouter: {e}")
        if _model_metadata_cache:
            return _model_metadata_cache
        disk_cache = _load_model_metadata_disk_cache()
        if disk_cache:
            _model_metadata_cache = disk_cache
            disk_age = _model_metadata_disk_cache_age_seconds()
            if disk_age is not None:
                _model_metadata_cache_time = time.time() - min(disk_age, _MODEL_CACHE_TTL)
            else:
                _model_metadata_cache_time = time.time() - _MODEL_CACHE_TTL + 1
            return _model_metadata_cache
        return {}


def fetch_endpoint_model_metadata(
    base_url: str,
    api_key: str = "",
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch model metadata from an OpenAI-compatible ``/models`` endpoint.

    This is used for explicit custom endpoints where hardcoded global model-name
    defaults are unreliable. Results are cached in memory per base URL.
    """
    normalized = _normalize_base_url(base_url)
    if not normalized or _is_openrouter_base_url(normalized):
        return {}

    if not force_refresh:
        cached = _endpoint_model_metadata_cache.get(normalized)
        cached_at = _endpoint_model_metadata_cache_time.get(normalized, 0)
        if cached is not None and (time.time() - cached_at) < _ENDPOINT_MODEL_CACHE_TTL:
            return cached

    candidates = [normalized]
    if normalized.endswith("/v1"):
        alternate = normalized[:-3].rstrip("/")
    else:
        alternate = normalized + "/v1"
    if alternate and alternate not in candidates:
        candidates.append(alternate)

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_error: Optional[Exception] = None

    if is_local_endpoint(normalized):
        try:
            if detect_local_server_type(normalized, api_key=api_key) == "lm-studio":
                server_url = _lmstudio_server_root(normalized)
                response = requests.get(
                    server_url.rstrip("/") + "/api/v1/models",
                    headers=headers,
                    timeout=(5, 10),
                    verify=_resolve_requests_verify(),
                )
                response.raise_for_status()
                payload = response.json()
                cache: Dict[str, Dict[str, Any]] = {}
                for model in payload.get("models", []):
                    if not isinstance(model, dict):
                        continue
                    model_id = model.get("key") or model.get("id")
                    if not model_id:
                        continue
                    entry: Dict[str, Any] = {"name": model.get("name", model_id)}

                    context_length = None
                    for inst in model.get("loaded_instances", []) or []:
                        if not isinstance(inst, dict):
                            continue
                        cfg = inst.get("config", {})
                        ctx = cfg.get("context_length") if isinstance(cfg, dict) else None
                        if isinstance(ctx, int) and ctx > 0:
                            context_length = ctx
                            break
                    if context_length is not None:
                        entry["context_length"] = context_length

                    max_completion_tokens = _extract_max_completion_tokens(model)
                    if max_completion_tokens is not None:
                        entry["max_completion_tokens"] = max_completion_tokens

                    pricing = _extract_pricing(model)
                    if pricing:
                        entry["pricing"] = pricing

                    _add_model_aliases(cache, model_id, entry)
                    alt_id = model.get("id")
                    if isinstance(alt_id, str) and alt_id and alt_id != model_id:
                        _add_model_aliases(cache, alt_id, entry)

                _endpoint_model_metadata_cache[normalized] = cache
                _endpoint_model_metadata_cache_time[normalized] = time.time()
                return cache
        except Exception as exc:
            last_error = exc

    for candidate in candidates:
        url = candidate.rstrip("/") + "/models"
        try:
            response = requests.get(url, headers=headers, timeout=(5, 10), verify=_resolve_requests_verify())
            response.raise_for_status()
            payload = response.json()
            cache: Dict[str, Dict[str, Any]] = {}
            for model in payload.get("data", []):
                if not isinstance(model, dict):
                    continue
                model_id = model.get("id")
                if not model_id:
                    continue
                entry: Dict[str, Any] = {"name": model.get("name", model_id)}
                context_length = _extract_context_length(model)
                if context_length is not None:
                    entry["context_length"] = context_length
                max_completion_tokens = _extract_max_completion_tokens(model)
                if max_completion_tokens is not None:
                    entry["max_completion_tokens"] = max_completion_tokens
                pricing = _extract_pricing(model)
                if pricing:
                    entry["pricing"] = pricing
                _add_model_aliases(cache, model_id, entry)

            # If this is a llama.cpp server, query /props for actual allocated context
            is_llamacpp = any(
                m.get("owned_by") == "llamacpp"
                for m in payload.get("data", []) if isinstance(m, dict)
            )
            if is_llamacpp:
                try:
                    # Try /v1/props first (current llama.cpp); fall back to /props for older builds
                    base = candidate.rstrip("/").replace("/v1", "")
                    _verify = _resolve_requests_verify()
                    props_resp = requests.get(base + "/v1/props", headers=headers, timeout=5, verify=_verify)
                    if not props_resp.ok:
                        props_resp = requests.get(base + "/props", headers=headers, timeout=5, verify=_verify)
                    if props_resp.ok:
                        props = props_resp.json()
                        gen_settings = props.get("default_generation_settings", {})
                        n_ctx = gen_settings.get("n_ctx")
                        model_alias = props.get("model_alias", "")
                        if n_ctx and model_alias and model_alias in cache:
                            cache[model_alias]["context_length"] = n_ctx
                except Exception:
                    pass

            _endpoint_model_metadata_cache[normalized] = cache
            _endpoint_model_metadata_cache_time[normalized] = time.time()
            return cache
        except Exception as exc:
            last_error = exc

    if last_error:
        logger.debug("Failed to fetch model metadata from %s/models: %s", normalized, last_error)
    _endpoint_model_metadata_cache[normalized] = {}
    _endpoint_model_metadata_cache_time[normalized] = time.time()
    return {}


def _resolve_endpoint_context_length(
    model: str,
    base_url: str,
    api_key: str = "",
) -> Optional[int]:
    """Resolve context length from an endpoint's live ``/models`` metadata."""
    endpoint_metadata = fetch_endpoint_model_metadata(base_url, api_key=api_key)
    matched = endpoint_metadata.get(model)
    if not matched:
        if len(endpoint_metadata) == 1:
            matched = next(iter(endpoint_metadata.values()))
        else:
            for key, entry in endpoint_metadata.items():
                if model in key or key in model:
                    matched = entry
                    break
    if matched:
        context_length = matched.get("context_length")
        if isinstance(context_length, int):
            return context_length
    return None


def _get_context_cache_path() -> Path:
    """Return path to the persistent context length cache file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "context_length_cache.yaml"


def _load_context_cache() -> Dict[str, int]:
    """Load the model+provider -> context_length cache from disk."""
    path = _get_context_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("context_lengths") or {}
    except Exception as e:
        logger.debug("Failed to load context length cache: %s", e)
        return {}


def _context_cache_key(model: str, base_url: str) -> str:
    """Canonical ``model@base_url`` key for the persistent context cache.

    Trailing slashes are stripped so ``http://host/v1`` and
    ``http://host/v1/`` share one entry instead of creating duplicates
    that can go stale independently.
    """
    return f"{model}@{(base_url or '').rstrip('/')}"


def save_context_length(model: str, base_url: str, length: int) -> None:
    """Persist a discovered context length for a model+provider combo.

    Cache key is ``model@base_url`` so the same model name served from
    different providers can have different limits.
    """
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    if cache.get(key) == length:
        return  # already stored
    cache[key] = length
    path = _get_context_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"context_lengths": cache}, f, default_flow_style=False)
        logger.info("Cached context length %s -> %s tokens", key, f"{length:,}")
    except Exception as e:
        logger.debug("Failed to save context length cache: %s", e)


def get_cached_context_length(model: str, base_url: str) -> Optional[int]:
    """Look up a previously discovered context length for model+provider."""
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    hit = cache.get(key)
    if hit is not None:
        return hit
    # Legacy rows written before key normalization may carry a trailing
    # slash — honor them rather than re-probing. Checked regardless of the
    # caller's slash form: the row's shape and the caller's shape can differ
    # in either direction (old slashed row + new normalized config, or the
    # reverse), so probe the literal form and the slashed canonical form.
    for legacy_key in (f"{model}@{base_url}", f"{key}/"):
        if legacy_key != key:
            hit = cache.get(legacy_key)
            if hit is not None:
                return hit
    return None


def _invalidate_cached_context_length(model: str, base_url: str) -> None:
    """Drop a stale cache entry so it gets re-resolved on the next lookup."""
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    # Invalidation must also drop the in-memory TTL probe entries for this
    # pair — otherwise the next resolution inside the TTL window reuses the
    # very value we just declared stale and re-persists it.
    bare = _strip_provider_prefix(model)
    stripped = (base_url or "").rstrip("/")
    _LOCAL_CTX_PROBE_CACHE.pop((bare, stripped), None)
    _LOCAL_CTX_PROBE_CACHE.pop(("ollama_show", bare, stripped), None)
    # Clear every key shape for this pair: canonical, the caller's literal
    # form, and the slashed legacy form — same set get_cached_context_length
    # consults, so a lookup can never resurrect a row invalidation missed.
    stale_keys = {key, f"{model}@{base_url}", f"{key}/"}
    if not any(k in cache for k in stale_keys):
        return
    for k in stale_keys:
        cache.pop(k, None)
    path = _get_context_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"context_lengths": cache}, f, default_flow_style=False)
    except Exception as e:
        logger.debug("Failed to invalidate context length cache entry %s: %s", key, e)


def get_next_probe_tier(current_length: int) -> Optional[int]:
    """Return the next lower probe tier, or None if already at minimum."""
    for tier in CONTEXT_PROBE_TIERS:
        if tier < current_length:
            return tier
    return None


def parse_context_limit_from_error(error_msg: str) -> Optional[int]:
    """Try to extract the actual context limit from an API error message.

    Many providers include the limit in their error text, e.g.:
      - "maximum context length is 32768 tokens"
      - "context_length_exceeded: 131072"
      - "Maximum context size 32768 exceeded"
      - "model's max context length is 65536"
    """
    error_lower = error_msg.lower()
    # Pattern: look for numbers near context-related keywords
    patterns = [
        r'max_model_len\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM: "max_model_len 32768", "=32768", ": 32768", "(32768)", "is 32768"
        r'maximum model length\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM alt: "maximum model length 131072", "... is 131072"
        r'(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})',
        r'context\s*(?:length|size|window)\s*(?:is|of|:)?\s*(\d{4,})',
        r'(\d{4,})\s*(?:token)?\s*(?:context|limit)',
        r'>\s*(\d{4,})\s*(?:max|limit|token)',  # "250000 tokens > 200000 maximum"
        r'(\d{4,})\s*(?:max(?:imum)?)\b',  # "200000 maximum"
    ]
    for pattern in patterns:
        match = re.search(pattern, error_lower)
        if match:
            limit = int(match.group(1))
            # Sanity check: must be a reasonable context length
            if 1024 <= limit <= 10_000_000:
                return limit
    return None


def get_context_length_from_provider_error(
    error_msg: str,
    current_context_length: int,
) -> Optional[int]:
    """Return a provider-reported lower context limit, if one is present.

    Context-overflow recovery must not invent a new model window size.  Some
    providers only say that the input exceeds the context window without
    reporting the actual maximum.  In that case callers should keep the
    configured context length and try compression only, rather than stepping
    down through guessed probe tiers (1M → 256K → 128K → ...).
    """
    parsed_limit = parse_context_limit_from_error(error_msg)
    if parsed_limit is None:
        return None
    if parsed_limit < current_context_length:
        return parsed_limit
    return None


def parse_available_output_tokens_from_error(error_msg: str) -> Optional[int]:
    """Detect an "output cap too large" error and return how many output tokens are available.

    Background — two distinct context errors exist:
      1. "Prompt too long"  — the INPUT itself exceeds the context window.
           Fix: compress history, and only reduce context_length if the
           provider explicitly reports the actual lower limit.
      2. "max_tokens too large" — input is fine, but input + requested_output > window.
           Fix: reduce max_tokens (the output cap) for this call.
           Do NOT touch context_length — the window hasn't shrunk.

    Anthropic's API returns errors like:
      "max_tokens: 32768 > context_window: 200000 - input_tokens: 190000 = available_tokens: 10000"

    Returns the number of output tokens that would fit (e.g. 10000 above), or None if
    the error does not look like a max_tokens-too-large error.
    """
    error_lower = error_msg.lower()

    # Must look like an output-cap error, not a prompt-length error.
    is_output_cap_error = (
        "max_tokens" in error_lower
        and ("available_tokens" in error_lower or "available tokens" in error_lower)
    ) or (
        # OpenRouter/Nous phrasing of the same condition.
        "in the output" in error_lower
        and "maximum context length" in error_lower
    ) or (
        # LM Studio / llama.cpp / some OpenAI-compatible servers:
        #   "This model's maximum context length is 65536 tokens. However, you
        #    requested 65536 output tokens and your prompt contains 77409
        #    characters ..."
        # The "requested N output tokens" phrasing means the OUTPUT cap is the
        # problem (the input itself fits) — reduce max_tokens, don't compress.
        "maximum context length" in error_lower
        and "requested" in error_lower
        and "output tokens" in error_lower
    ) or (
        # DashScope / Alibaba Cloud (Qwen) phrasing.  The provider rejects an
        # over-cap output request with a bounded range whose upper bound IS the
        # real max-output cap, e.g.
        #   "Range of max_tokens should be [1, 65536]"
        # The input itself fits — this is purely an output-cap error, so reduce
        # max_tokens and retry; do NOT compress.
        "range of max_tokens should be" in error_lower
    )
    if not is_output_cap_error:
        return None

    # DashScope / Alibaba range form: "Range of max_tokens should be [1, 65536]".
    # The upper bound is the available output cap.
    _m_range = re.search(
        r'range of max_tokens should be\s*\[\s*\d+\s*,\s*(\d+)\s*\]',
        error_lower,
    )
    if _m_range:
        _cap = int(_m_range.group(1))
        if _cap >= 1:
            return _cap

    # Extract the available_tokens figure.
    # Anthropic format: "… = available_tokens: 10000"
    patterns = [
        r'available_tokens[:\s]+(\d+)',
        r'available\s+tokens[:\s]+(\d+)',
        # fallback: last number after "=" in expressions like "200000 - 190000 = 10000"
        r'=\s*(\d+)\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, error_lower)
        if match:
            tokens = int(match.group(1))
            if tokens >= 1:
                return tokens

    # OpenRouter/Nous format: "maximum context length is N … (A of text input,
    # B of tool input, C in the output)". Available output = ctx - text - tool.
    _m_ctx = re.search(r'maximum context length is (\d+)', error_lower)
    _m_parts = re.search(
        r'\((\d+)\s+of text input,\s*(\d+)\s+of tool input,\s*(\d+)\s+in the output\)',
        error_lower,
    )
    if _m_ctx and _m_parts:
        _available = int(_m_ctx.group(1)) - int(_m_parts.group(1)) - int(_m_parts.group(2))
        if _available >= 1:
            return _available

    # LM Studio / llama.cpp style: context window is reported in tokens but the
    # prompt size is reported in CHARACTERS, e.g.
    #   "maximum context length is 65536 tokens ... your prompt contains 77409
    #    characters ...".
    # Estimate the input tokens conservatively (~3 chars/token, which
    # over-reserves the input so the retried output cap stays safely inside the
    # window) and leave the remainder of the window for output.
    _m_ctx_tok = re.search(r'maximum context length is (\d+)\s*token', error_lower)
    _m_chars = re.search(r'prompt contains (\d+)\s*character', error_lower)
    if _m_ctx_tok and _m_chars:
        _ctx = int(_m_ctx_tok.group(1))
        _est_input = (int(_m_chars.group(1)) + 2) // 3
        _available = _ctx - _est_input
        if _available >= 1:
            return _available

    # vLLM style: both the window and the prompt are reported in TOKENS, e.g.
    #   "This model's maximum context length is 131072 tokens. However, you
    #    requested 65536 output tokens and your prompt contains at least 65537
    #    input tokens, for a total of at least 131073 tokens. Please reduce
    #    the length of the input prompt or the number of requested output
    #    tokens."
    # Available output = window - input. When the input alone is at or over
    # the window this stays None, so the caller correctly falls through to
    # compression instead of futilely shrinking the output cap.
    _m_vllm_input = re.search(
        r'prompt contains (?:at least )?(\d+)\s*input tokens', error_lower
    )
    if _m_ctx_tok and _m_vllm_input:
        _available = int(_m_ctx_tok.group(1)) - int(_m_vllm_input.group(1))
        if _available >= 1:
            return _available

    return None


def is_output_cap_error(error_msg: str) -> bool:
    """Return True if a 400 is about the OUTPUT cap (max_tokens) being too large.

    This is the broader sibling of :func:`parse_available_output_tokens_from_error`:
    that function only returns a number when it can extract the available output
    budget from a *known* provider phrasing.  This one answers the cheaper
    yes/no question — "is this an output-cap error at all?" — across providers
    whose exact wording we may not yet parse a number from.

    Why this matters: an output-cap 400 is deterministic (every retry with the
    same ``max_tokens`` gets the identical rejection).  If such an error is
    misclassified as a context-overflow it gets routed into the compression
    loop, the compressor re-issues the call with the same oversized
    ``max_tokens``, the provider rejects it identically, and the session
    death-loops until "cannot compress further" (issue #55546, DashScope/Qwen:
    "Range of max_tokens should be [1, 65536]").  Compression cannot help an
    output-cap error — the input already fits.

    The signal: the error talks about ``max_tokens`` (or its aliases) as a
    cap/range/limit, and does NOT talk about the INPUT/prompt/context window
    being too long.  When both are present we defer to the context-overflow
    path (a real input overflow can also mention max_tokens).
    """
    error_lower = error_msg.lower()

    mentions_output_param = (
        "max_tokens" in error_lower
        or "max_output_tokens" in error_lower
        or "max_completion_tokens" in error_lower
    )
    if not mentions_output_param:
        return False

    # Phrasing that signals the OUTPUT cap specifically is the problem.
    output_cap_signal = (
        "range of max_tokens should be" in error_lower      # DashScope / Alibaba
        or "available_tokens" in error_lower                # Anthropic
        or "available tokens" in error_lower
        or ("in the output" in error_lower                  # OpenRouter / Nous
            and "maximum context length" in error_lower)
        or ("requested" in error_lower                      # LM Studio / llama.cpp
            and "output tokens" in error_lower)
        or "should be" in error_lower                       # generic "max_tokens should be <= N"
        or "less than or equal" in error_lower
        or "must be" in error_lower
    )
    if not output_cap_signal:
        return False

    # If the error ALSO clearly describes an oversized INPUT, it is a genuine
    # context overflow that happens to mention max_tokens — let the
    # context-overflow path handle it (it can compress the input).
    input_overflow_signal = (
        "prompt is too long" in error_lower
        or "prompt too long" in error_lower
        or "input is too long" in error_lower
        or "input token" in error_lower
        or "prompt length" in error_lower
        or "prompt contains" in error_lower
        or "reduce the length" in error_lower
    )
    return not input_overflow_signal


def _model_id_matches(candidate_id: str, lookup_model: str) -> bool:
    """Return True if *candidate_id* (from server) matches *lookup_model* (configured).

    Supports two forms:
    - Exact match:  "nvidia-nemotron-super-49b-v1" == "nvidia-nemotron-super-49b-v1"
    - Slug match:   "nvidia/nvidia-nemotron-super-49b-v1" matches "nvidia-nemotron-super-49b-v1"
                    (the part after the last "/" equals lookup_model)

    This covers LM Studio's native API which stores models as "publisher/slug"
    while users typically configure only the slug after the "local:" prefix.
    """
    if candidate_id == lookup_model:
        return True
    # Slug match: basename of candidate equals the lookup name
    if "/" in candidate_id and candidate_id.rsplit("/", 1)[1] == lookup_model:
        return True
    return False


def query_ollama_num_ctx(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Query an Ollama server for the model's context length.

    Returns the model's maximum context from GGUF metadata via ``/api/show``,
    or the explicit ``num_ctx`` from the Modelfile if set.  Returns None if
    the server is unreachable or not Ollama.

    This is the value that should be passed as ``num_ctx`` in Ollama chat
    requests to override the default 2048.
    """
    import httpx

    bare_model = _strip_provider_prefix(model)
    server_url = _localhost_to_ipv4(base_url.rstrip("/"))
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]

    try:
        server_type = detect_local_server_type(base_url, api_key=api_key)
    except Exception:
        return None
    if server_type != "ollama":
        return None

    headers = _auth_headers(api_key)

    try:
        with httpx.Client(timeout=3.0, headers=headers) as client:
            resp = client.post(f"{server_url}/api/show", json={"name": bare_model})
            if resp.status_code != 200:
                return None
            data = resp.json()

            # Prefer explicit num_ctx from Modelfile parameters (user override)
            params = data.get("parameters", "")
            if "num_ctx" in params:
                for line in params.split("\n"):
                    if "num_ctx" in line:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                return int(parts[-1])
                            except ValueError:
                                pass

            # Fall back to GGUF model_info context_length (training max)
            model_info = data.get("model_info", {})
            for key, value in model_info.items():
                if "context_length" in key and isinstance(value, (int, float)):
                    return int(value)
    except Exception:
        pass
    return None


def query_ollama_supports_vision(model: str, base_url: str, api_key: str = "") -> Optional[bool]:
    """Return True/False when Ollama ``/api/show`` reports vision support.

    Uses the ``capabilities`` field on Ollama 0.6.0+ and falls back to
    ``model_info.*.vision.block_count`` on older servers. Returns None when
    the server is unreachable, not Ollama, or the model is unknown.
    """
    import httpx

    bare_model = _strip_provider_prefix(model)
    if not bare_model or not base_url:
        return None

    try:
        if detect_local_server_type(base_url, api_key=api_key) != "ollama":
            return None
    except Exception:
        return None

    server_url = _localhost_to_ipv4(base_url.rstrip("/"))
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]

    headers = _auth_headers(api_key)

    try:
        with httpx.Client(timeout=3.0, headers=headers) as client:
            resp = client.post(f"{server_url}/api/show", json={"name": bare_model})
            if resp.status_code != 200:
                return None
            data = resp.json()
    except Exception:
        return None

    caps = data.get("capabilities")
    if isinstance(caps, list):
        if any(str(cap).lower() == "vision" for cap in caps):
            return True
        if caps:
            return False

    model_info = data.get("model_info")
    if isinstance(model_info, dict):
        for key in model_info:
            if "vision.block_count" in str(key).lower():
                return True

    return None


def _query_ollama_api_show(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Query an Ollama server's native ``/api/show`` for context length.

    Provider-agnostic: works against ANY Ollama-compatible server regardless
    of hostname — local Ollama, Ollama Cloud (``ollama.com``), custom Ollama
    hosting behind a reverse proxy, etc.  For non-Ollama servers the POST
    returns 404/405 quickly; the function handles errors gracefully.

    Results are cached in ``_LOCAL_CTX_PROBE_CACHE`` (same 30s TTL,
    positive-only — see ``_query_local_context_length``) so back-to-back
    resolutions during one startup issue a single POST instead of one per
    call site. Failures are never memoized: a server that isn't up yet must
    be re-probed once it comes up.

    For hosted servers the GGUF ``model_info.*.context_length`` is the
    authoritative source: the user can't set their own ``num_ctx``, and the
    OpenAI-compat ``/v1/models`` endpoint correctly omits ``context_length``
    per the OpenAI schema.

    Resolution order for hosted Ollama:
      1. ``model_info.*.context_length`` — GGUF training max (authoritative)
      2. ``parameters`` → ``num_ctx`` — server-side Modelfile override
    The order is flipped vs ``query_ollama_num_ctx()`` because local users
    control ``num_ctx`` themselves; hosted users can't.
    """
    import time as _time

    # Namespaced cache key: shares the TTL store with
    # _query_local_context_length but never collides with its (model, url)
    # keys — the two probes can return different values for the same pair.
    cache_key = ("ollama_show", _strip_provider_prefix(model), base_url.rstrip("/"))
    now = _time.monotonic()
    cached = _LOCAL_CTX_PROBE_CACHE.get(cache_key)
    if cached is not None and (now - cached[1]) < _LOCAL_CTX_PROBE_TTL_SECONDS:
        return cached[0]

    result = _query_ollama_api_show_uncached(model, base_url, api_key=api_key)
    if result:  # positive-only — never memoize a failed probe
        _LOCAL_CTX_PROBE_CACHE[cache_key] = (result, now)
    return result


def _query_ollama_api_show_uncached(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Uncached body of ``_query_ollama_api_show`` — one POST to ``/api/show``."""
    import httpx

    server_url = _localhost_to_ipv4(base_url.rstrip("/"))
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]

    headers = _auth_headers(api_key)

    try:
        with httpx.Client(timeout=5.0, headers=headers) as client:
            resp = client.post(f"{server_url}/api/show", json={"name": model})
            if resp.status_code != 200:
                return None
            data = resp.json()

            # Hosted Ollama: GGUF model_info is the real max — prefer it over
            # num_ctx which the Cloud operator may have capped arbitrarily.
            model_info = data.get("model_info", {})
            for key, value in model_info.items():
                if "context_length" in key and isinstance(value, (int, float)):
                    ctx = int(value)
                    if ctx >= 1024:
                        return ctx

            # Fall back to num_ctx from Modelfile parameters (rare on Cloud)
            params = data.get("parameters", "")
            if "num_ctx" in params:
                for line in params.split("\n"):
                    if "num_ctx" in line:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                ctx = int(parts[-1])
                                if ctx >= 1024:
                                    return ctx
                            except ValueError:
                                pass
    except Exception:
        pass
    return None


def _model_name_suggests_kimi(model: str) -> bool:
    """Return True if the model name looks like a Kimi-family model.

    Catches ``kimi-k2.6``, ``kimi-k2.5``, ``kimi-k2-thinking``,
    ``moonshotai/Kimi-K2.6``, and similar variants.  Used as a guard
    against stale OpenRouter metadata that underreports these models
    as 32K context when they actually support 262K+.
    """
    lower = model.lower()
    return lower.startswith("kimi") or "moonshot" in lower


def _model_name_suggests_minimax_m3(model: str) -> bool:
    """Return True if the model name looks like MiniMax M3.

    Catches ``MiniMax-M3``, ``minimax/minimax-m3``, and similar variants
    across surfaces (native MiniMax-M3, OpenRouter/Nous minimax/minimax-m3).
    Used as a guard against stale cache entries seeded by pre-catalog builds
    that resolved M3 via the generic ``minimax`` catch-all (204,800) before
    the ``minimax-m3`` (1M) entry existed in DEFAULT_CONTEXT_LENGTHS.
    """
    return "minimax-m3" in model.lower()


def _model_name_suggests_grok_4_3(model: str) -> bool:
    """Return True if the model name looks like a Grok 4.3 variant.

    Catches ``grok-4.3``, ``grok-4.3-latest``, and similar slugs.
    Used as a guard against stale cache entries seeded by pre-catalog builds
    that resolved grok-4.3 via the generic ``grok-4`` catch-all (256,000)
    before the ``grok-4.3`` (1M) entry was added to DEFAULT_CONTEXT_LENGTHS
    on 2026-05-15.
    """
    return "grok-4.3" in model.lower()


def _query_local_context_length(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Query a local server for the model's context length (short-TTL cached).

    The live-probe paths added for local endpoints (reconcile-on-hit and the
    pre-defaults step-7 probe) can fire this function several times in quick
    succession during one startup — banner display, ``/model`` switch,
    compressor ``update_model`` all resolve the same model. Each raw probe
    issues synchronous ``detect_local_server_type`` + query HTTP calls (bounded
    by the 3s httpx timeout), so an unreachable/slow local server would pay
    that cost repeatedly. A tiny in-process TTL cache collapses back-to-back
    probes for the same (model, base_url) into one network round-trip without
    persisting anything to disk (freshness across restarts is still handled by
    the reconcile logic, which probes again once the TTL expires).
    """
    import time as _time

    cache_key = (_strip_provider_prefix(model), base_url.rstrip("/"))
    now = _time.monotonic()
    cached = _LOCAL_CTX_PROBE_CACHE.get(cache_key)
    if cached is not None and (now - cached[1]) < _LOCAL_CTX_PROBE_TTL_SECONDS:
        return cached[0]

    result = _query_local_context_length_uncached(model, base_url, api_key=api_key)
    # Cache only positive results. A None/failure (server not up yet,
    # connection refused, timeout) must NOT be memoized — otherwise a probe
    # that fails during a startup race would suppress a legit retry seconds
    # later once the server is reachable. Positive-only caching still fully
    # bounds the hot-path probe rate (a reachable server returns a value and
    # gets cached); an unreachable one simply re-probes on the next call.
    if result:
        _LOCAL_CTX_PROBE_CACHE[cache_key] = (result, now)
    return result


def _query_local_context_length_uncached(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Query a local server for the model's context length."""
    import httpx

    # Strip recognised provider prefix (e.g., "local:model-name" → "model-name").
    # Ollama "model:tag" colons (e.g. "qwen3.5:27b") are intentionally preserved.
    model = _strip_provider_prefix(model)

    # Strip /v1 suffix to get the server root
    server_url = _localhost_to_ipv4(base_url.rstrip("/"))
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]
    lmstudio_url = _localhost_to_ipv4(_lmstudio_server_root(base_url))

    headers = _auth_headers(api_key)

    try:
        server_type = detect_local_server_type(base_url, api_key=api_key)
    except Exception:
        server_type = None

    try:
        with httpx.Client(timeout=3.0, headers=headers) as client:
            # Ollama: /api/show returns model details with context info
            if server_type == "ollama":
                resp = client.post(f"{server_url}/api/show", json={"name": model})
                if resp.status_code == 200:
                    data = resp.json()
                    # Prefer explicit num_ctx from Modelfile parameters: this is
                    # the *runtime* context Ollama will actually allocate KV cache
                    # for. The GGUF model_info.context_length is the training max,
                    # which can be larger than num_ctx — using it here would let
                    # Hermes grow conversations past the runtime limit and Ollama
                    # would silently truncate. Matches query_ollama_num_ctx().
                    params = data.get("parameters", "")
                    if "num_ctx" in params:
                        for line in params.split("\n"):
                            if "num_ctx" in line:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    try:
                                        return int(parts[-1])
                                    except ValueError:
                                        pass
                    # Fall back to GGUF model_info context_length (training max)
                    model_info = data.get("model_info", {})
                    for key, value in model_info.items():
                        if "context_length" in key and isinstance(value, (int, float)):
                            return int(value)

            # LM Studio native API: /api/v1/models returns max_context_length.
            # This is more reliable than the OpenAI-compat /v1/models which
            # doesn't include context window information for LM Studio servers.
            # Use _model_id_matches for fuzzy matching: LM Studio stores models as
            # "publisher/slug" but users configure only "slug" after "local:" prefix.
            if server_type == "lm-studio":
                resp = client.get(f"{lmstudio_url}/api/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("models", []):
                        if _model_id_matches(m.get("key", ""), model) or _model_id_matches(m.get("id", ""), model):
                            # Prefer loaded instance context (actual runtime value)
                            for inst in m.get("loaded_instances", []):
                                cfg = inst.get("config", {})
                                ctx = cfg.get("context_length")
                                if ctx and isinstance(ctx, (int, float)):
                                    return int(ctx)
                            break

            # LM Studio / vLLM / llama.cpp: try /v1/models/{model}
            resp = client.get(f"{server_url}/v1/models/{model}")
            if resp.status_code == 200:
                data = resp.json()
                # vLLM returns max_model_len
                ctx = data.get("max_model_len") or data.get("context_length") or data.get("max_tokens")
                if ctx and isinstance(ctx, (int, float)):
                    return int(ctx)

            # Try /v1/models and find the model in the list.
            # Use _model_id_matches to handle "publisher/slug" vs bare "slug".
            resp = client.get(f"{server_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models_list = data.get("data", [])
                for m in models_list:
                    if _model_id_matches(m.get("id", ""), model):
                        ctx = m.get("max_model_len") or m.get("context_length") or m.get("max_tokens")
                        if ctx and isinstance(ctx, (int, float)):
                            return int(ctx)
    except Exception:
        pass

    return None


def _normalize_model_version(model: str) -> str:
    """Normalize version separators for matching.

    Nous uses dashes: claude-opus-4-6, claude-sonnet-4-5
    OpenRouter uses dots: claude-opus-4.6, claude-sonnet-4.5
    Normalize both to dashes for comparison.
    """
    return model.replace(".", "-")


def _query_anthropic_context_length(model: str, base_url: str, api_key: str) -> Optional[int]:
    """Query Anthropic's /v1/models endpoint for context length.

    Only works with regular ANTHROPIC_API_KEY (sk-ant-api*).
    OAuth tokens (sk-ant-oat*) from Claude Code return 401.
    """
    if not api_key or api_key.startswith("sk-ant-oat"):
        return None  # OAuth tokens can't access /v1/models
    try:
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        url = f"{base}/v1/models?limit=1000"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        resp = requests.get(url, headers=headers, timeout=(5, 10), verify=_resolve_requests_verify())
        if resp.status_code != 200:
            return None
        data = resp.json()
        for m in data.get("data", []):
            if m.get("id") == model:
                ctx = m.get("max_input_tokens")
                if isinstance(ctx, int) and ctx > 0:
                    return ctx
    except Exception as e:
        logger.debug("Anthropic /v1/models query failed: %s", e)
    return None


# Known ChatGPT Codex OAuth context windows (observed via live
# chatgpt.com/backend-api/codex/models probe, Apr 2026). These are the
# `context_window` values, which are what Codex actually enforces — the
# direct OpenAI API has larger limits for the same slugs, but Codex OAuth
# caps lower (e.g. gpt-5.5 is 1.05M on the API, 272K on Codex).
#
# Used as a fallback when the live probe fails (no token, network error).
# Longest keys first so substring match picks the most specific entry.
_CODEX_OAUTH_CONTEXT_FALLBACK: Dict[str, int] = {
    "gpt-5.1-codex-max": 272_000,
    "gpt-5.1-codex-mini": 272_000,
    "gpt-5.3-codex": 272_000,
    # Spark runs on specialised low-latency hardware and exposes a smaller
    # 128k window than other Codex OAuth slugs. Listed explicitly so the
    # longest-key-first fallback resolves it correctly — substring match
    # on "gpt-5.3-codex" otherwise wins and reports 272k. Availability is
    # gated by ChatGPT Pro entitlement on the Codex backend.
    "gpt-5.3-codex-spark": 128_000,
    "gpt-5.2-codex": 272_000,
    "gpt-5.4-mini": 272_000,
    "gpt-5.6-sol": 272_000,
    "gpt-5.6-terra": 272_000,
    "gpt-5.6-luna": 272_000,
    "gpt-5.5": 272_000,
    "gpt-5.4": 272_000,
    "gpt-5.2": 272_000,
    "gpt-5": 272_000,
}


_codex_oauth_context_cache: Dict[str, int] = {}
_codex_oauth_context_cache_time: float = 0.0
_CODEX_OAUTH_CONTEXT_CACHE_TTL = 3600  # 1 hour


def _fetch_codex_oauth_context_lengths(access_token: str) -> Dict[str, int]:
    """Probe the ChatGPT Codex /models endpoint for per-slug context windows.

    Codex OAuth imposes its own context limits that differ from the direct
    OpenAI API (e.g. gpt-5.5 is 1.05M on the API, 272K on Codex). The
    `context_window` field in each model entry is the authoritative source.

    Returns a ``{slug: context_window}`` dict. Empty on failure.
    """
    global _codex_oauth_context_cache, _codex_oauth_context_cache_time
    now = time.time()
    if (
        _codex_oauth_context_cache
        and now - _codex_oauth_context_cache_time < _CODEX_OAUTH_CONTEXT_CACHE_TTL
    ):
        return _codex_oauth_context_cache

    try:
        resp = requests.get(
            "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=(5, 10),
            verify=_resolve_requests_verify(),
        )
        if resp.status_code != 200:
            logger.debug(
                "Codex /models probe returned HTTP %s; falling back to hardcoded defaults",
                resp.status_code,
            )
            return {}
        data = resp.json()
    except Exception as exc:
        logger.debug("Codex /models probe failed: %s", exc)
        return {}

    entries = data.get("models", []) if isinstance(data, dict) else []
    result: Dict[str, int] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        ctx = item.get("context_window")
        if isinstance(slug, str) and isinstance(ctx, int) and ctx > 0:
            result[slug.strip()] = ctx

    if result:
        _codex_oauth_context_cache = result
        _codex_oauth_context_cache_time = now
    return result


def _resolve_codex_oauth_context_length(
    model: str, access_token: str = ""
) -> Optional[int]:
    """Resolve a Codex OAuth model's real context window.

    Prefers a live probe of chatgpt.com/backend-api/codex/models (when we
    have a bearer token), then falls back to ``_CODEX_OAUTH_CONTEXT_FALLBACK``.
    """
    model_bare = _strip_provider_prefix(model).strip()
    if not model_bare:
        return None

    if access_token:
        live = _fetch_codex_oauth_context_lengths(access_token)
        if model_bare in live:
            return live[model_bare]
        # Case-insensitive match in case casing drifts
        model_lower = model_bare.lower()
        for slug, ctx in live.items():
            if slug.lower() == model_lower:
                return ctx

    # Fallback: longest-key-first substring match over hardcoded defaults.
    model_lower = model_bare.lower()
    for slug, ctx in sorted(
        _CODEX_OAUTH_CONTEXT_FALLBACK.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if slug in model_lower:
            return ctx

    return None


def _resolve_nous_context_length(
    model: str,
    base_url: str = "",
    api_key: str = "",
) -> Tuple[Optional[int], str]:
    """Resolve Nous Portal model context length.

    Tries the live Nous inference endpoint first (authoritative), then falls
    back to OpenRouter metadata with suffix/version matching.

    Nous model IDs are bare after prefix-stripping (e.g. 'qwen3.6-plus',
    'claude-opus-4-6') while OpenRouter uses prefixed IDs (e.g.
    'qwen/qwen3.6-plus', 'anthropic/claude-opus-4.6').  Version
    normalization (dot↔dash) is applied to handle name drifts.

    Returns ``(context_length, source)`` where ``source`` is one of:
      - ``"portal"``    — live /v1/models response (authoritative)
      - ``"openrouter"`` — OpenRouter cache fallback (non-authoritative;
        callers must NOT persist this to the on-disk cache or a single
        portal blip will freeze the wrong value in forever)
      - ``""``           — could not resolve
    """
    # Portal first — the Nous /models endpoint is authoritative for what our
    # infrastructure enforces and may differ from OR (e.g. OR reports 1M for
    # qwen3.6-plus; the portal correctly says 262144).  Fall back to the OR
    # catalog only if the portal doesn't list the model.
    if base_url:
        portal_ctx = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
        if portal_ctx is not None:
            return portal_ctx, "portal"

    metadata = fetch_model_metadata()

    def _safe_ctx(or_id: str, entry: dict) -> Optional[int]:
        ctx = entry.get("context_length")
        if ctx is None:
            return None
        if ctx <= 32768 and _model_name_suggests_kimi(or_id):
            logger.info(
                "Rejecting OpenRouter metadata context=%s for %r "
                "(Kimi-family underreport, Nous path); falling through to hardcoded defaults",
                ctx, or_id,
            )
            return None
        return ctx

    if model in metadata:
        ctx = _safe_ctx(model, metadata[model])
        if ctx is not None:
            return ctx, "openrouter"

    normalized = _normalize_model_version(model).lower()

    for or_id, entry in metadata.items():
        bare = or_id.split("/", 1)[1] if "/" in or_id else or_id
        if bare.lower() == model.lower() or _normalize_model_version(bare).lower() == normalized:
            ctx = _safe_ctx(or_id, entry)
            if ctx is not None:
                return ctx, "openrouter"

    model_lower = model.lower()
    for or_id, entry in metadata.items():
        bare = or_id.split("/", 1)[1] if "/" in or_id else or_id
        for candidate, query in [(bare.lower(), model_lower), (_normalize_model_version(bare).lower(), normalized)]:
            if candidate.startswith(query) and (
                len(candidate) == len(query) or candidate[len(query)] in "-:."
            ):
                ctx = _safe_ctx(or_id, entry)
                if ctx is not None:
                    return ctx, "openrouter"

    return None, ""


def get_model_context_length(
    model: str,
    base_url: str = "",
    api_key: str = "",
    config_context_length: int | None = None,
    provider: str = "",
    custom_providers: list | None = None,
) -> int:
    """Get the context length for a model.

    Resolution order:
    0. Explicit config override (model.context_length or custom_providers per-model)
    1. Persistent cache (previously discovered via probing).  Nous URLs
       bypass the cache here so step 5b can always reconcile against
       the authoritative portal /v1/models response.
    1b. AWS Bedrock static table (must precede custom-endpoint probe)
    2. Active endpoint metadata (/models for explicit custom endpoints)
    3. Local server query (for local endpoints)
    4. Anthropic /v1/models API (API-key users only, not OAuth)
    5. Provider-aware lookups (before generic OpenRouter cache):
       a. Copilot live /models API
       b. Nous: live /v1/models probe first (authoritative), then OR
          cache fallback with suffix/version normalisation.  Only
          portal-derived values are persisted to disk.
       c. Codex OAuth /models probe
       d. GMI /models endpoint
       e. Ollama native /api/show probe (any base_url, provider-agnostic)
       f. models.dev registry lookup (with :cloud/-cloud suffix fallback)
    6. OpenRouter live API metadata (Kimi-family 32k guard)
    7. Local server query (before hardcoded defaults for local endpoints)
    8. Hardcoded defaults (broad family patterns, longest-key-first)
    9. Default fallback (256K)"""
    # 0. Explicit config override — user knows best
    if config_context_length is not None and isinstance(config_context_length, int) and config_context_length > 0:
        return config_context_length

    # 0a. MoA virtual provider — ``model`` is a preset name, not a real model,
    # and ``base_url`` is the local virtual endpoint, so every probe below would
    # miss and fall through to the 256K default. The aggregator is the acting
    # model, so resolve the context window from the aggregator slot's real
    # provider+model instead. References are advisory-only and never bound the
    # acting context, so they're ignored here.
    if (provider or "").strip().lower() == "moa":
        try:
            from hermes_cli.config import load_config
            from hermes_cli.moa_config import resolve_moa_preset
            from hermes_cli.runtime_provider import resolve_runtime_provider

            preset = resolve_moa_preset(load_config().get("moa") or {}, model)
            agg = preset.get("aggregator") or {}
            agg_provider = str(agg.get("provider") or "").strip()
            agg_model = str(agg.get("model") or "").strip()
            if agg_model and agg_provider and agg_provider.lower() != "moa":
                rt = resolve_runtime_provider(requested=agg_provider, target_model=agg_model)
                return get_model_context_length(
                    agg_model,
                    base_url=rt.get("base_url", "") or "",
                    api_key=rt.get("api_key", "") or "",
                    provider=agg_provider,
                )
        except Exception:
            logger.debug("MoA aggregator context-length resolution failed", exc_info=True)
        # Fall through to the generic default if aggregator resolution failed.

    # 0b. custom_providers per-model override — check before any probe.
    # This closes the gap where /model switch and display paths used to fall
    # back to 128K despite the user having a per-model context_length set.
    # See #15779.
    if custom_providers and base_url and model:
        try:
            from hermes_cli.config import get_custom_provider_context_length
            cp_ctx = get_custom_provider_context_length(
                model=model,
                base_url=base_url,
                custom_providers=custom_providers,
            )
            if cp_ctx:
                return cp_ctx
        except Exception:
            pass  # fall through to probing

    # Normalise provider-prefixed model names (e.g. "local:model-name" →
    # "model-name") so cache lookups and server queries use the bare ID that
    # local servers actually know about.  Ollama "model:tag" colons are preserved.
    model = _strip_provider_prefix(model)

    # 1. Check persistent cache (model+provider)
    # LM Studio is excluded — its loaded context length is transient (the
    # user can reload the model with a different context_length at any time
    # via /api/v1/models/load), so a stale cached value would mask reloads.
    if base_url and not _skip_persistent_context_cache(base_url, provider):
        cached = get_cached_context_length(model, base_url)
        if cached is not None:
            # Invalidate stale Codex OAuth cache entries: pre-PR #14935 builds
            # resolved gpt-5.x to the direct-API value (e.g. 1.05M) via
            # models.dev and persisted it. Codex OAuth caps at 272K for every
            # slug, so any cached Codex entry at or above 400K is a leftover
            # from the old resolution path. Drop it and fall through to the
            # live /models probe in step 5 below.
            if provider == "openai-codex" and cached >= 400_000:
                logger.info(
                    "Dropping stale Codex cache entry %s@%s -> %s (pre-fix value); "
                    "re-resolving via live /models probe",
                    model, base_url, f"{cached:,}",
                )
                _invalidate_cached_context_length(model, base_url)
            # Invalidate stale 32k cache entries for Kimi-family models.
            elif cached <= 32768 and _model_name_suggests_kimi(model):
                logger.info(
                    "Dropping stale Kimi cache entry %s@%s -> %s (OpenRouter underreport); "
                    "re-resolving via hardcoded defaults",
                    model, base_url, f"{cached:,}",
                )
                _invalidate_cached_context_length(model, base_url)
            # Invalidate stale ≤204,800 cache entries for MiniMax-M3.  Pre-catalog
            # builds resolved M3 via the generic ``minimax`` catch-all (204,800)
            # and persisted it before the ``minimax-m3`` (1M) entry existed; that
            # stale value would otherwise stick forever here at step 1.  M3 is 1M,
            # so any sub-256K cached value for an M3 slug is a leftover — drop it
            # and fall through to the hardcoded default.
            elif cached <= 204_800 and _model_name_suggests_minimax_m3(model):
                logger.info(
                    "Dropping stale MiniMax-M3 cache entry %s@%s -> %s (pre-catalog value); "
                    "re-resolving via hardcoded defaults",
                    model, base_url, f"{cached:,}",
                )
                _invalidate_cached_context_length(model, base_url)
            # Invalidate stale ≤256,000 cache entries for Grok-4.3.  The
            # ``grok-4.3`` (1M) entry was added to DEFAULT_CONTEXT_LENGTHS on
            # 2026-05-15; prior to that, grok-4.3 slugs resolved via the
            # ``grok-4`` catch-all (256,000) and that value was persisted.
            # grok-4.3 is 1M, so any sub-262K cached value is a pre-catalog
            # leftover — drop it and fall through to the hardcoded default.
            elif cached <= 256_000 and _model_name_suggests_grok_4_3(model):
                logger.info(
                    "Dropping stale Grok-4.3 cache entry %s@%s -> %s (pre-catalog value); "
                    "re-resolving via hardcoded defaults",
                    model, base_url, f"{cached:,}",
                )
                _invalidate_cached_context_length(model, base_url)
            # Nous Portal: the portal /v1/models endpoint is authoritative.
            # Bypass the persistent cache so step 5b can always reconcile
            # against it — this corrects pre-fix entries seeded from the
            # OR catalog (the same OR underreport class that the Kimi/Qwen
            # DEFAULT_CONTEXT_LENGTHS overrides exist to mitigate) without
            # touching the on-disk file when the portal is unreachable.
            # The in-memory 300s endpoint metadata cache makes the per-call
            # cost amortise to ~0 within a process.
            elif _infer_provider_from_url(base_url) == "nous":
                logger.debug(
                    "Bypassing persistent cache for %s@%s (Nous portal authoritative)",
                    model, base_url,
                )
                # Fall through; step 5b reconciles and overwrites if portal responds.
            else:
                if is_local_endpoint(base_url):
                    return _reconcile_local_cached_context_length(
                        model, base_url, cached, api_key=api_key,
                    )
                return cached

    # 1b. AWS Bedrock — use static context length table.
    # Bedrock's ListFoundationModels API doesn't expose context window sizes,
    # so we maintain a curated table in bedrock_adapter.py that reflects
    # AWS-imposed limits (e.g. 200K for Claude models vs 1M on the native
    # Anthropic API).  This must run BEFORE the custom-endpoint probe at
    # step 2 — bedrock-runtime.<region>.amazonaws.com is not in
    # _URL_TO_PROVIDER, so it would otherwise be treated as a custom endpoint,
    # fail the /models probe (Bedrock doesn't expose that shape), and fall
    # back to the 128K default before reaching the original step 4b branch.
    if provider == "bedrock" or (
        base_url
        and base_url_hostname(base_url).startswith("bedrock-runtime.")
        and base_url_host_matches(base_url, "amazonaws.com")
    ):
        try:
            from agent.bedrock_adapter import get_bedrock_context_length
            return get_bedrock_context_length(model)
        except ImportError:
            pass  # boto3 not installed — fall through to generic resolution

    if provider == "novita" or (base_url and base_url_host_matches(base_url, "api.novita.ai")):
        ctx = _resolve_endpoint_context_length(model, base_url or "https://api.novita.ai/openai/v1", api_key=api_key)
        if ctx is not None:
            if base_url:
                save_context_length(model, base_url, ctx)
            return ctx

    # 2. Active endpoint metadata for truly custom/unknown endpoints.
    # Known providers (Copilot, OpenAI, Anthropic, etc.) skip this — their
    # /models endpoint may report a provider-imposed limit (e.g. Copilot
    # returns 128k) instead of the model's full context (400k).  models.dev
    # has the correct per-provider values and is checked at step 5+.
    if _is_custom_endpoint(base_url) and not _is_known_provider_base_url(base_url):
        context_length = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
        if context_length is not None:
            return context_length
        if not _is_known_provider_base_url(base_url):
            # 2b. Ollama native /api/show — any URL might be an Ollama server
            # (local, cloud, or custom hosting).  Non-Ollama servers return
            # 404/405 quickly.  Fall through on failure.
            ctx = _query_ollama_api_show(model, base_url, api_key=api_key)
            if ctx is not None:
                if not _skip_persistent_context_cache(base_url, provider):
                    save_context_length(model, base_url, ctx)
                return ctx
            # 3. Try querying local server directly
            if is_local_endpoint(base_url):
                local_ctx = _query_local_context_length(model, base_url, api_key=api_key)
                if local_ctx and local_ctx > 0:
                    if not _skip_persistent_context_cache(base_url, provider):
                        _maybe_cache_local_context_length(model, base_url, local_ctx)
                    return local_ctx
            logger.info(
                "Could not detect context length for model %r at %s — "
                "defaulting to %s tokens (probe-down). Set model.context_length "
                "in config.yaml to override.",
                model, base_url, f"{DEFAULT_FALLBACK_CONTEXT:,}",
            )
            # 3b. Before falling back to the hard 256K default, consult the
            # hardcoded catalog as a last resort.  A proxied/custom Anthropic
            # gateway (e.g. corporate proxy) fails the Ollama/local probes
            # above, but the model name may still match an entry in
            # DEFAULT_CONTEXT_LENGTHS (e.g. "claude-opus-4-8" → 1M).
            # Without this, the early return here short-circuits the catalog
            # lookup at step 8 and silently caps context at 256K.
            model_lower = model.lower()
            for default_model, length in sorted(
                DEFAULT_CONTEXT_LENGTHS.items(),
                key=lambda x: len(x[0]),
                reverse=True,
            ):
                if default_model in model_lower:
                    logger.info(
                        "Using hardcoded context length %s for model %r "
                        "(custom endpoint, catalog match on %r)",
                        f"{length:,}", model, default_model,
                    )
                    return length
            return DEFAULT_FALLBACK_CONTEXT

    # 4. Anthropic /v1/models API (only for regular API keys, not OAuth)
    if provider == "anthropic" or (
        base_url and base_url_hostname(base_url) == "api.anthropic.com"
    ):
        ctx = _query_anthropic_context_length(model, base_url or "https://api.anthropic.com", api_key)
        if ctx:
            return ctx

    # 4b. (Bedrock handled earlier at step 1b — before custom-endpoint probe.)

    # 5. Provider-aware lookups (before generic OpenRouter cache)
    # These are provider-specific and take priority over the generic OR cache,
    # since the same model can have different context limits per provider
    # (e.g. claude-opus-4.6 is 1M on Anthropic but 128K on GitHub Copilot).
    # If provider is generic (openrouter/custom/empty), try to infer from URL.
    effective_provider = provider
    if not effective_provider or effective_provider in {"openrouter", "custom"}:
        if base_url:
            inferred = _infer_provider_from_url(base_url)
            if inferred:
                effective_provider = inferred

    # 5a. Copilot live /models API — max_prompt_tokens from the user's account.
    # This catches account-specific models (e.g. claude-opus-4.6-1m) that
    # don't exist in models.dev. For models that ARE in models.dev, this
    # returns the provider-enforced limit which is what users can actually use.
    if effective_provider in {"copilot", "copilot-acp", "github-copilot"}:
        try:
            from hermes_cli.models import get_copilot_model_context
            ctx = get_copilot_model_context(model, api_key=api_key)
            if ctx:
                return ctx
        except Exception:
            pass  # Fall through to models.dev

    if effective_provider == "nous":
        ctx, source = _resolve_nous_context_length(
            model, base_url=base_url or "", api_key=api_key or ""
        )
        if ctx:
            # Persist ONLY portal-derived values.  Caching an OR-fallback
            # value here would freeze in a wrong number on the first portal
            # blip / auth glitch and step-1 would short-circuit it forever.
            # OR's catalog is community-maintained and is precisely why the
            # Kimi/Qwen DEFAULT_CONTEXT_LENGTHS overrides exist — we don't
            # want it leaking into the persistent cache for Nous URLs.
            if base_url and source == "portal":
                save_context_length(model, base_url, ctx)
            return ctx
    if effective_provider == "openai-codex":
        # Codex OAuth enforces lower context limits than the direct OpenAI
        # API for the same slug (e.g. gpt-5.5 is 1.05M on the API but 272K
        # on Codex). Authoritative source is Codex's own /models endpoint.
        codex_ctx = _resolve_codex_oauth_context_length(model, access_token=api_key or "")
        if codex_ctx:
            if base_url:
                save_context_length(model, base_url, codex_ctx)
            return codex_ctx
    if effective_provider == "gmi" and base_url:
        # GMI exposes authoritative context_length via /models, but it is not
        # in models.dev yet. Preserve that higher-fidelity endpoint lookup.
        ctx = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
        if ctx is not None:
            return ctx
    # 5e. Ollama native /api/show probe — runs for providers whose base_url
    # is NOT a known non-Ollama provider.  Ollama-compatible servers expose
    # this endpoint regardless of hostname (local Ollama, Ollama Cloud,
    # custom Ollama hosting).  The OpenAI-compat /v1/models endpoint
    # correctly omits context_length per the OpenAI schema, but /api/show
    # returns the authoritative GGUF model_info.context_length.
    # Known hosted providers (OpenRouter, Anthropic, OpenAI, …) are skipped:
    # they are definitively not Ollama, the POST always 404s, and the result
    # is never cached for them — so every fresh process used to pay a
    # ~300ms blocking HTTP round-trip on the first-turn critical path
    # (measured against openrouter.ai; worse on slow DNS).
    if base_url:
        _inferred_for_probe = _infer_provider_from_url(base_url)
        _skip_ollama_probe = (
            _inferred_for_probe is not None
            and "ollama" not in _inferred_for_probe
        )
        if not _skip_ollama_probe:
            ctx = _query_ollama_api_show(model, base_url, api_key=api_key)
            if ctx is not None:
                if not _skip_persistent_context_cache(base_url, provider):
                    save_context_length(model, base_url, ctx)
                return ctx
    # 5f. OpenRouter live /models metadata — authoritative for OpenRouter-routed
    # models. OpenRouter's catalog carries per-model context_length (e.g.
    # anthropic/claude-fable-5 -> 1M) and refreshes as new slugs ship, so it
    # must win over both models.dev (step 5g) and the hardcoded family catch-all
    # (step 8). Before this branch, an OpenRouter selection set
    # effective_provider="openrouter", which (a) made the models.dev lookup miss
    # brand-new slugs and (b) skipped the step-6 OR fallback (gated on `not
    # effective_provider`), so a fresh slug like claude-fable-5 fell through to
    # the generic "claude": 200K entry and under-reported a 1M window. Mirrors
    # the dedicated Nous/Copilot/GMI branches above.
    if effective_provider == "openrouter":
        metadata = fetch_model_metadata()
        entry = metadata.get(model)
        if entry:
            or_ctx = entry.get("context_length")
            # Guard against the known OpenRouter Kimi-family 32k underreport
            # (same class the hardcoded overrides exist to mitigate).
            if isinstance(or_ctx, int) and or_ctx > 0 and not (
                or_ctx == 32768 and _model_name_suggests_kimi(model)
            ):
                return or_ctx

    if effective_provider:
        from agent.models_dev import lookup_models_dev_context
        ctx = lookup_models_dev_context(effective_provider, model)
        if ctx:
            # MiniMax M3: models.dev reports 512K but actual context is 1M.
            # Prefer hardcoded catalog over stale probe value.
            if _model_name_suggests_minimax_m3(model):
                catalog = DEFAULT_CONTEXT_LENGTHS.get("minimax-m3")
                if catalog and ctx < catalog:
                    logger.info(
                        "Rejecting models.dev context=%s for %r "
                        "(MiniMax-M3 underreport); using hardcoded default %s",
                        ctx, model, f"{catalog:,}",
                    )
                    ctx = catalog
            return ctx

    # 6. OpenRouter live API metadata — provider-unaware fallback.
    # Only consulted when the provider is unknown (no effective_provider),
    # because OpenRouter data is community-maintained and can be incorrect
    # for models that belong to known providers with curated defaults.
    if not effective_provider:
        metadata = fetch_model_metadata()
        if model in metadata:
            or_ctx = metadata[model].get("context_length", DEFAULT_FALLBACK_CONTEXT)
            # Guard against stale OpenRouter metadata for Kimi-family models.
            if or_ctx == 32768 and _model_name_suggests_kimi(model):
                logger.info(
                    "Rejecting OpenRouter metadata context=%s for %r "
                    "(Kimi-family underreport); falling through to hardcoded defaults",
                    or_ctx, model,
                )
            else:
                return or_ctx

    # 7. Query local server before hardcoded defaults — model names like
    # ``Hermes-3-Llama-3.1-70B`` substring-match ``llama`` (131072) even when
    # vLLM is running at a lower ``--max-model-len`` (e.g. 32768 on limited VRAM).
    if base_url and is_local_endpoint(base_url):
        local_ctx = _query_local_context_length(model, base_url, api_key=api_key)
        if local_ctx and local_ctx > 0:
            if not _skip_persistent_context_cache(base_url, provider):
                _maybe_cache_local_context_length(model, base_url, local_ctx)
            return local_ctx

    # 8. Hardcoded defaults (fuzzy match — longest key first for specificity)
    # Only check `default_model in model` (is the key a substring of the input).
    # The reverse (`model in default_model`) causes shorter names like
    # "claude-sonnet-4" to incorrectly match "claude-sonnet-4-6" and return 1M.
    model_lower = model.lower()
    for default_model, length in sorted(
        DEFAULT_CONTEXT_LENGTHS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if default_model in model_lower:
            return length

    # 9. Default fallback — 256K
    return DEFAULT_FALLBACK_CONTEXT


async def get_model_context_length_async(
    model: str,
    base_url: str = "",
    api_key: str = "",
    config_context_length: int | None = None,
    provider: str = "",
    custom_providers: list | None = None,
) -> int:
    """Async variant of get_model_context_length.

    Offloads the entire synchronous resolution chain (which contains
    blocking HTTP calls via ``requests``) to a background thread so it
    does not freeze the asyncio event loop and cause Discord heartbeat
    timeouts.

    Shares all logic with the sync version — no code duplication.
    """
    import asyncio
    return await asyncio.to_thread(
        get_model_context_length,
        model,
        base_url=base_url,
        api_key=api_key,
        config_context_length=config_context_length,
        provider=provider,
        custom_providers=custom_providers,
    )


def estimate_tokens_rough(text: str) -> int:
    """Rough token estimate (~4 chars/token) for pre-flight checks.

    Uses ceiling division so short texts (1-3 chars) never estimate as
    0 tokens, which would cause the compressor and pre-flight checks to
    systematically undercount when many short tool results are present.
    """
    if not text:
        return 0
    return (len(text) + 3) // 4


def estimate_messages_tokens_rough(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate for a message list (pre-flight only).

    Image parts (base64 PNG/JPEG) are counted as a flat ~1500 tokens per
    image — the Anthropic pricing model — instead of counting raw base64
    character length. Without this, a single ~1MB screenshot would be
    estimated at ~250K tokens and trigger premature context compression.
    """
    _IMAGE_TOKEN_COST = 1500
    total_chars = 0
    image_tokens = 0
    for msg in messages:
        total_chars += _estimate_message_chars(msg)
        image_tokens += _count_image_tokens(msg, _IMAGE_TOKEN_COST)
    return ((total_chars + 3) // 4) + image_tokens


def _count_image_tokens(msg: Dict[str, Any], cost_per_image: int) -> int:
    """Count image-like content parts in a message; return their token cost."""
    count = 0
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in {"image", "image_url", "input_image"}:
                count += 1
    stashed = msg.get("_anthropic_content_blocks") if isinstance(msg, dict) else None
    if isinstance(stashed, list):
        for part in stashed:
            if isinstance(part, dict) and part.get("type") == "image":
                count += 1
    # Multimodal tool results that haven't been converted yet.
    if isinstance(content, dict) and content.get("_multimodal"):
        inner = content.get("content")
        if isinstance(inner, list):
            for part in inner:
                if isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
                    count += 1
    return count * cost_per_image


def _estimate_message_chars(msg: Dict[str, Any]) -> int:
    """Char count for token estimation, excluding base64 image data.

    Base64 images are counted via `_count_image_tokens` instead; including
    their raw chars here would massively overestimate token usage.
    """
    if not isinstance(msg, dict):
        return len(str(msg))
    shadow: Dict[str, Any] = {}
    for k, v in msg.items():
        if k == "_anthropic_content_blocks":
            continue
        if k == "content":
            if isinstance(v, list):
                cleaned = []
                for part in v:
                    if isinstance(part, dict):
                        if part.get("type") in {"image", "image_url", "input_image"}:
                            cleaned.append({"type": part.get("type"), "image": "[stripped]"})
                        else:
                            cleaned.append(part)
                    else:
                        cleaned.append(part)
                shadow[k] = cleaned
            elif isinstance(v, dict) and v.get("_multimodal"):
                shadow[k] = v.get("text_summary", "")
            else:
                shadow[k] = v
        else:
            shadow[k] = v
    return len(str(shadow))


def estimate_request_tokens_rough(
    messages: List[Dict[str, Any]],
    *,
    system_prompt: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Rough token estimate for a full chat-completions request.

    Includes the major payload buckets Hermes sends to providers:
    system prompt, conversation messages, and tool schemas.  With 50+
    tools enabled, schemas alone can add 20-30K tokens — a significant
    blind spot when only counting messages. Image content is counted
    at a flat per-image cost (see estimate_messages_tokens_rough).

    Note: A 15% safety margin is applied to the raw count because the
    character-count heuristic can underestimate the true token count by
    20-30%. This margin ensures auto-compression fires early enough to
    prevent requests from exceeding context_length after 3 retries.
    """
    total = 0
    if system_prompt:
        total += (len(system_prompt) + 3) // 4
    if messages:
        total += estimate_messages_tokens_rough(messages)
    if tools:
        total += _estimate_tools_tokens_rough(tools)
    # Safety margin: rough character-count estimate can underestimate
    # true prompt_tokens by 20-30%, causing auto-compression to fire
    # too late.  The 15% cushion triggers compression earlier so that
    # after 3 retries the request stays within context_length.
    return int(total * 1.15)


# NOTE: tool schemas can be large. Avoid repeated `str(tools)` conversions,
# which are CPU-heavy and can stall GUI event loops under GIL pressure.
#
# Keyed by ``id(tools)``. A long-lived gateway/desktop backend builds many
# transient tool lists over its lifetime, so the cache is bounded and evicts
# oldest-first (insertion-ordered dict) once it exceeds the cap. The cap is
# generous relative to how rarely toolsets are rebuilt within a process.
_TOOLS_TOKENS_CACHE: dict[int, Tuple[int, str, str, int]] = {}
_TOOLS_TOKENS_CACHE_MAX = 256


def _tool_name_for_cache(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str):
            return name
    name = tool.get("name")
    return name if isinstance(name, str) else ""


def _estimate_tools_tokens_rough(tools: List[Dict[str, Any]]) -> int:
    if not tools:
        return 0

    # Cache by list identity. Tools are rebuilt rarely (toolset changes),
    # but token estimates are requested frequently (preflight, compaction).
    key = id(tools)
    n = len(tools)
    first = _tool_name_for_cache(tools[0]) if n else ""
    last = _tool_name_for_cache(tools[-1]) if n else ""

    cached = _TOOLS_TOKENS_CACHE.get(key)
    if cached is not None:
        cached_n, cached_first, cached_last, cached_tokens = cached
        if cached_n == n and cached_first == first and cached_last == last:
            return cached_tokens

    # Fast, stable rough estimate: sum lengths of the major schema fields.
    # This avoids the pathological `str(tools)` path while still scaling with
    # schema size (descriptions + parameters dominate).
    total_chars = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict):
            name = fn.get("name") or ""
            desc = fn.get("description") or ""
            params = fn.get("parameters") or {}
        else:
            name = tool.get("name") or ""
            desc = tool.get("description") or ""
            params = tool.get("parameters") or {}

        if isinstance(name, str):
            total_chars += len(name)
        if isinstance(desc, str):
            total_chars += len(desc)
        # Parameters can be nested; JSON is closer to over-the-wire size than repr().
        try:
            total_chars += len(json.dumps(params, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            total_chars += len(str(params))

    tokens = (total_chars + 3) // 4
    # Bound the cache: drop the oldest entry when the cap is exceeded so a
    # long-running process can't accumulate an unbounded number of stale
    # ``id(tools)`` entries (id values are recycled after GC anyway).
    if len(_TOOLS_TOKENS_CACHE) >= _TOOLS_TOKENS_CACHE_MAX:
        _TOOLS_TOKENS_CACHE.pop(next(iter(_TOOLS_TOKENS_CACHE)), None)
    _TOOLS_TOKENS_CACHE[key] = (n, first, last, tokens)
    return tokens

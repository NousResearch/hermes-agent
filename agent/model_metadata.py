"""Model metadata, context lengths, and token estimation utilities.

Pure utility functions with no AIAgent dependency. Used by ContextCompressor
and run_agent.py for pre-flight context checks.
"""

import ipaddress
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import yaml

from utils import base_url_host_matches, base_url_hostname

from hermes_constants import OPENROUTER_MODELS_URL

logger = logging.getLogger(__name__)

# Provider names that can appear as a "provider:" prefix before a model ID.
# Only these are stripped — Ollama-style "model:tag" colons (e.g. "qwen3.5:27b")
# are preserved so the full model name reaches cache lookups and server queries.
_PROVIDER_PREFIXES: frozenset[str] = frozenset({
    "openrouter", "nous", "openai-codex", "copilot", "copilot-acp",
    "gemini", "ollama-cloud", "zai", "kimi-coding", "kimi-coding-cn", "stepfun", "minimax", "minimax-cn", "anthropic", "deepseek",
    "opencode-zen", "opencode-go", "ai-gateway", "kilocode", "alibaba",
    "qwen-oauth",
    "xiaomi",
    "arcee",
    "custom", "local",
    # Common aliases
    "google", "google-gemini", "google-ai-studio",
    "glm", "z-ai", "z.ai", "zhipu", "github", "github-copilot",
    "github-models", "kimi", "moonshot", "kimi-cn", "moonshot-cn", "claude", "deep-seek",
    "ollama",
    "stepfun", "opencode", "zen", "go", "vercel", "kilo", "dashscope", "aliyun", "qwen",
    "mimo", "xiaomi-mimo",
    "arcee-ai", "arceeai",
    "xai", "x-ai", "x.ai", "grok",
    "nvidia", "nim", "nvidia-nim", "nemotron",
    "qwen-portal",
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
_MODEL_CACHE_TTL = 3600
_endpoint_model_metadata_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
_endpoint_model_metadata_cache_time: Dict[str, float] = {}
_ENDPOINT_MODEL_CACHE_TTL = 300

# Descending tiers for context length probing when the model is unknown.
# We start at 128K (a safe default for most modern models) and step down
# on context-length errors until one works.
CONTEXT_PROBE_TIERS = [
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

# Thin fallback defaults — only broad model family patterns.
# These fire only when provider is unknown AND models.dev/OpenRouter/Anthropic
# all miss. Replaced the previous 80+ entry dict.
# For provider-specific context lengths, models.dev is the primary source.
DEFAULT_CONTEXT_LENGTHS = {
    # Anthropic Claude 4.6 (1M context) — bare IDs only to avoid
    # fuzzy-match collisions (e.g. "anthropic/claude-sonnet-4" is a
    # substring of "anthropic/claude-sonnet-4.6").
    # OpenRouter-prefixed models resolve via OpenRouter live API or models.dev.
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
    # GPT-5.5 (launched Apr 23 2026). 400k is the fallback for providers we
    # can't probe live. ChatGPT Codex OAuth actually caps lower (272k as of
    # Apr 2026) and is resolved via _resolve_codex_oauth_context_length().
    "gpt-5.5": 400000,
    "gpt-5.4-nano": 400000,           # 400k (not 1.05M like full 5.4)
    "gpt-5.4-mini": 400000,           # 400k (not 1.05M like full 5.4)
    "gpt-5.4": 1050000,               # GPT-5.4, GPT-5.4 Pro (1.05M context)
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
    # DeepSeek
    "deepseek": 128000,
    # Meta
    "llama": 131072,
    # Qwen — specific model families before the catch-all.
    # Official docs: https://help.aliyun.com/zh/model-studio/developer-reference/
    "qwen3-coder-plus": 1000000,  # 1M context
    "qwen3-coder": 262144,        # 256K context
    "qwen": 131072,
    # MiniMax — official docs: 204,800 context for all models
    # https://platform.minimax.io/docs/api-reference/text-anthropic-api
    "minimax": 204800,
    # GLM
    "glm": 202752,
    # xAI Grok — xAI /v1/models does not return context_length metadata,
    # so these hardcoded fallbacks prevent Hermes from probing-down to
    # the default 128k when the user points at https://api.x.ai/v1
    # via a custom provider. Values sourced from models.dev (2026-04).
    # Keys use substring matching (longest-first), so e.g. "grok-4.20"
    # matches "grok-4.20-0309-reasoning" / "-non-reasoning" / "-multi-agent-0309".
    "grok-code-fast": 256000,   # grok-code-fast-1
    "grok-4-1-fast": 2000000,   # grok-4-1-fast-(non-)reasoning
    "grok-2-vision": 8192,      # grok-2-vision, -1212, -latest
    "grok-4-fast": 2000000,     # grok-4-fast-(non-)reasoning
    "grok-4.20": 2000000,       # grok-4.20-0309-(non-)reasoning, -multi-agent-0309
    "grok-4": 256000,           # grok-4, grok-4-0709
    "grok-3": 131072,           # grok-3, grok-3-mini, grok-3-fast, grok-3-mini-fast
    "grok-2": 131072,           # grok-2, grok-2-1212, grok-2-latest
    "grok": 131072,             # catch-all (grok-beta, unknown grok-*)
    # Kimi
    "kimi": 262144,
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

_CONTEXT_LENGTH_KEYS = (
    "context_length",
    "context_window",
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
    "models.github.ai": "copilot",
    "api.fireworks.ai": "fireworks",
    "opencode.ai": "opencode-go",
    "api.x.ai": "xai",
    "integrate.api.nvidia.com": "nvidia",
    "api.xiaomimimo.com": "xiaomi",
    "xiaomimimo.com": "xiaomi",
    "ollama.com": "ollama-cloud",
}


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


def _is_known_provider_base_url(base_url: str) -> bool:
    return _infer_provider_from_url(base_url) is not None


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


def detect_local_server_type(base_url: str, api_key: str = "") -> Optional[str]:
    """Detect which local server is running at base_url by probing known endpoints.

    Returns one of: "ollama", "lm-studio", "vllm", "llamacpp", or None.
    """
    import httpx

    normalized = _normalize_base_url(base_url)
    server_url = normalized
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]

    headers = _auth_headers(api_key)

    try:
        with httpx.Client(timeout=2.0, headers=headers) as client:
            # LM Studio exposes /api/v1/models — check first (most specific)
            try:
                r = client.get(f"{server_url}/api/v1/models")
                if r.status_code == 200:
                    return "lm-studio"
            except Exception:
                pass
            # Ollama exposes /api/tags and responds with {"models": [...]}
            # LM Studio returns {"error": "Unexpected endpoint"} with status 200
            # on this path, so we must verify the response contains "models".
            try:
                r = client.get(f"{server_url}/api/tags")
                if r.status_code == 200:
                    try:
                        data = r.json()
                        if "models" in data:
                            return "ollama"
                    except Exception:
                        pass
            except Exception:
                pass
            # llama.cpp exposes /v1/props (older builds used /props without the /v1 prefix)
            try:
                r = client.get(f"{server_url}/v1/props")
                if r.status_code != 200:
                    r = client.get(f"{server_url}/props")  # fallback for older builds
                if r.status_code == 200 and "default_generation_settings" in r.text:
                    return "llamacpp"
            except Exception:
                pass
            # vLLM: /version
            try:
                r = client.get(f"{server_url}/version")
                if r.status_code == 200:
                    data = r.json()
                    if "version" in data:
                        return "vllm"
            except Exception:
                pass
    except Exception:
        pass

    return None


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
                if alias in normalized and normalized[alias] not in (None, ""):
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

    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=10)
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
        logger.debug("Fetched metadata for %s models from OpenRouter", len(cache))
        return cache

    except Exception as e:
        logging.warning(f"Failed to fetch model metadata from OpenRouter: {e}")
        return _model_metadata_cache or {}


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
                server_url = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized
                response = requests.get(
                    server_url.rstrip("/") + "/api/v1/models",
                    headers=headers,
                    timeout=10,
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
                    if context_length is None:
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
            response = requests.get(url, headers=headers, timeout=10)
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
                    props_resp = requests.get(base + "/v1/props", headers=headers, timeout=5)
                    if not props_resp.ok:
                        props_resp = requests.get(base + "/props", headers=headers, timeout=5)
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
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return data.get("context_lengths", {})
    except Exception as e:
        logger.debug("Failed to load context length cache: %s", e)
        return {}


def save_context_length(model: str, base_url: str, length: int) -> None:
    """Persist a discovered context length for a model+provider combo.

    Cache key is ``model@base_url`` so the same model name served from
    different providers can have different limits.
    """
    key = f"{model}@{base_url}"
    cache = _load_context_cache()
    if cache.get(key) == length:
        return  # already stored
    cache[key] = length
    path = _get_context_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump({"context_lengths": cache}, f, default_flow_style=False)
        logger.info("Cached context length %s -> %s tokens", key, f"{length:,}")
    except Exception as e:
        logger.debug("Failed to save context length cache: %s", e)


def get_cached_context_length(model: str, base_url: str) -> Optional[int]:
    """Look up a previously discovered context length for model+provider."""
    key = f"{model}@{base_url}"
    cache = _load_context_cache()
    return cache.get(key)


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


def parse_available_output_tokens_from_error(error_msg: str) -> Optional[int]:
    """Detect an "output cap too large" error and return how many output tokens are available.

    Background — two distinct context errors exist:
      1. "Prompt too long"  — the INPUT itself exceeds the context window.
           Fix: compress history and/or halve context_length.
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
    )
    if not is_output_cap_error:
        return None

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
    return None


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
    server_url = base_url.rstrip("/")
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


def _query_local_context_length(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Query a local server for the model's context length."""
    import httpx

    # Strip recognised provider prefix (e.g., "local:model-name" → "model-name").
    # Ollama "model:tag" colons (e.g. "qwen3.5:27b") are intentionally preserved.
    model = _strip_provider_prefix(model)

    # Strip /v1 suffix to get the server root
    server_url = base_url.rstrip("/")
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]

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
                resp = client.get(f"{server_url}/api/v1/models")
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
                            # Fall back to max_context_length (theoretical model max)
                            ctx = m.get("max_context_length") or m.get("context_length")
                            if ctx and isinstance(ctx, (int, float)):
                                return int(ctx)

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
        resp = requests.get(url, headers=headers, timeout=10)
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
    "gpt-5.2-codex": 272_000,
    "gpt-5.4-mini": 272_000,
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
            timeout=10,
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


def _resolve_nous_context_length(model: str) -> Optional[int]:
    """Resolve Nous Portal model context length via OpenRouter metadata.

    Nous model IDs are bare (e.g. 'claude-opus-4-6') while OpenRouter uses
    prefixed IDs (e.g. 'anthropic/claude-opus-4.6'). Try suffix matching
    with version normalization (dot↔dash).
    """
    metadata = fetch_model_metadata()  # OpenRouter cache
    # Exact match first
    if model in metadata:
        return metadata[model].get("context_length")

    normalized = _normalize_model_version(model).lower()

    for or_id, entry in metadata.items():
        bare = or_id.split("/", 1)[1] if "/" in or_id else or_id
        if bare.lower() == model.lower() or _normalize_model_version(bare).lower() == normalized:
            return entry.get("context_length")

    # Partial prefix match for cases like gemini-3-flash → gemini-3-flash-preview
    # Require match to be at a word boundary (followed by -, :, or end of string)
    model_lower = model.lower()
    for or_id, entry in metadata.items():
        bare = or_id.split("/", 1)[1] if "/" in or_id else or_id
        for candidate, query in [(bare.lower(), model_lower), (_normalize_model_version(bare).lower(), normalized)]:
            if candidate.startswith(query) and (
                len(candidate) == len(query) or candidate[len(query)] in "-:."
            ):
                return entry.get("context_length")

    return None


def get_model_context_length(
    model: str,
    base_url: str = "",
    api_key: str = "",
    config_context_length: int | None = None,
    provider: str = "",
) -> int:
    """Get the context length for a model.

    Resolution order:
    0. Explicit config override (model.context_length or custom_providers per-model)
    1. Persistent cache (previously discovered via probing)
    2. Active endpoint metadata (/models for explicit custom endpoints)
    3. Local server query (for local endpoints)
    4. Anthropic /v1/models API (API-key users only, not OAuth)
    5. OpenRouter live API metadata
    6. Nous suffix-match via OpenRouter cache
    7. models.dev registry lookup (provider-aware)
    8. Thin hardcoded defaults (broad family patterns)
    9. Default fallback (128K)
    """
    # 0. Explicit config override — user knows best
    if config_context_length is not None and isinstance(config_context_length, int) and config_context_length > 0:
        return config_context_length

    # Normalise provider-prefixed model names (e.g. "local:model-name" →
    # "model-name") so cache lookups and server queries use the bare ID that
    # local servers actually know about.  Ollama "model:tag" colons are preserved.
    model = _strip_provider_prefix(model)

    # 1. Check persistent cache (model+provider)
    if base_url:
        cached = get_cached_context_length(model, base_url)
        if cached is not None:
            return cached

    # 2. Active endpoint metadata for truly custom/unknown endpoints.
    # Known providers (Copilot, OpenAI, Anthropic, etc.) skip this — their
    # /models endpoint may report a provider-imposed limit (e.g. Copilot
    # returns 128k) instead of the model's full context (400k).  models.dev
    # has the correct per-provider values and is checked at step 5+.
    if _is_custom_endpoint(base_url) and not _is_known_provider_base_url(base_url):
        endpoint_metadata = fetch_endpoint_model_metadata(base_url, api_key=api_key)
        matched = endpoint_metadata.get(model)
        if not matched:
            # Single-model servers: if only one model is loaded, use it
            if len(endpoint_metadata) == 1:
                matched = next(iter(endpoint_metadata.values()))
            else:
                # Fuzzy match: substring in either direction
                for key, entry in endpoint_metadata.items():
                    if model in key or key in model:
                        matched = entry
                        break
        if matched:
            context_length = matched.get("context_length")
            if isinstance(context_length, int):
                return context_length
        if not _is_known_provider_base_url(base_url):
            # 3. Try querying local server directly
            if is_local_endpoint(base_url):
                local_ctx = _query_local_context_length(model, base_url, api_key=api_key)
                if local_ctx and local_ctx > 0:
                    save_context_length(model, base_url, local_ctx)
                    return local_ctx
            logger.info(
                "Could not detect context length for model %r at %s — "
                "defaulting to %s tokens (probe-down). Set model.context_length "
                "in config.yaml to override.",
                model, base_url, f"{DEFAULT_FALLBACK_CONTEXT:,}",
            )
            return DEFAULT_FALLBACK_CONTEXT

    # 4. Anthropic /v1/models API (only for regular API keys, not OAuth)
    if provider == "anthropic" or (
        base_url and base_url_hostname(base_url) == "api.anthropic.com"
    ):
        ctx = _query_anthropic_context_length(model, base_url or "https://api.anthropic.com", api_key)
        if ctx:
            return ctx

    # 4b. AWS Bedrock — use static context length table.
    # Bedrock's ListFoundationModels doesn't expose context window sizes,
    # so we maintain a curated table in bedrock_adapter.py.
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

    # 5. Provider-aware lookups (before generic OpenRouter cache)
    # These are provider-specific and take priority over the generic OR cache,
    # since the same model can have different context limits per provider
    # (e.g. claude-opus-4.6 is 1M on Anthropic but 128K on GitHub Copilot).
    # If provider is generic (openrouter/custom/empty), try to infer from URL.
    effective_provider = provider
    if not effective_provider or effective_provider in ("openrouter", "custom"):
        if base_url:
            inferred = _infer_provider_from_url(base_url)
            if inferred:
                effective_provider = inferred

    if effective_provider == "nous":
        ctx = _resolve_nous_context_length(model)
        if ctx:
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
    if effective_provider:
        from agent.models_dev import lookup_models_dev_context
        ctx = lookup_models_dev_context(effective_provider, model)
        if ctx:
            return ctx

    # 6. OpenRouter live API metadata (provider-unaware fallback)
    metadata = fetch_model_metadata()
    if model in metadata:
        return metadata[model].get("context_length", 128000)

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

    # 9. Query local server as last resort
    if base_url and is_local_endpoint(base_url):
        local_ctx = _query_local_context_length(model, base_url, api_key=api_key)
        if local_ctx and local_ctx > 0:
            save_context_length(model, base_url, local_ctx)
            return local_ctx

    # 10. Default fallback — 128K
    return DEFAULT_FALLBACK_CONTEXT


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
    """Rough token estimate for a message list (pre-flight only)."""
    total_chars = sum(len(str(msg)) for msg in messages)
    return (total_chars + 3) // 4


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
    blind spot when only counting messages.
    """
    total_chars = 0
    if system_prompt:
        total_chars += len(system_prompt)
    if messages:
        total_chars += sum(len(str(msg)) for msg in messages)
    if tools:
        total_chars += len(str(tools))
    return (total_chars + 3) // 4

"""Provider/model catalogs: discovery, caching, and identity helpers.

Origin module; cohesive clusters live in siblings and are re-imported here so
``hermes_cli.models.<name>`` stays the stable import/monkeypatch surface:
``models_catalog_static`` (curated tables, provider registry, aliases), ``models_reasoning_caps``,
``models_local`` (Ollama / LM Studio), ``models_pricing``, ``models_validate``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import urllib.parse
import urllib.request
import urllib.error
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeGuard

from hermes_cli import __version__ as _HERMES_VERSION
from hermes_cli.urllib_security import open_credentialed_url
from hermes_cli.models_catalog_static import (
    CANONICAL_PROVIDERS,
    OPENROUTER_MODELS,
    PREFERRED_SILENT_DEFAULT_MODEL,
    VERCEL_AI_GATEWAY_MODELS,
    _AGGREGATOR_PROVIDERS,
    _AZURE_FOUNDRY_RESPONSES_PREFIXES,
    _BORROWED_MODEL_PROVIDERS,
    _COPILOT_MODEL_ALIASES,
    _KEYLESS_STABLE_CACHE_PROVIDERS,
    _LIVE_FIRST_PICKER_PROVIDERS,
    _MODELS_DEV_PREFERRED,
    _OPENAI_FAST_MODE_PREFIXES,
    _OPENROUTER_VARIANT_SUFFIXES,
    _PROVIDER_ALIASES,
    _PROVIDER_LABELS,
    _PROVIDER_MODELS,
    _PROVIDER_RETIRED_ALIASES,
    _SILENT_DEFAULT_PROVIDERS,
    _xai_finalize_catalog)
from hermes_cli.models_reasoning_caps import (
    _OPENROUTER_CATALOG_URL,
    _seed_reasoning_caps)
from hermes_cli.models_local import (
    _OLLAMA_LOCAL_MODELS_CACHE,
    _OLLAMA_LOCAL_MODELS_CACHE_TTL,
    _OLLAMA_LOCAL_PROBE_FAILURE_CACHE,
    _OLLAMA_LOCAL_PROBE_REACHABLE,
    _get_ollama_base_url,
    _get_ollama_native_headers,
    _ollama_local_catalog,
    _ollama_probe_cache_key,
    _root_for_ollama_native_api,
    fetch_ollama_cloud_models)

logger = logging.getLogger(__name__)

# Identify ourselves so endpoints fronted by Cloudflare's Browser Integrity
# Check (error 1010) don't reject the default ``Python-urllib/*`` signature.
_HERMES_USER_AGENT = f"hermes-cli/{_HERMES_VERSION}"

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"
COPILOT_EDITOR_VERSION = "vscode/1.104.1"
COPILOT_REASONING_EFFORTS_GPT5 = ["minimal", "low", "medium", "high"]
COPILOT_REASONING_EFFORTS_O_SERIES = ["low", "medium", "high"]

def _urlopen_model_catalog_request(req: urllib.request.Request, *, timeout: float, ssl_context=None):
    """Open catalog requests without forwarding headers across origins."""
    return open_credentialed_url(req, timeout=timeout, ssl_context=ssl_context)


def _get_json(
    url: str, *, timeout: float, headers: Optional[dict[str, str]] = None, opener=None, **open_kwargs: Any
) -> Any:
    """GET ``url`` and parse the JSON body. ``opener`` defaults to the catalog opener (resolved at
    call time so monkeypatching ``_urlopen_model_catalog_request`` still applies). Raises on failure."""
    req = urllib.request.Request(url, headers=headers or {})
    with (opener or _urlopen_model_catalog_request)(req, timeout=timeout, **open_kwargs) as resp:
        return json.loads(resp.read().decode())


def _read_json_cache(path: Path, *, errors=Exception) -> Optional[dict]:
    """Load a JSON-object cache file; None when missing, unreadable, or not a dict."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except errors:
        return None
    return data if isinstance(data, dict) else None


def _write_json_cache(path: Path, data: Any, **dump_kwargs: Any) -> None:
    """Atomically persist a cache file (creating parents). Raises on failure — callers decide
    whether a failed cache write is worth logging."""
    from utils import atomic_json_write

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, data, **dump_kwargs)


def _merge_unique(primary: list[str], secondary: list[str], key=lambda m: str(m).lower()) -> list[str]:
    """``primary`` verbatim, then ``secondary`` entries whose ``key`` is new (deduped as it goes)."""
    merged, seen = list(primary), {key(m) for m in primary}
    for m in secondary:
        k = key(m)
        if k not in seen:
            seen.add(k)
            merged.append(m)
    return merged


def _custom_provider_ssl_context(base_url: str):
    """``ssl.SSLContext`` honoring a custom provider's ``ssl_ca_cert`` / ``ssl_verify`` (mirrors the
    httpx TLS resolution), or None so the urllib ``/models`` probe keeps the default policy."""
    if not base_url:
        return None
    try:
        from hermes_cli.config import get_custom_provider_tls_settings

        tls = get_custom_provider_tls_settings(base_url)
        if not tls:
            return None
        import ssl

        if tls.get("ssl_verify") is False:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        ca = tls.get("ssl_ca_cert")
        if isinstance(ca, str) and ca and os.path.isfile(ca):
            return ssl.create_default_context(cafile=ca)
    except Exception:
        return None  # never break discovery on a TLS-config lookup
    return None


# Process-lifetime picker lists refreshed from the live catalogs (see fetch_*_models).
    ("google/gemini-3-pro-preview",            ""),
    ("qwen/qwen3.7-max",                       ""),
    ("qwen/qwen3.7-plus",                      ""),
    ("qwen/qwen3.6-35b-a3b",                   ""),
_openrouter_catalog_cache: list[tuple[str, str]] | None = None
_ai_gateway_catalog_cache: list[tuple[str, str]] | None = None


def _codex_curated_models() -> list[str]:
    """Derive the openai-codex curated list from codex_models.py.

    Single source of truth: DEFAULT_CODEX_MODELS + forward-compat synthesis.
    This keeps the gateway /model picker in sync with the CLI `hermes model`
    flow without maintaining a separate static list.
    """
    from hermes_cli.codex_models import DEFAULT_CODEX_MODELS, _finalize_codex_models
    return _finalize_codex_models(list(DEFAULT_CODEX_MODELS))


# Static fallback for xAI when the models.dev disk cache is empty (fresh
# install, offline first run, etc.). Mirrors the xAI-direct model IDs from
# $HERMES_HOME/models_dev_cache.json as of 2026-04-28. Whenever xAI renames
# or retires a model, the disk cache picks it up on the next refresh and the
# fallback here only matters until that refresh lands.
#
# Models retired by xAI on May 15, 2026 are excluded — see
# https://docs.x.ai/developers/migration/may-15-retirement
# (grok-4, grok-4-0709, grok-4-fast{,-reasoning,-non-reasoning},
#  grok-4-1-fast{,-reasoning,-non-reasoning}, grok-code-fast-1 → grok-4.3).
_XAI_STATIC_FALLBACK: list[str] = [
    "grok-4.6",
    "grok-build-0.1",
    "grok-4.5",
    "grok-4.3",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-0309",
]

# Callable via xAI OAuth but omitted from models.dev and /v1/models listings.
_XAI_CURATED_EXTRAS: list[str] = [
    "grok-4.6",  # GA 2026-08 — kept until the models.dev disk cache refreshes
    "grok-4.5",  # GA 2026-07 — kept until the models.dev disk cache refreshes
    "grok-composer-2.5-fast",
]


_XAI_TOP_MODEL = "grok-4.6"


def _xai_promote_top(ids: list[str]) -> list[str]:
    """Pin the headline xAI model to the top of the curated list."""
    if _XAI_TOP_MODEL in ids:
        return [_XAI_TOP_MODEL] + [m for m in ids if m != _XAI_TOP_MODEL]
    return ids


def _xai_merge_curated_extras(ids: list[str]) -> list[str]:
    """Append Hermes-curated xAI models that are missing from models.dev."""
    out = list(ids)
    for extra in _XAI_CURATED_EXTRAS:
        if extra in out:
            continue
        # Keep the headline model pinned; slot extras immediately after it.
        insert_at = 1 if out and out[0] == _XAI_TOP_MODEL else len(out)
        out.insert(insert_at, extra)
    return out


def _xai_finalize_catalog(ids: list[str]) -> list[str]:
    return _xai_promote_top(_xai_merge_curated_extras(ids))


def _xai_curated_models() -> list[str]:
    """Offline curated floor for xAI / xAI OAuth pickers.

    Reads $HERMES_HOME/models_dev_cache.json directly (no network). Falls
    back to ``_XAI_STATIC_FALLBACK`` when the cache is empty or unreadable.
    """
    try:
        from agent.models_dev import _load_disk_cache
        data = _load_disk_cache()
        xai = data.get("xai") if isinstance(data, dict) else None
        models = xai.get("models") if isinstance(xai, dict) else None
        if isinstance(models, dict) and models:
            ids = [mid for mid in models.keys() if isinstance(mid, str)]
            if ids:
                return _xai_finalize_catalog(sorted(ids))
    except Exception:
        # Any failure (missing file, malformed JSON, import error)
        # falls through to the static list.
        pass
    return _xai_finalize_catalog(list(_XAI_STATIC_FALLBACK))


_PROVIDER_MODELS: dict[str, list[str]] = {
    "moa": ["default"],
    "nous": [
        # Anthropic
        "anthropic/claude-fable-5.1",
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-haiku-4.5",
        # OpenAI
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-sol-pro",
        "openai/gpt-5.6-terra",
        "openai/gpt-5.6-terra-pro",
        "openai/gpt-5.6-luna",
        "openai/gpt-5.6-luna-pro",
        "openai/gpt-5.5",
        "openai/gpt-5.5-pro",
        "openai/gpt-5.4-mini",
        # Google
        "google/gemini-3-pro-preview",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.8-flash",
        "google/gemini-3.7-flash",
        # xAI
        "x-ai/grok-4.6",
        # DeepSeek
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-pro-0813",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-flash-0731",
        # Qwen
        "qwen/qwen3.8-max",
        "qwen/qwen3.8-flash",
        "qwen/qwen3.7-max",
        "qwen/qwen3.7-plus",
        "qwen/qwen3.6-35b-a3b",
        # MoonshotAI
        "moonshotai/kimi-k3",
        # MiniMax
        "minimax/minimax-m3",
        # Z-AI
        "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash",
        "z-ai/glm-5.2",
        # Xiaomi
        "xiaomi/mimo-v2.5-pro",
        # Tencent
        "tencent/hy4-preview",
        "tencent/hy3",
        # StepFun
        "stepfun/step-3.7-flash",
        # NVIDIA
        "nvidia/nemotron-3-super-120b-a12b",
        # Sakana
        "sakana/fugu-ultra",
    ],
    # Native OpenAI Chat Completions (api.openai.com). Used by /model counts and
    # provider_model_ids fallback when /v1/models is unavailable.
    "openai": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-api": [
        "gpt-5.6-sol",
        "gpt-5.6-sol-pro",
        "gpt-5.6-terra",
        "gpt-5.6-terra-pro",
        "gpt-5.6-luna",
        "gpt-5.6-luna-pro",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-codex": _codex_curated_models(),
    "xai-oauth": _xai_curated_models(),
    "copilot-acp": [
        "copilot-acp",
    ],
    "copilot": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet-4.6",
        "claude-sonnet-5",
        "claude-sonnet-4",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
    ],
    "gemini": [
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite-preview",
    ],
    "zai": [
        "glm-5.3",
        "glm-5.3-flash",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "glm-5v-turbo",
        "glm-5-turbo",
        "glm-4.7",
        "glm-4.5",
        "glm-4.5-flash",
    ],
    "xai": _xai_curated_models(),
    "nvidia": [
        # NVIDIA flagship reasoning models
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        # Third-party agentic models hosted on build.nvidia.com
        # (map to OpenRouter defaults — users get familiar picks on NIM)
        "z-ai/glm-5.3",
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "minimaxai/minimax-m3",
    ],
    "kimi-coding": [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-for-coding",
        "kimi-for-coding-highspeed",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "kimi-coding-cn": [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "stepfun": [
        "step-3.5-flash",
        "step-3.5-flash-2603",
    ],
    "moonshot": [
        "kimi-k3",
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "minimax": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ],
    "minimax-oauth": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
    ],
    "minimax-cn": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ],
    "anthropic": [
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
    ],
    "deepseek": [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ],
    "xiaomi": [
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "mimo-v2-pro",
        "mimo-v2-omni",
        "mimo-v2-flash",
    ],
    "tencent-tokenhub": [
        "hy4-preview",
        "hy3",
        "hy3-preview",
    ],
    "tencent-tokenplan": [
        "hy4-preview",
        "hy3",
        "hy3-preview",
    ],
    "arcee": [
        "trinity-large-thinking",
        "trinity-large-preview",
        "trinity-mini",
    ],
    "gmi": [
        "zai-org/GLM-5.1-FP8",
        "deepseek-ai/DeepSeek-V3.2",
        "moonshotai/Kimi-K2.5",
        "google/gemini-3.1-flash-lite-preview",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
    ],
    # Synced against https://opencode.ai/docs/zen/ + live GET /zen/v1/models
    # (2026-08-20). Zen/Go are _LIVE_FIRST_PICKER_PROVIDERS, so this list is a
    # discovery floor — live entries lead in the picker and stale curated
    # names never pollute the top.
    "opencode-zen": [
        "x-preview-f-free",  # "Ox Alpha" stealth model — free, 1M ctx, ZDR
        "kimi-k3",
        "kimi-k2.5",
        "kimi-k2.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5",
        "gpt-5-codex",
        "gpt-5-nano",
        "claude-fable-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro",
        "gemini-3-flash",
        "grok-4.6",
        "grok-4.5",
        "grok-build-0.1",
        "muse-spark-1.2",
        "minimax-m3",
        "minimax-m2.7",
        "minimax-m2.5",
        "glm-5.3",
        "glm-5.3-flash",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "kimi-k2.7-code",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-flash-free",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "big-pickle",
        "mimo-v2.5-free",
        "hy3-free",
        "laguna-s-2.1-free",
        "nemotron-3-ultra-free",
        "nemotron-3.5-lightning-free",
        "muse-spark-1.2-contributor-free",
        "muse-spark-1.3-contributor-free",
    ],
    # OpenCode free tier — keyless (no OpenCode account needed). This is the
    # OFFLINE FLOOR only: provider_model_ids("opencode-free") revalidates live
    # against GET /zen/v1/models (keyless) and filters to the anonymous free
    # tier, so a relay-delisted model stops appearing in the picker and a
    # newly-live one becomes selectable without a release. This floor keeps the
    # picker populated when the relay is unreachable. Note: this floor may lag
    # the live relay — that is intentional; the live revalidation is the
    # source of truth when reachable. Known-delisted models are REMOVED from
    # the floor (x-preview-f-free delisted 2026-08-26 — offline fallback must
    # not offer a model that 401s). deepseek-v4-flash-free and mimo-v2.5-free
    # are back on the live list.
    "opencode-free": [
        "deepseek-v4-flash-free",
        "hy3-free",
        "mimo-v2.5-free",
        "laguna-s-2.1-free",
        "nemotron-3-ultra-free",
        "nemotron-3.5-lightning-free",
        "muse-spark-1.2-contributor-free",
        "muse-spark-1.3-contributor-free",
    ],
    # Synced against https://opencode.ai/docs/go/ + live GET /zen/go/v1/models
    # (2026-08-20).
    "opencode-go": [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "gpt-5.6-luna",
        "grok-4.5",
        "glm-5.3",
        "glm-5.3-flash",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "mimo-v2-pro",
        "mimo-v2-omni",
        "minimax-m3",
        "minimax-m2.7",
        "minimax-m2.5",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "qwen3.8-max",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "hy3",
        "hy3-preview",
        "muse-spark-1.2-contributor",
        "muse-spark-1.3-contributor",
        # Go-subscription twin of the Zen keyless Ox Alpha (live go/v1
        # catalog 2026-08-21; NOT keyless — Go relay requires a Go key).
        "ox-alpha-free",
    ],
    "kilocode": [
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
    ],
    # Alibaba DashScope Coding platform (coding-intl) — default endpoint.
    # Supports Qwen models + third-party providers (GLM, Kimi, MiniMax).
    # Users with classic DashScope keys should override DASHSCOPE_BASE_URL
    # to https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (OpenAI-compat)
    # or https://dashscope-intl.aliyuncs.com/apps/anthropic (Anthropic-compat).
    "alibaba": [
        # Qwen 千问系列 (DashScope / Qwen Cloud)
        "qwen3.8-max",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "kimi-k2.5",
        "qwen3.5-plus",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        # Third-party models available on coding-intl / DashScope
        "glm-5.2",
        "glm-5",
        "glm-4.7",
        "deepseek-v4-pro",
        "deepseek-v4-flash-0731",
        "MiniMax-M2.5",
    ],
    # Alibaba DashScope (China) — same platform as alibaba, domestic endpoint
    # (dashscope.aliyuncs.com); same catalog as the international tier.
    "alibaba-cn": [
        "qwen3.8-max",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "kimi-k2.5",
        "qwen3.5-plus",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        "glm-5.2",
        "glm-5",
        "glm-4.7",
        "deepseek-v4-pro",
        "deepseek-v4-flash-0731",
        "MiniMax-M2.5",
    ],
    # Alibaba Coding Plan — same platform as alibaba (DashScope coding-intl),
    # separate provider ID with its own base_url_env_var.
    "alibaba-coding-plan": [
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "qwen3-max-2026-01-23",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        "kimi-k2.5",
        "glm-5",
        "glm-4.7",
        "MiniMax-M2.5",
    ],
    # Alibaba Coding Plan (China) — domestic coding endpoint
    # (coding.dashscope.aliyuncs.com); same catalog as the international tier.
    "alibaba-coding-plan-cn": [
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "qwen3-max-2026-01-23",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        "kimi-k2.5",
        "glm-5",
        "glm-4.7",
        "MiniMax-M2.5",
    ],
    # Alibaba Token Plan (Personal Edition) — dedicated token-plan endpoint
    # (token-plan.ap-southeast-1.maas.aliyuncs.com), key tier `sk-sp-...`.
    # Catalog verified against a live Token Plan subscription (2026-08-03).
    "alibaba-token-plan": [
        "qwen3.8-max-preview",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v3.2",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
    ],
    # Alibaba Token Plan (China) — domestic token-plan endpoint
    # (token-plan.cn-beijing.maas.aliyuncs.com); same catalog as intl.
    "alibaba-token-plan-cn": [
        "qwen3.8-max-preview",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v3.2",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
    ],
    # Curated HF model list — only agentic models that map to OpenRouter defaults.
    "huggingface": [
        "moonshotai/Kimi-K2.5",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.5-35B-A3B",
        "deepseek-ai/DeepSeek-V3.2",
        "MiniMaxAI/MiniMax-M2.5",
        "zai-org/GLM-5",
        "XiaomiMiMo/MiMo-V2-Flash",
        "moonshotai/Kimi-K2-Thinking",
        "moonshotai/Kimi-K2.6",
    ],
    # AWS Bedrock — static fallback list used when dynamic discovery is
    # unavailable (no boto3, no credentials, or API error).  The agent
    # prefers live discovery via ListFoundationModels + ListInferenceProfiles.
    # Use inference profile IDs (us.*) since most models require them.
    "bedrock": [
        "us.anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "openai.gpt-5.5",
        "openai.gpt-5.6-sol",
        "openai.gpt-5.6-terra",
        "openai.gpt-5.6-luna",
        "us.amazon.nova-pro-v1:0",
        "us.amazon.nova-lite-v1:0",
        "us.amazon.nova-micro-v1:0",
        "deepseek.v3.2",
        "us.meta.llama4-maverick-17b-instruct-v1:0",
        "us.meta.llama4-scout-17b-instruct-v1:0",
    ],
    # Azure Foundry: user-provided endpoint and model.
    # Empty list because models depend on the endpoint configuration.
    "azure-foundry": [],
    # Google Vertex AI — static curated list.  Vertex's OpenAI-compatible
    # endpoint has no /models listing route, so without this entry the
    # /model picker only ever shows the currently-configured model.
    # Model IDs use the "google/" publisher prefix Vertex's openapi
    # endpoint expects (see hermes_cli/model_setup_flows.py).
    "vertex": [
        "google/gemini-3.1-pro-preview",
        "google/gemini-3-pro-preview",
        "google/gemini-3.5-flash",
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite-preview",
    ],
    "novita": [
        "moonshotai/kimi-k2.5",
        "minimax/minimax-m2.7",
        "zai-org/glm-5",
        "deepseek/deepseek-v3-0324",
        "deepseek/deepseek-r1-0528",
        "qwen/qwen3-235b-a22b-fp8",
    ],
}

# Vercel AI Gateway: derive the bare-model-id catalog from the curated
# ``VERCEL_AI_GATEWAY_MODELS`` snapshot so both the picker (tuples with descriptions)
# and the static fallback catalog (bare ids) stay in sync from a single
# source of truth.
_PROVIDER_MODELS["ai-gateway"] = [mid for mid, _ in VERCEL_AI_GATEWAY_MODELS]

# ---------------------------------------------------------------------------
# Nous Portal free-model helpers — the Portal models endpoint is the source of truth for what is
# offered (free or paid); we surface it as-is, no local allowlist filtering.
# ---------------------------------------------------------------------------


def _zero_priced(pricing: Any, keys: tuple[str, str], default: str) -> bool:
    """True when both pricing fields parse to 0 (missing fields read as ``default``)."""
    if not isinstance(pricing, dict):
        return False
    try:
        return all(float(pricing.get(k, default)) == 0 for k in keys)
    except (TypeError, ValueError):
        return False


def _is_model_free(model_id: str, pricing: dict[str, dict[str, str]]) -> bool:
    """Return True if *model_id* has zero-cost prompt AND completion pricing."""
    return bool(pricing.get(model_id)) and _zero_priced(pricing.get(model_id), ("prompt", "completion"), "1")


def partition_nous_models_by_tier(
    model_ids: list[str], pricing: dict[str, dict[str, str]], free_tier: bool
) -> tuple[list[str], list[str]]:
    """Split Nous models into (selectable, unavailable): free-tier users may only select free models
    (paid ones are returned as unavailable, shown grayed out)."""
    if not free_tier or not pricing:  # no pricing → can't determine, show everything
        return (model_ids, [])
    selectable = [mid for mid in model_ids if _is_model_free(mid, pricing)]
    return (selectable, [mid for mid in model_ids if mid not in selectable])


def _union_with_portal_recommendations(
    tier_key: str, curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str,
    *, force_refresh: bool, synthesize_free_pricing: bool,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Append the Portal's ``<tier_key>`` recommendations missing from ``curated_ids``.

    Curated models show first, Portal-only picks follow. Failures (network, parse, missing field)
    silently return the inputs unchanged — never block the picker on a Portal-side hiccup.
    """
    try:
        payload = fetch_nous_recommended_models(portal_base_url, force_refresh=force_refresh)
    except Exception:
        payload = None
    block = payload.get(tier_key) if isinstance(payload, dict) else None
    entries = block if isinstance(block, list) else []
    portal_ids = [name for entry in entries if (name := _extract_model_name(entry))]
    if not portal_ids:
        return (list(curated_ids), dict(pricing))

    augmented_pricing = dict(pricing)
    if synthesize_free_pricing:
        for mid in portal_ids:
            augmented_pricing.setdefault(mid, {"prompt": "0", "completion": "0"})
    seen = set(curated_ids)
    return (list(curated_ids) + [mid for mid in portal_ids if mid not in seen], augmented_pricing)


def union_with_portal_free_recommendations(
    curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str = "", *,
    force_refresh: bool = False) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Curated list + pricing plus the Portal's ``freeRecommendedModels``; Portal-only free picks get a
    synthetic $0 pricing entry so tier partitioning sees them as free."""
    return _union_with_portal_recommendations(
        "freeRecommendedModels", curated_ids, pricing, portal_base_url,
        force_refresh=force_refresh, synthesize_free_pricing=True)


def union_with_portal_paid_recommendations(
    curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str = "", *,
    force_refresh: bool = False) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Curated list plus the Portal's ``paidRecommendedModels``; ``pricing`` is deliberately left untouched."""
    return _union_with_portal_recommendations(
        "paidRecommendedModels", curated_ids, pricing, portal_base_url,
        force_refresh=force_refresh, synthesize_free_pricing=False)


# Free-tier detection cache, per profile — short so an account upgrade shows within minutes.
_FREE_TIER_CACHE_TTL: int = 180  # seconds
_free_tier_cache: dict[str, tuple[bool, float]] = {}  # profile key -> (result, timestamp)


def _pricing_profile_key() -> str:
    """Stable profile identity for process-local pricing caches."""
    from hermes_constants import hermes_home_key

    return hermes_home_key()


def get_cached_nous_free_tier() -> Optional[bool]:
    """This profile's live cached entitlement, or ``None`` if unknown/expired."""
    cached = _free_tier_cache.get(_pricing_profile_key())
    if cached is None or time.monotonic() - cached[1] >= _FREE_TIER_CACHE_TTL:
        return None
    return cached[0]


def check_nous_free_tier(*, force_fresh: bool = False, cached_only: bool = False) -> bool:
    """True only when the Nous Portal user is KNOWN to be free-tier (unknown/error → False so this
    never blocks users). Cached ``_FREE_TIER_CACHE_TTL`` seconds so an upgrade shows within minutes.
    ``cached_only`` returns the live cached answer or the fail-open ``False`` without contacting Portal."""
    now = time.monotonic()
    profile_key = _pricing_profile_key()
    if not force_fresh:
        cached_result = get_cached_nous_free_tier()
        if cached_result is not None:
            return cached_result
    if cached_only:
        return False
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        result = get_nous_portal_account_info(force_fresh=force_fresh).is_free_tier
    except Exception:
        result = False  # default to paid on error — don't block users
    _free_tier_cache[profile_key] = (result, now)
    return result


# ---------------------------------------------------------------------------
# Nous Portal recommended models — curated paid/free suggestions plus dedicated compaction (aux)
# and vision picks, TTL-cached per process. Fields read: {paid,free}RecommendedModels:
# [{modelName}], {paid,free}Recommended{Compaction,Vision}Model: {modelName} | null
# ---------------------------------------------------------------------------

NOUS_RECOMMENDED_MODELS_PATH = "/api/nous/recommended-models"
_NOUS_RECOMMENDED_CACHE_TTL: int = 600  # seconds (10 minutes)
# (result_dict, timestamp) keyed by portal_base_url so staging vs prod don't collide.
_nous_recommended_cache: dict[str, tuple[dict[str, Any], float]] = {}


def _nous_recommended_disk_path() -> "Path":
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "nous_recommended_cache.json"


def _read_nous_recommended_disk(base: str) -> dict[str, Any] | None:
    """Last-known-good payload for ``base`` from the per-base disk map
    ``{"<base>": {"data": {...}, "ts": <epoch>}}`` (staging and prod don't collide), or None."""
    blob = _read_json_cache(_nous_recommended_disk_path(), errors=(OSError, json.JSONDecodeError))
    entry = (blob or {}).get(base)
    data = entry.get("data") if isinstance(entry, dict) else None
    return data if isinstance(data, dict) and data else None


def _write_nous_recommended_disk(base: str, data: dict[str, Any]) -> None:
    """Merge ``data`` into the per-base disk map atomically; failures are debug-logged (the in-process
    cache still works)."""
    if not data:
        return
    path = _nous_recommended_disk_path()
    try:
        blob = _read_json_cache(path, errors=(OSError, json.JSONDecodeError)) or {}
        blob[base] = {"data": data, "ts": time.time()}
        _write_json_cache(path, blob, indent=2)
    except OSError as exc:
        logger.debug("nous recommended-models disk cache write failed: %s", exc)


def fetch_nous_recommended_models(
    portal_base_url: str = "", timeout: float = 5.0, *, force_refresh: bool = False
) -> dict[str, Any]:
    """Fetch the Portal's public ``/api/nous/recommended-models`` payload (no auth).

    Cached per portal URL for ``_NOUS_RECOMMENDED_CACHE_TTL`` seconds in process (``force_refresh``
    bypasses); a successful fetch is also persisted as last-known-good on disk, which serves a live
    failure so a transient Portal hiccup doesn't drop the recommendations.
    """
    base = (portal_base_url or "https://portal.nousresearch.com").rstrip("/")
    now = time.monotonic()
    cached = _nous_recommended_cache.get(base)
    if not force_refresh and cached is not None and now - cached[1] < _NOUS_RECOMMENDED_CACHE_TTL:
        return cached[0]
    try:
        data = _get_json(
            f"{base}{NOUS_RECOMMENDED_MODELS_PATH}", timeout=timeout, headers={"Accept": "application/json"}
        )
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    if data:
        _write_nous_recommended_disk(base, data)
    else:
        data = _read_nous_recommended_disk(base) or data
    _nous_recommended_cache[base] = (data, now)
    return data


def _resolve_nous_portal_url() -> str:
    """Best-effort lookup of the Portal base URL the user is authed against."""
    try:
        from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL, get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
        portal = str(state.get("portal_base_url") or "").strip()
        return (portal or str(DEFAULT_NOUS_PORTAL_URL)).rstrip("/")
    except Exception:
        return "https://portal.nousresearch.com"


def _extract_model_name(entry: Any) -> Optional[str]:
    """Pull the ``modelName`` field from a recommended-model entry, else None."""
    model_name = entry.get("modelName") if isinstance(entry, dict) else None
    return model_name.strip() if isinstance(model_name, str) and model_name.strip() else None


def get_nous_recommended_aux_model(
    *, vision: bool = False, free_tier: Optional[bool] = None, portal_base_url: str = "",
    force_refresh: bool = False) -> Optional[str]:
    """The Portal's recommended model for an auxiliary task: free tier → free pick only; paid tier →
    paid pick, falling back to the free one when the Portal returned ``null`` (staged rollouts)."""
    base = portal_base_url or _resolve_nous_portal_url()
    payload = fetch_nous_recommended_models(base, force_refresh=force_refresh)
    if not payload:
        return None
    if free_tier is None:
        try:
            free_tier = check_nous_free_tier()
        except Exception:
            free_tier = False  # assume paid on detection error — paid users see both fields anyway
    kind = "Vision" if vision else "Compaction"
    tiers = ("free",) if free_tier else ("paid", "free")
    return next((n for t in tiers if (n := _extract_model_name(payload.get(f"{t}Recommended{kind}Model")))), None)


def get_preferred_silent_default_model(provider: str = "openrouter") -> str:
    """Silent-default model id: the cached remote catalog's ``"default": true`` label (never hits the
    network — safe on hot paths), else :data:`PREFERRED_SILENT_DEFAULT_MODEL`."""
    try:
        from hermes_cli.model_catalog import get_default_model_from_cache
        labeled = get_default_model_from_cache(provider)
        if labeled:
            return labeled
    except Exception:
        pass
    return PREFERRED_SILENT_DEFAULT_MODEL


def pick_silent_default_model(model_ids: list[str], provider: str = "openrouter") -> str:
    """Catalog-labeled default when ``model_ids`` carries it, else the first entry, else "". Used by
    every surface that must choose a model without an interactive picker."""
    preferred = get_preferred_silent_default_model(provider)
    return preferred if preferred in model_ids else (model_ids[0] if model_ids else "")


def get_default_model_for_provider(provider: str) -> str:
    """Cost-safe default model for a provider, or "" — the NON-INTERACTIVE fallback when a provider
    is configured but no model was ever selected."""
    models = _PROVIDER_MODELS.get(provider, [])
    if provider in _SILENT_DEFAULT_PROVIDERS:
        preferred = get_preferred_silent_default_model(provider)
        # Trust the preferred default even without a static catalog (OpenRouter's picker list is
        # fetched live; its curated snapshot carries the default).
        if preferred and (preferred in models or not models):
            return preferred
    return models[0] if models else ""


def _openrouter_model_is_free(pricing: Any) -> bool:
    return _zero_priced(pricing, ("prompt", "completion"), "0")


def _openrouter_model_supports_tools(item: Any) -> bool:
    """True when ``supported_parameters`` advertises ``tools`` (hermes-agent is tool-calling-first).
    Permissive when the field is absent/malformed: some OpenRouter-compatible gateways (Nous Portal,
    private mirrors) don't populate it, and the picker must not silently empty for them.

    Ported from Kilo-Org/kilocode#9068.
    """
    params = item.get("supported_parameters") if isinstance(item, dict) else None
    return "tools" in params if isinstance(params, list) else True


# Reasoning-capability cache slots, one set per catalog (OpenRouter, Nous Portal). The logic
# lives in models_reasoning_caps and reads/writes these by name so tests can reset them here.
# ``*_cache``: model id → parsed caps for the process lifetime; ``*_failed_at``: monotonic time
# of the last failed fetch (60s re-fetch suppression); the flags are once-per-process guards.
_openrouter_reasoning_caps_cache: dict[str, Optional[dict[str, Any]]] | None = None
_openrouter_reasoning_caps_failed_at: float | None = None
_openrouter_caps_disk_checked = False
_openrouter_caps_warm_started = False
_nous_reasoning_caps_cache: dict[str, Optional[dict[str, Any]]] | None = None
_nous_reasoning_caps_failed_at: float | None = None
_nous_caps_disk_checked = False
_nous_caps_warm_started = False


from agent.reasoning_effort import clamp_effort as _clamp_effort


def clamp_reasoning_effort_to_supported(
    effort: Optional[str], supported_efforts: Optional[list[str]]) -> Optional[str]:
    """Thin wrapper over :func:`agent.reasoning_effort.clamp_effort`: keep a supported level verbatim,
    else the nearest WEAKER supported level (never silently escalate cost), else the weakest; unknown
    supported-sets and bespoke level names pass through unchanged."""
    return _clamp_effort(effort, supported_efforts)


def _fetch_live_catalog_index(url: str, timeout: float, opener) -> Optional[tuple[list, dict[str, dict[str, Any]]]]:
    """GET an OpenAI-style ``/models`` listing → ``(raw data array, {id: item})``, or None when the
    endpoint is unreachable or the payload has no ``data`` list."""
    try:
        payload = _get_json(url, timeout=timeout, headers={"Accept": "application/json"}, opener=opener)
    except Exception:
        return None
    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return None
    live_by_id = {
        mid: item for item in live_items if isinstance(item, dict) and (mid := str(item.get("id") or "").strip())
    }
    return live_items, live_by_id


def fetch_openrouter_models(
    timeout: float = 8.0, *, force_refresh: bool = False) -> list[tuple[str, str]]:
    """Return the curated OpenRouter picker list, refreshed from the live catalog when possible."""
    global _openrouter_catalog_cache

    if _openrouter_catalog_cache is not None and not force_refresh:
        return list(_openrouter_catalog_cache)

    # Remote catalog manifest first, in-repo snapshot when unreachable; the live /v1/models filter
    # (tool support, free pricing) is applied on top either way.
    try:
        from hermes_cli.model_catalog import get_curated_openrouter_models
        remote = get_curated_openrouter_models()
    except Exception:
        remote = None
    fallback = list(remote) if remote else list(OPENROUTER_MODELS)

    live = _fetch_live_catalog_index(_OPENROUTER_CATALOG_URL, timeout, _urlopen_model_catalog_request)
    if live is None:
        return list(_openrouter_catalog_cache or fallback)
    live_items, live_by_id = live

    # Free warm-up for the reasoning-capability cache: same payload the caps fetch would pull.
    global _openrouter_reasoning_caps_cache
    seeded = _seed_reasoning_caps(_OPENROUTER_CATALOG_URL, live_items)
    if _openrouter_reasoning_caps_cache is None and seeded is not None:
        _openrouter_reasoning_caps_cache = seeded

    curated: list[tuple[str, str]] = []
    silent_default = get_preferred_silent_default_model("openrouter")
    for preferred_id, _ in fallback:
        live_item = live_by_id.get(preferred_id)
        # Hide models without tool-calling support — selecting one fails at the first tool call.
        if live_item is None or not _openrouter_model_supports_tools(live_item):
            continue
        # Hide models that don't advertise tool-calling support — hermes-agent requires it and surfacing
        # them leads to immediate runtime failures when the user selects them. Ported from
        # Kilo-Org/kilocode#9068.
        if preferred_id == silent_default:
            desc = "default"  # keep the silent-default badge through the live refresh
        else:
            desc = "free" if _openrouter_model_is_free(live_item.get("pricing")) else ""
        curated.append((preferred_id, desc))

    if not curated:
        return list(_openrouter_catalog_cache or fallback)
    if not curated[0][1]:
        curated[0] = (curated[0][0], "recommended")
    _openrouter_catalog_cache = curated
    return list(curated)


def model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the OpenRouter model-id strings."""
    return [mid for mid, _ in fetch_openrouter_models(force_refresh=force_refresh)]


def get_curated_nous_model_ids() -> list[str]:
    """Curated Nous Portal model ids: the remote catalog manifest, else the in-repo
    ``_PROVIDER_MODELS["nous"]`` snapshot. Always a list."""
    try:
        from hermes_cli.model_catalog import get_curated_nous_models
        remote = get_curated_nous_models()
    except Exception:
        remote = None
    return list(remote or _PROVIDER_MODELS.get("nous", []))


def _ai_gateway_model_is_free(pricing: Any) -> bool:
    return _zero_priced(pricing, ("input", "output"), "0")


def fetch_ai_gateway_models(
    timeout: float = 8.0, *, force_refresh: bool = False) -> list[tuple[str, str]]:
    """Return the curated AI Gateway picker list, refreshed from the live catalog when possible."""
    global _ai_gateway_catalog_cache

    if _ai_gateway_catalog_cache is not None and not force_refresh:
        return list(_ai_gateway_catalog_cache)

    from hermes_constants import AI_GATEWAY_BASE_URL

    fallback = list(VERCEL_AI_GATEWAY_MODELS)
    live = _fetch_live_catalog_index(f"{AI_GATEWAY_BASE_URL.rstrip('/')}/models", timeout, urllib.request.urlopen)
    if live is None:
        return list(_ai_gateway_catalog_cache or fallback)
    _, live_by_id = live

    curated = [
        (pid, "free" if _ai_gateway_model_is_free(live_by_id[pid].get("pricing")) else "")
        for pid, _ in fallback if pid in live_by_id]
    if not curated:
        return list(_ai_gateway_catalog_cache or fallback)

    # A free Moonshot model in the live catalog is auto-promoted to #1 as "recommended".
    free_moonshot = next(
        (mid for mid, item in live_by_id.items()
         if mid.startswith("moonshotai/") and _ai_gateway_model_is_free(item.get("pricing"))),
        None)
    if free_moonshot:
        curated = [(free_moonshot, "recommended")] + [(mid, desc) for mid, desc in curated if mid != free_moonshot]
    else:
        curated[0] = (curated[0][0], "recommended")
    _ai_gateway_catalog_cache = curated
    return list(curated)


def ai_gateway_model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the AI Gateway model-id strings."""
    return [mid for mid, _ in fetch_ai_gateway_models(force_refresh=force_refresh)]


# ---------------------------------------------------------------------------
# Provider identity: ``provider:model`` parsing, auto-detection, labels
# ---------------------------------------------------------------------------

# All provider IDs and aliases valid on the left of the ``provider:model`` syntax.
_KNOWN_PROVIDER_NAMES: set[str] = set(_PROVIDER_LABELS) | set(_PROVIDER_ALIASES) | {"openrouter", "custom"}


_CONFIG_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError)


def _configured_custom_provider_ids() -> set[str]:
    """Return routable custom-provider IDs configured by the user."""
    ids = {"custom"}
    try:
        from hermes_cli.config import load_config
        from hermes_cli.providers import custom_provider_slug

        config = load_config()
        providers = config.get("providers", {})
        if isinstance(providers, dict):
            ids.update(custom_provider_slug(str(entry.get("name") or key), str(key))
                       for key, entry in providers.items() if isinstance(entry, dict))
        legacy = config.get("custom_providers", [])
        if isinstance(legacy, list):
            ids.update(
                custom_provider_slug(str(entry.get("name") or "")) for entry in legacy if isinstance(entry, dict))
    except _CONFIG_ERRORS:
        pass
    return ids


def _provider_has_credentials(pid: str) -> bool:
    try:
        from hermes_cli.auth import get_auth_status, has_usable_secret

        if pid == "custom":
            return bool((_get_custom_base_url() or "").strip())
        if pid == "openrouter":
            return has_usable_secret(os.getenv("OPENROUTER_API_KEY", ""))
        status = get_auth_status(pid)
        return bool(status.get("logged_in") or status.get("configured"))
    except Exception:
        return False


def list_available_providers() -> list[dict[str, str]]:
    """``{id, label, aliases, authenticated}`` for every provider usable with ``provider:model``,
    derived from :data:`CANONICAL_PROVIDERS` (shared with ``hermes model`` and ``/model``)."""
    aliases_for: dict[str, list[str]] = {}
    for alias, canonical in _PROVIDER_ALIASES.items():
        aliases_for.setdefault(canonical, []).append(alias)
    return [
        {
            "id": pid,
            "label": _PROVIDER_LABELS.get(pid, pid),
            "aliases": aliases_for.get(pid, []),
            "authenticated": _provider_has_credentials(pid)}
        for pid in [p.slug for p in CANONICAL_PROVIDERS] + ["custom"]]


def parse_model_input(raw: str, current_provider: str) -> tuple[str, str]:
    """Parse ``/model`` input into ``(provider, model)``. The colon is a provider delimiter only when
    the left side is a known provider/alias, so ``anthropic/claude-3.5-sonnet:beta`` stays a model."""
    stripped = raw.strip()
    colon = stripped.find(":")
    if colon > 0:
        provider_part = stripped[:colon].strip().lower()
        model_part = stripped[colon + 1:].strip()
        if provider_part and model_part and provider_part in _KNOWN_PROVIDER_NAMES:
            if provider_part == "custom":
                # Longest configured ``custom:<name>`` id that prefixes the input wins.
                lowered = stripped.lower()
                for custom_id in sorted(_configured_custom_provider_ids() - {"custom"}, key=len, reverse=True):
                    if lowered.startswith(f"{custom_id.lower()}:"):
                        return custom_id, stripped[len(custom_id) + 1 :].strip()
                # ``custom:local:qwen`` → ("custom:local", "qwen") for a configured named provider;
                # single-colon ``custom:qwen`` → ("custom", "qwen") as before.
                if ":" in model_part:
                    custom_name, actual_model = (part.strip() for part in model_part.split(":", 1))
                    if custom_name and actual_model:
                        if f"custom:{custom_name.lower()}" in _configured_custom_provider_ids():
                            return (f"custom:{custom_name.lower()}", actual_model)
                        return ("custom", model_part)
            return (normalize_provider(provider_part), model_part)
    return (current_provider, stripped)


def _get_custom_base_url() -> str:
    """The custom endpoint ``model.base_url`` from config.yaml."""
    return str(_get_model_config_dict().get("base_url", "")).strip()


def _get_provider_config_dict(provider: str) -> dict[str, Any]:
    """Return config.yaml providers.<provider>, or an empty dict."""
    key = str(provider or "").strip()
    if not key:
        return {}
    try:
        from hermes_cli.config import load_config
        providers_cfg = load_config().get("providers", {})
        if isinstance(providers_cfg, dict):
            entry = providers_cfg.get(key) or providers_cfg.get(key.lower())
            if isinstance(entry, dict):
                return entry
    except _CONFIG_ERRORS:
        pass
    return {}


def _get_model_config_dict() -> dict[str, Any]:
    """Return the main model config mapping, or an empty dict."""
    try:
        from hermes_cli.config import load_config
        model_cfg = load_config().get("model", {})
        if isinstance(model_cfg, dict):
            return model_cfg
    except Exception:
        pass
    return {}


def _base_url_looks_like_anthropic_messages(base_url: str) -> bool:
    normalized = str(base_url or "").strip().lower().rstrip("/")
    if not normalized:
        return False
    return urllib.parse.urlparse(normalized).path.rstrip("/").endswith(("/anthropic", "/anthropic/v1"))


def _anthropic_models_url(base_url: Optional[str] = None) -> str:
    endpoint = str(base_url or "https://api.anthropic.com").strip().rstrip("/")
    return endpoint + ("/models" if endpoint.endswith("/v1") else "/v1/models")


def curated_models_for_provider(
    provider: Optional[str],
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return ``(model_id, description)`` tuples for a provider's model list.

    Tries to fetch the live model list from the provider's API first,
    falling back to the static ``_PROVIDER_MODELS`` catalog if the API
    is unreachable.
    """
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_openrouter_models(force_refresh=force_refresh)

    # Try live API first (Codex, Nous, etc. all support /models)
    live = provider_model_ids(normalized)
    if live:
        return [(m, "") for m in live]

    # Fallback to static catalog
    models = _PROVIDER_MODELS.get(normalized, [])
    return [(m, "") for m in models]


def _provider_keys(provider: str) -> set[str]:
    key = (provider or "").strip().lower()
    normalized = normalize_provider(provider)
    return {k for k in (key, normalized) if k}


def _provider_catalog_names(provider: str) -> tuple[str, ...]:
    """Active picker models plus retired aliases recognized for detection."""
    return tuple(_PROVIDER_MODELS.get(provider, [])) + _PROVIDER_RETIRED_ALIASES.get(provider, ())


def _model_in_provider_catalog(name_lower: str, providers: set[str]) -> bool:
    return any(
        name_lower == model.lower()
        for provider in providers
        for model in _provider_catalog_names(provider))


def _openrouter_variant_base(model_id: str) -> Optional[str]:
    """Base model id when ``model_id`` carries a recognized OpenRouter routing-variant suffix
    (``x-ai/grok-4:nitro`` → ``x-ai/grok-4``), else ``None``."""
    base, sep, suffix = (model_id or "").rpartition(":")
    return base if sep and base and suffix.lower() in _OPENROUTER_VARIANT_SUFFIXES else None


def _resolve_static_model_alias(
    name_lower: str, current_keys: set[str]) -> Optional[tuple[str, str]]:
    """Resolve short aliases (e.g. sonnet/opus) using static catalogs only."""
    try:
        from hermes_cli.model_switch import MODEL_ALIASES
    except Exception:
        return None

    identity = MODEL_ALIASES.get(name_lower)
    if identity is None:
        return None

    def _match(provider: str) -> Optional[str]:
        prefix = f"{identity.vendor}/{identity.family}" if provider in _AGGREGATOR_PROVIDERS else identity.family
        prefix = prefix.lower()
        return next((m for m in _PROVIDER_MODELS.get(provider, []) if m.lower().startswith(prefix)), None)

    # Current provider first, then native vendors, then aggregators / borrow-list providers the user
    # is already on — so `sonnet` resolves to anthropic before any re-exposing provider.
    skip = current_keys | _AGGREGATOR_PROVIDERS | _BORROWED_MODEL_PROVIDERS
    candidates = [
        *current_keys, *(p for p in _PROVIDER_MODELS if p not in skip),
        *(p for p in _AGGREGATOR_PROVIDERS if p in current_keys),
        *(p for p in _BORROWED_MODEL_PROVIDERS if p in current_keys)]
    for provider in candidates:
        if matched := _match(provider):
            return provider, matched
    return None


def detect_static_provider_for_model(
    model_name: str, current_provider: str) -> Optional[tuple[str, str]]:
    """Auto-detect a provider from static catalogs only → ``(provider_id, model_name)`` (the name may
    be remapped by a static alias or a bare provider name), or ``None`` without a confident match."""
    name = (model_name or "").strip()
    if not name:
        return None

    name_lower = name.lower()
    current_keys = _provider_keys(current_provider)

    alias_match = _resolve_static_model_alias(name_lower, current_keys)
    if alias_match:
        return alias_match

    # Step 0: a bare provider name typed as the model (`/model nous`) is a provider switch to that
    # provider's default. Skip "custom" (no catalog) and "openrouter" (needs an explicit model).
    resolved_provider = _PROVIDER_ALIASES.get(name_lower, name_lower)
    if resolved_provider not in {"custom", "openrouter"}:
        default_models = _PROVIDER_MODELS.get(resolved_provider, [])
        if resolved_provider in _PROVIDER_LABELS and default_models and resolved_provider not in current_keys:
            # Cost-safe default, not ``default_models[0]``: metered aggregators list most-capable-first,
            # so [0] would silently escalate `/model nous` to the priciest flagship.
            return (resolved_provider, get_default_model_for_provider(resolved_provider) or default_models[0])

    # A model in the current provider's own catalog never suggests switching.
    if _model_in_provider_catalog(name_lower, current_keys):
        return None

    # Step 1: direct static-catalog match. Aggregators list other vendors' models — never
    # auto-switch TO them. A custom endpoint (custom / custom:*) is never auto-switched away
    # from: the user configured it deliberately and may serve the same model name there.
    if current_provider != "custom" and not current_provider.startswith("custom:"):
        for pid in _PROVIDER_MODELS:
            if pid in current_keys or pid in _AGGREGATOR_PROVIDERS or pid in _BORROWED_MODEL_PROVIDERS:
                continue
            if _model_in_provider_catalog(name_lower, {pid}):
                return (pid, name)

    # Borrow-list providers (re-expose other vendors' models) only after every native-vendor
    # catalog, and only when one is the current provider.
    for pid in _BORROWED_MODEL_PROVIDERS:
        if pid not in current_keys and _model_in_provider_catalog(name_lower, {pid}):
            return (pid, name)

    return None


def _configured_provider_ids() -> set[str]:
    """Provider ids (incl. ``custom:*``) from the user's ``providers:`` config block; empty when config
    is unreadable (callers fall through to built-in catalogs)."""
    try:
        from hermes_cli.config import load_config

        providers = (load_config() or {}).get("providers")
        if not isinstance(providers, dict):
            return set()
        return {key for pid in providers if (key := str(pid).strip().lower())}
    except Exception:
        return set()


def _resolve_provider_prefix(model_name: str) -> Optional[tuple[str, str]]:
    """Route an explicit ``vendor/model`` prefix (``nous/deepseek-v4-pro``, ``ollama/qwen3.5:4b``) to
    a provider the user defined in ``providers:`` (by raw name or alias) instead of the default.

    ``nous/deepseek-v4-pro`` or ``ollama/qwen3.5:4b`` should route to the named provider instead of falling
    back to the configured default (which silently sends non-default models to the wrong endpoint, #87189).
    """
    if "/" not in model_name:
        return None
    vendor, model = model_name.split("/", 1)
    vendor, model = vendor.strip().lower(), model.strip()
    if not vendor or not model:
        return None
    configured = _configured_provider_ids()
    # An explicitly named provider block (``ollama:``) wins over the alias table, which may
    # canonicalize the same name elsewhere (``ollama`` → ``custom``).
    for candidate in (vendor, _PROVIDER_ALIASES.get(vendor, vendor)):
        if candidate in configured:
            return (candidate, model)
    return None


def detect_provider_for_model(
    model_name: str, current_provider: str) -> Optional[tuple[str, str]]:
    """Auto-detect the best provider for a model name: static catalogs (bare provider name → its
    default; direct catalog match), then the OpenRouter catalog, then a configured ``vendor/`` prefix."""
    name = (model_name or "").strip()
    if not name:
        return None

    static_match = detect_static_provider_for_model(name, current_provider)
    if static_match:
        return static_match
    if _model_in_provider_catalog(name.lower(), _provider_keys(current_provider)):
        return None

    # OpenRouter catalog (exact slug, then bare model part).
    or_slug = _find_openrouter_slug(name)
    if or_slug:
        if current_provider != "openrouter" or or_slug != name:
            return ("openrouter", or_slug)
        return None  # already on openrouter with matching name

    # Explicit ``vendor/model`` prefix naming a configured provider — AFTER the OpenRouter lookup so
    # aggregator-native slugs (``deepseek/deepseek-chat``) keep their routing.
    return _resolve_provider_prefix(name)


def _find_openrouter_slug(model_name: str) -> Optional[str]:
    """Full OpenRouter slug for a bare or partial model name (exact slug first, then bare part)."""
    name_lower = model_name.strip().lower()
    if not name_lower:
        return None
    ids = model_ids()
    return (
        next((mid for mid in ids if name_lower == mid.lower()), None)
        or next((mid for mid in ids if "/" in mid and name_lower == mid.split("/", 1)[1].lower()), None)
    )


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to canonical ids. ``"auto"`` passes through — use
    ``hermes_cli.auth.resolve_provider()`` to resolve it from credentials."""
    normalized = (provider or "openrouter").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_label(provider: Optional[str]) -> str:
    """Return a human-friendly label for a provider id or alias."""
    original = (provider or "openrouter").strip()
    normalized = original.lower()
    if normalized == "auto":
        return "Auto"
    normalized = normalize_provider(normalized)
    return _PROVIDER_LABELS.get(normalized, original or "OpenRouter")


def _is_openai_fast_model(model_id: Optional[str]) -> bool:
    """OpenAI flagship eligible for Priority Processing. Codex-series excluded — the Codex Responses
    API doesn't accept ``service_tier``."""
    base = _strip_vendor_prefix(str(model_id or "")).split(":")[0]
    return bool(base) and "codex" not in base and base.startswith(tuple(_OPENAI_FAST_MODE_PREFIXES))


def _strip_vendor_prefix(model_id: str) -> str:
    """Lowercase and strip a ``vendor/`` prefix (``anthropic/claude-opus-4-6`` → ``claude-opus-4-6``)."""
    raw = str(model_id or "").strip().lower()
    return raw.split("/", 1)[1] if "/" in raw else raw


def model_supports_fast_mode(model_id: Optional[str]) -> bool:
    """Return whether Hermes should expose the /fast toggle for this model."""
    from agent.model_metadata import is_grok_46_family

    return (
        _is_anthropic_fast_model(model_id)
        or _is_openai_fast_model(model_id)
        or is_grok_46_family(str(model_id or "")))


def _is_anthropic_fast_model(model_id: Optional[str]) -> bool:
    """Accepts the Anthropic Fast Mode ``speed`` param (Opus 4.8 / Opus 5 only) — deliberately NOT a
    general "fast model" check: Opus 4.7 hard-400s on it, and dedicated ``…-fast`` ids select fast
    inference via the model field and must not also get it."""
    base = _strip_vendor_prefix(str(model_id or "")).split(":")[0]
    if not base.startswith("claude-") or "-fast" in base:
        return False
    return any(v in base for v in ("opus-4-8", "opus-4.8", "opus-5"))


def _fast_mode_route_supported(
    model_id: Optional[str], provider: Optional[str], base_url: Optional[str]) -> bool:
    """Only the first-party endpoint that bills for fast mode may receive its params."""
    from urllib.parse import urlparse

    from agent.model_metadata import is_grok_46_family

    if _is_anthropic_fast_model(model_id):
        allowed = {"anthropic": "api.anthropic.com"}
    elif is_grok_46_family(str(model_id or "")):
        allowed = {"xai": "api.x.ai"}
    else:
        allowed = {"openai": "api.openai.com", "openai-codex": "chatgpt.com"}
    if provider and normalize_provider(provider) not in allowed:
        return False
    host = (urlparse(str(base_url or "")).hostname or "").lower()
    return not host or host in allowed.values()


def resolve_fast_mode_overrides(
    model_id: Optional[str], *, provider: Optional[str] = None, base_url: Optional[str] = None
) -> dict[str, Any] | None:
    """Fast/priority request_overrides — ``{"speed": "fast"}`` (Anthropic Fast Mode) or
    ``{"service_tier": "priority"}`` (OpenAI / xAI Priority Processing) — or None if unsupported.
    With ``provider``/``base_url`` the route is gated too (``_fast_mode_route_supported``) so proxies
    never see the params. Single fast-mode gate for ``/fast`` and ``agent.fast_mode`` windows."""
    if not model_supports_fast_mode(model_id):
        return None
    if (provider or base_url) and not _fast_mode_route_supported(model_id, provider, base_url):
        return None
    return {"speed": "fast"} if _is_anthropic_fast_model(model_id) else {"service_tier": "priority"}


def _first_exchangeable_copilot_token(raw_tokens) -> str:
    """Exchange stored GitHub tokens in order; the first that validates AND exchanges wins (every
    entry is tried so a later valid token survives an earlier malformed one)."""
    from hermes_cli.copilot_auth import exchange_copilot_token, validate_copilot_token

    for raw in raw_tokens:
        raw = str(raw or "").strip()
        if not raw or not validate_copilot_token(raw)[0]:
            continue
        try:
            api_token = exchange_copilot_token(raw)[0]  # (api_token, expires_at, base_url)
        except Exception:
            continue
        if api_token:
            return api_token
    return ""


def _copilot_cli_config_tokens() -> list[str]:
    """``copilotTokens`` from the GitHub Copilot CLI's own plaintext store (JSONC — strip
    ``//``-comment lines), written by ``copilot login`` on hosts without an OS keychain."""
    cli_config = os.path.expanduser("~/.copilot/config.json")
    if not os.path.isfile(cli_config):
        return []
    with open(cli_config, "r", encoding="utf-8", errors="ignore") as fh:
        raw_text = "\n".join(
            line for line in fh.read().splitlines() if not line.lstrip().startswith("//"))
    data = json.loads(raw_text) if raw_text.strip() else {}
    tokens = data.get("copilotTokens")
    return list(tokens.values()) if isinstance(tokens, dict) else []


def _resolve_copilot_catalog_api_key() -> str:
    """Best-effort GitHub token for the Copilot catalog: env vars / ``gh auth token`` via
    ``resolve_api_key_provider_credentials``, then ``auth.json`` ``credential_pool.copilot[]``, then
    ``~/.copilot/config.json`` ``copilotTokens`` (the ACP CLI's own store). Without the latter two,
    keyless users see the picker fall back to the stale curated list on a silent 401."""
    def _pool_token() -> str:
        from hermes_cli.auth import read_credential_pool

        return _first_exchangeable_copilot_token(
            entry.get("access_token") for entry in read_credential_pool("copilot") if isinstance(entry, dict))

    sources = (
        lambda: _api_key_credentials("copilot")[0],
        _pool_token,
        lambda: _first_exchangeable_copilot_token(_copilot_cli_config_tokens()),
    )
    for source in sources:
        try:
            token = source()
        except Exception:
            continue
        if token:
            return token
    return ""


def _model_dedup_key(model_id: str) -> str:
    """Case-insensitive dedup key folded through the picker-search alias table, so a bare live wire
    id and its curated public slug (Kimi ``k3`` / ``kimi-k3``) don't both survive a merge."""
    key = str(model_id).strip().lower()
    try:
        from hermes_cli.model_search import model_alias_canonical
        return model_alias_canonical(key)
    except Exception:
        return key


def _merge_with_models_dev(provider: str, curated: list[str]) -> list[str]:
    """models.dev entries first (their order), then curated-only extras, case-insensitively deduped
    while preserving curated casing. Curated unchanged when models.dev is unreachable/empty."""
    try:
        from agent.models_dev import list_agentic_models
        mdev = list_agentic_models(provider)
    except Exception:
        mdev = []
    if not mdev:
        return list(curated)
    return _merge_unique(_merge_unique([], mdev), curated)


def _openai_discovery_base_url(provider: str) -> str:
    """OpenAI endpoint for model discovery, mirroring runtime precedence so discovery probes the SAME
    endpoint inference uses: ``$OPENAI_BASE_URL`` → config ``model.base_url`` (when the configured
    provider matches) → the canonical default."""
    env_raw = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if env_raw:
        return env_raw
    try:
        model_cfg = _get_model_config_dict()
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        same_provider = normalize_provider(provider) == normalize_provider(cfg_provider)
        if cfg_provider in ("openai", "openai-api") and same_provider:
            cfg_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            if cfg_url:
                return cfg_url
    except Exception:
        pass
    return "https://api.openai.com/v1"


def _codex_catalog(normalized: str, force_refresh: bool) -> list[str]:
    from hermes_cli.codex_models import get_codex_model_ids

    # Live OAuth token so the picker matches what ChatGPT lists for this account; hardcoded
    # catalog without a token / when unreachable.
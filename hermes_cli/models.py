"""Provider/model catalogs: discovery, caching, and identity helpers.

Origin module; cohesive clusters live in siblings and are re-imported here so
``hermes_cli.models.<name>`` stays the stable import/monkeypatch surface:
``models_catalog_static`` (curated tables, provider registry, aliases), ``models_reasoning_caps``,
``models_local`` (Ollama / LM Studio), ``models_pricing``, ``models_validate``.
"""

from __future__ import annotations

import contextvars
import copy
import json
import http.client
import logging
import os
import re
import sys
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
from hermes_cli.urllib_security import open_credentialed_url, url_origin
from utils import atomic_json_write, base_url_host_matches

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


# Fallback OpenRouter snapshot used when the live catalog is unavailable.
# (model_id, display description shown in menus)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    # Anthropic
    ("anthropic/claude-fable-5",               ""),
    ("anthropic/claude-opus-5",                ""),
    ("anthropic/claude-opus-5-fast",           "2x price, higher output speed"),
    ("anthropic/claude-opus-4.8",              ""),
    ("anthropic/claude-opus-4.8-fast",         "2x price, higher output speed"),
    ("anthropic/claude-sonnet-5",              ""),
    ("anthropic/claude-haiku-4.5",             ""),
    # OpenAI
    ("openai/gpt-5.6-sol",                     ""),
    ("openai/gpt-5.6-sol-pro",                 ""),
    ("openai/gpt-5.6-terra",                   ""),
    ("openai/gpt-5.6-terra-pro",               ""),
    ("openai/gpt-5.6-luna",                    ""),
    ("openai/gpt-5.6-luna-pro",                ""),
    ("openai/gpt-5.5",                         ""),
    ("openai/gpt-5.5-pro",                     ""),
    ("openai/gpt-5.4-mini",                    ""),
    # Google
    ("google/gemini-3.1-pro-preview",          ""),
    ("google/gemini-3.7-flash",                ""),
    # xAI
    ("x-ai/grok-4.6",                          ""),
    # DeepSeek
    ("deepseek/deepseek-v4-pro",               ""),
    ("deepseek/deepseek-v4-pro-0813",          "dated snapshot of v4-pro"),
    ("deepseek/deepseek-v4-flash",             ""),
    ("deepseek/deepseek-v4-flash-0731",        "dated snapshot of v4-flash"),
    # Qwen
    ("qwen/qwen3.8-max",                       ""),
    ("qwen/qwen3.8-flash",                     ""),
    # MoonshotAI
    ("moonshotai/kimi-k3",                     "recommended"),
    # MiniMax
    ("minimax/minimax-m3",                     ""),
    # Z-AI
    ("z-ai/glm-5.3",                           ""),
    ("z-ai/glm-5.3-flash",                     ""),
    ("z-ai/glm-5.2",                           "default"),
    # Xiaomi
    ("xiaomi/mimo-v2.5-pro",                   ""),
    # Tencent
    ("tencent/hy4-preview",                    ""),
    ("tencent/hy3",                            ""),
    # StepFun
    ("stepfun/step-3.7-flash",                 ""),
    # NVIDIA
    ("nvidia/nemotron-3-super-120b-a12b",      ""),
    # Meta
    ("meta/muse-spark-1.2",                    ""),
    # Sakana
    ("sakana/fugu-ultra",                      ""),
    # OpenRouter routers
    ("openrouter/pareto-code",                 "auto-routes to cheapest coder meeting openrouter.min_coding_score"),
    # Free tier
    ("thinkingmachines/inkling:free",          "free"),
    ("thinkingmachines/inkling-small:free",    "free"),
    ("minimax/minimax-m3:free",                "free"),
    ("z-ai/glm-5.2:free",                      "free"),
    ("poolside/laguna-s-2.1:free",             "free"),
    ("poolside/laguna-xs-2.1:free",            "free"),
    ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
    ("nvidia/nemotron-3-ultra-550b-a55b:free", "free"),
    ("nvidia/nemotron-3.5-lightning:free",     "free"),
]

_openrouter_catalog_cache: list[tuple[str, str]] | None = None

# The in-memory ``_openrouter_catalog_cache`` is per-process, so without a disk cache every cold
# picker open re-downloads the full ~686KB /api/v1/models catalog. The *curated* result
# (post-filter) is persisted under the same TTL the catalog manifest uses, so both layers go
# stale together.


def _openrouter_catalog_disk_ttl() -> float:
    """Same TTL as the catalog manifest this list is filtered from (honours ``model_catalog.ttl_minutes``)."""
    from hermes_cli.model_catalog import refresh_interval_seconds

    return refresh_interval_seconds()


def _openrouter_catalog_disk_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "openrouter_curated_catalog.json"


def _read_openrouter_catalog_disk() -> list[tuple[str, str]] | None:
    """Fresh curated catalog from disk, or None (missing, corrupt, expired, or empty)."""
    obj = _read_json_cache(_openrouter_catalog_disk_path())
    if obj is None:
        return None
    try:
        if time.time() - float(obj.get("fetched_at", 0)) > _openrouter_catalog_disk_ttl():
            return None
    except (TypeError, ValueError):
        return None
    items = obj.get("curated")
    if not isinstance(items, list):
        return None
    out = [(str(it[0]), str(it[1])) for it in items if isinstance(it, (list, tuple)) and len(it) == 2]
    return out or None


def _write_openrouter_catalog_disk(curated: list[tuple[str, str]]) -> None:
    try:
        _write_json_cache(
            _openrouter_catalog_disk_path(),
            {"fetched_at": time.time(), "curated": [list(c) for c in curated]})
    except Exception as exc:
        logger.debug("openrouter curated catalog disk write failed: %s", exc)
_ai_gateway_catalog_cache: list[tuple[str, str]] | None = None


def _codex_curated_models() -> list[str]:
    """Derive the openai-codex curated list from codex_models.py.

    Single source of truth: DEFAULT_CODEX_MODELS + forward-compat synthesis.
    This keeps the gateway /model picker in sync with the CLI `hermes model`
    flow without maintaining a separate static list.
    """
    from hermes_cli.codex_models import DEFAULT_CODEX_MODELS, _finalize_codex_models
    return _finalize_codex_models(list(DEFAULT_CODEX_MODELS))


def _zero_priced(pricing: Any, keys: tuple[str, str], default: str) -> bool:
    """True when both pricing fields parse to 0 (missing fields read as ``default``)."""
    if not isinstance(pricing, dict):
        return False
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
        "google/gemini-3.1-pro-preview",
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
        "gemini-3.6-flash",
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
    # Entries validated live against a GCP project (global region,
    # HTTP 200) as of 2026-07-21 (PR #68767).
    "vertex": [
        "google/gemini-3.1-pro-preview",
        "google/gemini-3-pro-preview",
        "google/gemini-3.6-flash",
        "google/gemini-3.5-flash",
        "google/gemini-3.5-flash-lite",
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite-preview",
        "google/gemini-3.1-flash-lite",
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
# Nous Portal free-model helper
# ---------------------------------------------------------------------------
# The Nous Portal models endpoint is the source of truth for which models
# are currently offered (free or paid). We trust whatever it returns and
# surface it to users as-is — no local allowlist filtering.


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
            # On any detection error, assume paid — paid users see both fields
            # anyway so this is a safe default that maximises model quality.
            free_tier = False

    if vision:
        paid_key, free_key = "paidRecommendedVisionModel", "freeRecommendedVisionModel"
    else:
        paid_key, free_key = "paidRecommendedCompactionModel", "freeRecommendedCompactionModel"

    # Preference order:
    #   free tier  → free only
    #   paid tier  → paid, then free (if paid field is null)
    candidates = [free_key] if free_tier else [paid_key, free_key]
    for key in candidates:
        name = _extract_model_name(payload.get(key))
        if name:
            return name
    return None


# ---------------------------------------------------------------------------
# Canonical provider list — single source of truth for provider identity.
# Every code path that lists, displays, or iterates providers derives from
# this list:  hermes model, /model, list_authenticated_providers.
#
# Fields:
#   slug        — internal provider ID (used in config.yaml, --provider flag)
#   label       — short display name
#   tui_desc    — longer description for the `hermes model` interactive picker
# ---------------------------------------------------------------------------

class ProviderEntry(NamedTuple):
    slug: str
    label: str
    tui_desc: str   # detailed description for `hermes model` TUI

CANONICAL_PROVIDERS: list[ProviderEntry] = [
    ProviderEntry("nous",           "Nous Portal",              "Nous Portal (Everything your agent needs, 300+ models with bundled tool use)"),
    ProviderEntry("fireworks",      "Fireworks AI",             "Fireworks AI (OpenAI-compatible direct model API)"),
    ProviderEntry("openrouter",     "OpenRouter",               "OpenRouter (Pay-per-use API aggregator)"),
    ProviderEntry("moa",            "Mixture of Agents",        "Mixture of Agents (named presets; aggregator acts after reference models)"),
    ProviderEntry("novita",         "NovitaAI",                 "NovitaAI (Cloud: Model API, Agent Sandbox, GPU Cloud)"),
    ProviderEntry("lmstudio",       "LM Studio",                "LM Studio (Local desktop app with built-in model server)"),
    ProviderEntry("anthropic",      "Anthropic",                "Anthropic (Claude models via API key or Claude Code)"),
    ProviderEntry("openai-codex",   "ChatGPT or Codex Subscription", "ChatGPT or Codex Subscription (Sign in with your ChatGPT account, uses Codex models)"),
    ProviderEntry("openai-api",     "OpenAI API",               "OpenAI API (api.openai.com, API key)"),
    ProviderEntry("alibaba",        "Qwen Cloud",               "Qwen Cloud / DashScope (Qwen + multi-provider)"),
    ProviderEntry("xai-oauth",      "xAI Grok OAuth (SuperGrok / Premium+)", "xAI Grok OAuth (SuperGrok / Premium+ subscription)"),
    ProviderEntry("xiaomi",         "Xiaomi MiMo",              "Xiaomi MiMo (MiMo-V2.5 and V2 models: pro, omni, flash)"),
    ProviderEntry("tencent-tokenhub", "Tencent TokenHub",       "Tencent TokenHub (Hy4 preview via tokenhub.tencentmaas.com)"),
    ProviderEntry("tencent-tokenplan", "Tencent TokenPlan",     "Tencent TokenPlan (Hy4 preview via api.lkeap.cloud.tencent.com, Anthropic Messages)"),
    ProviderEntry("nvidia",         "NVIDIA NIM",               "NVIDIA NIM (Nemotron models via build.nvidia.com or local NIM)"),
    ProviderEntry("copilot",        "GitHub Copilot",           "GitHub Copilot (Uses GITHUB_TOKEN or gh auth token)"),
    ProviderEntry("copilot-acp",    "GitHub Copilot ACP",       "GitHub Copilot ACP (Spawns copilot --acp --stdio)"),
    ProviderEntry("huggingface",    "Hugging Face",             "Hugging Face Inference Providers"),
    ProviderEntry("gemini",         "Google AI Studio",         "Google AI Studio (Native Gemini API)"),
    ProviderEntry("vertex",         "Google Vertex AI",         "Google Vertex AI (Gemini via GCP; OAuth2 service account or ADC, GCP billing/quotas)"),
    ProviderEntry("deepseek",       "DeepSeek",                 "DeepSeek (V3, R1, coder, direct API)"),
    ProviderEntry("xai",            "xAI",                      "xAI Grok (Direct API)"),
    ProviderEntry("zai",            "Z.AI / GLM",               "Z.AI / GLM (Zhipu direct API)"),
    ProviderEntry("kimi-coding",    "Kimi / Kimi Coding Plan",  "Kimi Coding Plan (api.kimi.com & Moonshot API)"),
    ProviderEntry("kimi-coding-cn", "Kimi / Moonshot (China)",  "Kimi / Moonshot China (Domestic direct API)"),
    ProviderEntry("stepfun",        "StepFun Step Plan",       "StepFun Step Plan (Agent / coding models via Step Plan API)"),
    ProviderEntry("minimax",        "MiniMax",                  "MiniMax (Global direct API)"),
    ProviderEntry("minimax-oauth",  "MiniMax (OAuth)",          "MiniMax via OAuth browser login (Coding Plan, minimax.io)"),
    ProviderEntry("minimax-cn",     "MiniMax (China)",          "MiniMax China (Domestic direct API)"),
    ProviderEntry("ollama-cloud",   "Ollama Cloud",             "Ollama Cloud (Cloud-hosted open models, ollama.com)"),
    ProviderEntry("arcee",          "Arcee AI",                 "Arcee AI (Trinity models, direct API)"),
    ProviderEntry("gmi",            "GMI Cloud",                "GMI Cloud (Multi-model direct API)"),
    ProviderEntry("kilocode",       "Kilo Code",                "Kilo Code (Kilo Gateway API)"),
    ProviderEntry("opencode-zen",   "OpenCode Zen",             "OpenCode Zen (Curated models, pay-as-you-go)"),
    ProviderEntry("opencode-go",    "OpenCode Go",              "OpenCode Go (Open models subscription)"),
    ProviderEntry("bedrock",        "AWS Bedrock",              "AWS Bedrock (Claude, Nova, Llama, DeepSeek; IAM or API key)"),
    ProviderEntry("azure-foundry",  "Azure Foundry",            "Azure Foundry (OpenAI-style or Anthropic-style endpoint, your Azure AI deployment)"),
    ProviderEntry("ai-gateway",     "Vercel AI Gateway",        "Vercel AI Gateway (Multi-model aggregator)"),
    ProviderEntry("qwen-oauth",     "Qwen OAuth (Portal)",      "Qwen OAuth (Reuses local Qwen CLI login)"),
]

# Auto-extend CANONICAL_PROVIDERS with any provider registered in providers/
# that is not already in the list above.  Adding plugins/model-providers/<name>/
# is sufficient to expose a new provider in the model picker, /model, and all
# downstream consumers — no edits to this file needed.
_canonical_slugs = {p.slug for p in CANONICAL_PROVIDERS}
try:
    from providers import list_providers as _list_providers_for_canonical
    for _pp in _list_providers_for_canonical():
        if _pp.name in _canonical_slugs:
            continue
        if _pp.auth_type in {"oauth_device_code", "oauth_external", "external_process", "aws_sdk", "copilot", "vertex"}:
            continue  # non-api-key flows need bespoke picker UX; skip auto-inject
        _label = _pp.display_name or _pp.name
        _desc = _pp.description or f"{_label} (direct API)"
        CANONICAL_PROVIDERS.append(ProviderEntry(_pp.name, _label, _desc))
        _canonical_slugs.add(_pp.name)
except Exception:
    pass

# Derived dicts — used throughout the codebase
_PROVIDER_LABELS = {p.slug: p.label for p in CANONICAL_PROVIDERS}
_PROVIDER_LABELS["custom"] = "Custom endpoint"  # special case: not a named provider


# ---------------------------------------------------------------------------
# Provider groups — DISPLAY ONLY
#
# Some vendors expose several Hermes provider slugs (one per endpoint /
# auth method: global API, China API, OAuth coding plan, ...). Listing every
# slug as a top-level row in the interactive `hermes model` / setup wizard /
# Telegram `/model` pickers makes that list long and noisy.
#
# These groups fold related slugs under one top-level row in INTERACTIVE
# PICKERS only. They do NOT change ``CANONICAL_PROVIDERS``, slug identity,
# the ``--provider`` flag, ``/model <provider:model>``, or any typed path —
# every member slug remains individually addressable. Grouping is a pure
# display affordance; ``group_providers()`` is the single fold used by all
# three picker surfaces so they stay consistent.
#
#   group_id -> (display_label, group_description, [member_slug, ...])
#
# ``group_description`` is a short blurb shown on the collapsed top-level group
# row in the interactive pickers (alongside the label). Member-specific detail
# lives in each member's ``tui_desc`` and shows in the drill-down sub-picker.
# Member order is the order shown inside the group submenu.
# ---------------------------------------------------------------------------
PROVIDER_GROUPS: dict[str, tuple[str, str, list[str]]] = {
    "kimi":     ("Kimi / Moonshot", "Coding Plan, Moonshot global & China endpoints", ["kimi-coding", "kimi-coding-cn"]),
    "minimax":  ("MiniMax",         "Global, OAuth Coding Plan & China endpoints",     ["minimax", "minimax-oauth", "minimax-cn"]),
    "xai":      ("xAI Grok",        "Direct API or SuperGrok / Premium+ OAuth",        ["xai", "xai-oauth"]),
    "google":   ("Google Gemini",   "Google AI Studio (API key)",                     ["gemini"]),
    "openai":   ("OpenAI",          "ChatGPT/Codex subscription or direct OpenAI API", ["openai-codex", "openai-api"]),
    "qwen":     ("Qwen",            "Qwen Cloud / DashScope, Coding Plan, Token Plan & Qwen CLI OAuth", ["alibaba", "alibaba-cn", "alibaba-coding-plan", "alibaba-coding-plan-cn", "alibaba-token-plan", "alibaba-token-plan-cn", "qwen-oauth"]),
    "opencode": ("OpenCode",        "Zen pay-as-you-go, Go subscription, or free tier", ["opencode-zen", "opencode-go", "opencode-free"]),
    "copilot":  ("GitHub Copilot",  "GitHub token API or copilot --acp process",       ["copilot", "copilot-acp"]),
    "tencent":  ("Tencent Hy",      "Hy4 / Hy3 via TokenHub & TokenPlan", ["tencent-tokenhub", "tencent-tokenplan"]),
}

# Reverse index: member slug -> group_id. Built once at import.
_SLUG_TO_GROUP: dict[str, str] = {
    slug: gid for gid, (_label, _desc, members) in PROVIDER_GROUPS.items() for slug in members
}


def provider_group_for_slug(slug: str) -> str:
    """Return the group_id a provider slug belongs to, or "" if ungrouped."""
    return _SLUG_TO_GROUP.get(str(slug or "").strip().lower(), "")


def group_providers(slugs):
    """Fold a flat ordered slug iterable into picker rows by provider group.

    DISPLAY ONLY. Used by every interactive picker (``hermes model``, the
    setup wizard, the Telegram ``/model`` keyboard) so grouping is identical
    across surfaces.

    Each returned row is a dict::

        {"kind": "single", "slug": <slug>}                       # ungrouped, or
                                                                  # 1-member group
        {"kind": "group", "group_id": <gid>, "label": <label>,
         "description": <desc>, "members": [<slug>, ...]}        # 2+ members

    Rules:
      * A group row appears at the position of its FIRST present member, in
        the input order. Subsequent members fold into that row (and are not
        emitted again).
      * Member order inside a group follows ``PROVIDER_GROUPS`` declaration,
        restricted to the members actually present in ``slugs``.
      * A group reduced to a single present member degrades to a ``single``
        row — no pointless one-item submenu.
      * Slugs not in any group pass through as ``single`` rows, order
        preserved.
      * Duplicate slugs in the input are ignored after first sight.
    """
    seen: set[str] = set()
    # Which present members each group has, in declaration order.
    group_members: dict[str, list[str]] = {}
    for gid, (_label, _desc, members) in PROVIDER_GROUPS.items():
        present = [m for m in members if m in set(slugs)]
        if present:
            group_members[gid] = present

    rows = []
    emitted_groups: set[str] = set()
    for slug in slugs:
        s = str(slug or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        gid = _SLUG_TO_GROUP.get(s, "")
        if not gid:
            rows.append({"kind": "single", "slug": s})
            continue
        if gid in emitted_groups:
            continue  # already folded at the first member's position
        emitted_groups.add(gid)
        members = group_members.get(gid, [s])
        if len(members) <= 1:
            rows.append({"kind": "single", "slug": members[0]})
        else:
            label, desc, _ = PROVIDER_GROUPS[gid]
            rows.append(
                {"kind": "group", "group_id": gid, "label": label,
                 "description": desc, "members": list(members)}
            )
    return rows


_PROVIDER_ALIASES = {
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "github": "copilot",
    "github-copilot": "copilot",
    "github-models": "copilot",
    "github-model": "copilot",
    "github-copilot-acp": "copilot-acp",
    "copilot-acp-agent": "copilot-acp",
    "google": "gemini",
    "google-gemini": "gemini",
    "google-ai-studio": "gemini",
    "google-vertex": "vertex",
    "vertex-ai": "vertex",
    "gcp-vertex": "vertex",
    "vertexai": "vertex",
    "kimi": "kimi-coding",
    "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn",
    "moonshot-cn": "kimi-coding-cn",
    "step": "stepfun",
    "stepfun-coding-plan": "stepfun",
    "arcee-ai": "arcee",
    "arceeai": "arcee",
    "gmi-cloud": "gmi",
    "gmicloud": "gmi",
    "fireworks-ai": "fireworks",
    "fw": "fireworks",
    "actual-computer": "actual",
    "actualcomputer": "actual",
    "aci": "actual",
    "nebius": "nebius-token-factory",
    "nebius-tokenfactory": "nebius-token-factory",
    "nebius-tf": "nebius-token-factory",
    "token-factory": "nebius-token-factory",
    "tokenfactory": "nebius-token-factory",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "minimax-portal": "minimax-oauth",
    "minimax-global": "minimax-oauth",
    "minimax_oauth": "minimax-oauth",
    "claude": "anthropic",
    "claude-code": "anthropic",
    "deep-seek": "deepseek",
    "opencode": "opencode-zen",
    "zen": "opencode-zen",
    "go": "opencode-go",
    "opencode-go-sub": "opencode-go",
    "free": "opencode-free",
    "opencode_free": "opencode-free",
    "aigateway": "ai-gateway",
    "vercel": "ai-gateway",
    "vercel-ai-gateway": "ai-gateway",
    "kilo": "kilocode",
    "kilo-code": "kilocode",
    "kilo-gateway": "kilocode",
    "dashscope": "alibaba",
    "aliyun": "alibaba",
    "qwen": "alibaba",
    "alibaba-cloud": "alibaba",
    "qwen-portal": "qwen-oauth",
    "hf": "huggingface",
    "hugging-face": "huggingface",
    "huggingface-hub": "huggingface",
    "novita-ai": "novita",
    "novitaai": "novita",
    "mimo": "xiaomi",
    "xiaomi-mimo": "xiaomi",
    "tencent": "tencent-tokenhub",
    "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub",
    "tencentmaas": "tencent-tokenhub",
    "tokenplan": "tencent-tokenplan",
    "tencent-lkeap": "tencent-tokenplan",
    "aws": "bedrock",
    "aws-bedrock": "bedrock",
    "amazon-bedrock": "bedrock",
    "amazon": "bedrock",
    "grok": "xai",
    "grok-oauth": "xai-oauth",
    "xai-oauth": "xai-oauth",
    "x-ai-oauth": "xai-oauth",
    "xai-grok-oauth": "xai-oauth",
    "x-ai": "xai",
    "x.ai": "xai",
    "nim": "nvidia",
    "nvidia-nim": "nvidia",
    "build-nvidia": "nvidia",
    "nemotron": "nvidia",
    "lmstudio": "lmstudio",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "ollama": "custom",  # bare "ollama" = local; use "ollama-cloud" for cloud
    "ollama_cloud": "ollama-cloud",
}


# In-repo fallback for the model Hermes silently lands on when the user never
# picked one (GUI onboarding confirm card, empty ``model.default``,
# provider-set-but-model-missing resolution). The AUTHORITATIVE source is the
# remote model catalog: the manifest labels exactly one entry per provider
# with ``"default": true`` (see get_default_model_from_cache in
# model_catalog.py), so maintainers can rotate the default without shipping a
# release. This constant is the offline/fresh-install fallback and MUST match
# the labeled entry in website/static/api/model-catalog.json. Deliberately a
# capable low-cost model rather than the curated lists' entry [0]: aggregator
# lists are ordered most-capable-first, so [0] is the priciest Anthropic
# flagship (claude-fable-5 / opus) — silently billing the most expensive model
# for traffic the user never opted into.
PREFERRED_SILENT_DEFAULT_MODEL = "z-ai/glm-5.2"


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


def recommended_nous_default_model() -> dict[str, Any]:
    """The model a Nous account lands on without choosing one, honouring the account's tier.

    Curated catalog plus the Portal's recommendations for the tier, narrowed to the org's policy,
    then (free tier) to the rows the tier may select, then :func:`pick_silent_default_model`.
    Contacts the Portal for a fresh tier read, so never call it on a hot path. Returns
    ``{"provider": "nous", "model": str, "free_tier": bool}``; ``model`` may be ``""`` when nothing
    is selectable (callers degrade). Shared by ``GET /api/model/recommended-default`` and the
    sign-in completion in ``hermes_cli.anon_auth`` so both land on the same model.
    """
    from hermes_cli import models_pricing as mp
    from hermes_cli.auth import get_provider_auth_state

    model_ids = get_curated_nous_model_ids()
    pricing = mp.get_pricing_for_provider("nous") or {}
    free_tier = check_nous_free_tier(force_fresh=True)
    try:
        portal_url = (get_provider_auth_state("nous") or {}).get("portal_base_url", "") or ""
    except Exception:
        portal_url = ""
    # Narrow to policy BEFORE the tier split, so a rescued id still has to pass the free/paid predicate.
    policy_allowed = mp.nous_policy_allowed_ids()
    union = union_with_portal_free_recommendations if free_tier else union_with_portal_paid_recommendations
    model_ids, pricing = union(model_ids, pricing, portal_url)
    model_ids = mp.restrict_to_nous_policy(model_ids, policy_allowed, rescue_empty=True)
    if free_tier:
        model_ids, _unavailable = partition_nous_models_by_tier(model_ids, pricing, free_tier=True)
    return {"provider": "nous", "model": pick_silent_default_model(model_ids, provider="nous"),
            "free_tier": bool(free_tier)}


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


# ── Disk mirror ────────────────────────────────────────────────────────
#
# The in-process caches are always cold in a short-lived process, and every
# consumer is on a hot path that must never block on HTTP — so without a disk
# copy, `hermes -p`, a cron job, or a freshly booted gateway answers
# "capability unknown" for its whole first turn and falls back to the
# conservative wire shape. Persisting the parsed catalog makes every run after
# the first correct from its first turn.
#
# One file holds every catalog, keyed by the URL it came from: OpenRouter and
# the Nous Portal list different models, and a staging Portal must not answer
# for production.
_REASONING_CAPS_DISK_TTL_SECONDS = 24 * 3600


def _reasoning_caps_disk_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "reasoning_caps.json"


def _read_reasoning_caps_disk() -> dict[str, Any]:
    try:
        with _reasoning_caps_disk_path().open(encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_reasoning_caps_disk(
    url: str,
) -> tuple[Optional[dict[str, Optional[dict[str, Any]]]], float]:
    """Return ``(caps, age_seconds)`` for *url*, or ``(None, 0.0)``."""
    entry = _read_reasoning_caps_disk().get(url)
    if not isinstance(entry, dict):
        return None, 0.0
    caps = entry.get("caps")
    if not isinstance(caps, dict) or not caps:
        return None, 0.0
    try:
        age = max(0.0, time.time() - float(entry.get("ts") or 0))
    except (TypeError, ValueError):
        age = float(_REASONING_CAPS_DISK_TTL_SECONDS)
    return {str(mid): model_caps for mid, model_caps in caps.items()}, age


def _save_reasoning_caps_disk(
    url: str, caps: dict[str, Optional[dict[str, Any]]]
) -> None:
    """Merge *url*'s catalog into the shared disk mirror, atomically."""
    try:
        data = _read_reasoning_caps_disk()
        data[url] = {"ts": time.time(), "caps": caps}
        path = _reasoning_caps_disk_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, data, indent=0, separators=(",", ":"))
    except Exception as exc:
        logger.debug("Failed to save reasoning-caps disk cache: %s", exc)


def _warm_reasoning_caps_async(refresh) -> None:
    """Run *refresh* in a background thread. Fire-and-forget.

    Called from hot paths that found the cache cold or the disk copy stale, so
    the next call — or, via the disk mirror, the next process — benefits
    without this turn ever blocking on HTTP. Callers own the once-per-process
    guard; the fetch keeps its own failure TTL. Skipped under pytest, where a
    mid-suite background fetch would make cache state, and therefore test
    behavior, timing-dependent.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    threading.Thread(
        target=refresh, name="reasoning-caps-warm", daemon=True
    ).start()


def _hydrate_reasoning_caps_from_disk(url: str, refresh):
    """The disk copy of *url*'s catalog, queueing *refresh* when it's stale.

    A copy past its TTL is still returned — a stale verdict beats no verdict,
    and reasoning capabilities change rarely — with a background refresh so
    the next run is current.
    """
    caps, age = _load_reasoning_caps_disk(url)
    if caps is None:
        return None
    if age >= _REASONING_CAPS_DISK_TTL_SECONDS:
        _warm_reasoning_caps_async(refresh)
    return caps


def _seed_reasoning_caps(
    url: str, items: Any
) -> Optional[dict[str, Optional[dict[str, Any]]]]:
    """Parse a ``/v1/models`` ``data`` array and mirror it for *url*.

    Takes the payload rather than fetching it, so the picker and pricing
    fetches — which pull the same document a capability fetch would — leave the
    mirror warm at no network cost. None when the array has no usable entries,
    which callers remember as a failure rather than caching as empty.
    """
    if not isinstance(items, list):
        return None
    caps_by_id: dict[str, Optional[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        caps_by_id[mid] = parse_openrouter_reasoning_capabilities(item)
    if not caps_by_id:
        return None
    _save_reasoning_caps_disk(url, caps_by_id)
    return caps_by_id


def _fetch_reasoning_caps_catalog(
    url: str, timeout: float
) -> Optional[dict[str, Optional[dict[str, Any]]]]:
    """Fetch one OpenRouter-shaped ``/v1/models`` catalog → per-model caps.

    Shared by every aggregator that serves OpenRouter's catalog schema
    (OpenRouter itself, Nous Portal). Returns None when the catalog is
    unreachable or carries no usable entries, so callers can remember the
    failure and fall back rather than caching an empty result.

    Sends a User-Agent because the Portal 403s anonymous catalog reads.
    """
    headers = {"Accept": "application/json", "User-Agent": _HERMES_USER_AGENT}
    try:
        req = urllib.request.Request(url, headers=headers)
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return None
    return _seed_reasoning_caps(url, payload.get("data"))


_OPENROUTER_CATALOG_URL = "https://openrouter.ai/api/v1/models"


def _fetch_openrouter_reasoning_caps(
    timeout: float = 6.0, *, force: bool = False
) -> Optional[dict[str, Optional[dict[str, Any]]]]:
    """Fetch + cache per-model reasoning capabilities from the live catalog.

    Returns None (without poisoning the cache) when the catalog is
    unreachable so callers can retry later and fall back in the meantime.
    Failed fetches are remembered for 60 seconds so hot per-turn callers
    don't pay an HTTP round-trip on every call while offline. *force* refetches
    past a cache populated from the disk mirror.
    """
    global _openrouter_reasoning_caps_cache, _openrouter_reasoning_caps_failed_at
    if _openrouter_reasoning_caps_cache is not None and not force:
        return _openrouter_reasoning_caps_cache
    if (
        _openrouter_reasoning_caps_failed_at is not None
        and (time.monotonic() - _openrouter_reasoning_caps_failed_at) < 60
    ):
        return None
    caps_by_id = _fetch_reasoning_caps_catalog(_OPENROUTER_CATALOG_URL, timeout)
    if caps_by_id is None:
        _openrouter_reasoning_caps_failed_at = time.monotonic()
        return None
    _openrouter_reasoning_caps_cache = caps_by_id
    return caps_by_id


def _refresh_openrouter_reasoning_caps() -> None:
    _fetch_openrouter_reasoning_caps(force=True)


def openrouter_model_reasoning_capabilities(
    model_id: Optional[str],
    *,
    timeout: float = 6.0,
    allow_fetch: bool = False,
) -> Optional[dict[str, Any]]:
    """Return live-catalog reasoning capabilities for an OpenRouter model.

    Tri-state contract for callers deciding whether to emit reasoning
    controls:
      - dict with ``supports_reasoning: True`` (+ ``supported_efforts``,
        ``mandatory``) — the route advertises reasoning controls;
      - dict with ``supports_reasoning: False`` — the catalog knows the model
        and it does NOT accept reasoning controls (definitive negative);
      - ``None`` — unknown: catalog not loaded yet, model not listed
        (private/custom route), or entry malformed. Callers should fall back
        to their static heuristics rather than treating this as a negative.

    By default this is a CACHE-ONLY lookup — safe on per-request hot paths
    (never blocks on HTTP). The cache is populated for free whenever
    ``fetch_openrouter_models()`` runs (model picker, setup), from the disk
    mirror a previous run left behind, by the non-blocking
    ``warm_openrouter_reasoning_caps_async()`` warmer, or by passing
    ``allow_fetch=True`` from non-latency-sensitive callers.
    """
    model = str(model_id or "").strip()
    if not model:
        return None
    caps_by_id = _openrouter_caps_cached()
    if caps_by_id is None and allow_fetch:
        caps_by_id = _fetch_openrouter_reasoning_caps(timeout=timeout)
    if caps_by_id is None:
        return None
    return caps_by_id.get(model)


_openrouter_caps_disk_checked = False
_openrouter_caps_warm_started = False
_nous_reasoning_caps_cache: dict[str, Optional[dict[str, Any]]] | None = None
_nous_reasoning_caps_failed_at: float | None = None
_nous_caps_disk_checked = False
_nous_caps_warm_started = False


def _openrouter_caps_cached() -> Optional[dict[str, Optional[dict[str, Any]]]]:
    """Cache-only OpenRouter caps: memory, else the disk mirror. Never HTTP."""
    global _openrouter_reasoning_caps_cache, _openrouter_caps_disk_checked
    if _openrouter_reasoning_caps_cache is None and not _openrouter_caps_disk_checked:
        _openrouter_caps_disk_checked = True
        _openrouter_reasoning_caps_cache = _hydrate_reasoning_caps_from_disk(
            _OPENROUTER_CATALOG_URL, _refresh_openrouter_reasoning_caps
        )
    return _openrouter_reasoning_caps_cache


def warm_openrouter_reasoning_caps_async() -> None:
    """Warm the OpenRouter reasoning-capability cache in the background."""
    global _openrouter_caps_warm_started
    if _openrouter_caps_warm_started or _openrouter_caps_cached() is not None:
        return
    _openrouter_caps_warm_started = True
    _warm_reasoning_caps_async(_refresh_openrouter_reasoning_caps)


# Nous Portal serves OpenRouter's catalog schema, so the same parser and
# tri-state contract apply. Kept in its own cache because the two catalogs
# list different models (and different capabilities for shared ids).
_nous_reasoning_caps_cache: dict[str, Optional[dict[str, Any]]] | None = None
_nous_reasoning_caps_failed_at: float | None = None


def nous_catalog_url() -> str:
    """The Portal ``/v1/models`` URL for the endpoint we actually talk to.

    Resolved through the documented ladder rather than pinned to production —
    ``NOUS_INFERENCE_BASE_URL`` → resolved credential base → prod — so a
    staging profile reads staging's capabilities. Reading prod's would answer
    the reasoning-mandatory question for the wrong deployment.
    """
    return f"{_resolve_nous_pricing_credentials()[1]}/v1/models"


def _fetch_nous_reasoning_caps(
    timeout: float = 6.0, *, force: bool = False
) -> Optional[dict[str, Optional[dict[str, Any]]]]:
    """Nous Portal counterpart of :func:`_fetch_openrouter_reasoning_caps`."""
    global _nous_reasoning_caps_cache, _nous_reasoning_caps_failed_at
    if _nous_reasoning_caps_cache is not None and not force:
        return _nous_reasoning_caps_cache
    if (
        _nous_reasoning_caps_failed_at is not None
        and (time.monotonic() - _nous_reasoning_caps_failed_at) < 60
    ):
        return None
    caps_by_id = _fetch_reasoning_caps_catalog(nous_catalog_url(), timeout)
    if caps_by_id is None:
        _nous_reasoning_caps_failed_at = time.monotonic()
        return None
    _nous_reasoning_caps_cache = caps_by_id
    return caps_by_id


def _refresh_nous_reasoning_caps() -> None:
    _fetch_nous_reasoning_caps(force=True)


def nous_model_reasoning_capabilities(
    model_id: Optional[str],
    *,
    timeout: float = 6.0,
    allow_fetch: bool = False,
) -> Optional[dict[str, Any]]:
    """Return live-catalog reasoning capabilities for a Nous Portal model.

    Same tri-state contract and cache-only default as
    :func:`openrouter_model_reasoning_capabilities`; warm the cache with
    :func:`warm_nous_reasoning_caps_async` from hot paths.
    """
    model = str(model_id or "").strip()
    if not model:
        return None
    caps_by_id = _nous_caps_cached()
    if caps_by_id is None and allow_fetch:
        caps_by_id = _fetch_nous_reasoning_caps(timeout=timeout)
    if caps_by_id is None:
        return None
    return caps_by_id.get(model)


_nous_caps_disk_checked = False
_nous_caps_warm_started = False


def _nous_caps_cached() -> Optional[dict[str, Optional[dict[str, Any]]]]:
    """Cache-only Portal caps: memory, else the disk mirror. Never HTTP.

    Guarded to one attempt per process because naming the catalog means
    resolving Portal credentials, which can itself reach the network to
    refresh a token — far too expensive for a caller that runs every turn.
    """
    global _nous_reasoning_caps_cache, _nous_caps_disk_checked
    if _nous_reasoning_caps_cache is None and not _nous_caps_disk_checked:
        _nous_caps_disk_checked = True
        _nous_reasoning_caps_cache = _hydrate_reasoning_caps_from_disk(
            nous_catalog_url(), _refresh_nous_reasoning_caps
        )
    return _nous_reasoning_caps_cache


def warm_nous_reasoning_caps_async() -> None:
    """Nous Portal counterpart of :func:`warm_openrouter_reasoning_caps_async`."""
    global _nous_caps_warm_started
    if _nous_caps_warm_started or _nous_caps_cached() is not None:
        return
    _nous_caps_warm_started = True
    _warm_reasoning_caps_async(_refresh_nous_reasoning_caps)


# Canonical low→high ordering used for nearest-level clamping. Kept as an
# alias of the single source of truth in ``agent.reasoning_effort``.
from agent.reasoning_effort import EFFORT_LADDER as _REASONING_EFFORT_ORDER
from agent.reasoning_effort import clamp_effort as _clamp_effort


def clamp_reasoning_effort_to_supported(
    effort: Optional[str], supported_efforts: Optional[list[str]]) -> Optional[str]:
    """Thin wrapper over :func:`agent.reasoning_effort.clamp_effort`: keep a supported level verbatim,
    else the nearest WEAKER supported level (never silently escalate cost), else the weakest; unknown
    supported-sets and bespoke level names pass through unchanged."""
    return _clamp_effort(effort, supported_efforts)

    Thin wrapper over the canonical policy in
    :func:`agent.reasoning_effort.clamp_effort` (single implementation for
    every transport and provider profile): keep a supported level verbatim,
    otherwise nearest WEAKER supported level (never silently escalate cost),
    weakest supported level when nothing weaker exists, pass through unknown
    supported-sets and bespoke level names unchanged.

    Ported from PrimeIntellect-ai/prime-agent#1258's thinking-level-map
    normalization.
    """
    return _clamp_effort(effort, supported_efforts)


def fetch_openrouter_models(
    timeout: float = 8.0, *, force_refresh: bool = False) -> list[tuple[str, str]]:
    """Return the curated OpenRouter picker list, refreshed from the live catalog when possible."""
    # The curated list is filtered from this profile's manifest (``model_catalog.*`` config, its
    # ``<home>/cache`` copy), so a routed profile keeps its own slot instead of the module one.
    from hermes_cli.models_profile_cache import profile_slot_get, profile_slot_set
    _me = sys.modules[__name__]
    cached = profile_slot_get(_me, "_openrouter_catalog_cache")

    if cached is not None and not force_refresh:
        return list(cached)

    # Cold process: serve from the persisted disk cache when fresh so the
    # picker doesn't re-download the full ~686KB catalog on every open.
    if not force_refresh:
        disk = _read_openrouter_catalog_disk()
        if disk:
            profile_slot_set(_me, "_openrouter_catalog_cache", disk)
            return list(disk)

    # Remote catalog manifest first, in-repo snapshot when unreachable; the live /v1/models filter
    # (tool support, free pricing) is applied on top either way.
    try:
        from hermes_cli.model_catalog import get_curated_openrouter_models
        remote = get_curated_openrouter_models()
    except Exception:
        remote = None
    fallback = list(remote) if remote else list(OPENROUTER_MODELS)

    try:
        req = urllib.request.Request(
            _OPENROUTER_CATALOG_URL,
            headers={"Accept": "application/json"},
        )
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return list(_openrouter_catalog_cache or fallback)

    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return list(_openrouter_catalog_cache or fallback)

    live_by_id: dict[str, dict[str, Any]] = {}
    for item in live_items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        live_by_id[mid] = item

    # Free warm-up for the reasoning-capability cache: this is the same payload
    # _fetch_openrouter_reasoning_caps would fetch, so parse it once here and
    # hot-path callers (openrouter_model_reasoning_capabilities) never need
    # their own HTTP round-trip.
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
        return list(cached or fallback)
    if not curated[0][1]:
        curated[0] = (curated[0][0], "recommended")
    profile_slot_set(_me, "_openrouter_catalog_cache", curated)
    _write_openrouter_catalog_disk(curated)
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
    live = _fetch_live_catalog_index(f"{AI_GATEWAY_BASE_URL.rstrip('/')}/models", timeout, _urlopen_model_catalog_request)
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


def compute_sale_discount(
    prompt: str,
    completion: str,
    original: Any,
) -> tuple[int, str, str] | None:
    """Derive sale chrome from gateway ``pricing.original`` when cheaper.

    Nous Portal-only feature: callers gate on the provider; this helper only
    sees ``original`` because the Nous fetch path opted in via
    ``include_sale_original=True``.

    Returns ``(discount_percent, was_prompt_raw, was_completion_raw)`` only when
    ``original`` is a dict and the current prompt (fallback: completion) rate
    is strictly below the corresponding original. Percent is
    ``round((1 - current/original) * 100)`` — never hardcoded, and a discount
    that rounds below 1% is treated as no sale (never render "-0%"). Returns
    ``None`` when there is no sale (missing/equal/invalid original), so UIs
    show normal prices.

    Free / $0 models are a special case: they are always "-100%" sale chrome
    (Teknium, Aug 2026 — the picker's discount column should say 100% off
    rather than sit blank on free rows). The ``was_*`` raws come from
    ``original`` when the gateway serves one and are empty strings otherwise;
    callers must skip the "was" segment when both are empty.
    """
    def _finite(raw: Any) -> float | None:
        try:
            n = float(raw)
        except (TypeError, ValueError):
            return None
        return n if n > 0 and n == n else None  # n == n rejects NaN

    def _nonneg(raw: Any) -> float | None:
        try:
            n = float(raw)
        except (TypeError, ValueError):
            return None
        return n if n >= 0 and n == n else None

    orig_dict = original if isinstance(original, dict) else {}
    was_prompt = orig_dict.get("prompt")
    was_completion = orig_dict.get("completion")

    # Free / $0 models: flat 100% off, with "was" prices only when the
    # gateway actually served an original (e.g. a :free sibling); a
    # natively-free model (stealth/ox-alpha) gets bare "-100%" chrome.
    cur_prompt_any = _nonneg(prompt) if prompt not in (None, "") else None
    cur_comp_any = _nonneg(completion) if completion not in (None, "") else None
    if cur_prompt_any == 0 and cur_comp_any in (0, None):
        return (
            100,
            str(was_prompt) if was_prompt not in (None, "") else "",
            str(was_completion) if was_completion not in (None, "") else "",
        )

    if not isinstance(original, dict):
        return None

    if was_prompt in (None, "") and was_completion in (None, ""):
        return None

    cur_prompt = _finite(prompt) if prompt not in (None, "") else None
    orig_prompt = _finite(was_prompt) if was_prompt not in (None, "") else None
    if cur_prompt is not None and orig_prompt is not None and cur_prompt < orig_prompt:
        pct = int(round((1.0 - (cur_prompt / orig_prompt)) * 100))
        if pct < 1:
            return None
        return (
            pct,
            str(was_prompt),
            str(was_completion) if was_completion not in (None, "") else "",
        )

    cur_comp = _finite(completion) if completion not in (None, "") else None
    orig_comp = _finite(was_completion) if was_completion not in (None, "") else None
    if cur_comp is not None and orig_comp is not None and cur_comp < orig_comp:
        pct = int(round((1.0 - (cur_comp / orig_comp)) * 100))
        if pct < 1:
            return None
        return (
            pct,
            str(was_prompt) if was_prompt not in (None, "") else "",
            str(was_completion),
        )

    return None


def fetch_models_with_pricing(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api",
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
    include_sale_original: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fetch ``/v1/models`` and return ``{model_id: {prompt, completion, ...}}``.

    Results are cached per *base_url* so repeated calls are free.
    Works with any OpenRouter-compatible endpoint (OpenRouter, Nous Portal).

    When *include_sale_original* is true (Nous Portal only) and the gateway
    advertises a global discount under ``pricing.original``, those
    pre-discount rates are copied through as a nested ``original`` dict so
    pickers can show sale chrome. Other providers never opt in — OpenRouter
    (and anything else sharing this helper) keeps the legacy
    ``{prompt, completion}`` shape even if a response happens to nest
    ``original``.
    """
    cache_key = (base_url or "").rstrip("/")
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    url = cache_key + "/v1/models"
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": _HERMES_USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return _cache_catalog(cache_key, {})

    # Same document the reasoning-capability fetch would pull, and every
    # picker/pricing surface goes through here — mirror it so a later hot-path
    # lookup (and the next process) has an answer without its own round-trip.
    _seed_reasoning_caps(url, payload.get("data"))

    result: dict[str, dict[str, Any]] = {}
    for item in payload.get("data", []):
        mid = item.get("id")
        pricing = item.get("pricing")
        if mid and isinstance(pricing, dict):
            entry: dict[str, Any] = {
                "prompt": str(pricing.get("prompt", "")),
                "completion": str(pricing.get("completion", "")),
            }
            if pricing.get("input_cache_read"):
                entry["input_cache_read"] = str(pricing["input_cache_read"])
            if pricing.get("input_cache_write"):
                entry["input_cache_write"] = str(pricing["input_cache_write"])
            # Sale chrome is Nous Portal-only. Never copy pricing.original for
            # OpenRouter / other OpenAI-compatible catalogs.
            if include_sale_original:
                original = pricing.get("original")
                if isinstance(original, dict):
                    orig_entry: dict[str, str] = {}
                    for key in (
                        "prompt",
                        "completion",
                        "input_cache_read",
                        "input_cache_write",
                    ):
                        if original.get(key) not in (None, ""):
                            orig_entry[key] = str(original[key])
                    if orig_entry.get("prompt") or orig_entry.get("completion"):
                        entry["original"] = orig_entry
            result[mid] = entry

    return _cache_catalog(cache_key, result)


def fetch_ai_gateway_pricing(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Fetch Vercel AI Gateway /v1/models and return hermes-shaped pricing.

    Vercel uses ``input`` / ``output`` field names; hermes's picker expects
    ``prompt`` / ``completion``. This translates. Cache read/write field names
    already match.
    """
    from hermes_constants import AI_GATEWAY_BASE_URL

    cache_key = AI_GATEWAY_BASE_URL.rstrip("/")
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    try:
        req = urllib.request.Request(
            f"{cache_key}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return _cache_catalog(cache_key, {})

    result: dict[str, dict[str, str]] = {}
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        pricing = item.get("pricing")
        if not (mid and isinstance(pricing, dict)):
            continue
        entry: dict[str, str] = {
            "prompt": str(pricing.get("input", "")),
            "completion": str(pricing.get("output", "")),
        }
        if pricing.get("input_cache_read"):
            entry["input_cache_read"] = str(pricing["input_cache_read"])
        if pricing.get("input_cache_write"):
            entry["input_cache_write"] = str(pricing["input_cache_write"])
        result[mid] = entry

    return _cache_catalog(cache_key, result)


def _resolve_openrouter_api_key() -> str:
    """Best-effort OpenRouter API key for pricing fetch."""
    return os.getenv("OPENROUTER_API_KEY", "").strip()


_DEFAULT_NOUS_INFERENCE_BASE = "https://inference-api.nousresearch.com"


def _resolve_nous_pricing_credentials() -> tuple[str, str]:
    """Return ``(api_key, base_url)`` for Nous Portal pricing.

    The Nous inference ``/v1/models`` endpoint exposes pricing without
    authentication, so the api_key is best-effort: when runtime credential
    resolution fails (expired refresh token, missing auth.json, etc.) we
    still return a usable inference base URL so the picker keeps working
    with anonymous pricing data.  Free-tier users in particular need this
    — pricing drives the free/paid partition, and silently returning empty
    pricing because of an auth blip makes the picker look broken ("No free
    models currently available").

    Base URL precedence (mirrors runtime credential resolution):
    1. ``NOUS_INFERENCE_BASE_URL`` env override (staging / preview)
    2. Resolved runtime credential ``base_url``
    3. Production default

    Without (1), a staging profile's sale ``pricing.original`` never
    reaches the pickers — the anonymous fallback would hit prod, which
    has no ``original`` field.
    """
    env_base = None
    try:
        from hermes_cli.auth import _nous_inference_env_override

        env_base = _nous_inference_env_override()
    except Exception:
        env_base = None

    api_key = ""
    creds_base = ""
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials()
        if creds:
            api_key = creds.get("api_key", "") or ""
            creds_base = (creds.get("base_url", "") or "").strip()
    except Exception:
        pass

    base_url = (env_base or creds_base or _DEFAULT_NOUS_INFERENCE_BASE).rstrip("/")
    # Credential bases arrive with or without the ``/v1`` suffix. Callers
    # append their own path, so hand back the bare origin.
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return (api_key, base_url)


def get_pricing_for_provider(provider: str, *, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """Return live pricing for providers that support it (openrouter, nous, ai-gateway, novita)."""
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_models_with_pricing(
            api_key=_resolve_openrouter_api_key(),
            base_url="https://openrouter.ai/api",
            force_refresh=force_refresh,
        )
    if normalized == "ai-gateway":
        return fetch_ai_gateway_pricing(force_refresh=force_refresh)
    if normalized == "novita":
        return _fetch_novita_pricing(force_refresh=force_refresh)
    if normalized == "deepinfra":
        return _fetch_deepinfra_pricing(force_refresh=force_refresh)
    if normalized == "fireworks":
        return _fireworks_pricing_from_models_dev(force_refresh=force_refresh)
    if normalized == "nous":
        api_key, base_url = _resolve_nous_pricing_credentials()
        if base_url:
            return fetch_models_with_pricing(
                api_key=api_key,
                base_url=base_url,
                force_refresh=force_refresh,
                # Sale chrome (pricing.original) is Nous Portal-only.
                include_sale_original=True,
            )
    return {}


def _fireworks_pricing_from_models_dev(
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Derive Fireworks picker pricing from the models.dev registry cache.

    No dedicated network fetch: ``fetch_models_dev()`` already maintains an
    in-memory + disk cache (1h TTL) that every picker surface shares, so this
    is a pure dict transform on the picker path — no added latency and no
    per-render network call. Results are additionally memoized in
    ``_pricing_cache`` so repeated menu renders within a process are free.

    models.dev publishes Fireworks costs in USD per 1M tokens; the shared
    pricing formatter expects per-token strings, so divide by 1M.
    """
    cache_key = "models.dev/fireworks"
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    result: dict[str, dict[str, str]] = {}
    try:
        from hermes_cli.auth import get_auth_status, has_usable_secret

        if pid == "custom":
            return bool((_get_custom_base_url() or "").strip())
        if pid == "openrouter":
            from hermes_cli.model_switch import _scoped_key_env
            return has_usable_secret(_scoped_key_env("OPENROUTER_API_KEY"))
        status = get_auth_status(pid)
        return bool(status.get("logged_in") or status.get("configured"))
    except Exception:
        return False


def _configured_custom_provider_ids() -> set[str]:
    """Return routable custom-provider IDs configured by the user."""
    ids = {"custom"}
    try:
        from hermes_cli.config import load_config
        from hermes_cli.providers import custom_provider_slug

        config = load_config()
        providers = config.get("providers", {})
        if isinstance(providers, dict):
            for key, entry in providers.items():
                if isinstance(entry, dict):
                    ids.add(custom_provider_slug(str(entry.get("name") or key), str(key)))
        legacy = config.get("custom_providers", [])
        if isinstance(legacy, list):
            for entry in legacy:
                if isinstance(entry, dict):
                    ids.add(custom_provider_slug(str(entry.get("name") or "")))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError):
        pass
    return ids

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
                lowered = stripped.lower()
                for custom_id in sorted(
                    _configured_custom_provider_ids() - {"custom"},
                    key=len,
                    reverse=True,
                ):
                    prefix = f"{custom_id.lower()}:"
                    if lowered.startswith(prefix):
                        return custom_id, stripped[len(custom_id) + 1 :].strip()
            # Support custom:name:model triple syntax for named custom
            # providers.  ``custom:local:qwen`` → ("custom:local", "qwen").
            # Single colon ``custom:qwen`` → ("custom", "qwen") as before.
            if provider_part == "custom" and ":" in model_part:
                second_colon = model_part.find(":")
                custom_name = model_part[:second_colon].strip()
                actual_model = model_part[second_colon + 1:].strip()
                if custom_name and actual_model:
                    custom_id = f"custom:{custom_name.lower()}"
                    if custom_id in _configured_custom_provider_ids():
                        return (custom_id, actual_model)
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


def _get_provider_config_dict(provider: str) -> dict[str, Any]:
    """Return config.yaml providers.<provider>, or an empty dict."""
    key = str(provider or "").strip()
    if not key:
        return {}
    try:
        from hermes_cli.config import load_config
        config = load_config()
        providers_cfg = config.get("providers", {})
        if isinstance(providers_cfg, dict):
            entry = providers_cfg.get(key) or providers_cfg.get(key.lower())
            if isinstance(entry, dict):
                return entry
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError):
        pass
    return {}


def _root_for_ollama_native_api(base_url: str) -> str:
    """Convert an OpenAI-style Ollama base URL to the native API root."""
    root = str(base_url or "").strip().rstrip("/")
    if root.startswith(":"):
        root = "http://127.0.0.1" + root
    elif root and "://" not in root:
        root = "http://" + root
    for suffix in ("/api/tags", "/v1/models", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    return root


def _normalize_openai_base_url(base_url: Optional[str]) -> str:
    """Add a usable HTTP scheme without changing an OpenAI API path."""
    value = str(base_url or "").strip()
    if value.startswith(":"):
        return "http://127.0.0.1" + value
    if value and "://" not in value:
        return "http://" + value
    return value


def _get_ollama_base_url() -> str:
    """Resolve the local Ollama-compatible endpoint URL.

    Prefer explicit config under ``providers.ollama.base_url`` because this is
    how local Ollama-compatible endpoints can be wired without changing the
    active model provider. Fall back to active ``model.base_url`` only when the
    active provider is ollama/custom, then to Ollama's local default.
    """
    provider_cfg = _get_provider_config_dict("ollama")
    configured = (
        provider_cfg.get("base_url", "")
        or provider_cfg.get("api", "")
        or provider_cfg.get("url", "")
        or ""
    )
    if configured:
        return str(configured).strip()

    model_cfg = _get_model_config_dict()
    model_provider = str(model_cfg.get("provider", "") or "").strip().lower()
    model_base = str(model_cfg.get("base_url", "") or "").strip()
    if model_provider == "ollama" and model_base:
        return model_base
    if model_provider == "custom" and model_base:
        # Only reuse the active bare custom endpoint when it is actually
        # Ollama-compatible. Otherwise a user working against an unrelated
        # OpenAI-compatible endpoint would make the Ollama picker probe that
        # endpoint's /api/tags and hide their local Ollama catalog.
        try:
            if should_use_ollama_native_catalog("custom", model_base):
                return model_base
        except (OSError, RuntimeError, TypeError, ValueError):
            pass

    env_host = os.getenv("OLLAMA_HOST", "").strip()
    if env_host:
        if env_host.startswith(":") and not env_host.startswith("::"):
            env_host = "127.0.0.1" + env_host
        elif env_host.startswith("[") and env_host.endswith("]"):
            env_host = f"{env_host}:11434"
        elif "://" in env_host:
            try:
                parsed = urllib.parse.urlsplit(env_host)
                if parsed.hostname and parsed.port is None:
                    hostname = parsed.hostname
                    if ":" in hostname and not hostname.startswith("["):
                        hostname = f"[{hostname}]"
                    userinfo = (
                        parsed.netloc.rsplit("@", 1)[0] + "@"
                        if "@" in parsed.netloc
                        else ""
                    )
                    env_host = parsed._replace(
                        netloc=f"{userinfo}{hostname}:11434"
                    ).geturl()
            except ValueError:
                pass
        elif env_host.count(":") > 1 and not env_host.startswith("["):
            env_host = f"[{env_host}]:11434"
        elif ":" not in env_host:
            env_host = f"{env_host}:11434"
        return env_host
    return "http://localhost:11434"


def _get_ollama_request_headers() -> dict[str, str]:
    """Return configured headers and credentials for native Ollama requests."""
    entry = _get_provider_config_dict("ollama")
    raw = entry.get("extra_headers")
    try:
        from hermes_cli.config import normalize_extra_headers

        result = normalize_extra_headers(raw)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        result = {}

    api_key = str(entry.get("api_key") or "").strip()
    if not api_key:
        key_env = str(
            entry.get("key_env") or entry.get("api_key_env") or ""
        ).strip()
        api_key = os.getenv(key_env, "").strip() if key_env else ""
    if api_key:
        if not any(key.lower() == "authorization" for key in result):
            result["Authorization"] = f"Bearer {api_key}"
    return result


def _get_ollama_native_headers(
    base_url: Optional[str],
    *,
    api_key: Optional[str] = None,
) -> dict[str, str]:
    """Resolve Ollama credentials and headers for one endpoint origin."""
    entry = _get_provider_config_dict("ollama")
    configured_base = str(
        entry.get("base_url") or entry.get("api") or entry.get("url") or ""
    ).strip()
    explicit_key = str(api_key or "").strip()
    configured_matches = bool(
        configured_base
        and base_url
        and _same_ollama_native_root(base_url, configured_base)
    )
    if not configured_matches and not explicit_key:
        return {}
    headers = _get_ollama_request_headers() if configured_matches else {}
    if explicit_key:
        # A provider-specific key must not inherit any configured Authorization
        # variant from the Ollama origin when both share a native root.
        for key in tuple(headers):
            if key.lower() == "authorization":
                del headers[key]
        headers["Authorization"] = f"Bearer {explicit_key}"
    return headers


_OLLAMA_LOCAL_MODELS_CACHE_TTL: int = 300  # seconds (5 minutes)
_OLLAMA_LOCAL_MODELS_CACHE: dict[str, tuple[tuple[str, ...], float]] = {}
_OLLAMA_LOCAL_PROBE_FAILURE_CACHE: dict[str, float] = {}
_OLLAMA_LOCAL_PROBE_REACHABLE: dict[str, bool] = {}
_OLLAMA_LOCAL_PROBE_FAILURE_TTL: int = 30
_OLLAMA_LOCAL_CACHE_MAX_ENTRIES: int = 256


def _evict_related_ollama_cache_entries(key: str) -> None:
    _OLLAMA_LOCAL_MODELS_CACHE.pop(key, None)
    _OLLAMA_LOCAL_PROBE_REACHABLE.pop(key, None)
    for failure_key in list(_OLLAMA_LOCAL_PROBE_FAILURE_CACHE):
        if failure_key == key or failure_key.startswith(f"{key}|timeout:"):
            _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.pop(failure_key, None)


def _remember_ollama_cache(cache: dict[str, Any], key: str, value: Any) -> None:
    if key not in cache and len(cache) >= _OLLAMA_LOCAL_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(cache))
        _evict_related_ollama_cache_entries(
            oldest_key.split("|timeout:", 1)[0]
        )
    cache[key] = value


def _ollama_probe_cache_key(root: str, headers: Optional[dict[str, str]]) -> str:
    cache_key = root
    if headers:
        import hashlib

        normalized_headers = sorted(
            (str(key).lower(), str(value)) for key, value in headers.items()
        )
        header_blob = json.dumps(
            normalized_headers, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8", errors="replace")
        header_fingerprint = hashlib.blake2b(header_blob, digest_size=8).hexdigest()
        cache_key = f"{root}|headers:{header_fingerprint}"
    return cache_key


def probe_ollama_local_models(
    base_url: Optional[str] = None,
    timeout: float = 2.0,
    headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Probe local Ollama-compatible models from native ``/api/tags``.

    Returns ``None`` when the endpoint cannot be reached or returns malformed
    data, and a list (possibly empty) when ``/api/tags`` was reachable. Stock
    Ollama exposes its authoritative local model catalog at ``/api/tags``;
    OpenAI-compatible ``/v1/models`` is not required for local Ollama servers.
    """
    root = _root_for_ollama_native_api(base_url or _get_ollama_base_url())
    if not root:
        return None
    cache_key = _ollama_probe_cache_key(root, headers)
    failure_key = f"{cache_key}|timeout:{float(timeout):.3f}"
    cached = _OLLAMA_LOCAL_MODELS_CACHE.get(cache_key)
    if cached is not None:
        cached_models, cached_at = cached
        if time.monotonic() - cached_at < _OLLAMA_LOCAL_MODELS_CACHE_TTL:
            return list(cached_models)
    failed_at = _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.get(failure_key)
    if failed_at is not None:
        if time.monotonic() - failed_at < _OLLAMA_LOCAL_PROBE_FAILURE_TTL:
            return None
        _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.pop(failure_key, None)

    try:
        url = root.rstrip("/") + "/api/tags"
        request_headers = {"User-Agent": _HERMES_USER_AGENT}
        request_headers.update(headers or {})
        req = urllib.request.Request(url, headers=request_headers)
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except (
        ValueError,
        OSError,
        TimeoutError,
        http.client.HTTPException,
        urllib.error.URLError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, False
        )
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_FAILURE_CACHE, failure_key, time.monotonic()
        )
        return None

    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, False
        )
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_FAILURE_CACHE, failure_key, time.monotonic()
        )
        return None

    models: list[str] = []
    seen: set[str] = set()
    for item in raw_models:
        if isinstance(item, dict):
            model_id = str(item.get("model") or item.get("name") or "").strip()
        else:
            _remember_ollama_cache(
                _OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, False
            )
            _remember_ollama_cache(
                _OLLAMA_LOCAL_PROBE_FAILURE_CACHE, failure_key, time.monotonic()
            )
            return None
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(model_id)
    if raw_models and not models:
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, False
        )
        _remember_ollama_cache(
            _OLLAMA_LOCAL_PROBE_FAILURE_CACHE, failure_key, time.monotonic()
        )
        return None
    _remember_ollama_cache(_OLLAMA_LOCAL_PROBE_REACHABLE, cache_key, True)
    _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.pop(failure_key, None)
    _remember_ollama_cache(
        _OLLAMA_LOCAL_MODELS_CACHE,
        cache_key,
        (tuple(models), time.monotonic()),
    )
    return models


def fetch_ollama_local_models(
    base_url: Optional[str] = None,
    timeout: float = 2.0,
    headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Fetch local Ollama-compatible models, preserving probe failure as ``None``."""
    return probe_ollama_local_models(base_url, timeout, headers=headers)


def _same_ollama_native_root(left: str, right: str) -> bool:
    """Return True when two Ollama/OpenAI-style base URLs share an API root."""
    left_root = _root_for_ollama_native_api(left).rstrip("/")
    right_root = _root_for_ollama_native_api(right).rstrip("/")
    if not left_root or not right_root:
        return False
    try:
        left_parts = urllib.parse.urlsplit(left_root)
        right_parts = urllib.parse.urlsplit(right_root)
        return (
            url_origin(left_root) == url_origin(right_root)
            and left_parts.path.rstrip("/") == right_parts.path.rstrip("/")
        )
    except (AttributeError, ValueError):
        return False


def should_use_ollama_native_catalog(
    provider: Optional[str],
    base_url: Optional[str],
    headers: Optional[dict[str, str]] = None,
) -> bool:
    """Return True when model discovery should use local Ollama ``/api/tags``.

    Bare ``ollama`` is normalized to ``custom`` elsewhere so runtime paths can
    share the OpenAI-compatible chat client. For model discovery and validation,
    however, local Ollama's authoritative model list is ``/api/tags``. Use that
    path when the caller explicitly asked for Ollama, when the base URL matches
    configured ``providers.ollama.base_url``, or when an ambiguous custom URL on
    Ollama's default local port actually serves ``/api/tags``. Ordinary custom
    endpoints stay on the existing OpenAI-compatible ``/models`` probe path.
    """
    requested = str(provider or "").strip().lower()
    root = _root_for_ollama_native_api(base_url or "")
    if root:
        try:
            host = (urllib.parse.urlparse(root).hostname or "").lower()
            if host == "ollama.com" or host.endswith(".ollama.com"):
                return False
        except ValueError:
            pass

    known_non_local_providers = {
        "openrouter",
        "nous",
        "anthropic",
        "openai",
        "openai-codex",
        "gemini",
        "ollama-cloud",
    }
    if requested in known_non_local_providers:
        return False

    if requested == "ollama":
        if not root:
            return False
        configured = _get_provider_config_dict("ollama")
        configured_base = str(
            configured.get("base_url")
            or configured.get("api")
            or configured.get("url")
            or ""
        ).strip()
        if configured_base and not _same_ollama_native_root(root, configured_base):
            return probe_ollama_local_models(root, timeout=0.5, headers=headers) is not None
        return True

    provider_cfg = _get_provider_config_dict("ollama")
    configured_ollama_base_url = str(
        provider_cfg.get("base_url", "")
        or provider_cfg.get("api", "")
        or provider_cfg.get("url", "")
        or ""
    ).strip()
    if configured_ollama_base_url and _same_ollama_native_root(root, configured_ollama_base_url):
        return True

    if not root:
        return False

    local_like_providers = {"", "custom", "local", "llamacpp", "llama.cpp", "llama-cpp", "vllm"}
    if requested not in local_like_providers and not requested.startswith("custom:"):
        return False

    if requested == "custom:ollama" or requested.endswith("-ollama"):
        return True

    try:
        parsed = urllib.parse.urlparse(root)
        if parsed.port != 11434:
            return False
    except ValueError:
        return False

    return probe_ollama_local_models(root, timeout=0.5, headers=headers) is not None


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


def _anthropic_models_url(base_url: Optional[str] = None, *, after_id: Optional[str] = None) -> str:
    """Anthropic ``/v1/models`` page URL. The endpoint is cursor-paginated with a default page of
    20 (smaller than the live catalog), so every request asks for the maximum page size and
    ``after_id`` continues from a previous page's ``last_id``."""
    endpoint = str(base_url or "https://api.anthropic.com").strip().rstrip("/")
    url = endpoint + ("/models" if endpoint.endswith("/v1") else "/v1/models")
    params = {"limit": "1000"}
    if after_id:
        params["after_id"] = after_id
    return url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)


_ANTHROPIC_MODELS_MAX_PAGES = 20


def _anthropic_next_cursor(page: Any, seen_cursors: set[str]) -> Optional[str]:
    """``last_id`` to continue from, or None when the page is final or the server repeats a
    cursor (which would otherwise loop forever)."""
    if not isinstance(page, dict) or page.get("has_more") is not True:
        return None
    last_id = page.get("last_id")
    if not isinstance(last_id, str) or not last_id or last_id in seen_cursors:
        return None
    seen_cursors.add(last_id)
    return last_id


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
        for model in _provider_catalog_names(provider)
    )


_AGGREGATOR_PROVIDERS = frozenset(
    {"nous", "openrouter", "ai-gateway", "copilot", "kilocode"}
)

# OpenRouter request-time routing variants (docs: guides/routing/model-variants).
# These suffixes are per-request routing modifiers valid on ANY model id —
# ":nitro" sorts the endpoint pool by throughput and admits priority-tier
# endpoints, ":floor" sorts by price and admits flex-tier endpoints, ":exacto"
# applies quality-first provider sorting, ":online" attaches the web plugin.
# They are never separate catalog entries: /models lists only the base id.
# NOT in this set: ":free", ":batch", ":thinking", ":extended" — those ARE
# distinct catalog SKUs that appear in /models when they exist, so absence
# from the listing is authoritative for them and the direct-membership check
# above handles the valid ones.
_OPENROUTER_VARIANT_SUFFIXES = frozenset({"nitro", "floor", "exacto", "online"})


def _openrouter_variant_base(model_id: str) -> Optional[str]:
    """Return the base model id when ``model_id`` carries a recognized
    OpenRouter routing-variant suffix (e.g. ``x-ai/grok-4:nitro`` →
    ``x-ai/grok-4``), else ``None``."""
    base, sep, suffix = (model_id or "").rpartition(":")
    if not sep or not base:
        return None
    if suffix.lower() in _OPENROUTER_VARIANT_SUFFIXES:
        return base
    return None

# Subscription/OAuth providers whose catalogs RE-EXPOSE other vendors' models
# would be listed here (tried only as a last resort for bare short-alias
# resolution, after every native-vendor catalog, so they never hijack an alias
# away from the model's native vendor). None are currently defined.
_BORROWED_MODEL_PROVIDERS: frozenset[str] = frozenset()

# Providers whose live /v1/models endpoint is the authoritative catalog, so the
# curated list is a discovery-only fallback. For these, the picker merges
# live-first (live entries lead, curated-only entries append). Every OTHER
# provider keeps curated-first (commit 658ac1d86, #46309) so a deliberately
# surfaced newest model stays at the top even when the live API lags. OpenCode
# Zen / Go re-expose dozens of upstream vendors and rotate them frequently, so
# their stale curated entries must not pollute the top of the picker. (#49129)
_LIVE_FIRST_PICKER_PROVIDERS: frozenset[str] = frozenset(
    {"opencode-zen", "opencode-go"}
)


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
    """Auto-detect the best provider for a model name: the current provider's live catalog, static
    catalogs (bare provider name → its default; direct catalog match), then the OpenRouter catalog,
    then a configured ``vendor/`` prefix.

    Never hands back a provider the user holds no credentials for: an unauthenticated guess is
    skipped and the ladder continues (``None`` = stay on the current provider). Exceptions: the user
    NAMED the provider (``/model nous``), or there is no current provider yet (``auto``) — then the
    first guess is returned so the credential step fails loudly instead of silently ignoring input."""
    from hermes_cli.models_detect import (
        current_provider_catalog_match, current_provider_owns_vendor, provider_has_credentials)

    name = (model_name or "").strip()
    if not name:
        return None

    # The current provider's LIVE catalog outranks every static guess: a model it already serves
    # (Codex early-access ids, Portal-only slugs, Ollama Cloud models absent from _PROVIDER_MODELS)
    # must never re-route the session to another vendor or to metered OpenRouter.
    served = current_provider_catalog_match(name, current_provider)
    if served is not None:
        return (current_provider, served) if served != name else None
    # Live catalog unavailable or lagging: the vendor's own id on the vendor's first-party provider
    # is still a selection — an aggregator relisting it is not grounds to switch.
    if current_provider_owns_vendor(name, current_provider):
        return None

    no_selection = (current_provider or "").strip().lower() in {"", "auto"}
    for candidate in _detection_candidates(name, current_provider):
        if candidate is None:
            return None  # the current catalog owns this name
        if no_selection or candidate[0] == current_provider or provider_has_credentials(candidate[0]):
            return candidate
        if _PROVIDER_ALIASES.get(name.lower(), name.lower()) == candidate[0]:
            return candidate  # explicitly named provider: let the credential step report it
        logger.debug("Skipping auto-switch of '%s' to %s: no credentials configured", name, candidate[0])
    # A ``vendor/model`` prefix naming a provider the user DECLARED in ``providers:`` is a selection,
    # not a guess — hand it back even before its key is wired up.
    return _resolve_provider_prefix(name)


def _detection_candidates(name: str, current_provider: str):
    """Yield ``(provider, model)`` guesses in ladder order; ``None`` means the current provider's own
    catalog owns the name (stop, stay)."""
    static_match = detect_static_provider_for_model(name, current_provider)
    if static_match:
        yield static_match
    if _model_in_provider_catalog(name.lower(), _provider_keys(current_provider)):
        yield None
        return

    # OpenRouter catalog (exact slug, then bare model part).
    or_slug = _find_openrouter_slug(name)
    if or_slug:
        if current_provider == "openrouter" and or_slug == name:
            yield None  # already on openrouter with matching name
            return
        yield ("openrouter", or_slug)

    # Explicit ``vendor/model`` prefix naming a configured provider — AFTER the OpenRouter lookup so
    # aggregator-native slugs (``deepseek/deepseek-chat``) keep their routing.
    prefixed = _resolve_provider_prefix(name)
    if prefixed:
        yield prefixed


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

    Tries live API endpoints for providers that support them (Codex, Nous),
    falling back to static lists. For providers in ``_MODELS_DEV_PREFERRED``
    (opencode-go/zen, xiaomi, deepseek, smaller inference providers, etc.),
    models.dev entries are merged on top of curated so new models released
    on the platform appear in ``/model`` without a Hermes release.
    """
    requested = str(provider or "").strip().lower()
    if requested == "ollama":
        if force_refresh:
            _OLLAMA_LOCAL_MODELS_CACHE.clear()
            _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.clear()
            _OLLAMA_LOCAL_PROBE_REACHABLE.clear()
        base_url = _get_ollama_base_url()
        headers = _get_ollama_native_headers(base_url)
        use_native = should_use_ollama_native_catalog(
            "ollama", base_url, headers=headers
        )
        if use_native:
            if headers:
                native_models = fetch_ollama_local_models(base_url, headers=headers)
            else:
                native_models = fetch_ollama_local_models(base_url)
            native_key = _ollama_probe_cache_key(
                _root_for_ollama_native_api(base_url), headers or None
            )
            if native_models or _OLLAMA_LOCAL_PROBE_REACHABLE.get(native_key) is True:
                return native_models or []
        else:
            # Non-native Ollama-compatible endpoints (including Ollama Cloud)
            # retain the generic OpenAI-compatible catalog path.
            pass
        # gateways that expose only OpenAI-style /v1/models.
        config = _get_provider_config_dict("ollama")
        fallback_key = str(config.get("api_key") or "").strip()
        if not fallback_key:
            key_env = str(config.get("key_env") or "").strip()
            fallback_key = os.getenv(key_env, "").strip() if key_env else ""
        fallback_base = _normalize_openai_base_url(
            config.get("base_url") or base_url
        )
        fallback_headers = _get_ollama_native_headers(
            fallback_base, api_key=fallback_key
        )
        fallback_models = fetch_api_models(
            fallback_key,
            fallback_base,
            headers=fallback_headers or None,
        )
        return fallback_models or []

    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return model_ids(force_refresh=force_refresh)
    if normalized == "openai-codex":
        from hermes_cli.codex_models import get_codex_model_ids

        # Pass the live OAuth access token so the picker matches whatever
        # ChatGPT lists for this account right now (new models appear without
        # a Hermes release). Falls back to the hardcoded catalog if no token
        # or the endpoint is unreachable.
        access_token = None
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
            access_token = creds.get("api_key")
        except Exception:
            access_token = None
        return get_codex_model_ids(access_token=access_token)
    if normalized in {"copilot", "copilot-acp"}:
        try:
            live = _fetch_github_models(_resolve_copilot_catalog_api_key())
            if live:
                return live
        except Exception:
            pass
        if normalized == "copilot-acp":
            return list(_PROVIDER_MODELS.get("copilot", []))
    if normalized == "nous":
        # Try live Nous Portal /models endpoint
        try:
            from hermes_cli.auth import fetch_nous_models, resolve_nous_runtime_credentials
            creds = resolve_nous_runtime_credentials()
            if creds:
                live = fetch_nous_models(api_key=creds.get("api_key", ""), inference_base_url=creds.get("base_url", ""))
                if live:
                    return live
        except Exception:
            pass
        # Live failed (or no creds). Fall back to the docs-hosted manifest
        # — NOT the in-repo _PROVIDER_MODELS["nous"] snapshot — so newly
        # added Portal models still surface without a Hermes release.
        manifest_ids = get_curated_nous_model_ids()
        if manifest_ids:
            return manifest_ids
    if normalized == "stepfun":
        try:
            from hermes_cli.auth import resolve_api_key_provider_credentials

            creds = resolve_api_key_provider_credentials("stepfun")
            api_key = str(creds.get("api_key") or "").strip()
            base_url = str(creds.get("base_url") or "").strip()
            if api_key and base_url:
                live = fetch_api_models(api_key, base_url)
                if live:
                    return live
        except Exception:
            pass
    if normalized == "anthropic":
        model_cfg = _get_model_config_dict()
        cfg_provider = normalize_provider(str(model_cfg.get("provider", "") or ""))
        if cfg_provider == "anthropic":
            cfg_base_url = str(model_cfg.get("base_url", "") or "").strip()
            cfg_api_key = str(model_cfg.get("api_key", "") or "").strip()
        else:
            cfg_base_url = ""
            cfg_api_key = ""
        live = _fetch_anthropic_models(
            base_url=cfg_base_url or None,
            api_key=cfg_api_key or None,
        )
        if live:
            if cfg_base_url:
                return live
            # The live /v1/models dump lags newly-routed curated aliases
            # (e.g. claude-fable-5, which is reachable on Anthropic before it
            # is enumerated by the models endpoint). Surface curated entries
            # first, then append any live-only models, so a fresh curated
            # model never disappears just because the API hasn't listed it yet.
            curated = list(_PROVIDER_MODELS.get("anthropic", []))
            merged = list(curated)
            merged_lower = {m.lower() for m in curated}
            for m in live:
                if m.lower() not in merged_lower:
                    merged.append(m)
                    merged_lower.add(m.lower())
            return merged
        return list(_PROVIDER_MODELS.get("anthropic", []))
    if normalized == "ai-gateway":
        live = _fetch_ai_gateway_models()
        if live:
            return live
    if normalized == "deepinfra":
        # DeepInfra's generic /models endpoint mixes chat, image, video,
        # speech, and embedding models. The tagged catalog helper is the only
        # safe source for the chat picker, including its empty/failure result.
        return _fetch_deepinfra_models(force_refresh=force_refresh) or []
    if normalized == "ollama-cloud":
        live = fetch_ollama_cloud_models(force_refresh=force_refresh)
        if live:
            return live
    if normalized in ("openai", "openai-api"):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            base = _openai_discovery_base_url(normalized)
            # Custom OpenAI-compatible endpoints (proxies, gateways, self-hosted)
            # may serve a small curated catalog — use the live list verbatim so
            # discovery works. But the official OpenAI hosts (canonical AND the
            # data-residency regional hosts, which serve the identical dump)
            # return 120+ entries of embeddings, whisper, tts, dall-e,
            # moderation and legacy chat models — none of which belong in the
            # agent model picker. For official hosts, intersect the live list
            # with our curated agentic catalog so ``/model`` matches what
            # ``hermes model`` shows.
            from hermes_cli.providers import is_official_openai_host

            is_default_openai = is_official_openai_host(base)
            try:
                live = fetch_api_models(api_key, base)
                if live:
                    if is_default_openai:
                        live_lower = {m.lower() for m in live}
                        curated = list(_PROVIDER_MODELS.get(normalized, []))
                        # Keep curated order; only surface curated models the
                        # account actually has access to.
                        filtered = [m for m in curated if m.lower() in live_lower]
                        if filtered:
                            return filtered
                        # Account serves none of the curated models (rare —
                        # e.g. org without GPT-5 access). Fall back to curated
                        # so the picker still offers sane defaults.
                        return curated or live
                    return live
            except Exception:
                pass
    if normalized == "gmi":
        try:
            from hermes_cli.auth import resolve_api_key_provider_credentials

            creds = resolve_api_key_provider_credentials("gmi")
            api_key = str(creds.get("api_key") or "").strip()
            base_url = str(creds.get("base_url") or "").strip()
            if api_key and base_url:
                live = fetch_api_models(api_key, base_url)
                if live:
                    return live
        except Exception:
            pass
    if normalized == "custom":
        base_url = _get_custom_base_url()
        if base_url:
            model_cfg = _get_model_config_dict()
            # Try common API key env vars for custom endpoints
            api_key = (
                str(model_cfg.get("api_key", "") or "").strip()
                or os.getenv("CUSTOM_API_KEY", "")
                or os.getenv("OPENAI_API_KEY", "")
                or os.getenv("OPENROUTER_API_KEY", "")
            )
            api_mode = "anthropic_messages" if _base_url_looks_like_anthropic_messages(base_url) else None
            live = fetch_api_models(api_key, base_url, api_mode=api_mode)
            if live:
                return live
    # Bedrock uses live discovery keyed by the resolved AWS region so that
    # EU/AP users see eu.*/ap.* model IDs instead of the static us.* list.
    # Note: early return intentionally skips _MODELS_DEV_PREFERRED merge
    # below — bedrock is not expected to appear in that table.
    if normalized == "bedrock":
        try:
            from agent.bedrock_adapter import bedrock_model_ids_or_none
            ids = bedrock_model_ids_or_none()
            if ids is not None:
                return ids
        except Exception:
            pass

    # OpenCode Free: keyless live catalog, revalidated against the Zen relay
    # every TTL. models.dev's cost.input==0 filter lags reality
    # (deepseek-v4-flash-free stayed "free" there after its promo ended and the
    # relay began 401ing keyless requests), so we filter the live /zen/v1/models
    # dump to the anonymous-servable `*-free` tier ourselves and fall back to
    # the curated _PROVIDER_MODELS floor only when the live fetch fails or is
    # empty. This is what keeps a relay-delisted model (e.g. x-preview-f-free)
    # from lingering in the picker until a release re-syncs the snapshot.
    if normalized == "opencode-free":
        return _fetch_opencode_free_models(
            force_refresh=force_refresh
        ) or list(_PROVIDER_MODELS.get(normalized, []))

    # ── Profile-based generic live fetch (all simple api-key providers) ──
    # Handles any provider registered in providers/ with auth_type="api_key".
    # Replaces per-provider copy-paste blocks (stepfun, gmi, zai, etc.).
    try:
        from hermes_cli.auth import resolve_codex_runtime_credentials

        access_token = resolve_codex_runtime_credentials(refresh_if_expiring=True).get("api_key")
    except Exception:
        access_token = None
    return get_codex_model_ids(access_token=access_token)


_COPILOT_ACP_SESSION_MEMO_TTL = 300.0  # 5 min; SWR disk cache handles the rest
_COPILOT_ACP_SESSION_FAIL_TTL = 30.0  # failed probes re-probe quickly so a fresh CLI login is picked up
_copilot_acp_session_memo: Optional[tuple[float, float, Optional[list[str]]]] = None  # (at, ttl, models)


def _copilot_acp_session_models(force_refresh: bool) -> Optional[list[str]]:
    """Enabled models from a signed-in ``copilot --acp`` session, memoized for a few minutes —
    successes AND failures. Model-switch validation (``models_validate._static_catalog``) reads
    this uncached on every ``/model`` switch, and each miss is a CLI spawn + handshake (up to the
    probe timeout), so without the memo every switch paid a subprocess. A failed probe is
    memoized much more briefly so a user who signs in to the CLI right after a miss is picked up
    on the next switch (or immediately via ``/model --refresh``, which clears this memo)."""
    global _copilot_acp_session_memo
    now = time.monotonic()
    memo = _copilot_acp_session_memo
    if not force_refresh and memo is not None and now - memo[0] < memo[1]:
        return memo[2]
    from providers import get_provider_profile

    try:
        live = get_provider_profile("copilot-acp").fetch_models() or None
    except Exception:
        logger.debug("copilot-acp session model discovery failed", exc_info=True)
        live = None
    _copilot_acp_session_memo = (now, _COPILOT_ACP_SESSION_MEMO_TTL if live else _COPILOT_ACP_SESSION_FAIL_TTL, live)
    return live


def _copilot_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    if normalized == "copilot-acp" and (live := _copilot_acp_session_models(force_refresh)):
        return live
    try:
        live = _fetch_github_models(_resolve_copilot_catalog_api_key())
        if live:
            return live
    except Exception:
        pass
    return list(_PROVIDER_MODELS.get("copilot", [])) if normalized == "copilot-acp" else None


def _nous_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    try:
        from hermes_cli.auth import fetch_nous_models, resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials()
        if creds:
            live = fetch_nous_models(api_key=creds.get("api_key", ""), inference_base_url=creds.get("base_url", ""))
            if live:
                return live
    except Exception:
        pass
    # Live failed / no creds: the docs-hosted manifest — NOT the in-repo snapshot — so newly added
    # Portal models still surface without a Hermes release.
    return get_curated_nous_model_ids() or None


def _api_key_credentials(normalized: str) -> tuple[str, str]:
    """``(api_key, base_url)`` from ``resolve_api_key_provider_credentials``; empty strings on any miss."""
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials(normalized)
        return str(creds.get("api_key") or "").strip(), str(creds.get("base_url") or "").strip()
    except Exception:
        return "", ""


def _api_key_provider_live(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    """Live /v1/models for a simple api-key provider (stepfun, gmi); None on any miss."""
    api_key, base_url = _api_key_credentials(normalized)
    if not (api_key and base_url):
        return None
    try:
        return fetch_api_models(api_key, base_url) or None
    except Exception:
        return None


def _anthropic_catalog(normalized: str, force_refresh: bool) -> list[str]:
    model_cfg = _get_model_config_dict()
    cfg_base_url = cfg_api_key = ""
    if normalize_provider(str(model_cfg.get("provider", "") or "")) == "anthropic":
        cfg_base_url = str(model_cfg.get("base_url", "") or "").strip()
        cfg_api_key = str(model_cfg.get("api_key", "") or "").strip()
    live = _fetch_anthropic_models(base_url=cfg_base_url or None, api_key=cfg_api_key or None)
    curated = list(_PROVIDER_MODELS.get("anthropic", []))
    if not live:
        return curated
    # The live /v1/models dump lags newly-routed curated aliases (reachable before enumerated):
    # curated first, then live-only extras, so a fresh curated model never disappears.
    return live if cfg_base_url else _merge_unique(curated, live)


def _openai_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    base = _openai_discovery_base_url(normalized)
    # Custom OpenAI-compatible endpoints serve a small curated catalog — use it verbatim. Official
    # OpenAI hosts (canonical and data-residency regional) return 120+ embeddings/whisper/tts/…
    # entries, so intersect with the curated agentic catalog so ``/model`` matches ``hermes model``.
    # Model not in live /v1/models — check the curated catalog before rejecting. Providers may omit models
    # from their live listing that are still valid (stale cache, partial rollout, gated previews). Use the
    # pure-catalog helper (no extra live fetch) so we only accept models Hermes actually ships. (#46850)
    # Their /v1/models listing is access-scoped and authoritative — a model absent from it is one this key
    # CANNOT serve, so the curated soft-accept would manufacture a selection that 400s at first use. Custom
    # OpenAI-compatible proxies keep the fallback (incomplete listings are common there).
    from hermes_cli.providers import is_official_openai_host

    try:
        live = fetch_api_models(api_key, base)
    except Exception:
        live = None
    if not live:
        return None
    if not is_official_openai_host(base):
        return live
    live_lower = {m.lower() for m in live}
    curated = list(_PROVIDER_MODELS.get(normalized, []))
    # Curated order, only models the account has access to; an account serving none of them (rare)
    # falls back to curated so the picker still offers sane defaults.
    discovered = [m for m in curated if m.lower() in live_lower]
    # Astra is intentionally absent from offline/static catalogs: the official API's
    # account-scoped /models response is the only source that may advertise it.
    discovered.extend(m for m in live if is_astra_model(m))
    return discovered or curated or live


def _custom_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    base_url = _get_custom_base_url()
    if not base_url:
        return None
    model_cfg = _get_model_config_dict()
    # Try common API key env vars for custom endpoints.
    api_key = (
        str(model_cfg.get("api_key", "") or "").strip()
        or os.getenv("CUSTOM_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or os.getenv("OPENROUTER_API_KEY", ""))
    api_mode = "anthropic_messages" if _base_url_looks_like_anthropic_messages(base_url) else None
    return fetch_api_models(api_key, base_url, api_mode=api_mode) or None


def _bedrock_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    # Live discovery keyed by the resolved AWS region so EU/AP users see eu.*/ap.* ids.
    try:
        from agent.bedrock_adapter import bedrock_model_ids_or_none

        return bedrock_model_ids_or_none()
    except Exception:
        return None


def _opencode_free_catalog(normalized: str, force_refresh: bool) -> list[str]:
    # Live keyless catalog filtered to the anonymous-servable `*-free` tier ourselves (models.dev's
    # cost.input==0 lags reality); the curated floor applies only when the live fetch fails/is empty.
    return _fetch_opencode_free_models(force_refresh=force_refresh) or list(_PROVIDER_MODELS.get(normalized, []))


# Per-provider catalog sources tried before the generic profile fetch. A fetcher returning None
# falls through to the profile/curated path; a list is returned as-is (even empty).
_PROVIDER_CATALOG_FETCHERS: dict[str, Any] = {
    "openrouter": lambda normalized, force_refresh: model_ids(force_refresh=force_refresh),
    "openai-codex": _codex_catalog,
    "copilot": _copilot_catalog,
    "copilot-acp": _copilot_catalog,
    "nous": _nous_catalog,
    "stepfun": _api_key_provider_live,
    "gmi": _api_key_provider_live,
    "anthropic": _anthropic_catalog,
    "ai-gateway": lambda normalized, force_refresh: _fetch_ai_gateway_models() or None,
    # DeepInfra's generic /models mixes chat, image, video, speech and embedding models; the tagged
    # catalog helper is the only safe source for the chat picker, including its empty/failure result.
    "deepinfra": lambda normalized, force_refresh: _fetch_deepinfra_models(force_refresh=force_refresh) or [],
    "ollama-cloud": lambda normalized, force_refresh: fetch_ollama_cloud_models(force_refresh=force_refresh) or None,
    "openai": _openai_catalog,
    "openai-api": _openai_catalog,
    "custom": _custom_catalog,
    "bedrock": _bedrock_catalog,
    "opencode-free": _opencode_free_catalog}


def _profile_live_catalog(normalized: str) -> Optional[list[str]]:
    """Generic live fetch for any provider registered in providers/ with ``auth_type="api_key"``.

    Live results are merged with the curated list so models the live endpoint omits still appear:
    curated-first by default so the newest curated models lead when the live API lags;
    ``_LIVE_FIRST_PICKER_PROVIDERS`` (OpenCode Zen/Go, authoritative live API) live-first so stale
    curated entries stop polluting the top. Plugin providers without a static entry use the
    profile's ``fallback_models`` as the curated list (Fireworks lists an image model first).
    """
    from providers import get_provider_profile

    profile = get_provider_profile(normalized)
    if not (profile and profile.auth_type == "api_key" and profile.base_url):
        return None
    api_key, base_url = _api_key_credentials(normalized)
    live = profile.fetch_models(api_key=api_key, base_url=base_url or profile.base_url or None) if api_key else None
    if not live:
        return list(profile.fallback_models) if profile.fallback_models else None
    curated = list(_PROVIDER_MODELS.get(normalized, [])) or list(profile.fallback_models or ())
    if not curated:
        return live
    primary, secondary = (live, curated) if normalized in _LIVE_FIRST_PICKER_PROVIDERS else (curated, live)
    return _merge_unique(primary, secondary, key=_model_dedup_key)


def provider_model_ids(provider: Optional[str], *, force_refresh: bool = False) -> list[str]:
    """Best known model catalog for a provider: per-provider live fetchers, then the generic profile
    fetch, then the static list (merged with models.dev for ``_MODELS_DEV_PREFERRED`` providers)."""
    requested = str(provider or "").strip().lower()
    if requested == "ollama":
        return _ollama_local_catalog(force_refresh)

    normalized = normalize_provider(provider)
    fetcher = _PROVIDER_CATALOG_FETCHERS.get(normalized)
    if fetcher is not None:
        models = fetcher(normalized, force_refresh)
        if models is not None:
            return models
    try:
        models = _profile_live_catalog(normalized)
    except Exception:
        models = None
    if models is not None:
        return models

    # Merge static curated list with live API results so models that the live endpoint omits (stale cache,
    # partial rollout) still appear in the picker. Single providers (kimi, zai) use curated-first (commit
    # 658ac1d86) to surface newest models even when live API lags (#46309). OpenCode Zen / Go are different:
    # their live API is the authoritative catalog, so they merge live-first — live entries lead and stale
    # curated entries no longer pollute the top of the picker. (#49129) Plugin providers with no static
    # _PROVIDER_MODELS entry fall back to the profile's curated fallback_models so their agentic picks lead
    # the picker instead of whatever the live catalog happens to return first (e.g. Fireworks lists an image
    # model, flux-*, ahead of its chat models).
    curated_static = list(_PROVIDER_MODELS.get(normalized, []))
    if normalized not in _MODELS_DEV_PREFERRED:
        return curated_static
    merged = _merge_with_models_dev(normalized, curated_static)
    return _xai_finalize_catalog(merged) if normalized in {"xai", "xai-oauth"} else merged


# ---------------------------------------------------------------------------
# Disk cache for provider_model_ids() — keeps /model picker fast (otherwise every open re-fetches
# every authed provider's /v1/models). One JSON file at $HERMES_HOME/provider_models_cache.json;
# entries keyed by credential fingerprint (rotate OPENAI_API_KEY → entry invalidates); 1h TTL;
# only NON-EMPTY results are cached so a transient failure is never pinned; any read/write error
# degrades silently to a live fetch.
# ---------------------------------------------------------------------------

_PROVIDER_MODELS_CACHE_TTL = 3600  # 1h
# Providers whose catalog is served with NO credential and therefore gets a
# stable (constant) credential fingerprint in the disk cache. The opencode-free
# catalog is anonymous — its freshness comes from TTL revalidation, not from
# user-rotatable credentials — so folding in unrelated auth.json mtimes would
# only needlessly bust the SWR cache.
_KEYLESS_STABLE_CACHE_PROVIDERS = frozenset({"opencode-free"})
# Stale-while-revalidate window: an expired-but-same-credentials entry is
# served IMMEDIATELY (picker opens stay instant) while a background daemon
# thread re-fetches the live catalog and rewrites the disk cache for the
# next open. Beyond this bound the entry is considered too old to trust and
# the caller blocks on a live fetch as before. Rationale: the /model picker's
# provider listing runs 8-9 serial /v1/models round-trips (~2-3s) whenever
# the 1h TTL lapses mid-session — model catalogs change on release timescales,
# not hourly, so serving hour-old data while refreshing off-thread is strictly
# better than stalling every picker surface (CLI, TUI, dashboard, gateway).
_PROVIDER_MODELS_STALE_SERVE_MAX = 7 * 24 * 3600  # 7d

# Cache keys with a background SWR refresh in flight — dedupes concurrent refreshes.
_swr_refresh_inflight: set = set()
_swr_refresh_lock = threading.Lock()


def _cache_entry(fp: str, models: list[str], at: Optional[float] = None) -> dict:
    """One provider row of the disk cache: credential fingerprint, write time, model ids."""
    return {"fp": fp, "at": time.time() if at is None else at, "models": list(models)}


def _ollama_native_probe_reachable() -> bool:
    """Whether the configured local Ollama root answered the native ``/api/tags`` probe (an empty
    catalog from a reachable server is authoritative; a failed probe is not)."""
    base_url = _get_ollama_base_url()
    headers = _get_ollama_native_headers(base_url) or None
    probe_key = _ollama_probe_cache_key(_root_for_ollama_native_api(base_url), headers)
    return _OLLAMA_LOCAL_PROBE_REACHABLE.get(probe_key) is True


def _spawn_swr_refresh(cache_key: str, refresh_fn=None) -> None:
    """Fire-and-forget daemon refresh of *cache_key*'s cache entry, at most one in flight per key.
    Failures are swallowed — the stale entry stays served until a later refresh succeeds.
    ``refresh_fn`` (no-args → fresh entry dict or None) lets ``custom:<base_url>`` keys from
    :func:`cached_fetch_api_models` reuse the same inflight-dedupe scaffolding."""
    # Under a routed profile the inflight key includes the home: the same provider slug names a
    # different disk cache and credential set per profile, so one profile's refresh must not
    # suppress another's. Unscoped keeps the bare key (tests inspect the set by slug).
    from hermes_constants import get_hermes_home_override, hermes_home_key
    inflight_key = cache_key if get_hermes_home_override() is None else (hermes_home_key(), cache_key)
    with _swr_refresh_lock:
        if inflight_key in _swr_refresh_inflight:
            return
        _swr_refresh_inflight.add(inflight_key)

    def _default_refresh():
        live = provider_model_ids(cache_key, force_refresh=True)
        if not live and cache_key == "ollama":
            base_url = _get_ollama_base_url()
            headers = _get_ollama_native_headers(base_url) or None
            probe_key = _ollama_probe_cache_key(
                _root_for_ollama_native_api(base_url), headers
            )
            if _OLLAMA_LOCAL_PROBE_REACHABLE.get(probe_key) is True:
                return {
                    "fp": _credential_fingerprint(cache_key),
                    "at": time.time(),
                    "models": [],
                }
        if not live:
            return None
        return {
            "fp": _credential_fingerprint(cache_key),
            "at": time.time(),
            "models": list(live),
        }

    def _refresh() -> None:
        try:
            entry = (refresh_fn or _default_refresh)()
            if entry:
                _store_cache_entry(cache_key, entry)
        except Exception:
            logger.debug("SWR refresh failed for %s", cache_key, exc_info=True)
        finally:
            with _swr_refresh_lock:
                _swr_refresh_inflight.discard(inflight_key)

    # copy_context: the refresh must read the calling profile's credentials and write ITS disk cache.
    ctx = contextvars.copy_context()
    threading.Thread(target=lambda: ctx.run(_refresh), daemon=True, name=f"model-cache-swr-{cache_key}").start()


def _provider_models_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "provider_models_cache.json"


def _credential_fingerprint(provider: str) -> str:
    """Short hash of the credentials ``provider_model_ids(provider)`` would see right now: api-key /
    base-url env vars from ``PROVIDER_REGISTRY`` plus the mtimes of ``auth.json`` and external
    credential files (OAuth re-auth busts the cache without parsing every file shape)."""
    import hashlib

    # Keyless providers serve the catalog anonymously: nothing the user rotates should invalidate
    # the entry, so a stable fingerprint keeps the SWR cache alive and busts only on TTL expiry.
    if (provider or "").strip().lower() in _KEYLESS_STABLE_CACHE_PROVIDERS:
        return "keyless:" + (provider or "").strip().lower()

    parts: list[str] = []

    # Keyless providers have no credential to fingerprint: the catalog is
    # served anonymously, so nothing the user rotates (env vars, auth files,
    # base URLs) should invalidate the cached entry. A stable fingerprint keeps
    # the SWR disk cache alive across unrelated re-auths and only busts on TTL
    # expiry — matching how the live catalog genuinely changes.
    if (provider or "").strip().lower() in _KEYLESS_STABLE_CACHE_PROVIDERS:
        return "keyless:" + (provider or "").strip().lower()

    # Env vars from PROVIDER_REGISTRY for this slug
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        pcfg = PROVIDER_REGISTRY.get(provider)
        if pcfg is not None:
            for ev in getattr(pcfg, "api_key_env_vars", ()) or ():
                parts.append(f"{ev}={os.environ.get(ev, '')}")
            bev = getattr(pcfg, "base_url_env_var", "") or ""
            if bev:
                parts.append(f"{bev}={os.environ.get(bev, '')}")
    except Exception:
        pass

    # config.yaml's model.base_url changes the endpoint discovery probes (data-residency hosts)
    # without touching any env var, so it must change the fingerprint too.
    if provider in ("openai", "openai-api"):
        try:
            parts.append(f"effective_base={_openai_discovery_base_url(provider)}")
        except Exception:
            pass

    if provider == "ollama":
        parts.append(f"OLLAMA_HOST={_os.environ.get('OLLAMA_HOST', '')}")
        provider_cfg = _get_provider_config_dict("ollama")
        parts.append(
            "providers.ollama.base_url="
            f"{provider_cfg.get('base_url', '') or provider_cfg.get('api', '') or provider_cfg.get('url', '')}"
        )
        parts.append(f"providers.ollama.api_key={provider_cfg.get('api_key', '')}")
        key_env = provider_cfg.get("key_env") or provider_cfg.get("api_key_env") or ""
        parts.append(f"providers.ollama.key_env={key_env}")
        if key_env:
            parts.append(f"{key_env}={_os.environ.get(str(key_env), '')}")
        model_cfg = _get_model_config_dict()
        parts.append(
            "model.provider="
            f"{model_cfg.get('provider', '')}|model.base_url={model_cfg.get('base_url', '')}"
        )
        parts.append(
            "providers.ollama.extra_headers="
            + json.dumps(provider_cfg.get("extra_headers", {}), sort_keys=True, default=str)
        )

    # OAuth / external-file mtimes that change on re-auth
    try:
        from hermes_constants import get_hermes_home
        for rel in ("auth.json", "credentials.json"):
            p = get_hermes_home() / rel
            try:
                parts.append(f"{rel}@{p.stat().st_mtime_ns}")
            except FileNotFoundError:
                parts.append(f"{rel}@missing")
            except Exception:
                pass
    except Exception:
        pass

    def _mtime_part(label: str, path) -> None:
        try:
            parts.append(f"{label}@{os.stat(path).st_mtime_ns}")
        except FileNotFoundError:
            parts.append(f"{label}@missing")
        except Exception:
            pass

    try:
        from hermes_constants import get_hermes_home
        for rel in ("auth.json", "credentials.json"):
            _mtime_part(rel, get_hermes_home() / rel)
    except Exception:
        pass
    for rel in ("~/.codex/auth.json", "~/.claude/.credentials.json",
                "~/.config/github-copilot/hosts.json", "~/.minimax/credentials.json"):
        path = os.path.expanduser(rel)
        _mtime_part(path, path)

    blob = "|".join(parts).encode("utf-8", errors="replace")
    # blake2b, not sha256: fingerprint only (collisions = a harmless cache miss), and CodeQL's
    # weak-sensitive-data-hashing rule flags sha256 over env vars named *API_KEY*/*TOKEN*.
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _load_provider_models_cache() -> dict:
    """Return the full cache dict, or {} on any error."""
    try:
        return _read_json_cache(_provider_models_cache_path()) or {}
    except Exception:
        return {}


_cache_write_lock = threading.Lock()


def _save_provider_models_cache(data: dict) -> None:
    """Persist the cache dict. Best-effort — silent on any error."""
    try:
        _write_json_cache(_provider_models_cache_path(), data, indent=None)
    except Exception:
        pass


def _store_cache_entry(cache_key: str, entry: dict, cache: Optional[dict] = None) -> None:
    """Write one row into the disk cache (reloading the latest state unless ``cache`` is given)."""
    if cache is None:
        cache = _load_provider_models_cache()
    cache[cache_key] = entry
    _save_provider_models_cache(cache)


def update_provider_cache_entry(provider: str, models: list[str]) -> None:
    """Thread-safe single-entry update for parallel prefetch workers: load-modify-save under a lock
    so concurrent fetches don't clobber each other's rows. Best-effort, silent on any error."""
    try:
        normalized = normalize_provider(provider) or (provider or "")
        if not normalized or not models:
            return
        fp = _credential_fingerprint(normalized)
        with _cache_write_lock:
            _store_cache_entry(normalized, _cache_entry(fp, models))
    except Exception:
        pass


def _normalized_cache_slug(provider: Optional[str]) -> str:
    """``ollama`` stays a raw slug (its alias would canonicalize to ``custom``); everything else normalizes."""
    requested = str(provider or "").strip().lower()
    return requested if requested == "ollama" else (normalize_provider(provider) or (provider or ""))

    Hits the cache when fresh; otherwise calls the live function and
    persists a non-empty result. Always returns a list (never None).
    """
    requested = str(provider or "").strip().lower()
    normalized = requested if requested == "ollama" else (normalize_provider(provider) or (provider or ""))
    if not normalized:
        return []
    if normalized == "ollama":
        ttl_seconds = min(ttl_seconds, _OLLAMA_LOCAL_MODELS_CACHE_TTL)

    cache = _load_provider_models_cache()
    fp = _credential_fingerprint(normalized)
    entry = cache.get(normalized)
    now = time.time()

    allow_empty_ollama = normalized == "ollama"
    if not force_refresh and _cache_entry_valid(entry, fp, allow_empty=allow_empty_ollama):
        age = now - entry["at"]
        if age < ttl_seconds:
            return list(entry["models"])
        # Empty native catalogs are authoritative only for the short native
        # TTL. Re-probe after expiry so newly pulled models become visible;
        # do not serve an empty row through the generic stale window.
        if entry["models"] and age < _PROVIDER_MODELS_STALE_SERVE_MAX:
            # Stale-while-revalidate: serve the expired entry immediately so
            # interactive picker opens never block on serial /v1/models
            # round-trips; refresh the cache off-thread for the next open.
            _spawn_swr_refresh(normalized)
            return list(entry["models"])

    live = provider_model_ids(normalized, force_refresh=force_refresh)
    if live:
        _store_cache_entry(normalized, _cache_entry(fp, live, now), cache)
        return list(live)

    if normalized == "ollama":
        base_url = _get_ollama_base_url()
        headers = _get_ollama_native_headers(base_url) or None
        probe_key = _ollama_probe_cache_key(
            _root_for_ollama_native_api(base_url), headers
        )
        if _OLLAMA_LOCAL_PROBE_REACHABLE.get(probe_key) is True:
            # A reachable empty native catalog is authoritative for the short
            # native TTL; do not resurrect a stale disk catalog.
            cache[normalized] = {"fp": fp, "at": now, "models": []}
            _save_provider_models_cache(cache)
            return []

        # A failed/non-native probe is not authoritative. Preserve a stale
        # catalog rather than blanking the picker during a transient outage.
        if (
            isinstance(entry, dict)
            and entry.get("fp") == fp
            and isinstance(entry.get("models"), list)
            and entry["models"]
        ):
            return list(entry["models"])
        return []

    # Live fetch returned nothing. If we have a stale entry with the
    # SAME fingerprint, prefer it over an empty result — stale data
    # beats no data when the network is flaky.
    if _cache_entry_valid(entry, fp):
        return [model for model in entry["models"] if not _model_requires_account_discovery(normalized, model)]
    return []


def clear_provider_models_cache(provider: Optional[str] = None) -> None:
    """Drop one provider's cache entry, or wipe the whole cache (``provider=None``). Used by
    ``/model --refresh`` and ``hermes model --refresh``."""
    try:
        # Native Ollama tags are keyed by root URL rather than provider slug.
        # A targeted refresh for a custom local-Ollama endpoint cannot identify
        # the right root from the provider name alone, so clear this small
        # in-process cache on every explicit provider-cache refresh.
        _OLLAMA_LOCAL_MODELS_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_REACHABLE.clear()
        if provider is None:
            path = _provider_models_cache_path()
            if path.exists():
                path.unlink()
            return
        cache = _load_provider_models_cache()
        requested = str(provider or "").strip().lower()
        normalized = requested if requested == "ollama" else (normalize_provider(provider) or provider or "")
        changed = False
        if normalized in cache:
            del cache[normalized]
            changed = True
        if changed:
            _save_provider_models_cache(cache)
    except Exception:
        pass


def _resolve_anthropic_pool_catalog_credentials() -> tuple[str, str]:
    """Return a read-only API-key pool credential for model discovery.

    ``resolve_anthropic_token()`` intentionally ignores ``api_key`` pool
    entries because its runtime contract is OAuth-oriented. The model catalog
    supports regular ``x-api-key`` auth, so it needs a narrow fallback that
    preserves the credential's configured endpoint instead of sending a
    proxy-scoped key to Anthropic's public host.
    """
    try:
        from agent.credential_pool import AUTH_TYPE_API_KEY
        from hermes_cli.auth import read_credential_pool

        for entry in read_credential_pool("anthropic"):
            if not isinstance(entry, dict):
                continue
            if entry.get("auth_type") != AUTH_TYPE_API_KEY:
                continue
            token = str(entry.get("access_token") or "").strip()
            if not token:
                continue
            endpoint = str(
                entry.get("base_url") or entry.get("inference_base_url") or ""
            ).strip()
            return token, endpoint
    except Exception:
        pass
    return "", ""


def _fetch_anthropic_models(
    timeout: float = 5.0,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[list[str]]:
    """Fetch available models from the Anthropic /v1/models endpoint.

    Uses resolve_anthropic_token() to find credentials (env vars, OAuth,
    or Claude Code auto-discovery) unless api_key is provided explicitly. If
    those sources are empty, a read-only API-key credential_pool entry is used.
    Returns sorted model IDs or None.
    """
    try:
        from agent.credential_pool import AUTH_TYPE_API_KEY
        from hermes_cli.auth import read_credential_pool

        for entry in read_credential_pool("anthropic"):
            if not isinstance(entry, dict) or entry.get("auth_type") != AUTH_TYPE_API_KEY:
                continue
            token = str(entry.get("access_token") or "").strip()
            if token:
                return token, str(entry.get("base_url") or entry.get("inference_base_url") or "").strip()
    except Exception:
        pass
    return "", ""


def _fetch_anthropic_models(
    timeout: float = 5.0, *, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> Optional[list[str]]:
    """Sorted model ids from the Anthropic /v1/models endpoint, or None. Credentials: explicit
    ``api_key``, else ``resolve_anthropic_token()`` (env / OAuth / Claude Code), else a read-only
    API-key credential_pool entry."""
    try:
        from agent.anthropic_credentials import resolve_anthropic_token, _is_oauth_token
    except ImportError:
        return None

    resolved_base_url = base_url
    token = (api_key or "").strip() or resolve_anthropic_token()
    if not token:
        # A pool credential and its endpoint are one security boundary. Never
        # pair the selected pool key with a caller-provided model endpoint.
        token, resolved_base_url = _resolve_anthropic_pool_catalog_credentials()
    if not token:
        return None

    headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
    is_oauth = _is_oauth_token(token)
    if is_oauth:
        headers["Authorization"] = f"Bearer {token}"
        from agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS, _CONTEXT_1M_BETA
        headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
    else:
        headers["x-api-key"] = token

    def _do_request(h: dict[str, str]):
        req = urllib.request.Request(
            _anthropic_models_url(resolved_base_url),
            headers=h,
        )
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        try:
            data = _get_json(url, timeout=timeout, headers=headers)
        except urllib.error.HTTPError as http_err:
            # OAuth subscriptions that 400 the 1M context beta ("long context beta is not yet
            # available for this subscription"): retry once without it; re-raise anything else.
            if not (is_oauth and http_err.code == 400):
                raise
            try:
                body_text = http_err.read().decode(errors="ignore").lower()
            except Exception:
                body_text = ""
            if not ("long context beta" in body_text and "not yet available" in body_text):
                raise
            headers["anthropic-beta"] = ",".join(
                [b for b in _COMMON_BETAS if b != _CONTEXT_1M_BETA] + list(_OAUTH_ONLY_BETAS)
            )
            data = _get_json(url, timeout=timeout, headers=headers)
        models = [m["id"] for m in data.get("data", []) if m.get("id")]
        seen_cursors: set[str] = set()
        for _page in range(_ANTHROPIC_MODELS_MAX_PAGES):
            cursor = _anthropic_next_cursor(data, seen_cursors)
            if cursor is None:
                break
            data = _get_json(_anthropic_models_url(resolved_base_url, after_id=cursor), timeout=timeout, headers=headers)
            models.extend(m["id"] for m in data.get("data", []) if m.get("id"))
        models = list(dict.fromkeys(models))
        # opus, then sonnet, then haiku; alphabetical within tier.
        return sorted(models, key=lambda m: ("opus" not in m, "sonnet" not in m, "haiku" not in m, m))
    except Exception as e:
        logger.debug("Failed to fetch Anthropic models: %s", e)
        return None


def _payload_items(payload: Any) -> list[dict[str, Any]]:
    data = payload.get("data", []) if isinstance(payload, dict) else payload
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def copilot_default_headers(*, is_agent_turn: bool = True) -> dict[str, str]:
    """Standard headers for Copilot API requests."""
    try:
        from hermes_cli.copilot_auth import copilot_request_headers
        return copilot_request_headers(is_agent_turn=is_agent_turn)
    except ImportError:
        return {
            "Editor-Version": COPILOT_EDITOR_VERSION,
            "User-Agent": "HermesAgent/1.0",
            "Openai-Intent": "conversation-edits",
            "x-initiator": "agent" if is_agent_turn else "user"}


_COPILOT_CHAT_ENDPOINTS = {"/chat/completions", "/responses", "/v1/messages"}


def _copilot_catalog_item_is_text_model(
    item: dict[str, Any], *, ignore_picker_flag: bool = False) -> bool:
    if not str(item.get("id") or "").strip():
        return False
    if not ignore_picker_flag and item.get("model_picker_enabled") is False:
        return False
    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        model_type = str(capabilities.get("type") or "").strip().lower()
        if model_type and model_type != "chat":
            return False
    supported_endpoints = item.get("supported_endpoints")
    if isinstance(supported_endpoints, list):
        endpoints = {e for endpoint in supported_endpoints if (e := str(endpoint).strip())}
        if endpoints and not endpoints & _COPILOT_CHAT_ENDPOINTS:
            return False
    return True


def _copilot_text_models(items: list[dict[str, Any]], *, ignore_picker_flag: bool = False) -> list[dict[str, Any]]:
    """Chat-capable catalog rows, deduped by id, in catalog order."""
    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in items:
        model_id = str(item.get("id") or "").strip()
        if model_id in seen_ids:
            continue
        if not _copilot_catalog_item_is_text_model(item, ignore_picker_flag=ignore_picker_flag):
            continue
        seen_ids.add(model_id)
        models.append(item)
    return models


# Short-TTL cache of the filtered GitHub Copilot /models catalog (picker + context/normalize helpers
# share it). Keyed by the api_key of the successful fetch so a credential swap never serves the
# previous account's catalog; monotonic clock; lock-free (a race at worst duplicates one fetch).
_github_model_catalog_cache: Optional[list[dict[str, Any]]] = None
_github_model_catalog_cache_key: Optional[str] = None
_github_model_catalog_cache_time: float = 0.0
_GITHUB_MODEL_CATALOG_CACHE_TTL = 300  # 5 minutes


def fetch_github_model_catalog(
    api_key: Optional[str] = None, timeout: float = 5.0) -> Optional[list[dict[str, Any]]]:
    """Fetch the live GitHub Copilot model catalog for this account."""
    global _github_model_catalog_cache, _github_model_catalog_cache_key
    global _github_model_catalog_cache_time

    if (
        _github_model_catalog_cache is not None
        and _github_model_catalog_cache_key == api_key
        and (time.monotonic() - _github_model_catalog_cache_time) < _GITHUB_MODEL_CATALOG_CACHE_TTL
    ):
        return copy.deepcopy(_github_model_catalog_cache)  # deep: callers must not mutate cached dicts

    attempts: list[dict[str, str]] = []
    if api_key:
        attempts.append({**copilot_default_headers(), "Authorization": f"Bearer {api_key}"})
    attempts.append(copilot_default_headers())

    for headers in attempts:
        try:
            items = _payload_items(_get_json(COPILOT_MODELS_URL, timeout=timeout, headers=headers))
        except Exception:
            continue
        models = _copilot_text_models(items)
        if not models and items:
            # GitHub has been observed returning ``model_picker_enabled: false`` for EVERY model on
            # some accounts, which would strand the picker on the stale curated fallback. The flag
            # is a display hint, not an availability contract — retry without it (chat/endpoint
            # checks still apply).
            models = _copilot_text_models(items, ignore_picker_flag=True)
        if models:
            _github_model_catalog_cache = copy.deepcopy(models)
            _github_model_catalog_cache_key = api_key
            _github_model_catalog_cache_time = time.monotonic()
            return models
    return None


# ─── Copilot catalog context-window helpers ───

# Module-level cache: {model_id: max_prompt_tokens}
_copilot_context_cache: dict[str, int] = {}
_copilot_context_cache_time: float = 0.0
_copilot_context_cache_key: Optional[str] = None  # fingerprint of the api_key the entry was fetched with
_COPILOT_CONTEXT_CACHE_TTL = 3600  # 1 hour


def get_copilot_model_context(model_id: str, api_key: Optional[str] = None) -> Optional[int]:
    """``max_prompt_tokens`` for a Copilot model from the live /models API (cached in-process 1h; a
    miss on a fresh cache does not re-fetch), or None."""
    global _copilot_context_cache, _copilot_context_cache_time, _copilot_context_cache_key

    # Keyed on the credential like fetch_github_model_catalog: the catalog (and its limits) is
    # per-account, so another profile's token must not be served this entry.
    from agent.credential_persistence import fingerprint_secret_value
    key_fp = fingerprint_secret_value(api_key)
    if (_copilot_context_cache and _copilot_context_cache_key == key_fp
            and (time.time() - _copilot_context_cache_time < _COPILOT_CONTEXT_CACHE_TTL)):
        return _copilot_context_cache.get(model_id)

    catalog = fetch_github_model_catalog(api_key=api_key)
    if not catalog:
        return None
    cache: dict[str, int] = {}
    for item in catalog:
        mid = str(item.get("id") or "").strip()
        max_prompt = ((item.get("capabilities") or {}).get("limits") or {}).get("max_prompt_tokens")
        if mid and isinstance(max_prompt, int) and max_prompt > 0:
            cache[mid] = max_prompt
    _copilot_context_cache = cache
    _copilot_context_cache_time = time.time()
    _copilot_context_cache_key = key_fp
    return cache.get(model_id)


def _is_github_models_base_url(base_url: Optional[str]) -> bool:
    return (base_url or "").strip().rstrip("/").lower().startswith(
        (COPILOT_BASE_URL, "https://models.github.ai/inference", "https://models.inference.ai.azure.com")
    )


def _fetch_github_models(api_key: Optional[str] = None, timeout: float = 5.0) -> Optional[list[str]]:
    catalog = fetch_github_model_catalog(api_key=api_key, timeout=timeout)
    return [item.get("id", "") for item in catalog if item.get("id")] if catalog else None


def _copilot_catalog_ids(
    catalog: Optional[list[dict[str, Any]]] = None, api_key: Optional[str] = None) -> set[str]:
    if catalog is None and api_key:
        catalog = fetch_github_model_catalog(api_key=api_key)
    return {mid for item in (catalog or []) if (mid := str(item.get("id") or "").strip())}


def normalize_copilot_model_id(
    model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return ""

    catalog_ids = _copilot_catalog_ids(catalog=catalog, api_key=api_key)
    alias = _COPILOT_MODEL_ALIASES.get(raw)
    if alias:
        return alias

    candidates = [raw]
    if "/" in raw:
        candidates.append(raw.split("/", 1)[1].strip())
    if raw.endswith(("-mini", "-nano", "-chat")):
        candidates.append(raw[:-5])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in _COPILOT_MODEL_ALIASES:
            return _COPILOT_MODEL_ALIASES[candidate]
        if candidate in catalog_ids:
            return candidate

    if "/" in raw:
        stripped = raw.split("/", 1)[1].strip()
        # Enterprise BYOK custom models expose ``owner/sub/model`` ids (two
        # slashes). A strip guess that still contains "/" cannot be a Copilot
        # id, so pass the input through untouched instead of corrupting it.
        if stripped and "/" not in stripped:
            return stripped
    return raw


def _github_reasoning_efforts_for_model_id(model_id: str) -> list[str]:
    raw = (model_id or "").strip().lower()
    if raw.startswith(("openai/o1", "openai/o3", "openai/o4", "o1", "o3", "o4")):
        return list(COPILOT_REASONING_EFFORTS_O_SERIES)
    normalized = normalize_copilot_model_id(model_id).lower()
    if normalized.startswith("gpt-5"):
        return list(COPILOT_REASONING_EFFORTS_GPT5)
    return []


def _should_use_copilot_responses_api(model_id: str) -> bool:
    """opencode's ``shouldUseCopilotResponsesApi``: GPT-5+ uses the Responses API except
    ``gpt-5-mini``; non-GPT models (Claude, Gemini, ...) use Chat Completions."""
    match = re.match(r"^gpt-(\d+)", model_id)
    return bool(match) and int(match.group(1)) >= 5 and not model_id.startswith("gpt-5-mini")


def copilot_model_api_mode(
    model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None) -> str:
    """API mode for a Copilot model from the id pattern (opencode's approach). Copilot's Claude models
    go through its OpenAI-compatible chat endpoint, not the native Anthropic adapter: the catalog may
    advertise /v1/messages but the Copilot token/header scheme lives in the OpenAI client path."""
    if catalog is None and api_key:  # fetch once so normalize + endpoint check share it
        catalog = fetch_github_model_catalog(api_key=api_key)
    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if normalized and _should_use_copilot_responses_api(normalized):
        return "codex_responses"
    return "chat_completions"


def azure_foundry_model_api_mode(model_name: Optional[str]) -> Optional[str]:
    """``"codex_responses"`` for families that only accept the Responses API on Azure Foundry (GPT-5.x
    incl. gpt-5-mini, codex, o1/o3/o4), else None. Any ``vendor/`` prefix is stripped first."""
    raw = str(model_name or "").strip().lower().rsplit("/", 1)[-1]
    return "codex_responses" if raw and raw.startswith(tuple(_AZURE_FOUNDRY_RESPONSES_PREFIXES)) else None


_OPENCODE_FAMILIES = ("opencode-free", "opencode-go", "opencode-zen")


def opencode_provider_family(provider_id: Optional[str]) -> Optional[str]:
    """Resolve a provider id (canonical or prefixed) to its OpenCode family, or None.

    Returns ``"opencode-zen"`` or ``"opencode-go"`` for the built-in providers AND for custom providers
    whose name extends a family slug (e.g. ``opencode-go-bridge`` pointing at
    ``https://opencode.ai/zen/go/v1``, issue #85589). Matching is case-insensitive. Custom family providers
    need the same per-model api_mode routing and /v1 base-url normalization as the built-ins — this
    predicate is the single owner of that family-membership question; do not re-implement it inline.
    """
    raw = str(provider_id or "").strip().lower()
    if not raw:
        return None
    canonical = normalize_provider(provider_id)
    if canonical in _OPENCODE_FAMILIES:
        return canonical
    return next((f for f in _OPENCODE_FAMILIES if raw.startswith(f)), None)


def opencode_provider_family(provider_id: Optional[str]) -> Optional[str]:
    """Resolve a provider id to its OpenCode family, or None.

    Returns ``"opencode-zen"`` or ``"opencode-go"`` for the built-in
    providers AND for custom providers whose name extends a family slug
    (e.g. ``opencode-go-bridge`` pointing at ``https://opencode.ai/zen/go/v1``,
    issue #85589). Matching is case-insensitive. Custom family providers
    need the same per-model api_mode routing and /v1 base-url normalization
    as the built-ins — this predicate is the single owner of that
    family-membership question; do not re-implement it inline.

    ``opencode-go`` is checked before ``opencode-zen`` but the two slugs are
    not prefixes of each other, so order is cosmetic.
    """
    raw = str(provider_id or "").strip().lower()
    if not raw:
        return None
    canonical = normalize_provider(provider_id)
    if canonical in {"opencode-zen", "opencode-go", "opencode-free"}:
        return canonical
    if raw.startswith("opencode-free"):
        return "opencode-free"
    if raw.startswith("opencode-go"):
        return "opencode-go"
    if raw.startswith("opencode-zen"):
        return "opencode-zen"
    return None


def normalize_opencode_model_id(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Normalize OpenCode config IDs to the bare model slug used in API requests."""
    family = opencode_provider_family(provider_id)
    current = str(model_id or "").strip()
    if not current or family is None:
        return current

    prefix = f"{provider_id}/" if provider_id else f"{family}/"
    if current.lower().startswith(prefix.lower()):
        return current[len(prefix):]
    fallback_prefix = f"{family}/"
    if current.lower().startswith(fallback_prefix.lower()):
        return current[len(fallback_prefix):]
    return current


# OpenCode Zen free-tier models (``*-free`` slugs, e.g. x-preview-f-free /
# "Ox Alpha", plus unsuffixed free models like big-pickle) are served
# ANONYMOUSLY on the Zen relay: a request with no Authorization header
# succeeds, while ANY non-empty bearer the relay doesn't recognize is
# rejected with 401 "Invalid API key" — including our "no-key-required"
# placeholder and OpenCode GO subscription keys (the Go relay doesn't serve
# the free tier at all: "Model x is not supported").
# Verified live 2026-08-21 against POST /zen/v1/chat/completions.
OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER = "opencode-zen-free-keyless"
_OPENCODE_ZEN_FREE_BASE_URL = "https://opencode.ai/zen/v1"

# Free-tier models whose slug does NOT carry the ``-free`` suffix.
# (big-pickle is OpenCode's rotating free stealth slot.)
_OPENCODE_KEYLESS_EXTRA_SLUGS = frozenset({"big-pickle"})

# Models whose slug carries ``-free`` but are NOT anonymous-servable: they are
# KEYED (Go-subscription) models and must be excluded from the keyless free
# catalog even though the suffix looks free. ox-alpha-free is the Go relay's
# subscription twin of the Zen keyless Ox Alpha (verified 2026-08-21).
_OPENCODE_FREE_KEYED_SUFFIX_MODELS = frozenset({"ox-alpha-free"})

# In-process memo for _fetch_opencode_free_models(): (fetched_at, ids-or-None).
# Direct provider_model_ids("opencode-free") callers (model validation, healing)
# can run several times per resolution — without this each would block on a
# network round-trip. Failures are memoized too (negative caching) so an
# unreachable relay doesn't stall every validation for `timeout` seconds.
_opencode_free_live_memo: Optional[tuple[float, Optional[list[str]]]] = None
_OPENCODE_FREE_LIVE_MEMO_TTL = 300.0  # 5 min; SWR disk cache handles the rest


def is_opencode_zen_free_model(model_id: Optional[str]) -> bool:
    """True when ``model_id`` is an OpenCode Zen free-tier slug.

    Matches the ``*-free`` suffix plus the known unsuffixed free slugs
    (``big-pickle``). Tolerates provider-prefixed ids
    (``opencode-zen/x-preview-f-free``). The Go catalog serves no free
    models (verified 2026-08-21), so this identifies the Zen free tier
    across the OpenCode family.
    """
    bare = str(model_id or "").strip().rsplit("/", 1)[-1].lower()
    if not bare:
        return False
    return bare.endswith("-free") or bare in _OPENCODE_KEYLESS_EXTRA_SLUGS


def opencode_zen_free_headers() -> dict:
    """Client default_headers for anonymous OpenCode Zen free-tier requests.

    ``Authorization: ""`` overrides the OpenAI SDK's ``Bearer <api_key>``
    header so the placeholder key never reaches the wire — the Zen relay
    accepts anonymous requests for free models but 401s any unknown bearer.
    Attribution headers mirror the opencode provider profile.
    """
    try:
        from hermes_cli import __version__ as _v
    except Exception:
        _v = "0"
    return {
        "Authorization": "",
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
        "User-Agent": f"HermesAgent/{_v}",
    }


def _fetch_opencode_free_models(
    timeout: float = 8.0, *, force_refresh: bool = False
) -> Optional[list[str]]:
    """Fetch the live keyless OpenCode Free catalog from the Zen relay.

    GETs ``{_OPENCODE_ZEN_FREE_BASE_URL}/models`` ANONYMOUSLY (the free tier
    rejects any unrecognized Authorization bearer with 401) and filters the
    dump to the anonymous-servable ``*-free`` tier. Returns ``None`` on any
    network/auth/parse failure so callers fall back to the curated
    ``_PROVIDER_MODELS["opencode-free"]`` floor; an empty filtered result is
    treated as a failure for the same reason (a relay with zero free models is
    not worth trusting over the floor).

    A short in-process memo (``_OPENCODE_FREE_LIVE_MEMO_TTL``) keeps direct
    ``provider_model_ids("opencode-free")`` callers — model validation runs
    it several times per resolution — from issuing one blocking network
    round-trip each. The picker's cross-process freshness still comes from
    the SWR disk cache one layer up; ``force_refresh=True`` (the SWR refresh
    path) bypasses and repopulates the memo.

    The Zen ``/models`` dump also lists paid/subscription IDs (e.g. Go
    ``ox-alpha-free`` is KEYED despite the suffix), so a bare ``*-free`` suffix
    filter is not safe on its own — this mirrors the existing
    ``opencode_zen_free_runtime`` contract, which uses membership in the
    verified keyless catalog as the routing criterion.
    """
    import urllib.request

    from hermes_cli.urllib_security import open_credentialed_url

    now = time.time()
    if not force_refresh:
        memo = _opencode_free_live_memo
        if memo is not None and now - memo[0] < _OPENCODE_FREE_LIVE_MEMO_TTL:
            return list(memo[1]) if memo[1] else None

    url = f"{_OPENCODE_ZEN_FREE_BASE_URL.rstrip('/')}/models"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    for k, v in opencode_zen_free_headers().items():
        if k.lower() != "authorization":  # never send a bearer keylessly
            req.add_header(k, v)
    try:
        with open_credentialed_url(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        items = data if isinstance(data, list) else data.get("data", [])
    except Exception:
        _set_opencode_free_live_memo(None)
        return None
    ids = [m["id"] for m in items if isinstance(m, dict) and isinstance(m.get("id"), str)]
    # Filter to the anonymous-servable free tier. The Zen dump can contain
    # keyed/Go IDs; only the verified free set belongs in the keyless picker.
    live_free = [
        mid
        for mid in ids
        if mid.lower().endswith("-free")
        and mid.lower() not in _OPENCODE_FREE_KEYED_SUFFIX_MODELS
    ]
    result = live_free if live_free else None
    _set_opencode_free_live_memo(result)
    return result


def _set_opencode_free_live_memo(ids: Optional[list[str]]) -> None:
    global _opencode_free_live_memo
    _opencode_free_live_memo = (time.time(), list(ids) if ids else None)


def _opencode_free_known_model_slugs() -> set[str]:
    """Lowercased keyless free-tier slugs known right now — WITHOUT network I/O.

    Union of the static ``_PROVIDER_MODELS["opencode-free"]`` floor, the
    in-process live memo, and the SWR disk-cache entry. Used by the
    ``opencode_zen_free_runtime`` healing path, which runs during model
    resolution and must never block on a live fetch. Union (not replacement)
    so a stale cache can only widen healing, never silently disable it.
    """
    known = {m.lower() for m in _PROVIDER_MODELS.get("opencode-free", [])}
    memo = _opencode_free_live_memo
    if memo is not None and memo[1]:
        known.update(m.lower() for m in memo[1])
    try:
        entry = _load_provider_models_cache().get("opencode-free") or {}
        known.update(str(m).lower() for m in entry.get("models", []) or [])
    except Exception:
        pass
    return known


def opencode_zen_free_runtime(provider_id: Optional[str], model_id: Optional[str]) -> Optional[dict]:
    """Keyless runtime entry for an OpenCode Zen free-tier model, or None.

    Returns a resolve_runtime_provider-shaped dict pinning the request to the
    Zen relay with the keyless placeholder whenever:

    - ``provider_id`` is ``opencode-free`` (the dedicated keyless provider —
      EVERY model on it routes anonymously; that is the provider's contract), or
    - ``provider_id`` is any other OpenCode-family provider and ``model_id``
      is in the VERIFIED keyless catalog (``_PROVIDER_MODELS["opencode-free"]``)
      — heals a free-model selection made under opencode-zen/opencode-go,
      whose keys the free tier rejects.

    Membership, not the ``-free`` suffix, is the heal criterion: the suffix
    stopped being a reliable keyless signal when ``ox-alpha-free`` appeared
    on the Go relay as a KEYED subscription model (2026-08-21) — suffix-based
    healing would have routed it to a Zen relay that doesn't serve it.
    Membership means the union of the cached LIVE keyless catalog (in-process
    memo / SWR disk cache — never a blocking fetch on this hot path) and the
    static floor, so a newly-live free model heals without a release.
    """
    family = opencode_provider_family(provider_id)
    if family is None:
        return None
    if family != "opencode-free":
        bare = normalize_opencode_model_id(provider_id, model_id).strip().lower()
        if bare not in _opencode_free_known_model_slugs():
            return None
    normalized = normalize_opencode_model_id(provider_id, model_id)
    api_mode = opencode_model_api_mode("opencode-zen", normalized)
    base_url = normalize_opencode_base_url(
        "opencode-zen", api_mode, _OPENCODE_ZEN_FREE_BASE_URL
    )
    return {
        "provider": family,
        "api_mode": api_mode,
        "base_url": base_url,
        "api_key": OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER,
        "default_headers": opencode_zen_free_headers(),
        "source": "opencode-zen-free-keyless",
    }


def opencode_model_api_mode(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Determine the API mode for an OpenCode Zen / Go model.

    OpenCode routes different models behind different API surfaces:

    - GPT-5 / Codex / Grok models on Zen use ``/v1/responses``
    - GPT / Grok models on Go (gpt-5.6-luna, grok-4.5) use ``/v1/responses``
    - Muse Spark on Go and Zen uses ``/v1/responses`` (chat/completions 503s)
    - Claude models on Zen use ``/v1/messages``
    - MiniMax and Qwen models on Go use ``/v1/messages``
    - GLM / Kimi / DeepSeek / MiMo on Go use ``/v1/chat/completions``
    - Qwen models on Zen use ``/v1/messages``
    - Other Zen models (Gemini, GLM, Kimi, MiniMax, DeepSeek, etc.) use
      ``/v1/chat/completions``

    This follows the published OpenCode docs for Zen and Go endpoints
    (https://opencode.ai/docs/zen/ and https://opencode.ai/docs/go/).
    """
    family = opencode_provider_family(provider_id)
    # opencode-free is Zen-hosted (the free tier lives on the Zen relay),
    # so it shares Zen's per-model endpoint routing.
    if family == "opencode-free":
        family = "opencode-zen"
    normalized = normalize_opencode_model_id(provider_id, model_id).lower()
    if not normalized:
        return "chat_completions"

    if family == "opencode-go":
        if normalized.startswith("gpt-") or normalized.startswith("grok-"):
            # GPT and Grok models on Go (gpt-5.6-luna, grok-4.5) are served
            # via /v1/responses per the published Go endpoint table, same as
            # GPT/Grok on Zen: https://opencode.ai/docs/go/#endpoints
            return "codex_responses"
        if normalized.startswith("muse-spark"):
            # Muse Spark (standard + contributor) is Responses-only on Go.
            # /v1/chat/completions returns HTTP 503 with an empty assistant
            # message; /v1/responses completes. See opencode.ai/docs/go.
            return "codex_responses"
        if normalized.startswith("minimax-"):
            return "anthropic_messages"
        if normalized.startswith("qwen"):
            # All Qwen models on Go (qwen3.7-max, qwen3.7-plus, qwen3.6-plus)
            # are served via /v1/messages per the published Go endpoint table.
            return "anthropic_messages"
        return "chat_completions"

    if family == "opencode-zen":
        if normalized.startswith("claude-"):
            return "anthropic_messages"
        if normalized.startswith("gpt-") or normalized.startswith("grok-"):
            # GPT-5/Codex and all Grok models on Zen (grok-4.6, grok-4.5,
            # grok-build-0.1) are served via /v1/responses per the Zen
            # endpoint table.
            return "codex_responses"
        if normalized.startswith("muse-spark"):
            # Standard Muse Spark on Zen is served via /v1/responses:
            # https://opencode.ai/docs/zen/#endpoints
            return "codex_responses"
        if normalized.startswith("qwen"):
            # Qwen models on Zen moved to /v1/messages per the published
            # Zen endpoint table.
            return "anthropic_messages"
        return "chat_completions"

    return "chat_completions"


def normalize_opencode_base_url(
    provider_id: Optional[str], api_mode: Optional[str], base_url: Optional[str]) -> str:
    """Normalize an OpenCode Zen / Go base URL for the API mode. Must be SYMMETRIC: the anthropic-
    stripped URL gets persisted to ``model.base_url`` after switching into an anthropic-routed model,
    and chat/codex modes heal it by re-adding ``/v1`` — but only on opencode.ai hosts, so custom
    ``OPENCODE_*_BASE_URL`` proxies are left alone."""
    url = str(base_url or "").strip().rstrip("/")
    if not url or opencode_provider_family(provider_id) is None:
        return url
    if opencode_provider_family(provider_id) is None:
        return url

    import re as _re

    if api_mode == "anthropic_messages":
        return re.sub(r"/v1$", "", url)
    if url.endswith("/v1"):
        return url
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        host = ""
    return url + "/v1" if host == "opencode.ai" or host.endswith(".opencode.ai") else url


def github_model_reasoning_efforts(
    model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None) -> list[str]:
    """Return supported reasoning-effort levels for a Copilot-visible model."""
    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if not normalized:
        return []

    if catalog is None and api_key:
        catalog = fetch_github_model_catalog(api_key=api_key)
    catalog_entry = next((item for item in catalog if item.get("id") == normalized), None) if catalog else None
    if catalog_entry is not None:
        capabilities = catalog_entry.get("capabilities")
        if isinstance(capabilities, dict):
            # Structured catalog: the advertised list is authoritative (empty when absent).
            supports = capabilities.get("supports")
            efforts = supports.get("reasoning_effort") if isinstance(supports, dict) else None
            if not isinstance(efforts, list):
                return []
            return list(dict.fromkeys(e for effort in efforts if (e := str(effort).strip().lower())))
        # Legacy list-shaped capabilities: only a "reasoning" tag unlocks the pattern defaults.
        if "reasoning" not in {str(c).strip().lower() for c in catalog_entry.get("capabilities", [])}:
            return []
    return _github_reasoning_efforts_for_model_id(str(model_id or normalized))


# Negative cache: monotonic timestamp of the last fully-failed probe, keyed
# by ``host:port`` so both URL candidates (``/v1`` + root) share one entry.
# Without this, an unreachable endpoint (TCP blackhole — SYN draws no reply,
# so every attempt burns its full connect timeout) makes every picker open /
# chat turn re-pay the timeout per candidate, and the sequential stalls stack
# past 10s while the Desktop sits on a spinner with no error (#81123). Short
# TTL collapses the burst but still picks up recovery without a restart.
# Mirrors _deepinfra_catalog_neg_cache.
_probe_neg_cache: dict[str, float] = {}
_PROBE_NEG_TTL = 60.0  # seconds


def _probe_neg_key(base_url: str) -> Optional[str]:
    """``host:port`` for *base_url* (both URL candidates share one entry), or None without a host."""
    from utils import base_url_origin

    _, host, port = base_url_origin(base_url)
    return f"{host}:{port}" if host else None


def _probe_result(
    models, probed_url, resolved_base_url, suggested_base_url=None, used_fallback=False
) -> dict[str, Any]:
    return {
        "models": models,
        "probed_url": probed_url,
        "resolved_base_url": resolved_base_url,
        "suggested_base_url": suggested_base_url,
        "used_fallback": used_fallback}


def probe_api_models(
    api_key: Optional[str], base_url: Optional[str], timeout: float = 5.0,
    api_mode: Optional[str] = None, request_headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Probe a ``/models`` endpoint with light URL heuristics (``base`` then ``base±/v1``).
    ``anthropic_messages`` mode sends ``x-api-key`` + ``anthropic-version`` instead of a bearer; the
    ``data[].id`` response shape is identical. ``models`` is None when no candidate answered."""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return _probe_result(None, None, "")
    if _is_github_models_base_url(normalized):
        models = _fetch_github_models(api_key=api_key, timeout=timeout)
        return _probe_result(models, COPILOT_MODELS_URL, COPILOT_BASE_URL)

    alternate_base = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized + "/v1"
    candidates: list[tuple[str, bool]] = [(normalized, False)]
    if alternate_base and alternate_base != normalized:
        candidates.append((alternate_base, True))

    tried: list[str] = []
    _neg_key = _probe_neg_key(normalized)
    if _neg_key is not None:
        _neg_seen = _probe_neg_cache.get(_neg_key)
        if _neg_seen is not None and (time.monotonic() - _neg_seen) < _PROBE_NEG_TTL:
            return _probe_result(
                None, normalized.rstrip("/") + "/models", normalized,
                alternate_base if alternate_base != normalized else None)
    headers: dict[str, str] = {"User-Agent": _HERMES_USER_AGENT}
    if urllib.parse.urlparse(normalized).hostname == "generativelanguage.googleapis.com":
        headers["X-Goog-Api-Client"] = f"hermes-agent/{_HERMES_VERSION}"
    if api_key and api_mode == "anthropic_messages":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if normalized.startswith(COPILOT_BASE_URL):
        headers.update(copilot_default_headers())
    if isinstance(request_headers, dict):
        # Per-provider custom headers can contain secrets: merge last so endpoint config wins; never log.
        from hermes_cli.config import normalize_extra_headers

        headers.update(normalize_extra_headers(request_headers))

    # Only thread ssl_context when a per-provider TLS override applies; public endpoints keep the
    # original 2-arg call so existing call-seam mocks stay valid.
    _open_kwargs: dict[str, Any] = {}
    _ssl_context = _custom_provider_ssl_context(normalized)
    if _ssl_context is not None:
        _open_kwargs["ssl_context"] = _ssl_context
    reachable = False
    for candidate_base, is_fallback in candidates:
        url = candidate_base.rstrip("/") + "/models"
        tried.append(url)
        try:
            data = _get_json(url, timeout=timeout, headers=headers, **_open_kwargs)
        except urllib.error.HTTPError:
            # The host answered: an auth/404 failure is not unreachability, and a user fixing
            # their key must not be served a cached "no models" for the next TTL window.
            reachable = True
            continue
        except Exception:
            continue
        if _neg_key is not None:
            _probe_neg_cache.pop(_neg_key, None)
        return _probe_result(
            [m.get("id", "") for m in data.get("data", [])], url, candidate_base.rstrip("/"),
            alternate_base if alternate_base != candidate_base else normalized, is_fallback)

    if _neg_key is not None and not reachable:
        _probe_neg_cache[_neg_key] = time.monotonic()
    return _probe_result(
        None, tried[0] if tried else normalized.rstrip("/") + "/models", normalized,
        alternate_base if alternate_base != normalized else None)


# Legacy id-regex filter for items with no surface tag; unreachable (deletable) once every catalog
# entry carries an explicit ``chat``/``embed``/``image-gen``/``tts``/``stt`` tag.
_DEEPINFRA_EXCLUDE_RE = re.compile(
    r"(?i)(embed|rerank|whisper|stable-diffusion|flux|sdxl|"
    r"tts|bark|speech|image-gen|clip|vit-|dpt-)")

# Surface tags say *what kind of model* this is. Absent all of them, the tags array only carries
# capability tags (``reasoning``, ``vision``, …) and the chat surface falls back to id-regex inference.
_DEEPINFRA_SURFACE_TAGS: frozenset[str] = frozenset({
    "chat", "embed", "image-gen", "tts", "stt", "video-gen"})

_DEEPINFRA_DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
_DEEPINFRA_MODELS_QUERY = "filter=true&sort_by=hermes"

# Full tagged catalog keyed by base URL; every surface filter reads it so one round-trip serves all.
_deepinfra_catalog_cache: dict[str, list[dict]] = {}

# Negative cache (monotonic time of the last failed fetch per base URL) so an unreachable catalog
# doesn't make every surface helper eat the full timeout in turn. Short TTL so connectivity recovers.
_deepinfra_catalog_neg_cache: dict[str, float] = {}
_DEEPINFRA_CATALOG_NEG_TTL = 60.0  # seconds


def _deepinfra_env(key: str) -> str:
    """Profile-scoped ``.env``/environ read: under a multiplexed turn the launch env is not this profile's."""
    from hermes_cli.config import get_env_value_prefer_dotenv
    return (get_env_value_prefer_dotenv(key) or "").strip()


def _deepinfra_catalog_url() -> tuple[str, str]:
    """Return ``(cache_key, full_url)`` for the DeepInfra catalog endpoint. The key carries the
    api-key fingerprint: the catalog is user-scoped (private fine-tunes), so two profiles with
    different keys must not share an entry."""
    base = (_deepinfra_env("DEEPINFRA_BASE_URL") or _DEEPINFRA_DEFAULT_BASE_URL).rstrip("/")
    from agent.credential_persistence import fingerprint_secret_value
    fp = fingerprint_secret_value(_deepinfra_env("DEEPINFRA_API_KEY")) or "anon"
    return f"{base}#{fp}", f"{base}/models?{_DEEPINFRA_MODELS_QUERY}"


def _fetch_deepinfra_catalog(
    *, timeout: float = 5.0, force_refresh: bool = False) -> Optional[list[dict]]:
    """Raw DeepInfra catalog list (chat, embed, image-gen, TTS, STT in one response), cached per base
    URL. A Bearer token is attached when available so user-scoped catalogs (private fine-tunes) show."""
    cache_key, url = _deepinfra_catalog_url()
    if not force_refresh:
        if cache_key in _deepinfra_catalog_cache:
            return _deepinfra_catalog_cache[cache_key]
        last_fail = _deepinfra_catalog_neg_cache.get(cache_key)
        if last_fail is not None and (time.monotonic() - last_fail) < _DEEPINFRA_CATALOG_NEG_TTL:
            return None

    headers: dict[str, str] = {"User-Agent": _HERMES_USER_AGENT}
    api_key = _deepinfra_env("DEEPINFRA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        payload = _get_json(url, timeout=timeout, headers=headers)
    except Exception:
        _deepinfra_catalog_neg_cache[cache_key] = time.monotonic()
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        _deepinfra_catalog_neg_cache[cache_key] = time.monotonic()
        return None
    _deepinfra_catalog_cache[cache_key] = data
    _deepinfra_catalog_neg_cache.pop(cache_key, None)
    return data


def _fetch_deepinfra_models_by_tag(
    tag: str, *, timeout: float = 5.0, force_refresh: bool = False) -> Optional[list[dict]]:
    """DeepInfra ``{"id", "metadata"}`` items whose ``metadata.tags`` includes *tag*. Items with no
    surface tag fall through to the legacy id-regex exclusion (chat surface only — embed/image-gen/
    tts/stt cannot be inferred from an id). ``None`` on network failure."""
    data = _fetch_deepinfra_catalog(timeout=timeout, force_refresh=force_refresh)
    if data is None:
        return None
    matched: list[dict] = []
    for item in data:
        mid = item.get("id")
        raw_metadata = item.get("metadata")
        if not mid or raw_metadata is None:  # metadata None = listed-but-not-served stub
            continue
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        raw_tags = metadata.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else []
        if any(t in _DEEPINFRA_SURFACE_TAGS for t in tags):
            hit = tag in tags
        else:
            hit = tag == "chat" and not _DEEPINFRA_EXCLUDE_RE.search(mid)
        if hit:
            matched.append({"id": mid, "metadata": metadata})
    return matched


def _fetch_deepinfra_models(
    timeout: float = 5.0, *, force_refresh: bool = False) -> Optional[list[str]]:
    """DeepInfra chat-model ids (string-list contract for :func:`provider_model_ids`); ``None`` on
    network failure or when no chat-tagged id exists."""
    items = _fetch_deepinfra_models_by_tag("chat", timeout=timeout, force_refresh=force_refresh)
    return ([item["id"] for item in items] or None) if items is not None else None


def deepinfra_model_ids(tag: str, *, force_refresh: bool = False) -> list[str]:
    """Return DeepInfra model ids carrying surface *tag* (``[]`` on failure)."""
    items = _fetch_deepinfra_models_by_tag(tag, force_refresh=force_refresh)
    return [item["id"] for item in items] if items else []


def deepinfra_base_url(section: Optional[dict] = None) -> str:
    """DeepInfra base URL: config-section ``base_url`` → ``DEEPINFRA_BASE_URL`` env → default; stripped."""
    candidate = section.get("base_url") if isinstance(section, dict) else None
    value = candidate or _deepinfra_env("DEEPINFRA_BASE_URL") or _DEEPINFRA_DEFAULT_BASE_URL
    return str(value).strip().rstrip("/")


def _fetch_ai_gateway_models(timeout: float = 5.0) -> Optional[list[str]]:
    """Fetch available language models with tool-use from AI Gateway."""
    api_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.getenv("AI_GATEWAY_BASE_URL", "").strip()
    if not base_url:
        from hermes_constants import AI_GATEWAY_BASE_URL
        base_url = AI_GATEWAY_BASE_URL

    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": _HERMES_USER_AGENT}
    try:
        url = base_url.rstrip("/") + "/models"
        data = _get_json(url, timeout=timeout, headers=headers)
        return [
            m["id"] for m in data.get("data", [])
            if m.get("id") and m.get("type") == "language" and "tool-use" in (m.get("tags") or [])]
    except Exception:
        return None


def fetch_api_models(
    api_key: Optional[str], base_url: Optional[str], timeout: float = 5.0,
    api_mode: Optional[str] = None, headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Fetch the list of available model IDs from the provider's ``/models`` endpoint."""
    result = probe_api_models(api_key, base_url, timeout=timeout, api_mode=api_mode, request_headers=headers)
    return result.get("models")


def _custom_endpoint_fingerprint(
    api_key: Any, api_mode: Optional[str], headers: Optional[dict[str, str]]) -> str:
    """Custom endpoints have no ``PROVIDER_REGISTRY`` slug, so hash exactly what callers pass to
    :func:`fetch_api_models`: a rotated ``api_key``, changed ``api_mode`` or edited ``extra_headers``
    each bust the cache entry. blake2b for the same CodeQL rationale as ``_credential_fingerprint``."""
    import hashlib

    from agent.command_token_source import CommandTokenSource
    identity = api_key.cache_identity if isinstance(api_key, CommandTokenSource) else api_key
    blob = "|".join((identity or "", api_mode or "", json.dumps(headers or {}, sort_keys=True)))
    return hashlib.blake2b(blob.encode("utf-8", errors="replace"), digest_size=8).hexdigest()


def _cache_entry_valid(
    entry: Any,
    fp: str,
    *,
    allow_empty: bool = False,
) -> "TypeGuard[dict[str, Any]]":
    """True when *entry* is a well-formed cache row for fingerprint *fp*.

    Requires a numeric ``at`` so corrupt disk state (hand-edited JSON with
    ``"at": "yesterday"`` or ``null``) degrades to a cache miss / live fetch
    instead of raising out of the wrapper. Empty model lists are valid only
    for callers that explicitly opt into an authoritative empty catalog.
    """
    return (
        isinstance(entry, dict)
        and entry.get("fp") == fp
        and isinstance(entry.get("models"), list)
        and (allow_empty or bool(entry["models"]))
        and isinstance(entry.get("at"), (int, float))
        and not isinstance(entry.get("at"), bool))


def cached_fetch_api_models(
    api_key: Any, base_url: Optional[str], *, timeout: float = 5.0,
    api_mode: Optional[str] = None, headers: Optional[dict[str, str]] = None,
    force_refresh: bool = False, cache_only: bool = False,
    fetch_models=None,
    ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL) -> Optional[list[str]]:
    """Disk-cached :func:`fetch_api_models` for custom endpoints. ``cache_only`` callers (GUI picker
    opens that must not block on a stopped local endpoint) still get a warm catalog instead of
    collapsing to the config-declared subset. ``fetch_models`` supplies native-aware discovery
    without minting a command token before cache admission."""
    from hermes_cli.model_switch_providers import _NativePickerModelList

    def _catalog(entry):
        return (_NativePickerModelList if entry.get("native_catalog") else list)(entry["models"])

    def _entry(live, at=None):
        return {**_cache_entry(fp, live, at), "native_catalog": isinstance(live, _NativePickerModelList)}

    def _live():
        if fetch_models is not None:
            return fetch_models()
        from agent.command_token_source import materialize_probe_api_key
        return fetch_api_models(materialize_probe_api_key(api_key), base_url, timeout=timeout, api_mode=api_mode, headers=headers)

    normalized_url = str(base_url or "").strip().rstrip("/").lower()
    if not normalized_url:  # nothing to key the cache on
        return None if cache_only else _live()

    # Key on URL AND credential fingerprint: N ``custom_providers`` rows can share one proxy URL
    # with distinct keys (#106184). A URL-only key let the last probe overwrite its siblings'
    # slot, so every other same-URL row failed the fingerprint check, got an empty catalog and
    # vanished from the no-probe pickers.
    fp = _custom_endpoint_fingerprint(api_key, api_mode, headers)
    cache_key = f"custom:{normalized_url}#{fp}"
    cache = _load_provider_models_cache()
    entry = cache.get(cache_key)
    now = time.time()
    valid = not force_refresh and _cache_entry_valid(entry, fp, allow_empty=isinstance(entry, dict) and entry.get("native_catalog") is True)

    if valid:
        age = now - entry["at"]
        if age < ttl_seconds:
            return _catalog(entry)
        if age < _PROVIDER_MODELS_STALE_SERVE_MAX:
            # Stale-while-revalidate: serve now, refresh off-thread for the next open. cache_only
            # opens (GUI pickers that must not block on a stopped local server) take the same
            # non-blocking refresh: without it a locally loaded model stayed invisible for the
            # whole 7-day stale window unless the user found "Refresh Models" (#71169 class).
            def _refresh_custom():
                live = _live()
                return _entry(live) if live or isinstance(live, _NativePickerModelList) else None

            _spawn_swr_refresh(cache_key, _refresh_custom)
            return _catalog(entry)

    if cache_only:
        return None

    live = _live()
    if live or isinstance(live, _NativePickerModelList):
        stored = _entry(live, now)
        _store_cache_entry(cache_key, stored, cache)
        return _catalog(stored)
    # Live returned nothing (offline, timeout, auth hiccup): a stale same-fingerprint entry beats it.
    if _cache_entry_valid(entry, fp, allow_empty=isinstance(entry, dict) and entry.get("native_catalog") is True):
        return _catalog(entry)
    return live


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import NamedTuple  # noqa: F401,E402
from difflib import get_close_matches  # noqa: F401,E402
import http.client  # noqa: F401,E402

def is_nous_free_tier(account_info: dict[str, Any]) -> bool:
    """Return True if the account info indicates a free (unpaid) tier.

    Prefer the Portal's explicit ``paid_service_access.allowed`` entitlement
    decision.  Legacy payloads fall back to ``subscription.monthly_charge == 0``.
    Returns False when both signals are missing or unparseable.
    """
    paid_access = account_info.get("paid_service_access")
    if isinstance(paid_access, dict):
        allowed = paid_access.get("allowed")
        if isinstance(allowed, bool):
            return not allowed
        paid = paid_access.get("paid_access")
        if isinstance(paid, bool):
            return not paid

    sub = account_info.get("subscription")
    if not isinstance(sub, dict):
        return False
    charge = sub.get("monthly_charge")
    if charge is None:
        return False
    try:
        return float(charge) == 0
    except (TypeError, ValueError):
        return False

_OPENCODE_KEYLESS_EXTRA_SLUGS = frozenset({"big-pickle"})

def is_opencode_zen_free_model(model_id: Optional[str]) -> bool:
    """True when ``model_id`` is an OpenCode Zen free-tier slug.

    Matches the ``*-free`` suffix plus the known unsuffixed free slugs
    (``big-pickle``). Tolerates provider-prefixed ids
    (``opencode-zen/x-preview-f-free``). The Go catalog serves no free
    models (verified 2026-08-21), so this identifies the Zen free tier
    across the OpenCode family.
    """
    bare = str(model_id or "").strip().rsplit("/", 1)[-1].lower()
    if not bare:
        return False
    return bare.endswith("-free") or bare in _OPENCODE_KEYLESS_EXTRA_SLUGS


def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """
    Validate a ``/model`` value for the active provider.


    Returns a dict with:
      - accepted: whether the CLI should switch to the requested model now
      - persist: whether it is safe to save to config
      - recognized: whether it matched a known provider catalog
      - message: optional warning / guidance for the user
    """
    requested = (model_name or "").strip()
    normalized = normalize_provider(provider)
    if normalized == "openrouter" and base_url and not base_url_host_matches(base_url, "openrouter.ai"):
        normalized = "custom"
    requested_for_lookup = requested
    if normalized == "copilot":
        requested_for_lookup = normalize_copilot_model_id(
            requested,
            api_key=api_key,
        ) or requested

    if not requested:
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model name cannot be empty.",
        }

    if normalized == "moa":
        try:
            from hermes_cli.config import load_config
            from hermes_cli.moa_config import normalize_moa_config

            cfg = normalize_moa_config(load_config().get("moa") or {})
            if requested in cfg["presets"]:
                return {"accepted": True, "persist": True, "recognized": True, "message": None}
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": f"MoA preset `{requested}` was not found. Run `hermes moa list`.",
            }
        except Exception as exc:
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": f"Could not read MoA presets: {exc}",
            }

    if any(ch.isspace() for ch in requested):
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model names cannot contain spaces.",
        }

    if normalized == "lmstudio":
        from hermes_cli.auth import AuthError
        # Use probe_lmstudio_models so we can distinguish None (unreachable
        # / malformed response) from [] (reachable, but no chat-capable models
        # are loaded). fetch_lmstudio_models collapses both to [].
        try:
            models = probe_lmstudio_models(api_key=api_key, base_url=base_url)
        except AuthError as exc:
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": (
                    f"{exc} Set `LM_API_KEY` (or update it) to match the server's bearer token."
                ),
            }
        if models is None:
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": f"Could not reach LM Studio's `/api/v1/models` to validate `{requested}`.",
            }
        if not models:
            return {
                "accepted": False, "persist": False, "recognized": False,
                "message": (
                    f"LM Studio is reachable but no chat-capable models are loaded. "
                    f"Load `{requested}` in LM Studio (Developer tab → Load Model) and try again."
                ),
            }
        if requested_for_lookup in set(models):
            return {"accepted": True, "persist": True, "recognized": True, "message": None}
        return {
            "accepted": False, "persist": False, "recognized": False,
            "message": f"Model `{requested}` was not found in LM Studio's model listing.",
        }

    if str(provider or "").strip().lower() == "ollama" and not base_url:
        base_url = _get_ollama_base_url()
    ollama_base_url = base_url
    configured_ollama_base_url = str(
        (
            _get_provider_config_dict("ollama").get("base_url")
            or _get_provider_config_dict("ollama").get("api")
            or _get_provider_config_dict("ollama").get("url")
            or ""
        )
    ).strip()
    configured_headers_allowed = not (
        configured_ollama_base_url
        and not _same_ollama_native_root(ollama_base_url or "", configured_ollama_base_url)
    )
    if headers is not None:
        ollama_headers = {}
        if configured_headers_allowed:
            ollama_headers.update(
                _get_ollama_native_headers(ollama_base_url, api_key=api_key)
            )
        for key in tuple(ollama_headers):
            if key.lower() == "authorization":
                del ollama_headers[key]
        ollama_headers.update(headers)
        caller_has_authorization = any(
            key.lower() == "authorization" for key in headers
        )
        if api_key and not caller_has_authorization:
            for key in tuple(ollama_headers):
                if key.lower() == "authorization":
                    del ollama_headers[key]
            ollama_headers["Authorization"] = f"Bearer {api_key}"
    elif configured_headers_allowed:
        ollama_headers = _get_ollama_native_headers(ollama_base_url, api_key=api_key)
    else:
        ollama_headers = {}
    if should_use_ollama_native_catalog(
        provider, ollama_base_url, headers=ollama_headers
    ):
        ollama_models = probe_ollama_local_models(
            ollama_base_url, headers=ollama_headers
        )
        if ollama_models is None:
            # A failed native probe is not authoritative; fall back to the
            # existing OpenAI-compatible catalog before accepting blindly.
            ollama_models = probe_api_models(
                api_key,
                _normalize_openai_base_url(ollama_base_url),
                request_headers=ollama_headers,
            ).get("models")
        if ollama_models is None:
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: could not reach this Ollama endpoint's `/api/tags` model listing to validate `{requested}`. "
                    "Hermes will save the model name, but local Ollama model discovery could not verify it."
                ),
            }
        if requested_for_lookup in set(ollama_models):
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        suggestions = get_close_matches(requested_for_lookup, ollama_models, n=3, cutoff=0.5)
        suggestion_text = ""
        if suggestions:
            suggestion_text = "\n  Similar local Ollama models: " + ", ".join(f"`{s}`" for s in suggestions)
        empty_hint = " No models are currently listed by `/api/tags`." if not ollama_models else ""
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": (
                f"Note: `{requested}` was not found in this Ollama endpoint's `/api/tags` model listing."
                f"{empty_hint} It may still work if the server supports hidden or aliased models."
                f"{suggestion_text}"
            ),
        }

    if normalized == "custom" or normalized.startswith("custom:"):
        # Try probing with correct auth for the api_mode.
        if api_mode == "anthropic_messages":
            probe = probe_api_models(
                api_key,
                base_url,
                api_mode=api_mode,
                request_headers=headers,
            )
        else:
            probe = probe_api_models(
                api_key,
                base_url,
                request_headers=headers,
            )
        api_models = probe.get("models")
        if api_models is not None:
            if requested_for_lookup in set(api_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }

            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }

            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)

            message = (
                f"Note: `{requested}` was not found in this custom endpoint's model listing "
                f"({probe.get('probed_url')}). It may still work if the server supports hidden or aliased models."
                f"{suggestion_text}"
            )
            if probe.get("used_fallback"):
                message += (
                    f"\n  Endpoint verification succeeded after trying `{probe.get('resolved_base_url')}`. "
                    f"Consider saving that as your base URL."
                )

            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": message,
            }

        message = (
            f"Note: could not reach this custom endpoint's model listing at `{probe.get('probed_url')}`. "
            f"Hermes will still save `{requested}`, but the endpoint should expose `/models` for verification."
        )
        if api_mode == "anthropic_messages":
            message += (
                "\n  Many Anthropic-compatible proxies do not implement the Models API "
                "(GET /v1/models).  The model name has been accepted without verification."
            )
        if probe.get("suggested_base_url"):
            message += f"\n  If this server expects `/v1`, try base URL: `{probe.get('suggested_base_url')}`"

        return {
            "accepted": api_mode == "anthropic_messages",
            "persist": True,
            "recognized": False,
            "message": message,
        }

    # Providers with non-standard catalog validation — /v1/models probing is not the right path.
    if normalized in {"openai-codex", "xai-oauth"}:
        try:
            catalog_models = provider_model_ids(normalized)
        except Exception:
            catalog_models = []
        # Ineligible ``-900k`` aliases (e.g. `gpt-5.5-900k`) must be rejected
        # BEFORE the hidden-slug soft-accept below: the suffix is a Hermes
        # picker convention, so an unknown `*-900k` name can never be a real
        # hidden provider slug — soft-accepting one silently runs at 272K on
        # a different model than the user thinks (#92797 review).
        if normalized == "openai-codex":
            from agent.model_metadata import (
                CODEX_CONTEXT_VARIANT_SUFFIX,
                is_codex_context_variant,
            )
            _req_lower = requested_for_lookup.strip().lower()
            if (
                _req_lower.endswith(CODEX_CONTEXT_VARIANT_SUFFIX)
                and requested_for_lookup not in set(catalog_models)
            ):
                if is_codex_context_variant(requested_for_lookup):
                    # Valid variant that a stale catalog hasn't synthesized
                    # yet. Accept it directly — falling through would let the
                    # typo auto-corrector "fix" it to the base slug and
                    # silently drop the large-context opt-in.
                    return {
                        "accepted": True,
                        "persist": True,
                        "recognized": True,
                        "message": None,
                    }
                _base_guess = requested_for_lookup[: -len(CODEX_CONTEXT_VARIANT_SUFFIX)]
                return {
                    "accepted": False,
                    "persist": False,
                    "recognized": False,
                    "message": (
                        f"`{requested}` is not a valid large-context variant — "
                        f"`{_base_guess}` enforces the standard 272K window on "
                        f"Codex, so no `-900k` option exists for it. Pick the "
                        f"base model, or a verified variant from the `/model` "
                        f"picker (e.g. `gpt-5.6-sol-900k`)."
                    ),
                }
        if catalog_models:
            if requested_for_lookup in set(catalog_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, catalog_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
            suggestions = get_close_matches(requested_for_lookup, catalog_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)
            provider_label = "OpenAI Codex" if normalized == "openai-codex" else "xAI Grok OAuth (SuperGrok / Premium+)"
            # Plausibility gate (#45006): the soft-accept (#16172 / #19729) exists
            # for entitlement-gated *hidden* slugs the curated listing hasn't
            # caught up with — but those are always the provider's own family
            # (openai-codex -> gpt-*; xai-oauth -> grok-*). Accepting an
            # unrelated typed name (e.g. `qwen3.5-4b`, `llama-3.1-8b`) here turns
            # what should be an actionable "did you mean --provider <x>?" error
            # into a confusing success that 400s on the next turn. Only soft-
            # accept names that share the provider's family prefix; reject the
            # rest with guidance to pin the right provider.
            _family_prefixes = {
                "openai-codex": ("gpt-", "codex-", "o1", "o3", "o4"),
                "xai-oauth": ("grok-",),
            }.get(normalized, ())
            _lower = requested_for_lookup.strip().lower()
            _plausible = (not _family_prefixes) or any(
                _lower.startswith(p) for p in _family_prefixes
            )
            if not _plausible:
                return {
                    "accepted": False,
                    "persist": False,
                    "recognized": False,
                    "message": (
                        f"`{requested}` doesn't look like a {provider_label} model "
                        f"and isn't in its listing, so it was not accepted. If it "
                        f"belongs to another configured provider, switch with "
                        f"`--provider <slug>` (or select it from the `/model` "
                        f"picker)."
                        f"{suggestion_text}"
                    ),
                }
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in the {provider_label} model listing. "
                    "It may still work if your account has access to a newer or hidden model ID."
                    f"{suggestion_text}"
                ),
            }

    # MiniMax providers don't expose a /models endpoint — validate against
    # the static catalog instead, similar to openai-codex.
    if normalized in {"minimax", "minimax-cn"}:
        try:
            catalog_models = provider_model_ids(normalized)
        except Exception:
            catalog_models = []
        if catalog_models:
            # Case-insensitive lookup (catalog uses mixed case like MiniMax-M2.7)
            catalog_lower = {m.lower(): m for m in catalog_models}
            if requested_for_lookup.lower() in catalog_lower:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            # Auto-correct close matches (case-insensitive)
            catalog_lower_list = list(catalog_lower.keys())
            auto = get_close_matches(requested_for_lookup.lower(), catalog_lower_list, n=1, cutoff=0.9)
            if auto:
                corrected = catalog_lower[auto[0]]
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": corrected,
                    "message": f"Auto-corrected `{requested}` → `{corrected}`",
                }
            suggestions = get_close_matches(requested_for_lookup.lower(), catalog_lower_list, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{catalog_lower[s]}`" for s in suggestions)
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in the MiniMax catalog."
                    f"{suggestion_text}"
                    "\n  MiniMax does not expose a /models endpoint, so Hermes cannot verify the model name."
                    "\n  The model may still work if it exists on the server."
                ),
            }

    # Native Anthropic provider: /v1/models requires x-api-key (or Bearer for
    # OAuth) plus anthropic-version headers.  The generic OpenAI-style probe
    # below uses plain Bearer auth and 401s against Anthropic, so dispatch to
    # the native fetcher which handles both API keys and Claude-Code OAuth
    # tokens.  (The api_mode=="anthropic_messages" branch below handles the
    # Messages-API transport case separately.)
    if normalized == "anthropic":
        anthropic_models = _fetch_anthropic_models(
            base_url=base_url or None,
            api_key=api_key or None,
        )
        if anthropic_models is not None:
            if requested_for_lookup in set(anthropic_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            auto = get_close_matches(requested_for_lookup, anthropic_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
            suggestions = get_close_matches(requested, anthropic_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)
            # Accept anyway — Anthropic sometimes gates newer/preview models
            # (e.g. snapshot IDs, early-access releases) behind accounts
            # even though they aren't listed on /v1/models.
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in Anthropic's /v1/models listing. "
                    f"It may still work if you have early-access or snapshot IDs."
                    f"{suggestion_text}"
                ),
            }
        # _fetch_anthropic_models returned None — no token resolvable or
        # network failure.  Fall through to the generic warning below.

    # Anthropic Messages API: many proxies don't implement /v1/models.
    # Try probing with correct auth; if it fails, accept with a warning.
    if api_mode == "anthropic_messages":
        api_models = fetch_api_models(api_key, base_url, api_mode=api_mode)
        if api_models is not None:
            if requested_for_lookup in set(api_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
        # Probe failed or model not found — accept anyway (proxy likely
        # doesn't implement the Anthropic Models API).
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": (
                f"Note: could not verify `{requested}` against this endpoint's "
                f"model listing.  Many Anthropic-compatible proxies do not "
                f"implement GET /v1/models.  The model name has been accepted "
                f"without verification."
            ),
        }

    # Probe the live API to check if the model actually exists
    api_models = fetch_api_models(api_key, base_url)

    if api_models is not None:
        # Gemini's OpenAI-compat /v1beta/openai/models endpoint returns IDs
        # prefixed with "models/" (e.g. "models/gemini-2.5-flash") — native
        # Gemini-API convention.  Our curated list and user input both use
        # the bare ID, so a direct set-membership check drops every known
        # Gemini model.  Strip the prefix before comparison.  See #12532.
        if normalized == "gemini":
            api_models = [
                m[len("models/"):] if isinstance(m, str) and m.startswith("models/") else m
                for m in api_models
            ]
        if requested_for_lookup in set(api_models):
            # API confirmed the model exists
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        # OpenRouter routing variants (":nitro", ":floor", ...) are request-time
        # modifiers, not catalog entries — /models lists only the base id.
        # Validate the BASE against the listing but preserve the suffixed id,
        # and do this BEFORE fuzzy auto-correction: get_close_matches would
        # otherwise "correct" `model:nitro` → `model` and silently strip the
        # user's routing opt-in.
        _variant_base = (
            _openrouter_variant_base(requested_for_lookup)
            if normalized == "openrouter"
            else None
        )
        if _variant_base is not None and _variant_base in set(api_models):
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        else:
            # API responded but model is not listed.  Accept anyway —
            # the user may have access to models not shown in the public
            # listing (e.g. Z.AI Pro/Max plans can use glm-5 on coding
            # endpoints even though it's not in /models).  Warn but allow.

            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }

            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)

            # Model not in live /v1/models — check the curated catalog
            # before rejecting.  Providers may omit models from their live
            # listing that are still valid (stale cache, partial rollout,
            # gated previews).  Use the pure-catalog helper (no extra live
            # fetch) so we only accept models Hermes actually ships.  (#46850)
            #
            # EXCEPTION: official OpenAI hosts (canonical api.openai.com and
            # the data-residency regional hosts).  Their /v1/models listing is
            # access-scoped and authoritative — a model absent from it is one
            # this key CANNOT serve, so the curated soft-accept would
            # manufacture a selection that 400s at first use.  Custom
            # OpenAI-compatible proxies keep the fallback (incomplete
            # listings are common there).
            _openai_listing_is_authoritative = False
            if normalized in ("openai", "openai-api"):
                from hermes_cli.providers import is_official_openai_host

                _openai_listing_is_authoritative = is_official_openai_host(base_url)
            if not _openai_listing_is_authoritative and _model_in_provider_catalog(
                (_variant_base or requested_for_lookup).lower(),
                _provider_keys(normalized),
            ):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": (
                        f"Note: `{requested}` was not found in the live /v1/models listing "
                        f"but exists in the curated catalog — accepted."
                    ),
                }

            # Nous provider: also check the Portal's live
            # /api/nous/recommended-models feed. That feed can list a model
            # (e.g. a newly-promoted free/paid recommendation) before it's
            # been added to the hardcoded _PROVIDER_MODELS["nous"] curated
            # list or the docs-hosted catalog manifest has been rebuilt.
            # `hermes chat` already accepts these models via
            # union_with_portal_free/paid_recommendations() at model-list
            # build time; this mirrors that same source of truth for the
            # per-message /model validation path (messaging platform
            # pickers, /model command), which previously only checked the
            # curated catalog and rejected valid Portal-recommended models.
            if normalized == "nous":
                try:
                    portal_payload = fetch_nous_recommended_models(
                        _resolve_nous_portal_url()
                    )
                    portal_model_names = {
                        name.lower()
                        for tier in ("freeRecommendedModels", "paidRecommendedModels")
                        for entry in (portal_payload.get(tier) or [])
                        if (name := _extract_model_name(entry))
                    }
                except Exception:
                    portal_model_names = set()
                if requested_for_lookup.lower() in portal_model_names:
                    return {
                        "accepted": True,
                        "persist": True,
                        "recognized": True,
                        "message": (
                            f"Note: `{requested}` was not found in the live /v1/models "
                            f"listing but is a current Nous Portal recommendation — accepted."
                        ),
                    }

        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": (
                f"Model `{requested}` was not found in this provider's model listing."
                f"{suggestion_text}"
            ),
        }

    # api_models is None — couldn't reach API.  Accept and persist,
    # but warn so typos don't silently break things.

    # Bedrock: use our own discovery instead of HTTP /models endpoint.
    # Bedrock's bedrock-runtime URL doesn't support /models — it uses the
    # AWS SDK control plane (ListFoundationModels + ListInferenceProfiles).
    if normalized == "bedrock":
        try:
            from agent.bedrock_adapter import discover_bedrock_models, resolve_bedrock_runtime_region
            region = resolve_bedrock_runtime_region()
            discovered = discover_bedrock_models(region)
            discovered_ids = {m["id"] for m in discovered}
            if requested in discovered_ids:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            # Not in discovered list — still accept (user may have custom
            # inference profiles or cross-account access), but warn.
            suggestions = get_close_matches(requested, list(discovered_ids), n=3, cutoff=0.4)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in Bedrock model discovery for {region}. "
                    f"It may still work with custom inference profiles or cross-account access."
                    f"{suggestion_text}"
                ),
            }
        except Exception:
            pass  # Fall through to generic warning

    # Static-catalog fallback: when the /models probe was unreachable,
    # validate against the curated list from provider_model_ids() — same
    # pattern as the openai-codex and minimax branches above.  This keeps
    # /model switches working in the gateway for providers whose /models
    # endpoint is temporarily unreachable or returns a non-JSON payload.
    # Without this block, validate_requested_model would reject every model
    # on such providers, switch_model() would return success=False, and
    # the gateway would never write to _session_model_overrides.
    provider_label = _PROVIDER_LABELS.get(normalized, normalized)
    try:
        catalog_models = provider_model_ids(normalized)
    except Exception:
        catalog_models = []

    if catalog_models:
        catalog_lower = {m.lower(): m for m in catalog_models}
        if requested_for_lookup.lower() in catalog_lower:
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        # OpenRouter routing-variant suffixes: validate the base id against
        # the catalog, keep the suffixed id (same rule as the live-listing
        # path above — variants never appear as catalog entries).
        if normalized == "openrouter":
            _cat_variant_base = _openrouter_variant_base(requested_for_lookup)
            if (
                _cat_variant_base is not None
                and _cat_variant_base.lower() in catalog_lower
            ):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
        catalog_lower_list = list(catalog_lower.keys())
        auto = get_close_matches(
            requested_for_lookup.lower(), catalog_lower_list, n=1, cutoff=0.9
        )
        if auto:
            corrected = catalog_lower[auto[0]]
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "corrected_model": corrected,
                "message": f"Auto-corrected `{requested}` → `{corrected}`",
            }
        suggestions = get_close_matches(
            requested_for_lookup.lower(), catalog_lower_list, n=3, cutoff=0.5
        )
        suggestion_text = ""
        if suggestions:
            suggestion_text = "\n  Similar models: " + ", ".join(
                f"`{catalog_lower[s]}`" for s in suggestions
            )
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": (
                f"Note: `{requested}` was not found in the {provider_label} curated catalog "
                f"and the /models endpoint was unreachable.{suggestion_text}"
                f"\n  The model may still work if it exists on the provider."
            ),
        }

    # No catalog available — accept with a warning, matching the comment's
    # stated intent ("Accept and persist, but warn").
    return {
        "accepted": True,
        "persist": True,
        "recognized": False,
        "message": (
            f"Note: could not reach the {provider_label} API to validate `{requested}`. "
            f"If the service isn't down, this model may not be valid."
        ),
    }

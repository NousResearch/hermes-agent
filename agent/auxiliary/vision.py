"""Vision-specific resolution for the auxiliary client (F-C3 step 2).

The vision path has its own auto-selection order (OpenRouter → Nous →
Codex → Anthropic → custom endpoint), its own "which backends are
available right now?" question (used by setup + tool gating), and
its own override precedence (direct endpoint overrides > explicit
provider > auto). Text-only auxiliary tasks don't use any of this,
so it reads more cleanly as its own module.

Everything here used to live at the bottom of ``auxiliary_client.py``
next to the text resolution chain. The move is verbatim — names,
argument order, and return shapes are preserved. The public symbols
are re-exported from ``agent.auxiliary`` so legacy callers
(``tools/vision_tools.py``, ``tests/agent/test_auxiliary_client.py``)
keep working without an edit, and the shim at
``agent/auxiliary_client.py`` continues to surface them through its
``from agent.auxiliary.base import *`` + private-symbol re-export
block.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from openai import OpenAI

from agent.auxiliary.base import (
    _get_cached_client,
    _resolve_task_provider_model,
    _to_async_client,
    _try_anthropic,
    _try_codex,
    _try_custom_endpoint,
    _try_nous,
    _try_openrouter,
    resolve_provider_client,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_VISION_AUTO_PROVIDER_ORDER",
    "_normalize_vision_provider",
    "_resolve_strict_vision_backend",
    "_strict_vision_backend_available",
    "_preferred_main_vision_provider",
    "get_available_vision_backends",
    "resolve_vision_provider_client",
    "get_vision_auxiliary_client",
    "get_async_vision_auxiliary_client",
]


_VISION_AUTO_PROVIDER_ORDER = (
    "openrouter",
    "nous",
    "openai-codex",
    "anthropic",
    "custom",
)


def _normalize_vision_provider(provider: Optional[str]) -> str:
    provider = (provider or "auto").strip().lower()
    if provider == "codex":
        return "openai-codex"
    if provider == "main":
        return "custom"
    return provider


def _resolve_strict_vision_backend(provider: str) -> Tuple[Optional[Any], Optional[str]]:
    provider = _normalize_vision_provider(provider)
    if provider == "openrouter":
        return _try_openrouter()
    if provider == "nous":
        return _try_nous()
    if provider == "openai-codex":
        return _try_codex()
    if provider == "anthropic":
        return _try_anthropic()
    if provider == "custom":
        return _try_custom_endpoint()
    return None, None


def _strict_vision_backend_available(provider: str) -> bool:
    return _resolve_strict_vision_backend(provider)[0] is not None


def _preferred_main_vision_provider() -> Optional[str]:
    """Return the selected main provider when it is also a supported vision backend."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            provider = _normalize_vision_provider(model_cfg.get("provider", ""))
            if provider in _VISION_AUTO_PROVIDER_ORDER:
                return provider
    except Exception:
        pass
    return None


def get_available_vision_backends() -> List[str]:
    """Return the currently available vision backends in auto-selection order.

    This is the single source of truth for setup, tool gating, and runtime
    auto-routing of vision tasks. The selected main provider is preferred when
    it is also a known-good vision backend; otherwise Hermes falls back through
    the standard conservative order.
    """
    ordered = list(_VISION_AUTO_PROVIDER_ORDER)
    preferred = _preferred_main_vision_provider()
    if preferred in ordered:
        ordered.remove(preferred)
        ordered.insert(0, preferred)
    return [provider for provider in ordered if _strict_vision_backend_available(provider)]


def resolve_vision_provider_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    async_mode: bool = False,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Resolve the client actually used for vision tasks.

    Direct endpoint overrides take precedence over provider selection. Explicit
    provider overrides still use the generic provider router for non-standard
    backends, so users can intentionally force experimental providers. Auto mode
    stays conservative and only tries vision backends known to work today.
    """
    requested, resolved_model, resolved_base_url, resolved_api_key = _resolve_task_provider_model(
        "vision", provider, model, base_url, api_key
    )
    requested = _normalize_vision_provider(requested)

    def _finalize(resolved_provider: str, sync_client: Any, default_model: Optional[str]):
        if sync_client is None:
            return resolved_provider, None, None
        final_model = resolved_model or default_model
        if async_mode:
            async_client, async_model = _to_async_client(sync_client, final_model)
            return resolved_provider, async_client, async_model
        return resolved_provider, sync_client, final_model

    if resolved_base_url:
        client, final_model = resolve_provider_client(
            "custom",
            model=resolved_model,
            async_mode=async_mode,
            explicit_base_url=resolved_base_url,
            explicit_api_key=resolved_api_key,
        )
        if client is None:
            return "custom", None, None
        return "custom", client, final_model

    if requested == "auto":
        ordered = list(_VISION_AUTO_PROVIDER_ORDER)
        preferred = _preferred_main_vision_provider()
        if preferred in ordered:
            ordered.remove(preferred)
            ordered.insert(0, preferred)

        for candidate in ordered:
            sync_client, default_model = _resolve_strict_vision_backend(candidate)
            if sync_client is not None:
                return _finalize(candidate, sync_client, default_model)
        logger.debug("Auxiliary vision client: none available")
        return None, None, None

    if requested in _VISION_AUTO_PROVIDER_ORDER:
        sync_client, default_model = _resolve_strict_vision_backend(requested)
        return _finalize(requested, sync_client, default_model)

    client, final_model = _get_cached_client(requested, resolved_model, async_mode)
    if client is None:
        return requested, None, None
    return requested, client, final_model


def get_vision_auxiliary_client() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for vision/multimodal auxiliary tasks."""
    _, client, final_model = resolve_vision_provider_client(async_mode=False)
    return client, final_model


def get_async_vision_auxiliary_client():
    """Return (async_client, model_slug) for async vision consumers."""
    _, client, final_model = resolve_vision_provider_client(async_mode=True)
    return client, final_model

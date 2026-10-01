"""Canonical fast-mode capability and route-gating metadata."""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from models.metadata.context import is_anthropic_fast_mode_model, is_grok_46_family
from providers.identity import normalize_provider

_OPENAI_FAST_MODE_PREFIXES: tuple[str, ...] = ("gpt-", "o1", "o3", "o4")


def _strip_vendor_prefix(model_id: str) -> str:
    raw = str(model_id or "").strip().lower()
    return raw.split("/", 1)[1] if "/" in raw else raw


def _is_openai_fast_model(model_id: Optional[str]) -> bool:
    base = _strip_vendor_prefix(str(model_id or "")).split(":")[0]
    return bool(base) and "codex" not in base and base.startswith(_OPENAI_FAST_MODE_PREFIXES)


def model_supports_fast_mode(model_id: Optional[str]) -> bool:
    """Whether Hermes should expose the /fast toggle for this model."""
    return (
        is_anthropic_fast_mode_model(model_id)
        or _is_openai_fast_model(model_id)
        or is_grok_46_family(str(model_id or ""))
    )


def _fast_mode_route_supported(
    model_id: Optional[str],
    provider: Optional[str],
    base_url: Optional[str],
) -> bool:
    if is_anthropic_fast_mode_model(model_id):
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
    model_id: Optional[str],
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, str] | None:
    """Fast/priority request overrides for a supported first-party route."""
    if not model_supports_fast_mode(model_id):
        return None
    if (provider or base_url) and not _fast_mode_route_supported(model_id, provider, base_url):
        return None
    return (
        {"speed": "fast"}
        if is_anthropic_fast_mode_model(model_id)
        else {"service_tier": "priority"}
    )


__all__ = ["model_supports_fast_mode", "resolve_fast_mode_overrides"]

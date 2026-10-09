"""Registered provider runtime refresh hooks; CLI dispatch remains at the edge."""
from __future__ import annotations
from typing import Any, Callable
def plugin_refresh_hook(provider: str) -> Callable[[Any], Any] | None:
    try:
        from providers import get_provider_profile
    except Exception:
        return None
    hook = getattr(get_provider_profile(provider), "refresh_credential", None)
    return hook if callable(hook) else None

"""
Image Generation Provider Registry
==================================

Central map of registered providers. Populated by plugins at import-time via
``PluginContext.register_image_gen_provider()``; consumed by the
``image_generate`` tool to dispatch each call to the active backend.

Active selection
----------------
The active provider is chosen by ``image_gen.provider`` in ``config.yaml``.
If unset, :func:`get_active_provider` applies fallback logic:

1. If exactly one provider is registered, use it.
2. Otherwise if a provider named ``fal`` is registered, use it (legacy
   default — matches pre-plugin behavior).
3. Otherwise return ``None`` (the tool surfaces a helpful error pointing
   the user at ``hermes tools``).
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from agent.image_gen_provider import ImageGenProvider
from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)


_providers: Dict[str, ImageGenProvider] = {}
_scoped_providers: Dict[str, Dict[str, ImageGenProvider]] = {}
_lock = threading.Lock()


def register_provider(provider: ImageGenProvider, *, scope: Optional[str] = None) -> None:
    """Register an image generation provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — this makes hot-reload scenarios (tests, dev loops)
    behave predictably.
    """
    if not isinstance(provider, ImageGenProvider):
        raise TypeError(
            f"register_provider() expects an ImageGenProvider instance, "
            f"got {type(provider).__name__}"
        )
    raw_name = provider.name
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("Image gen provider .name must be a non-empty string")
    name = raw_name.strip()
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        existing = target.get(name)
        target[name] = provider
    if existing is not None:
        logger.debug("Image gen provider '%s' re-registered (was %r)", name, type(existing).__name__)
    else:
        logger.debug("Registered image gen provider '%s' (%s)", name, type(provider).__name__)


def list_providers(*, scope: Optional[str] = None) -> List[ImageGenProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        merged = dict(_providers)
        merged.update(_scoped_providers.get(scope or hermes_home_key(), {}))
        items = list(merged.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str, *, scope: Optional[str] = None) -> Optional[ImageGenProvider]:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    with _lock:
        key = name.strip()
        return _scoped_providers.get(scope or hermes_home_key(), {}).get(key) or _providers.get(key)


def snapshot_registration(
    name: str, *, scope: Optional[str] = None
) -> Optional[ImageGenProvider]:
    with _lock:
        target = _providers if scope is None else _scoped_providers.get(scope, {})
        return target.get(name.strip())


def restore_registration(
    name: str,
    current: ImageGenProvider,
    previous: Optional[ImageGenProvider],
    *,
    scope: Optional[str] = None,
) -> bool:
    """Restore a plugin registration only when *current* is still installed."""
    key = name.strip()
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        if target.get(key) is not current:
            return False
        if previous is None:
            target.pop(key, None)
        else:
            target[key] = previous
        if scope is not None and not target:
            _scoped_providers.pop(scope, None)
    return True


def get_active_provider() -> Optional[ImageGenProvider]:
    """Resolve the currently-active provider.

    Reads ``image_gen.provider`` from config.yaml; falls back per the
    module docstring.

    **Availability semantics** (mirrors :mod:`agent.web_search_registry`):

    - When ``image_gen.provider`` is explicitly set, the configured
      provider is returned even if :meth:`ImageGenProvider.is_available`
      reports False — the dispatcher surfaces a precise "X_API_KEY is not
      set" error rather than silently switching backends.
    - When ``image_gen.provider`` is unset, the fallback path (single-
      provider shortcut and the FAL legacy preference) is filtered by
      ``is_available()`` so we don't pick a provider the user has no
      credentials for.
    """
    configured: Optional[str] = None
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            raw = section.get("provider")
            if isinstance(raw, str) and raw.strip():
                configured = raw.strip()
    except Exception as exc:
        logger.debug("Could not read image_gen.provider from config: %s", exc)

    # The managed "Nous Subscription" selection is serviced by the FAL
    # plugin through the managed fal-queue gateway (the legacy FAL pipeline
    # routes managed when the stored selection is "nous").
    if configured:
        try:
            from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER

            if configured.lower() == NOUS_MANAGED_PROVIDER:
                configured = "fal"
        except Exception:  # pragma: no cover — helpers are in-repo
            pass

    with _lock:
        snapshot = dict(_providers)
        snapshot.update(_scoped_providers.get(hermes_home_key(), {}))

    def _is_available_safe(p: ImageGenProvider) -> bool:
        """Wrap ``is_available()`` so a buggy provider doesn't kill resolution."""
        try:
            return bool(p.is_available())
        except Exception as exc:  # noqa: BLE001
            logger.debug("image_gen provider %s.is_available() raised %s", p.name, exc)
            return False

    # 1. Explicit config wins — return regardless of is_available() so the
    #    user gets a precise downstream error message rather than a silent
    #    backend switch.
    if configured:
        provider = snapshot.get(configured)
        if provider is not None:
            return provider
        logger.debug(
            "image_gen.provider='%s' configured but not registered; falling back",
            configured,
        )

    # 2. Fallback: single registered provider — but only if it's actually
    #    available (no credentials = don't surface it as "active").
    available = [p for p in snapshot.values() if _is_available_safe(p)]
    if len(available) == 1:
        return available[0]

    # 3. Fallback: prefer legacy FAL for backward compat, when available.
    fal = snapshot.get("fal")
    if fal is not None and _is_available_safe(fal):
        return fal

    return None


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    with _lock:
        _providers.clear()
        _scoped_providers.clear()

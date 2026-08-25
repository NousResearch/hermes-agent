"""Image generation provider registry.

Populated by plugins at import-time via ``PluginContext.register_image_gen_provider()``;
the ``image_generate`` tool dispatches to :func:`get_active_provider`. Selection is
``image_gen.provider`` in config.yaml; when unset: the single *available* provider,
else ``fal`` if registered and available (legacy default), else ``None`` (the tool
points the user at ``hermes tools``).
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.image_gen_provider import ImageGenProvider
from agent.provider_registry import ProviderRegistry, configured_provider_name, is_available_safe

logger = logging.getLogger(__name__)


_registry: ProviderRegistry[ImageGenProvider] = ProviderRegistry(
    label="Image gen", provider_cls=ImageGenProvider, logger=logger,
)
_registry.export(globals())


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
        if snapshot.get(configured) is not None:
            return snapshot[configured]
        logger.debug("image_gen.provider='%s' configured but not registered; falling back", configured)

    def _available(p: ImageGenProvider) -> bool:
        return is_available_safe(p, logger, "image_gen provider %s.is_available() raised %s")

    available = [p for p in snapshot.values() if _available(p)]
    if len(available) == 1:
        return available[0]
    fal = snapshot.get("fal")
    return fal if fal is not None and _available(fal) else None


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Dict  # noqa: F401,E402
from typing import List  # noqa: F401,E402
import threading  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'hermes_home_key': ('hermes_constants', 'hermes_home_key'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

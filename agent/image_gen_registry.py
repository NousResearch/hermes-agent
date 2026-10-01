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
    """Resolve the currently-active provider. Availability semantics (mirrors
    :mod:`agent.web_search_registry`): an explicitly configured provider is returned
    even if ``is_available()`` is False, so the dispatcher surfaces a precise
    "X_API_KEY is not set" error instead of silently switching backends; only the
    unconfigured fallback path is filtered by availability."""
    configured = configured_provider_name("image_gen", logger)
    snapshot = _registry.merged()
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

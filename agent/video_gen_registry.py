"""Video Generation Provider Registry.

Populated by plugins via ``PluginContext.register_video_gen_provider()``;
consumed by the ``video_generate`` tool. The active provider is
``video_gen.provider`` from ``config.yaml``; a configured-but-unregistered name
fails closed. If unset, the single *available* registered provider is used
(mirrors ``agent/image_gen_registry.py`` minus its legacy ``fal`` preference)
so a box with credentials for only one backend auto-selects it; otherwise None
and the tool points the user at ``hermes tools``.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.provider_registry import ProviderRegistry, configured_provider_name, is_available_safe
from agent.video_gen_provider import VideoGenProvider

logger = logging.getLogger(__name__)


_registry: ProviderRegistry[VideoGenProvider] = ProviderRegistry(
    label="Video gen", provider_cls=VideoGenProvider, logger=logger,
)
_registry.export(globals())


def get_active_provider() -> Optional[VideoGenProvider]:
    """Resolve the currently-active provider (see module docstring)."""
    configured = configured_provider_name("video_gen", logger)
    snapshot = _registry.merged()

    if configured:
        provider = snapshot.get(configured)
        if provider is None:
            logger.debug(
                "video_gen.provider='%s' configured but not registered; failing closed", configured
            )
        return provider

    available = [
        p for p in snapshot.values()
        if is_available_safe(p, logger, "video_gen provider %s.is_available() raised %s")
    ]
    return available[0] if len(available) == 1 else None


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

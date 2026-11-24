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

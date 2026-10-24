"""Profile-scoped plugin registration for bounded inbound origin projections."""

from functools import cached_property
from typing import Callable


class PluginOriginManagerMixin:
    @cached_property
    def _inbound_origin_projectors(self) -> dict[str, tuple[str, Callable, Callable]]:
        return {}


class PluginOriginContextMixin:
    def register_inbound_origin_projector(self, *, platform: str, capture: Callable, validate: Callable):
        """Register one source validator for a platform in this plugin's profile."""
        if self._load_abandoned:
            raise self._refuse("an inbound origin projector after load timed out")
        if not isinstance(platform, str) or not platform or not callable(capture) or not callable(validate):
            raise self._refuse("an invalid inbound origin projector")
        key = platform.lower()
        if key != platform or key in self._manager._inbound_origin_projectors:
            raise self._refuse(f"a duplicate or invalid inbound origin projector for {platform!r}")
        entry = (self.plugin_id, capture, validate)
        return self._register_entry(
            "inbound_origin_projector", key, self._manager._inbound_origin_projectors,
            entry, "Plugin %s registered inbound origin projector: %s", key,
        )

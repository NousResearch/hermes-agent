"""Host capability accessors exposed through the public plugin context."""

from __future__ import annotations

from functools import cached_property
from typing import Any


class PluginContextCapabilitiesMixin:
    """Create the host-owned capability facades available to a plugin."""

    @cached_property
    def desktop(self):
        """Narrow desktop media capability facade for plugins.

        Availability means the desktop event sink is reachable; each request
        still checks the live session and transport authority in the gateway.
        """
        from hermes_cli.plugin_desktop import DesktopPluginCapability
        return DesktopPluginCapability()

    @property
    def llm(self) -> Any:
        """Host-owned :class:`agent.plugin_llm.PluginLlm` facade: completions on the user's active
        model/auth. Overrides (model, agent id, auth profile) are fail-closed, gated by
        ``plugins.entries.<plugin_id>.llm.*``."""
        if self._llm is None:
            from agent.plugin_llm import PluginLlm
            self._llm = PluginLlm(plugin_id=self.plugin_id)
        return self._llm

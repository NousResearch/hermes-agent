"""Native Discord interaction registration API and per-manager registry."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Callable, Optional

from hermes_cli.plugins_loader import _serialized_replacement

logger = logging.getLogger("hermes_cli.plugins")


class PluginInteractionRegistryMixin:
    """Lazily created registry, separate from cross-platform legacy slash commands."""

    @property
    def _plugin_discord_interaction_commands(self) -> dict[str, dict]:
        return self.__dict__.setdefault("_plugin_discord_interaction_commands_store", {})


class PluginInteractionRegistrationMixin:
    """Additive PluginContext method for typed Discord native interactions."""

    @_serialized_replacement
    def register_discord_interaction_command(
        self, name: str, handler: Callable, description: str = "", args_hint: str = "",
    ) -> Optional[object]:
        """Register ``handler(raw_args, interaction_context)`` for a native Discord interaction.

        The handler returns explicitly routed ``DiscordInteractionReply`` values. The supplied
        context has platform-verified identity and interaction/server/channel IDs; plugins remain
        responsible for their own authorization. Existing ``register_command`` and text/``!``
        behavior is unchanged.
        """
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning("Plugin '%s' tried to register an interaction command with an empty name.",
                           self.manifest.name)
            return None
        with suppress(Exception):
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                logger.warning("Plugin '%s' interaction command '/%s' conflicts with a built-in command.",
                               self.manifest.name, clean)
                return None
        if clean in self._manager._plugin_commands:
            logger.warning("Plugin '%s' interaction command '/%s' conflicts with a legacy plugin command.",
                           self.manifest.name, clean)
            return None
        hint = (args_hint or "").strip()
        entry = {
            "handler": handler,
            "description": description or "Discord plugin interaction",
            "plugin": self.manifest.name,
            "plugin_key": self.plugin_id,
            "args_hint": hint,
        }
        return self._register_entry(
            "discord_interaction_command", clean, self._manager._plugin_discord_interaction_commands,
            entry, "Plugin %s registered Discord interaction command: /%s", clean,
        )


def get_plugin_discord_interaction_commands() -> dict[str, dict]:
    """Return opt-in native Discord interaction commands, separate from legacy slash dispatch."""
    from hermes_cli.plugins import _ensure_plugins_discovered

    return _ensure_plugins_discovered()._plugin_discord_interaction_commands

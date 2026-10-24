"""PluginContext delegation capabilities, backed by the host lifecycle service."""
from functools import cached_property
from typing import Any
from hermes_cli.plugins_loader import _serialized_replacement


class SubagentExtensionContextMethods:
    @cached_property
    def subagent_lifecycle(self) -> Any:
        """Plugin-safe subagent lifecycle service: serializable handles and immutable snapshots,
        never a live agent or private registry."""
        from agent.subagent_lifecycle import SubagentLifecycleService, get_active_subagent_parent
        return SubagentLifecycleService(get_active_subagent_parent)

    @_serialized_replacement
    def register_subagent_extension(self, extension):
        """Install one data-only delegation extension for this profile; unload restores its predecessor."""
        from hermes_cli.plugins import PluginToolOverrideError
        if not self._tool_override_allowed("delegate_task"):
            raise PluginToolOverrideError(
                f"Plugin {self.plugin_id!r} requires plugins.entries.{self.plugin_id}.allow_tool_override: true "
                "to extend delegate_task.")
        from agent.subagent_extension import SubagentExtension
        if self._wrong_type(extension, SubagentExtension, "subagent extension", "a"):
            return None
        if not hasattr(self._manager, "_subagent_extensions"):
            self._manager._subagent_extensions = {}
        return self._register_entry("subagent_extension", "default", self._manager._subagent_extensions,
                                    extension, "Plugin %s registered a subagent extension")

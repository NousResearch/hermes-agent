"""Tool registrars on ``PluginContext``: ``register_tool`` and ``extend_tool_schema``.

Bound into the class body (``from hermes_cli.plugins_tools import ...``), so they are ordinary
ctx methods with the same abandoned-load guard as every registrar.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from hermes_cli.plugins_ledger import PluginRegistration
from hermes_cli.plugins_loader import _serialized_replacement

# Same logger as the rest of the ctx surface, so plugin-load diagnostics stay in one place.
logger = logging.getLogger("hermes_cli.plugins")

__all__ = ["extend_tool_schema", "register_tool"]


@_serialized_replacement
def register_tool(
    self, name: str, toolset: str, schema: dict, handler: Callable,
    check_fn: Callable | None = None, requires_env: list | None = None, is_async: bool = False,
    description: str = "", emoji: str = "", override: bool = False,
    *, max_result_size_chars: float | None = None,
    dynamic_schema_overrides: Callable | None = None, execution_timeout: float | None = None,
) -> Optional[PluginRegistration]:
    """Register a tool in the global registry and track it as plugin-provided. ``override=True``
    replaces a same-named built-in (without it a name claimed by another toolset is rejected) and
    needs operator opt-in via ``plugins.entries.<plugin_id>.allow_tool_override: true`` — otherwise
    any enabled plugin could silently replace a privileged built-in like ``write_file`` (#23194).
    Result caps, dynamic schema overrides and ``execution_timeout`` (seconds, raises the tool's
    dispatch deadline) pass through to the registry unchanged."""
    if override and not self._tool_override_allowed(name):
        from hermes_cli.plugins import PluginToolOverrideError
        raise PluginToolOverrideError(
            f"Plugin {self.manifest.name!r} cannot override built-in tool {name!r}. Set "
            f"plugins.entries.{self.plugin_id}.allow_tool_override: true "
            f"in config.yaml to allow this plugin to replace built-in tools."
        )
    from tools.registry import registry
    scope = self._manager.scope_key
    previous = registry.snapshot_registration(name, scope=scope)
    if previous is None and not override and registry.get_entry(name, scope=scope) is not None:
        logger.warning("Plugin %s tried to shadow global tool %s without override=True",
                       self.manifest.name, name)
        return None
    registry.register(
        name=name, toolset=toolset, schema=schema, handler=handler, check_fn=check_fn,
        requires_env=requires_env, is_async=is_async, description=description, emoji=emoji,
        max_result_size_chars=max_result_size_chars,
        dynamic_schema_overrides=dynamic_schema_overrides, execution_timeout=execution_timeout,
        override=override, scope=scope,
    )
    registered = registry.snapshot_registration(name, scope=scope)
    handle = None
    if registered is not None and registered is not previous and registered.handler is handler:
        self._manager._plugin_tool_names.add(name)
        handle = self._manager._track_scoped_registration(
            self.manifest, "tool", name, registry, registered, previous,
            finalize=lambda: self._manager._remove_tool_name_if_unowned(name),
        )
    logger.debug("Plugin %s registered tool: %s%s", self.manifest.name, name,
                 " (override)" if override else "")
    return handle


def extend_tool_schema(
    self, tool_name: str, extender: Callable[[dict], Optional[dict]],
) -> PluginRegistration:
    """Rewrite *tool_name*'s schema each time tool definitions are built, for this plugin's profile.

    *extender* receives a deep copy of the resolved function schema (``{name, description,
    parameters}``, after the tool's own ``dynamic_schema_overrides`` and earlier extenders, in
    registration order) and returns the new schema, or ``None`` to leave it unchanged. ``name`` must
    not change. An extender that raises is logged once and skipped. It binds to the tool *name*, so
    it applies even if that tool registers after this plugin, and it is removed on unload.

    Definitions are memoized (``model_tools.get_tool_definitions``): they recompute when a tool or
    extender is (de)registered, config.yaml changes, or the profile/toolset selection changes. An
    extender reading other state sees its output reused until one of those happens. The schema only
    changes what the model is shown: argument coercion still reads the static schema, and the tool's
    handler must accept (or a ``pre_tool_call`` hook must consume) any field the extender adds.
    """
    if not callable(extender):
        raise TypeError("extend_tool_schema extender must be callable")
    from tools.registry import registry
    remove = registry.add_schema_extender(tool_name, extender, scope=self._manager.scope_key)
    logger.debug("Plugin %s extended tool schema: %s", self.manifest.name, tool_name)
    return self._track("tool_schema_extender", tool_name, remove)

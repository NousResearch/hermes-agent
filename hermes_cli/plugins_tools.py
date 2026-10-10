"""PluginContext tool registration and disposable session ownership."""

from __future__ import annotations

from typing import Callable, Optional

from hermes_cli.plugins_ledger import PluginRegistration
from hermes_cli.plugins_loader import _serialized_replacement


class PluginToolContextMixin:
    """Public tool registration surface shared by PluginContext."""

    @_serialized_replacement
    def register_tool(
        self, name: str, toolset: str, schema: dict, handler: Callable,
        check_fn: Callable | None = None, requires_env: list | None = None, is_async: bool = False,
        description: str = "", emoji: str = "", override: bool = False,
        _owner_session_key: str | None = None,
    ) -> Optional[PluginRegistration]:
        """Register a tool in the global registry and track it as plugin-provided. ``override=True``
        replaces a same-named built-in (without it a name claimed by another toolset is rejected) and
        needs operator opt-in via ``plugins.entries.<plugin_id>.allow_tool_override: true`` — otherwise
        any enabled plugin could silently replace a privileged built-in like ``write_file``.

        ``override=True`` against a built-in tool requires the operator to opt in via
        ``plugins.entries.<plugin_id>.allow_tool_override: true`` in config.yaml — mirrors the trust gate
        pattern used for ``ctx.llm`` provider/model overrides (#23194).
        """
        from hermes_cli.plugins import PluginToolOverrideError, logger

        if override and not self._tool_override_allowed(name):
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
            override=override, scope=scope, owner_session_key=_owner_session_key,
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


    def session_toolset(
        self, session_key: str, *, name: str, description: str = "", direct: bool = False,
    ):
        """Create a disposable toolset owned by one gateway session.

        Register tools before the session's first agent turn. Hermes freezes
        membership when it resolves that session's schema so the conversation
        keeps a byte-stable prompt-cache prefix.
        """
        from hermes_cli.plugin_session_toolsets import PluginSessionToolset
        handle = PluginSessionToolset(self, session_key, name, description, direct=direct)
        registration = self._track("session_toolset", handle.name, handle._dispose_resources)
        handle._bind_ownership_registration(registration)
        return handle


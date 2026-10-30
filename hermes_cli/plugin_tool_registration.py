"""Focused implementation for PluginContext tool registration."""

from typing import Callable, Optional

from hermes_cli.plugins import (
    PluginToolOverrideError,
    _plugin_home_scope,
    _plugin_settings_entry,
    logger,
)
from hermes_cli.plugins_ledger import PluginRegistration


def _gateway_source_consent(ctx) -> Callable[[], bool]:
    """Build the per-registration, profile-scoped source-consent check."""
    manager = ctx._manager
    manifest = ctx.manifest
    plugin_id = ctx.plugin_id

    def consent() -> bool:
        try:
            if manifest.source == "bundled":
                return False
            with _plugin_home_scope(manager.home_path):
                if manager._runs_in_plugin_host(manifest) is not True:
                    return False
                from hermes_cli import plugins as plugins_module

                entry = _plugin_settings_entry(
                    plugins_module.load_config_readonly() or {}, plugin_id
                )
            return entry is not None and entry.get("allow_gateway_source_context") is True
        except Exception:  # health: allow BLE001 -- deny source context on any consent callback failure
            logger.exception(
                "Gateway source consent check failed; denying source context for %s",
                plugin_id,
            )
            return False

    return consent


def register_plugin_tool(
    ctx,
    name: str,
    toolset: str,
    schema: dict,
    handler: Callable,
    check_fn: Callable | None = None,
    requires_env: list | None = None,
    is_async: bool = False,
    description: str = "",
    emoji: str = "",
    override: bool = False,
    requires_gateway_source_context: bool = False,
) -> Optional[PluginRegistration]:
    """Register the tool and preserve PluginContext's registration lifecycle."""
    if override and not ctx._tool_override_allowed(name):
        raise PluginToolOverrideError(
            f"Plugin {ctx.manifest.name!r} cannot override built-in tool {name!r}. Set "
            f"plugins.entries.{ctx.plugin_id}.allow_tool_override: true "
            f"in config.yaml to allow this plugin to replace built-in tools."
        )

    from tools.registry import registry

    scope = ctx._manager.scope_key
    previous = registry.snapshot_registration(name, scope=scope)
    if previous is None and not override and registry.get_entry(name, scope=scope) is not None:
        logger.warning(
            "Plugin %s tried to shadow global tool %s without override=True",
            ctx.manifest.name,
            name,
        )
        return None

    source_gate = (
        _gateway_source_consent(ctx)
        if requires_gateway_source_context is True
        else None
    )
    registry.register(
        name=name,
        toolset=toolset,
        schema=schema,
        handler=handler,
        check_fn=check_fn,
        requires_env=requires_env,
        is_async=is_async,
        description=description,
        emoji=emoji,
        override=override,
        scope=scope,
        gateway_source_consent=source_gate,
        requires_gateway_source_context=requires_gateway_source_context is True,
    )

    registered = registry.snapshot_registration(name, scope=scope)
    handle = None
    if registered is not None and registered is not previous and registered.handler is handler:
        ctx._manager._plugin_tool_names.add(name)
        handle = ctx._manager._track_scoped_registration(
            ctx.manifest,
            "tool",
            name,
            registry,
            registered,
            previous,
            finalize=lambda: ctx._manager._remove_tool_name_if_unowned(name),
        )
    logger.debug(
        "Plugin %s registered tool: %s%s",
        ctx.manifest.name,
        name,
        " (override)" if override else "",
    )
    return handle

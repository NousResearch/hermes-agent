"""Authenticated gateway command context. Constructed only at the slash dispatch sink."""
from dataclasses import dataclass
import inspect


@dataclass(frozen=True)
class PluginCommandContext:
    _runner: object
    _event: object
    plugin_id: str
    _registration_context: object
    _command_name: str
    _command_entry: dict

    @property
    def source(self):
        from copy import deepcopy
        return deepcopy(self._event.source)

    def start_side_run(self, prompt, config):
        from hermes_cli.plugin_side_runs import SideRunConfig
        options = SideRunConfig.from_mapping(config)
        manager = self._registration_context._manager
        # An async handler can resume after unload/reload. Lease validation and acquiring its
        # cleanup registration must be atomic with that teardown, not just checked at dispatch.
        with manager._discovery_lock:
            if manager._plugin_commands.get(self._command_name) is not self._command_entry:
                raise ValueError("Plugin command is no longer active")
            return self._service().start(self._event, self.plugin_id, prompt, options,
                                         registration_context=self._registration_context)

    def cancel_side_run(self, session_id):
        return self._service().cancel(self._event.source, session_id, plugin_id=self.plugin_id)

    def _service(self):
        from gateway.side_runs import SideRunService
        service = getattr(self._runner, "_plugin_side_runs", None)
        if service is None:
            service = SideRunService(self._runner)
            self._runner._plugin_side_runs = service
        return service


def accepts_context(handler):
    try:
        parameter = inspect.signature(handler).parameters.get("context")
    except (TypeError, ValueError):
        # Opaque builtins retain the original one-argument command contract.
        return False
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
    }


async def dispatch_plugin_command(runner, event, source, command):
    if not command:
        return False, None
    from hermes_cli.plugins import get_plugin_commands, get_plugin_command_handler
    name = command.replace("_", "-")
    handler = get_plugin_command_handler(name)
    if handler is None:
        return False, None
    entry = get_plugin_commands().get(name)
    denied = runner._check_slash_access(source, name)
    if denied is not None:
        return True, denied
    if not event.allow_gateway_control or event.internal:
        return True, "Plugin command requires an authenticated user command."
    try:
        kwargs = {}
        if accepts_context(handler):
            kwargs["context"] = PluginCommandContext(runner, event, entry["plugin_key"], entry["context"], name, entry)
        result = handler(event.get_command_args().strip(), **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return True, str(result) if result is not None else None
    except Exception:
        # Plugin/provider exceptions can contain credentials and must never become model prompts.
        return True, "Plugin command failed. Check its configuration and host compatibility."

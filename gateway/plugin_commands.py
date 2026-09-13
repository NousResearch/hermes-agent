"""Native slash callbacks bound to the messaging source's profile and owner."""

from contextlib import nullcontext
import inspect

from hermes_cli.plugins_command import invoke_plugin_command, plugin_command_context


async def dispatch_plugin_command(runner, event, source, command):
    """Return (handled, output), creating an owner only for a registered command."""
    from gateway.run import _async_profile_runtime_scope
    from hermes_cli.plugins import get_plugin_command_handler

    scope = (_async_profile_runtime_scope(runner._resolve_profile_home_for_source(source))
             if getattr(runner.config, "multiplex_profiles", False) else nullcontext())
    async with scope:
        handler = get_plugin_command_handler(command.replace("_", "-"))
        if handler is None:
            return False, None
        entry = await runner.async_session_store.get_or_create_session(source)
        context = plugin_command_context(
            session_id=entry.session_id, task_id=entry.session_id,
            stored_session_id=entry.session_id, surface="gateway")
        result = invoke_plugin_command(handler, event.get_command_args().strip(), **context)
        if inspect.isawaitable(result):
            result = await result
        return True, str(result) if result else None

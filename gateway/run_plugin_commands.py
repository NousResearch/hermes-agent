"""Gateway adapter for plugin slash-command dispatch."""

from __future__ import annotations

import functools
import inspect
import logging

from gateway.session import SessionSource, build_session_context
from gateway.platforms.event import MessageEvent

logger = logging.getLogger("gateway.run")


async def dispatch_gateway_plugin_command(runner, event: MessageEvent, source: SessionSource, command):
    """Dispatch a plugin command with optional source while preserving gateway execution context."""
    if not command:
        return False, None, command
    try:
        from hermes_cli.plugin_command_dispatch import invoke_plugin_command
        from hermes_cli.plugins import get_plugin_command_handler

        plugin_handler = get_plugin_command_handler(command.replace("_", "-"))
        if plugin_handler:
            # Session context is already bound for handlers that use get_session_env().
            # Sync handlers stay off the gateway loop to preserve watchdog liveness.
            plugin_context = build_session_context(source, runner.config)
            plugin_context.session_key = runner._session_key_for_source(source)
            user_args = event.get_command_args().strip()
            with runner._session_env_scope(plugin_context):
                if inspect.iscoroutinefunction(plugin_handler):
                    result = await invoke_plugin_command(plugin_handler, user_args, source=source)
                else:
                    result = await runner._run_in_executor_with_context(
                        functools.partial(
                            invoke_plugin_command, plugin_handler, user_args, source=source
                        )
                    )
                    if inspect.isawaitable(result):
                        result = await result
            return True, str(result) if result else None, command
    except Exception:
        logger.exception("Plugin command dispatch failed")
    return False, None, command

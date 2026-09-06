"""Authenticated context for plugin slash-command handlers."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional


@dataclass(frozen=True, slots=True)
class PluginCommandContext:
    """Immutable identity of the admitted gateway actor invoking a plugin command."""

    platform: str
    user_id: Optional[str]
    chat_id: str
    chat_type: str
    scope_id: Optional[str] = None
    profile: Optional[str] = None


_PLUGIN_COMMAND_CONTEXT: ContextVar[Optional[PluginCommandContext]] = ContextVar(
    "hermes_plugin_command_context", default=None
)


def get_plugin_command_context() -> Optional[PluginCommandContext]:
    """Return the current authenticated gateway command context, or ``None``."""
    return _PLUGIN_COMMAND_CONTEXT.get()


async def call_plugin_command_handler(
    handler: Callable[[str], Any], raw_args: str, *, context: Optional[PluginCommandContext] = None
) -> Any:
    """Invoke a plugin command while binding its task-local authenticated context."""
    token = _PLUGIN_COMMAND_CONTEXT.set(context)
    try:
        result = handler(raw_args)
        if inspect.isawaitable(result):
            return await result
        return result
    finally:
        _PLUGIN_COMMAND_CONTEXT.reset(token)

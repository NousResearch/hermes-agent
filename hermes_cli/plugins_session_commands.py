"""Ledger-backed command and boolean policy registration bodies."""

from contextlib import suppress
import inspect
import logging
import re
from typing import Any, Callable

logger = logging.getLogger("hermes_cli.plugins")

def register_command(
    ctx, name: str, handler: Callable, description: str = "", args_hint: str = "",
    argument_mode: str | None = None,
) -> Any:
    """Register an in-session slash command (``/name``); handler ``fn(raw_args: str) -> str | None``
    (sync or async). ``args_hint`` (e.g. ``"<file>"``) lets adapters like Discord surface an argument
    field; without it the command registers parameterless there but still accepts trailing text."""
    clean = name.lower().strip().lstrip("/").replace(" ", "-")
    if not clean:
        logger.warning("Plugin '%s' tried to register a command with an empty name.", ctx.manifest.name)
        return
    with suppress(Exception):  # reject if it conflicts with a built-in command
        from hermes_cli.commands import resolve_command
        if resolve_command(clean) is not None:
            logger.warning("Plugin '%s' tried to register command '/%s' which conflicts "
                           "with a built-in command. Skipping.", ctx.manifest.name, clean)
            return
    hint = (args_hint or "").strip()
    entry = {
        "handler": handler, "description": description or "Plugin command",
        "plugin": ctx.manifest.name, "plugin_key": ctx.plugin_id, "args_hint": hint,
        "argument_mode": argument_mode if argument_mode in {"options", "text", "mixed"}
        else ("text" if hint else None),
    }
    return ctx._register_entry("command", clean, ctx._manager._plugin_commands, entry,
                                "Plugin %s registered command: /%s", clean)


def register_session_command(
    ctx, name: str, handler: Callable, *, prepare: Callable,
    description: str = "", args_hint: str = "", subcommands: tuple = (),
    advertise_in_gateway_menu: bool = True,
) -> Any:
    """Register a parsed, host-authorized session command; see session_commands."""
    from hermes_cli.session_commands import SessionCommandHandler
    clean = name.lower().strip().lstrip("/").replace(" ", "-")
    handle = ctx.register_command(
        clean, SessionCommandHandler(clean, prepare, handler), description, args_hint)
    if handle is not None:
        entry = ctx._manager._plugin_commands[clean]
        entry.update(subcommands=tuple(subcommands),
                     advertise_in_gateway_menu=advertise_in_gateway_menu)
    return handle


def register_session_policy(
    ctx, key: str, *, config_path: str, apply: Callable, default: bool = False,
) -> Any:
    """Register one durable tri-state boolean policy and its narrow engine callback."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", key):
        raise ValueError("invalid session policy key")
    if (not isinstance(config_path, str) or not config_path or not callable(apply)
            or inspect.iscoroutinefunction(apply) or type(default) is not bool):
        raise ValueError("session policy needs config_path, callback, and bool default")
    if key in ctx._manager._session_policies:
        raise ValueError(f"session policy already registered: {key}")
    entry = {"config_path": config_path, "apply": apply, "default": default,
             "plugin_key": ctx.plugin_id}
    return ctx._register_entry("session_policy", key, ctx._manager._session_policies,
                                entry, "Plugin %s registered session policy: %s", key)

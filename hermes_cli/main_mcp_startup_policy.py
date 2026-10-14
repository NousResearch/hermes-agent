"""CLI startup classification: which ``hermes`` commands own their MCP startup.

Split out of ``hermes_cli/main.py`` (the facade) so the command-classification
predicates live beside each other and stop growing the entrypoint. ``main.py``
re-exports every name here, so existing patches on the facade still intercept them.
"""

from __future__ import annotations

import os


_AGENT_COMMANDS = {None, "chat", "acp", "rl"}
_AGENT_SUBCOMMANDS = {
    "cron": ("cron_command", {"run", "tick"}),
    "gateway": ("gateway_command", {"run"}),
    "mcp": ("mcp_action", {"serve"}),
}


def _is_tui_chat_launch(args) -> bool:
    if getattr(args, "tui", False) or os.environ.get("HERMES_TUI") == "1":
        return True
    # The chat path decides TUI-vs-classic via _resolve_use_tui (--cli/--tui
    # flags, TTY gate, HERMES_TUI env, display.interface config). Bare
    # `hermes`/`hermes chat` with a TUI display config was previously missed
    # here, so the wrapper pre-warmed its own MCP discovery while the TUI
    # gateway (spawned moments later) ran a second one — an idle stdio MCP
    # server copy held dead for the whole session. Only chat commands can
    # launch the TUI; other commands (mcp serve, gateway, acp, cron) keep
    # their own discovery behavior untouched.
    if getattr(args, "command", None) not in {None, "chat"}:
        return False
    # Late-bind through the facade: `hermes_cli.main` re-exports this predicate and
    # `_resolve_use_tui`, so a monkeypatch on either facade name is the one observed here
    # (same convention as the moved cli_* mixin bodies).
    from hermes_cli import main as _facade

    return _facade._resolve_use_tui(args)


def _agent_subcommand_selected(args) -> bool:
    """True for ``cron run/tick``, ``gateway run``, ``mcp serve`` (see _AGENT_SUBCOMMANDS)."""
    _sub_attr, _sub_set = _AGENT_SUBCOMMANDS.get(args.command, (None, None))
    return bool(_sub_attr and getattr(args, _sub_attr, None) in _sub_set)


def _command_has_dedicated_mcp_startup(args) -> bool:
    """acp / gateway run / cron run|tick own their MCP startup on the runtime path."""
    return args.command == "acp" or (
        args.command != "mcp" and _agent_subcommand_selected(args)
    )


def _is_mcp_stdio_server_launch(args) -> bool:
    """True for ``hermes mcp serve``: the stdio *server* mode, not an agent runtime turn.

    It serves a fixed, hand-registered tool set (``mcp_serve._TOOL_NAMES``) and never
    calls ``discover_mcp_tools``, so pre-loading every configured external MCP server
    here spawns processes whose tools the server cannot expose (#30757).
    """
    return args.command == "mcp" and getattr(args, "mcp_action", None) == "serve"


def _should_background_mcp_startup(args) -> bool:
    return not _is_tui_chat_launch(args) and args.command in {None, "chat", "rl"}
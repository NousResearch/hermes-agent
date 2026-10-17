"""Plugin TUI frontends: ``ctx.register_tui_frontend(name, bundle_dir)``.

A frontend is a Node bundle (``<bundle_dir>/dist/entry.js``) that speaks the same stdio gateway
protocol as ``ui-tui``. ``hermes chat --tui --frontend <name>`` runs it in place of the built-in
TUI, through the same launcher (profile, cwd, resume, worktree, shared-gateway attach, exit
summary), so a plugin never re-enters ``hermes_cli.main``.

The registration is stored as a CLI-command entry, so the ``hermes <name>`` shorthand is wired by
the plugin CLI discovery that already runs for a non-built-in first word, and unload/disable
follow the ledger like any other CLI command.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

# EX_TEMPFAIL from sysexits.h: "this frontend cannot run in this terminal". The launcher then
# runs the built-in TUI once in its place.
FRONTEND_UNAVAILABLE_EXIT = 75

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def tui_frontend_entry(ctx: Any, name: str, bundle_dir: str | os.PathLike, help: str) -> dict:
    """The ``_cli_commands`` entry for one frontend; a malformed name is a plugin bug (ValueError)."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"TUI frontend name {name!r} must match {_NAME.pattern} (it is also `hermes <name>`)")
    return {
        "name": name, "help": help or f"Chat in the {name} TUI frontend",
        "description": f"`hermes {name} [FLAGS]` runs `hermes chat --tui --frontend {name} [FLAGS]`; "
                       "every `hermes chat` flag works.",
        "setup_fn": lambda parser: _setup_shorthand(parser, name), "handler_fn": None,
        "plugin": ctx.manifest.name, "plugin_key": ctx.plugin_id,
        "tui_frontend": Path(bundle_dir).expanduser().resolve(),
    }


def _setup_shorthand(parser: Any, name: str) -> None:
    """``hermes <name>`` takes the chat flags and parses as that chat launch. ``command="chat"``
    overrides the subcommand name argparse stored, so startup gating (plugin/MCP discovery, the
    metrics surface) treats it exactly like ``hermes chat``."""
    from hermes_cli._parser import add_chat_arguments
    from hermes_cli.main import cmd_chat

    add_chat_arguments(parser)
    parser.set_defaults(func=cmd_chat, command="chat", tui=True, tui_frontend=name)


def _frontends() -> dict[str, dict]:
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    discover_plugins()  # idempotent: `hermes <name>` already ran it while building the parser
    return {k: v for k, v in get_plugin_manager()._cli_commands.items() if "tui_frontend" in v}


def find_tui_frontend(name: str) -> tuple[Optional[dict], list[str]]:
    """(entry for *name* or None, every registered frontend name)."""
    frontends = _frontends()
    return frontends.get(name), sorted(frontends)

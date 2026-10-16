"""Plugin-provided CLI commands for ``hermes_cli.main``: attaching one descriptor (top-level, or as a
sub-verb under a built-in), and the manifest-declared fast path that attaches the command argv names
importing only its own plugin. ``builtins`` is main's ``_BUILTIN_SUBCOMMANDS``, passed in so this
module never imports the CLI facade."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Collection

logger = logging.getLogger(__name__)


def builtin_verb_subparsers(subparsers, parent: str, builtins: Collection[str]):
    """The sub-verb action of built-in command ``parent`` (``auth`` → its ``add``/``list``/...), or None."""
    parent_parser = subparsers.choices.get(parent) if parent in builtins else None
    actions = getattr(parent_parser, "_actions", ())
    return next((a for a in actions if isinstance(a, argparse._SubParsersAction)), None)


def attach_plugin_cli_command(subparsers, cmd_info, builtins: Collection[str]) -> None:
    """Register one plugin-provided command from its descriptor: top-level, or under the built-in
    named by ``cmd_info["parent"]``."""
    if cmd_info.get("parent"):
        subparsers = builtin_verb_subparsers(subparsers, cmd_info["parent"], builtins)
        if subparsers is None:
            raise ValueError(f"CLI command parent {cmd_info['parent']!r} is not a built-in command with sub-verbs")
    plugin_parser = subparsers.add_parser(
        cmd_info["name"],
        help=cmd_info["help"],
        description=cmd_info.get("description", ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cmd_info["setup_fn"](plugin_parser)
    if cmd_info.get("handler_fn") is not None:
        plugin_parser.set_defaults(func=cmd_info["handler_fn"])


def declared_cli_command_argv(subparsers, builtins: Collection[str]) -> str | None:
    """The ``cli_command_key`` argv could name as a manifest-declared plugin command, else None: an
    unknown first token (``hermes <verb>``) or an unknown sub-verb of a built-in (``hermes auth
    <verb>``). Known built-in verbs return None and pay nothing."""
    from hermes_cli._parser import command_argv

    args = command_argv(sys.argv[1:])
    if not args:
        return None
    if args[0] not in builtins:
        return args[0]
    verbs = builtin_verb_subparsers(subparsers, args[0], builtins)
    if verbs is None or len(args) < 2 or args[1] in verbs.choices:
        return None
    return f"{args[0]} {args[1]}"


def attach_declared_plugin_cli_command(subparsers, key: str, builtins: Collection[str]) -> bool:
    """Attach the ``plugin.yaml``-declared command spelled ``key``, importing only its plugin.

    Found by reading manifests, never by importing plugins; ``key`` comes from
    :func:`declared_cli_command_argv`, so it never shadows a built-in. False when nothing declares
    ``key`` or it failed to attach.
    """
    try:
        from hermes_cli.plugins import cli_command_key, discover_declared_cli_commands

        commands = discover_declared_cli_commands()
    except Exception:  # a broken manifest must never take the CLI down; full discovery still runs
        logger.warning("Declared plugin CLI scan failed", exc_info=True)
        return False
    for cmd_info in commands:
        if cli_command_key(cmd_info["name"], cmd_info["parent"]) != key:
            continue
        try:
            attach_plugin_cli_command(subparsers, cmd_info, builtins)
        except Exception:  # a plugin's setup_fn is third-party code; fall back to full discovery
            logger.warning("Plugin CLI command %r failed to attach", key, exc_info=True)
            return False
        return True
    return False

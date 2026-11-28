"""Plugin-provided top-level ``hermes`` commands (issue #130607).

Each plugin registers descriptors through ``PluginContext.register_cli_command``; this module
turns them into argparse subparsers. It lives beside ``main.py`` rather than inside it so the
facade keeps shrinking.

The registration loop is isolated per command on purpose: it used to run inside one ``try``, so
the first descriptor that raised — a handler-only command registered with ``setup_fn=None``, a
``setup_fn`` that raises, or a name colliding with a built-in registered later — unwound the
loop. Every remaining plugin's command then vanished from the parser table and the CLI answered
``'<cmd>' is not a hermes command`` for all of them, with the exception recorded at DEBUG only.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

#: Exit code for a plugin command whose parser could not be built (argparse's own error code).
_SKIPPED_EXIT_CODE = 2


def _skipped_command(name: str):
    """Handler for a subparser whose ``setup_fn`` raised.

    argparse offers no supported way to remove a subparser that already exists, so leaving it
    without a handler made ``hermes <name>`` fall through to the top-level help and exit 0 as if it
    had worked. Failing loudly is the honest outcome.
    """

    def _fail(_args) -> int:
        logger.error(
            "Plugin CLI command %r is unavailable: its argparse setup failed at startup", name
        )
        return _SKIPPED_EXIT_CODE

    _fail.__name__ = f"_skipped_{name.replace('-', '_')}"
    return _fail


def _raw_argv_handler(handler_fn):
    """Adapt a handler-only descriptor's raw-argv callable to the ``args.func(args)`` contract.

    These plugins dispatch themselves (``sub = args[0] if args else ""``), so they need the trailing
    tokens, not an argparse ``Namespace``. Handing them the Namespace raised
    ``TypeError: 'Namespace' object is not subscriptable`` from inside the plugin, and with no
    positional to collect the trailing tokens argparse rejected ``hermes <cmd> status`` outright.
    """

    def _run(args):
        return handler_fn(list(getattr(args, "plugin_argv", []) or []))

    _run.__name__ = "plugin_raw_argv_handler"
    return _run


def attach_plugin_cli_command(subparsers, cmd_info) -> None:
    """Register one plugin-provided top-level command from its descriptor."""
    plugin_parser = subparsers.add_parser(
        cmd_info["name"],
        help=cmd_info["help"],
        description=cmd_info.get("description", ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    setup_fn = cmd_info.get("setup_fn")
    handler_fn = cmd_info.get("handler_fn")
    if setup_fn is None:
        # ``register_cli_command`` allows a handler-only command whose plugin dispatches on raw
        # argv (setup_fn=None); calling it unconditionally raised TypeError. Keep the handler, and
        # give it a home for the trailing tokens it expects.
        logger.debug(
            "Plugin command %r registers no setup_fn; dispatching it on raw argv",
            cmd_info.get("name"),
        )
        if handler_fn is not None:
            plugin_parser.add_argument("plugin_argv", nargs=argparse.REMAINDER)
            plugin_parser.set_defaults(func=_raw_argv_handler(handler_fn))
        return
    try:
        setup_fn(plugin_parser)
    except Exception:
        plugin_parser.set_defaults(func=_skipped_command(cmd_info["name"]))
        raise
    if handler_fn is not None:
        plugin_parser.set_defaults(func=handler_fn)


def attach_plugin_cli_command_safely(subparsers, cmd_info) -> bool:
    """Attach one plugin command, keeping its failure local to that command.

    A parser created before the failure stays in the table (argparse builds it before ``setup_fn``
    runs), now carrying a handler that exits non-zero and names the command. What matters is that
    every other plugin's command still attaches, and that the skipped one is named at WARNING
    rather than swallowed at DEBUG.
    """
    try:
        attach_plugin_cli_command(subparsers, cmd_info)
        return True
    except Exception as exc:
        logger.warning(
            "Plugin CLI command %r was skipped: %s",
            cmd_info.get("name"),
            exc,
            exc_info=True,
        )
        return False


def register_plugin_cli_commands(subparsers) -> None:
    """Register plugin-provided top-level commands (each plugin builds its own argparse tree).

    Skipped when the invocation targets a known built-in — eagerly importing every bundled
    plugin module costs 500-650ms.
    """
    # Imported here, not at module scope: ``hermes_cli.main`` is the caller and owns these.
    from hermes_cli.main import (
        _first_positional_argv,
        _plugin_cli_discovery_needed,
        _resolve_deferred_platform_cli_command,
    )

    if not _plugin_cli_discovery_needed():
        return
    try:
        from plugins.memory import discover_plugin_cli_commands
        from hermes_cli.plugins import discover_plugins, get_plugin_manager
        from hermes_cli.left_core_migration import recover_for_command

        recover_for_command(_first_positional_argv())
        seen_plugin_commands = set()
        for cmd_info in discover_plugin_cli_commands():
            # Only a name that actually attached blocks a later provider of the same command.
            if attach_plugin_cli_command_safely(subparsers, cmd_info):
                seen_plugin_commands.add(cmd_info["name"])

        discover_plugins()
        # The invoked platform may still be a deferred entry; import it so its
        # register_cli_command side effect runs before we read _cli_commands.
        # See #54678.
        _resolve_deferred_platform_cli_command(_first_positional_argv())
        for cmd_info in get_plugin_manager()._cli_commands.values():
            if cmd_info["name"] not in seen_plugin_commands:
                attach_plugin_cli_command_safely(subparsers, cmd_info)
    except Exception as exc:
        # Discovery failing removes every plugin command at once; say so rather than repeating the
        # DEBUG-only silence this module exists to fix.
        logger.warning(
            "Plugin CLI discovery failed; plugin commands are unavailable: %s", exc, exc_info=True
        )

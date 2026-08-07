"""CLI commands for the Hindsight memory plugin.

Discovered by ``plugins.memory.discover_plugin_cli_commands()`` when
``memory.provider`` is ``hindsight`` and registered as the top-level
``hermes hindsight`` command.  Keep module-level imports stdlib-only —
this file is imported during argparse setup, before any provider SDK
is needed.
"""

from __future__ import annotations

import argparse
from datetime import datetime


def _iso_date(value: str) -> str:
    """argparse ``type=`` validator for YYYY-MM-DD flags."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc
    return value


def _non_negative_int(value: str) -> int:
    """argparse ``type=`` validator rejecting negatives."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return number


def register_cli(subparser) -> None:
    """Build the ``hermes hindsight`` argparse subcommand tree.

    Called by the plugin CLI registration system during argparse setup.
    The *subparser* is the parser for ``hermes hindsight``.
    """
    subs = subparser.add_subparsers(dest="hindsight_command")
    parser = subs.add_parser(
        "import-sessions",
        help="Backfill historical Hermes sessions from state.db into Hindsight",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show import candidates without calling Hindsight",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip Hindsight documents whose IDs already exist; recommended for reruns",
    )
    parser.add_argument(
        "--bank-id", default=None,
        help="Target Hindsight bank (overrides config bank_id/bank_id_template)",
    )
    parser.add_argument(
        "--since", type=_iso_date,
        help="Only import sessions on or after YYYY-MM-DD",
    )
    parser.add_argument(
        "--until", type=_iso_date,
        help="Only import sessions on or before YYYY-MM-DD",
    )
    parser.add_argument(
        "--days", type=_non_negative_int,
        help="Only import sessions from the last N days",
    )
    parser.add_argument(
        "--limit", type=_non_negative_int,
        help="Limit number of sessions considered",
    )
    parser.add_argument(
        "--retain-timeout", type=_non_negative_int, default=600,
        help="Per-session retain timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--doc-id-prefix", default="",
        help="Prefix to add before each session_id document_id",
    )
    parser.add_argument(
        "--extra-tags", default="hermes-backfill",
        help="Comma-separated tags to add (default: hermes-backfill)",
    )


def hindsight_command(args: argparse.Namespace) -> None:
    """Route hindsight subcommands."""
    sub = getattr(args, "hindsight_command", None)
    if sub == "import-sessions":
        from plugins.memory.hindsight.import_sessions import (
            handle_import_sessions_command,
        )

        handle_import_sessions_command(args)
        return
    print("  Usage: hermes hindsight import-sessions [options]")
    print("  Available: import-sessions")

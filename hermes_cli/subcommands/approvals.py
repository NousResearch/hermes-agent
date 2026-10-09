"""``hermes approvals`` subcommand parser.

Follows the cron/security pattern: parser construction lives here, the
handler is injected by ``main.py`` so this module never imports ``main``
(cycle avoidance).
"""

from __future__ import annotations

from typing import Callable


def build_approvals_parser(subparsers, *, cmd_approvals: Callable) -> None:
    """Attach the ``approvals`` subcommand to ``subparsers``."""
    approvals_parser = subparsers.add_parser(
        "approvals",
        help="Approval-prompt tools (mine history into allowlist proposals)",
        description=(
            "Tools for the dangerous-command approval system. "
            "`hermes approvals suggest` mines past approval decisions from "
            "the session database and proposes command_allowlist entries so "
            "repeatedly-approved commands stop prompting."
        ),
    )
    approvals_subparsers = approvals_parser.add_subparsers(
        dest="approvals_command",
        metavar="<subcommand>",
    )

    suggest_parser = approvals_subparsers.add_parser(
        "suggest",
        help="Propose command_allowlist entries from past approvals",
        description=(
            "Scan the session database for dangerous-classified commands "
            "that ran with user approval, rank the recurring patterns, and "
            "print a numbered allowlist proposal. Nothing is written unless "
            "--apply is given. Destructive classes (recursive delete, sudo, "
            "disk writes, credential edits, ...) are never proposed."
        ),
    )
    suggest_parser.add_argument(
        "--apply",
        dest="apply_indices",
        metavar="N[,M...]",
        help="Merge the numbered proposals (from a prior run) into "
        "command_allowlist in config.yaml",
    )
    suggest_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text",
    )
    suggest_parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="How far back to scan session history (default: 90; 0 = all)",
    )
    suggest_parser.add_argument(
        "--min-count",
        dest="min_count",
        type=int,
        default=2,
        help="Minimum approval count for a pattern to be proposed (default: 2)",
    )
    suggest_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of proposals to show (default: 20)",
    )
    suggest_parser.add_argument(
        "--db",
        help="Path to an alternate session database (default: ~/.hermes/state.db)",
    )
    suggest_parser.set_defaults(func=cmd_approvals)
    approvals_parser.set_defaults(func=cmd_approvals)

"""``hermes backup`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_backup_parser(subparsers, *, cmd_backup: Callable) -> None:
    """Attach the ``backup`` subcommand to ``subparsers``."""
    # =========================================================================
    # backup command
    # =========================================================================
    backup_parser = subparsers.add_parser(
        "backup",
        help="Back up Hermes home directory to a zip file",
        description="Create a zip archive of your entire Hermes configuration, "
        "skills, sessions, and data (excludes the hermes-agent codebase). "
        "Use --quick for a fast snapshot of just critical state files.",
    )
    # -o/--output writes a full archive at that path; --quick instead stores a
    # state snapshot into HERMES_HOME/state-snapshots/ and never writes an
    # archive, so the two cannot combine. Passing both used to accept -o and
    # silently never write the file (exit 0, nothing created) (#98369); make it
    # a clear argparse error instead.
    _backup_target = backup_parser.add_mutually_exclusive_group()
    _backup_target.add_argument(
        "-o",
        "--output",
        help="Output path for the zip file (default: ~/hermes-backup-<timestamp>.zip); not used with --quick",
    )
    _backup_target.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="Quick snapshot: only critical state files (config, state.db, .env, auth, cron)",
    )
    backup_parser.add_argument(
        "-l", "--label", help="Label for the snapshot (only used with --quick)"
    )
    backup_parser.set_defaults(func=cmd_backup)

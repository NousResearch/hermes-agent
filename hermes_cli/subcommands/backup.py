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
        "Additional exclusions can be set with backup.exclusions in config.yaml, "
        "--exclude, or --exclude-from. Use --quick for a fast snapshot of just "
        "critical state files.",
    )
    backup_parser.add_argument(
        "-o",
        "--output",
        help="Output path for the zip file (default: ~/hermes-backup-<timestamp>.zip)",
    )
    backup_parser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="Quick snapshot: only critical state files (config, state.db, .env, auth, cron)",
    )
    backup_parser.add_argument(
        "-l", "--label", help="Label for the snapshot (only used with --quick)"
    )
    backup_parser.add_argument(
        "--exclude",
        action="append",
        metavar="PATTERN",
        help=(
            "Exclude a directory name, relative path, or glob from full backups; "
            "repeatable. Also configurable as backup.exclusions in config.yaml."
        ),
    )
    backup_parser.add_argument(
        "--exclude-from",
        action="append",
        metavar="FILE",
        help=(
            "Read full-backup exclusions from a text file, one pattern per line; "
            "repeatable. Also configurable as backup.exclusions_file in config.yaml."
        ),
    )
    backup_parser.set_defaults(func=cmd_backup)

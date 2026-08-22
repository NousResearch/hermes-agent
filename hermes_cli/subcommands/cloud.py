"""``hermes cloud`` subcommand parser.

Hermes Cloud migration verbs. Handler injected to avoid importing ``main``
(cycle avoidance, same shape as ``subcommands/sync.py``).
"""

from __future__ import annotations

import argparse
from typing import Callable


def build_cloud_parser(subparsers, *, cmd_cloud: Callable) -> None:
    """Attach the ``cloud`` subcommand (and its sub-actions) to ``subparsers``."""
    cloud_parser = subparsers.add_parser(
        "cloud",
        help="Migrate this Hermes setup to Hermes Cloud",
        description=(
            "Package this machine's Hermes setup (settings, skills, memory, "
            "SOUL, cron jobs, and optionally chat history) for import into a "
            "Hermes Cloud instance."
        ),
        epilog=(
            "Examples:\n"
            "  hermes cloud export                        write a migration bundle to the current directory\n"
            "  hermes cloud export --include-history      also include conversation history (state.db)\n"
            "  hermes cloud export -o ~/Desktop/mv.zip    write to a specific path\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cloud_sub = cloud_parser.add_subparsers(dest="cloud_command")

    export_parser = cloud_sub.add_parser(
        "export",
        help="Export this Hermes setup as a migration bundle",
        description=(
            "Create a zip of your Hermes configuration, skills, memory, SOUL, "
            "and cron jobs, ready to import into a Hermes Cloud instance. "
            "Secret files (.env, auth.json) are excluded unless "
            "--include-secrets is given; chat history (state.db) is excluded "
            "unless --include-history is given."
        ),
    )
    export_parser.add_argument(
        "-o",
        "--output",
        help="Output path for the zip (default: ./hermes_cloud_migration_<host>_<timestamp>.zip)",
    )
    export_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite an existing bundle at the output path",
    )
    export_parser.add_argument(
        "--include-secrets",
        action="store_true",
        help="Include .env and auth.json in the bundle (to be used with care: "
        "the archive then carries live credentials, and the cloud importer "
        "still discards them)",
    )
    export_parser.add_argument(
        "--include-history",
        action="store_true",
        help="Include chat history (state.db) in the bundle",
    )
    export_parser.set_defaults(func=cmd_cloud)
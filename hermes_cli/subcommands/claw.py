"""``hermes claw`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_claw_parser(subparsers, *, cmd_claw: Callable) -> None:
    """Attach the ``claw`` subcommand to ``subparsers``."""
    claw_parser = subparsers.add_parser(
        "claw",
        help="OpenClaw migration tools",
        description="Migrate settings, memories, skills, and API keys from OpenClaw to HT AI Agent",
    )
    claw_subparsers = claw_parser.add_subparsers(dest="claw_action")

    # claw migrate
    claw_migrate = claw_subparsers.add_parser(
        "migrate",
        help="Migrate from OpenClaw to HT AI Agent",
        description="Import settings, memories, skills, and API keys from an OpenClaw installation. "
        "Always shows a preview before making changes.",
    )
    claw_migrate.add_argument(
        "--source", help="Path to OpenClaw directory (default: ~/.openclaw)"
    )
    claw_migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only — stop after showing what would be migrated",
    )
    claw_migrate.add_argument(
        "--preset",
        choices=["user-data", "full"],
        default="full",
        help="Migration preset (default: full). Neither preset imports secrets — "
        "pass --migrate-secrets to include API keys.",
    )
    claw_migrate.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: refuse to apply when the plan has conflicts)",
    )
    claw_migrate.add_argument(
        "--migrate-secrets",
        action="store_true",
        help="Include allowlisted secrets (TELEGRAM_BOT_TOKEN, API keys, etc.). "
        "Required even under --preset full.",
    )
    claw_migrate.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the pre-migration zip snapshot of ~/.hermes/ (by default a "
        "single restore-point archive is written to ~/.hermes/backups/ "
        "before apply; restorable with 'ht import').",
    )
    claw_migrate.add_argument(
        "--workspace-target", help="Absolute path to copy workspace instructions into"
    )
    claw_migrate.add_argument(
        "--skill-conflict",
        choices=["skip", "overwrite", "rename"],
        default="skip",
        help="How to handle skill name conflicts (default: skip)",
    )
    claw_migrate.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompts"
    )

    # claw cleanup
    claw_cleanup = claw_subparsers.add_parser(
        "cleanup",
        aliases=["clean"],
        help="Archive leftover OpenClaw directories after migration",
        description="Scan for and archive leftover OpenClaw directories to prevent state fragmentation",
    )
    claw_cleanup.add_argument(
        "--source", help="Path to a specific OpenClaw directory to clean up"
    )
    claw_cleanup.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be archived without making changes",
    )
    claw_cleanup.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompts"
    )
    claw_parser.set_defaults(func=cmd_claw)

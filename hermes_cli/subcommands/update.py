"""``hermes update`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_update_parser(subparsers, *, cmd_update: Callable) -> None:
    """Attach the ``update`` subcommand to ``subparsers``."""
    # =========================================================================
    # update command
    # =========================================================================
    update_parser = subparsers.add_parser(
        "update",
        help="Update Hermes Agent to the latest version",
        description="Pull the latest changes from git and reinstall dependencies",
    )
    update_parser.add_argument(
        "--gateway",
        action="store_true",
        default=False,
        help="Gateway mode: use file-based IPC for prompts instead of stdin (used internally by /update)",
    )
    update_parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Check whether an update is available without installing anything",
    )
    update_parser.add_argument(
        "--no-backup",
        action="store_true",
        default=False,
        help="Skip ALL pre-update backups for this run (both the quick state snapshot and the full zip; overrides updates.pre_update_backup)",
    )
    update_parser.add_argument(
        "--backup",
        action="store_true",
        default=False,
        help="Force a FULL pre-update backup (quick state snapshot + HERMES_HOME zip) for this run, regardless of updates.pre_update_backup",
    )
    update_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Assume yes for interactive prompts (config migration, stash restore). API-key entry is skipped; run 'hermes config migrate' separately for those.",
    )
    update_parser.add_argument(
        "--branch",
        default=None,
        metavar="NAME",
        help=(
            "Update against this branch instead of the default (main). "
            "If the local checkout is on a different branch, hermes will "
            "switch to the requested branch first (auto-stashing any "
            "uncommitted changes)."
        ),
    )
    update_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Windows: proceed with the update even when another hermes.exe is detected. The concurrent process will likely cause WinError 32 warnings and may leave a reboot-deferred .exe replacement. Does NOT bypass the venv-process guard (see --force-venv).",
    )
    update_parser.add_argument(
        "--force-venv",
        action="store_true",
        default=False,
        help="Windows: mutate the venv even while other processes are running from its interpreter (desktop backend, gateway, terminals). Those processes keep native .pyd files locked, so the dependency sync will likely fail partway and strand the install half-updated. Use only if you know the detected holders are false positives.",
    )

    update_subparsers = update_parser.add_subparsers(dest="update_subcommand")
    update_auto_parser = update_subparsers.add_parser(
        "auto",
        help="Auto-update run-now and scheduler commands",
        description=(
            "Auto-update commands. run-now reuses the existing update flow. "
            "enable installs a user-level schedule (launchd on macOS, systemd "
            "user timer on Linux when available). These commands do not install "
            "WebUI hooks, rollback, or a Hermes daemon."
        ),
    )
    update_auto_subparsers = update_auto_parser.add_subparsers(
        dest="auto_subcommand",
        required=True,
    )

    update_auto_enable = update_auto_subparsers.add_parser(
        "enable",
        help="Enable scheduled auto-update",
        description=(
            "Enable a user-level schedule. The update time runs `hermes update "
            "auto run-now`; optional plan times run a check-only notice so the "
            "same auto-update manager can announce an upcoming update before "
            "applying it."
        ),
    )
    update_auto_enable.add_argument(
        "--time",
        required=True,
        metavar="HH:MM",
        help="Local 24-hour time to run the scheduled update, for example 04:00.",
    )
    update_auto_enable.add_argument(
        "--plan-time",
        action="append",
        default=[],
        metavar="HH:MM",
        help=(
            "Optional local 24-hour time to run a check-only planned-update "
            "notice. Can be passed more than once."
        ),
    )
    update_auto_enable.set_defaults(
        func=lambda args: __import__(
            "hermes_cli.update_auto",
            fromlist=["cmd_auto_enable"],
        ).cmd_auto_enable(args)
    )

    update_auto_disable = update_auto_subparsers.add_parser(
        "disable",
        help="Disable scheduled auto-update",
        description=(
            "Disable and remove only the Hermes auto-update user-level schedule. "
            "This does not remove unrelated launchd, systemd, or cron jobs."
        ),
    )
    update_auto_disable.set_defaults(
        func=lambda args: __import__(
            "hermes_cli.update_auto",
            fromlist=["cmd_auto_disable"],
        ).cmd_auto_disable(args)
    )

    update_auto_status = update_auto_subparsers.add_parser(
        "status",
        help="Show auto-update scheduler and last-run status",
        description=(
            "Show the auto-update status file, including the last run, planned "
            "notice, version fields, backup path, error, scheduler type/path, "
            "and log path."
        ),
    )
    update_auto_status.set_defaults(
        func=lambda args: __import__(
            "hermes_cli.update_auto",
            fromlist=["cmd_auto_status"],
        ).cmd_auto_status(args)
    )

    update_auto_plan = update_auto_subparsers.add_parser(
        "plan",
        help="Check for updates and print a concise planned-update notice",
        description=(
            "Run the check-only half of auto-update. If an update is available, "
            "print a concise notice suitable for cron/launchd/systemd delivery; "
            "no files are changed and the real update is left for the scheduled "
            "run-now time."
        ),
    )
    update_auto_plan.add_argument(
        "--branch",
        default=None,
        metavar="NAME",
        help="Check this branch instead of the default (main).",
    )
    update_auto_plan.set_defaults(
        func=lambda args: __import__(
            "hermes_cli.update_auto",
            fromlist=["cmd_auto_plan"],
        ).cmd_auto_plan(args)
    )

    update_auto_run_scheduled = update_auto_subparsers.add_parser(
        "run-scheduled",
        help="Run the scheduled auto-update dispatcher",
        description=(
            "Internal scheduler entrypoint. Dispatches to plan or run-now based "
            "on the configured update time and plan-time values."
        ),
    )
    update_auto_run_scheduled.set_defaults(
        func=lambda args: __import__(
            "hermes_cli.update_auto",
            fromlist=["cmd_auto_run_scheduled"],
        ).cmd_auto_run_scheduled(args)
    )

    update_auto_run_now = update_auto_subparsers.add_parser(
        "run-now",
        help="Run one manual non-agentic auto-update pass now",
        description=(
            "Run one manual non-agentic update pass now. This reuses the existing "
            "`hermes update` flow, requires a pre-update backup before applying "
            "changes, uses no LLM/model calls, and writes status/log files."
        ),
    )
    update_auto_run_now.add_argument(
        "--branch",
        default=None,
        metavar="NAME",
        help="Update against this branch instead of the default (main).",
    )
    update_auto_run_now.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Pass through the existing Windows concurrent-process override.",
    )
    update_auto_run_now.set_defaults(
        func=lambda args: __import__(
            "hermes_cli.update_auto",
            fromlist=["cmd_auto_run_now"],
        ).cmd_auto_run_now(args)
    )

    update_parser.set_defaults(func=cmd_update)

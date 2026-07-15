"""``hermes payments`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_payments_parser(subparsers, *, cmd_payments: Callable) -> None:
    """Attach the ``payments`` subcommand family to ``subparsers``."""
    payments_parser = subparsers.add_parser(
        "payments",
        help="Payments review sync helpers",
        description=(
            "Run the Gmail payments sync immediately or install a recurring "
            "Hermes cron job that keeps the canonical payments review store fresh."
        ),
    )
    payments_subparsers = payments_parser.add_subparsers(dest="payments_command")

    sync = payments_subparsers.add_parser(
        "sync-gmail",
        help="Sync Gmail payment candidates into the canonical payments review store",
    )
    sync.add_argument("--query", default=None, help="Override the default Gmail search query")
    sync.add_argument("--max-results", type=int, default=None, help="Maximum Gmail threads to inspect")

    shadow_sync = payments_subparsers.add_parser(
        "shadow-sync-gmail",
        help="Run Gmail sync using the existing pipeline, then mirror results into inbox_items shadow storage",
    )
    shadow_sync.add_argument("--query", default=None, help="Override the default Gmail search query")
    shadow_sync.add_argument("--max-results", type=int, default=None, help="Maximum Gmail threads to inspect")

    payments_subparsers.add_parser(
        "shadow-report",
        help="Compare the legacy payments queue against the inbox_items shadow store",
    )

    shadow_schedule = payments_subparsers.add_parser(
        "schedule-shadow-sync",
        help="Install or update a recurring Hermes cron job for Gmail shadow sync",
    )
    shadow_schedule.add_argument(
        "schedule",
        nargs="?",
        default=None,
        help="Cron schedule like 'every 6h', '30m', or '0 */6 * * *'",
    )
    shadow_schedule.add_argument("--query", default=None, help="Override the default Gmail search query")
    shadow_schedule.add_argument("--max-results", type=int, default=None, help="Maximum Gmail threads to inspect")
    shadow_schedule.add_argument("--name", default=None, help="Friendly cron job name")
    shadow_schedule.add_argument(
        "--run-now",
        action="store_true",
        help="Run one shadow sync immediately after installing or updating the cron job",
    )

    schedule = payments_subparsers.add_parser(
        "schedule-gmail-sync",
        help="Install or update a recurring Hermes cron job for Gmail payments sync",
    )
    schedule.add_argument(
        "schedule",
        nargs="?",
        default=None,
        help="Cron schedule like 'every 1h', '30m', or '0 * * * *'",
    )
    schedule.add_argument("--query", default=None, help="Override the default Gmail search query")
    schedule.add_argument("--max-results", type=int, default=None, help="Maximum Gmail threads to inspect")
    schedule.add_argument("--name", default=None, help="Friendly cron job name")
    schedule.add_argument(
        "--run-now",
        action="store_true",
        help="Run one sync immediately after installing or updating the cron job",
    )

    payments_parser.set_defaults(func=cmd_payments)

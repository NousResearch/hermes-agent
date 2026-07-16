"""``hermes notify`` and ``hermes omi`` subcommand parsers.

Handlers injected to avoid importing ``main`` (mirrors the webhook parser).
"""

from __future__ import annotations

from typing import Callable


def build_notify_parser(subparsers, *, cmd_notify: Callable) -> None:
    """Attach the ``notify`` subcommand to ``subparsers``."""
    notify_parser = subparsers.add_parser(
        "notify",
        help="Inspect and tune the notification / attention budget",
        description=(
            "View today's proactive-notification budget usage and give the "
            "governor feedback so it learns which categories to surface."
        ),
    )
    notify_subparsers = notify_parser.add_subparsers(dest="notify_action")

    notify_subparsers.add_parser(
        "status", help="Show today's budget usage and per-category thresholds"
    )

    keep = notify_subparsers.add_parser(
        "keep", help="Mark a category as wanted (lowers its threshold)"
    )
    keep.add_argument("category", help="Notification category, e.g. omi_commitment")

    mute = notify_subparsers.add_parser(
        "mute", help="Mark a category as unwanted (raises its threshold)"
    )
    mute.add_argument("category", help="Notification category, e.g. cron:<job_id>")

    notify_parser.set_defaults(func=cmd_notify)


def build_omi_parser(subparsers, *, cmd_omi: Callable) -> None:
    """Attach the ``omi`` subcommand to ``subparsers``."""
    omi_parser = subparsers.add_parser(
        "omi",
        help="Omi wearable commitment extraction",
        description=(
            "Scan the Omi wearable transcript for commitments you personally "
            "made and file them as kanban cards. Opt-in (consent required)."
        ),
    )
    omi_subparsers = omi_parser.add_subparsers(dest="omi_action")

    omi_subparsers.add_parser("scan", help="Run the commitment scan now")
    omi_subparsers.add_parser(
        "enable", help="Enable the scan (opt-in) and show how to schedule it"
    )
    omi_subparsers.add_parser("disable", help="Disable the scan")

    omi_parser.set_defaults(func=cmd_omi)


def build_threads_parser(subparsers, *, cmd_threads: Callable) -> None:
    """Attach the ``threads`` (stalled-thread follow-up) subcommand."""
    threads_parser = subparsers.add_parser(
        "threads",
        help="Stalled-thread / open-commitment follow-up",
        description=(
            "Find open loops — commitments past due or untouched, and threads "
            "awaiting your reply — and nudge you once via a batched digest. "
            "Opt-in (consent required)."
        ),
    )
    threads_subparsers = threads_parser.add_subparsers(dest="threads_action")

    threads_subparsers.add_parser("scan", help="Run the stalled-thread scan now")
    threads_subparsers.add_parser(
        "list", help="Show open-loop candidates without nudging (dry run)"
    )
    threads_subparsers.add_parser(
        "enable", help="Enable the scan (opt-in) and show how to schedule it"
    )
    threads_subparsers.add_parser("disable", help="Disable the scan")

    threads_parser.set_defaults(func=cmd_threads)

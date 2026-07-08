"""``hermes webhook`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_webhook_parser(subparsers, *, cmd_webhook: Callable) -> None:
    """Attach the ``webhook`` subcommand to ``subparsers``."""
    # =========================================================================
    # webhook command
    # =========================================================================
    webhook_parser = subparsers.add_parser(
        "webhook",
        help="Manage dynamic webhook subscriptions",
        description="Create, list, and remove webhook subscriptions for event-driven agent activation",
    )
    webhook_subparsers = webhook_parser.add_subparsers(dest="webhook_action")

    wh_sub = webhook_subparsers.add_parser(
        "subscribe", aliases=["add"], help="Create a webhook subscription"
    )
    wh_sub.add_argument("name", help="Route name (used in URL: /webhooks/<name>)")
    wh_sub.add_argument(
        "--prompt", default="", help="Prompt template with {dot.notation} payload refs"
    )
    wh_sub.add_argument(
        "--events", default="", help="Comma-separated event types to accept"
    )
    wh_sub.add_argument(
        "--actions",
        default="",
        help="Comma-separated action whitelist within the event, e.g. "
        "opened,synchronize,reopened for pull_request. Omit to accept any "
        "action (including closed/labeled/review re-fires).",
    )
    wh_sub.add_argument(
        "--ignore-senders",
        default="",
        help="Comma-separated GitHub logins to ignore (case-insensitive). "
        "Set this to the account used for --deliver github_comment on this "
        "route so the agent doesn't reply to its own comments and loop.",
    )
    wh_sub.add_argument(
        "--mark-own-comments",
        action="store_true",
        help="Alternative to --ignore-senders for when the bot posts as the "
        "same GitHub account a human also comments from. Appends an "
        "invisible marker to every comment this route posts, and ignores "
        "incoming comments/reviews carrying that marker, so the agent still "
        "responds to your comments but not its own. Makes bot comments "
        "identifiable via the raw markdown source (not the rendered view).",
    )
    wh_sub.add_argument("--description", default="", help="What this subscription does")
    wh_sub.add_argument(
        "--skills", default="", help="Comma-separated skill names to load"
    )
    wh_sub.add_argument(
        "--deliver",
        default="log",
        help="Delivery target: log, telegram, discord, slack, etc.",
    )
    wh_sub.add_argument(
        "--deliver-chat-id",
        default="",
        help="Target chat ID for cross-platform delivery",
    )
    wh_sub.add_argument(
        "--deliver-repo",
        default="",
        help="GitHub repo (org/name) for deliver: github_comment. Supports "
        "{dot.notation} payload refs, e.g. {repository.full_name}.",
    )
    wh_sub.add_argument(
        "--deliver-pr-number",
        default="",
        help="GitHub PR or issue number for deliver: github_comment. Supports "
        "{dot.notation} payload refs, e.g. {number} or {issue.number}.",
    )
    wh_sub.add_argument(
        "--secret", default="", help="HMAC secret (auto-generated if omitted)"
    )
    wh_sub.add_argument(
        "--deliver-only",
        action="store_true",
        help="Skip the agent — deliver the rendered prompt directly as the "
        "message. Zero LLM cost. Requires --deliver to be a real target "
        "(not 'log').",
    )

    webhook_subparsers.add_parser(
        "list", aliases=["ls"], help="List all dynamic subscriptions"
    )

    wh_rm = webhook_subparsers.add_parser(
        "remove", aliases=["rm"], help="Remove a subscription"
    )
    wh_rm.add_argument("name", help="Subscription name to remove")

    wh_test = webhook_subparsers.add_parser(
        "test", help="Send a test POST to a webhook route"
    )
    wh_test.add_argument("name", help="Subscription name to test")
    wh_test.add_argument(
        "--payload", default="", help="JSON payload to send (default: test payload)"
    )

    webhook_parser.set_defaults(func=cmd_webhook)

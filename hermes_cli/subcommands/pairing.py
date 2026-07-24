"""``hermes pairing`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_pairing_parser(subparsers, *, cmd_pairing: Callable) -> None:
    """Attach the ``pairing`` subcommand to ``subparsers``."""
    pairing_parser = subparsers.add_parser(
        "pairing",
        help="Manage DM pairing codes for user authorization",
        description="Approve or revoke user access via pairing codes",
    )
    pairing_sub = pairing_parser.add_subparsers(dest="pairing_action")

    pairing_sub.add_parser("list", help="Show pending + approved users")

    pairing_approve_parser = pairing_sub.add_parser(
        "approve", help="Approve a pairing code or pending user"
    )
    pairing_approve_parser.add_argument(
        "platform", help="Platform name (telegram, discord, slack, whatsapp)"
    )
    pairing_approve_parser.add_argument(
        "code", help="Pairing code to approve (or user_id with --by-user-id)"
    )
    pairing_approve_parser.add_argument(
        "--by-user-id",
        action="store_true",
        default=False,
        help="Treat the code argument as a user_id instead — useful when the "
        "pairing code DM was not delivered (e.g. WeChat/Weixin). The user_id "
        "is shown in 'hermes pairing list'.",
    )

    pairing_revoke_parser = pairing_sub.add_parser("revoke", help="Revoke user access")
    pairing_revoke_parser.add_argument("platform", help="Platform name")
    pairing_revoke_parser.add_argument("user_id", help="User ID to revoke")

    pairing_sub.add_parser("clear-pending", help="Clear all pending codes")
    pairing_parser.set_defaults(func=cmd_pairing)

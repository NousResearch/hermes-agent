"""``hermes groups`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_groups_parser(subparsers, *, cmd_groups: Callable) -> None:
    """Attach the ``groups`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "groups", help="Choose which messaging chats can control your Group Chats",
        description="Allow, list or revoke the messaging chats that can control your Group Chats "
                    "with /group. The gateway must be running.")
    sub = parser.add_subparsers(dest="groups_action")
    allow = sub.add_parser("allow", help="Allow the chat that showed this code after /group")
    allow.add_argument("code", help="The code from the chat, for example K7Q2-M9XF")
    allow.add_argument("--yes", action="store_true", help="Allow without asking for confirmation")
    sub.add_parser("chats", help="List the chats that can control your Group Chats")
    revoke = sub.add_parser("revoke", help="Stop a chat from controlling your Group Chats")
    revoke.add_argument("chat", help="The chat ID that 'hermes groups chats' shows")
    parser.set_defaults(func=cmd_groups)

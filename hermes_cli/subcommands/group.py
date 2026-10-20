"""``hermes group`` subcommand parser (Bot Mode group chats)."""

from __future__ import annotations

from typing import Callable


def build_group_parser(subparsers, *, cmd_group: Callable) -> None:
    """Attach the ``group`` subcommand to ``subparsers``."""
    group_parser = subparsers.add_parser(
        "group",
        help="Manage Bot Mode group chats (the desktop Bots tab rooms)",
    )
    group_sub = group_parser.add_subparsers(dest="group_action")

    group_sub.add_parser("bots", help="List available bots (profiles)")
    group_sub.add_parser("list", help="List groups and their members")

    create = group_sub.add_parser("create", help="Create a group chat")
    create.add_argument("name")
    create.add_argument("-b", "--bots", required=True, help="Comma-separated member bots (2-6)")

    info = group_sub.add_parser("info", help="Show one group")
    info.add_argument("name")

    add = group_sub.add_parser("add", help="Add members to a group")
    add.add_argument("name")
    add.add_argument("-b", "--bots", required=True, help="Comma-separated bots to add")

    remove = group_sub.add_parser("remove", help="Remove members from a group")
    remove.add_argument("name")
    remove.add_argument("-b", "--bots", required=True, help="Comma-separated bots to remove")

    rename = group_sub.add_parser("rename", help="Rename a group")
    rename.add_argument("old")
    rename.add_argument("new")

    disband = group_sub.add_parser("disband", help="Disband a group")
    disband.add_argument("name")

    group_parser.set_defaults(func=cmd_group)

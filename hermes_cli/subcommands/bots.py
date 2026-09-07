"""``hermes bots`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_bots_parser(subparsers, *, cmd_bots: Callable) -> None:
    """Attach non-interactive Bot Mode profile management commands."""
    parser = subparsers.add_parser(
        "bots",
        help="Manage Bot Mode profiles used by $Name chains",
    )
    actions = parser.add_subparsers(dest="bots_action")

    actions.add_parser("list", help="List bot profiles")

    create = actions.add_parser("create", help="Create a bot profile")
    create.add_argument("name", help="Bot nickname/profile id")
    create.add_argument("--model", required=True, help="Model id")
    create.add_argument("--provider", required=True, help="Hermes provider id")
    create.add_argument(
        "--system-prompt",
        required=True,
        help="Bot system prompt (stored as the profile SOUL.md)",
    )
    create.add_argument(
        "--disabled",
        action="store_true",
        help="Create the profile disabled for $ chains",
    )

    info = actions.add_parser("info", help="Show one bot profile")
    info.add_argument("name")

    configure = actions.add_parser(
        "configure", help="Update model, provider, system prompt, or status"
    )
    configure.add_argument("name")
    configure.add_argument("--model")
    configure.add_argument("--provider")
    configure.add_argument("--system-prompt")
    status = configure.add_mutually_exclusive_group()
    status.add_argument("--enable", action="store_true")
    status.add_argument("--disable", action="store_true")

    for action in ("enable", "disable"):
        status_parser = actions.add_parser(action, help=f"{action.title()} a bot")
        status_parser.add_argument("name")

    rename = actions.add_parser("rename", help="Rename a bot profile")
    rename.add_argument("old_name")
    rename.add_argument("new_name")

    remove = actions.add_parser("remove", help="Remove a bot profile")
    remove.add_argument("name")
    remove.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm removal (required; this command never prompts)",
    )

    parser.set_defaults(func=cmd_bots)

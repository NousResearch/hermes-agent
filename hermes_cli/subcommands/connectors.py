"""``hermes connectors`` subcommand parser."""

from __future__ import annotations


def build_connectors_parser(subparsers) -> None:
    """Attach the ``connectors`` subcommand to ``subparsers``."""
    from hermes_cli.connectors_cmd import cmd_connectors

    parser = subparsers.add_parser(
        "connectors", help="Manage hosted app connectors and their accounts",
        description="List, connect, rename and remove the app accounts Hermes uses through Nous Portal.")
    actions = parser.add_subparsers(dest="connectors_action")

    actions.add_parser("list", aliases=["ls"], help="Show every connector and whether it is connected")

    accounts = actions.add_parser("accounts", help="Show connected accounts")
    accounts.add_argument("app", nargs="?", help="Only show accounts of this app")
    accounts.add_argument("--all", action="store_true", help="Also show retired accounts")

    connect = actions.add_parser("connect", help="Connect an app account in your browser")
    connect.add_argument("app", help="App to connect, for example gmail")
    connect.add_argument("--alias", help="Name for this account, for example work")
    connect.add_argument("--reconnect", action="store_true", help="Repair an account that stopped working")

    rename = actions.add_parser("rename", help="Rename an account")
    rename.add_argument("app", help="App the account belongs to")
    rename.add_argument("name", help="Current name or label of the account")
    rename.add_argument("--to", required=True, help="New name for the account")

    disconnect = actions.add_parser("disconnect", help="Remove an account")
    disconnect.add_argument("app", help="App the account belongs to")
    disconnect.add_argument("--alias", help="Name or label of the account to remove")
    disconnect.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")

    parser.set_defaults(func=cmd_connectors)

"""``hermes connectors`` subcommand parser."""

from __future__ import annotations


def build_connectors_parser(subparsers) -> None:
    """Attach the ``connectors`` subcommand to ``subparsers``."""
    from hermes_cli.connectors_cmd import cmd_connectors

    parser = subparsers.add_parser(
        "connectors", help="Manage hosted app connectors and their accounts",
        description="List, connect, rename and remove the app accounts Hermes uses through Nous Portal, "
                    "or call one of their tools from a script.")
    actions = parser.add_subparsers(dest="connectors_action")

    actions.add_parser("list", aliases=["ls"], help="Show every connector and whether it is connected")

    accounts = actions.add_parser("accounts", help="Show connected accounts")
    accounts.add_argument("app", nargs="?", help="Only show accounts of this app")
    accounts.add_argument("--all", action="store_true", help="Also show retired accounts")

    connect = actions.add_parser("connect", help="Connect an app account in your browser")
    connect.add_argument("app", help="App to connect, for example gmail")
    connect.add_argument(
        "--alias", help="Name for a new account (for example work); with --reconnect, the name or label to repair")
    connect.add_argument("--reconnect", action="store_true", help="Repair an account that stopped working")
    connect.add_argument(
        "--force", action="store_true", help="With --reconnect, sign in again even if the account is connected")

    rename = actions.add_parser("rename", help="Rename an account")
    rename.add_argument("app", help="App the account belongs to")
    rename.add_argument("name", help="Current name or label of the account")
    rename.add_argument("--to", required=True, help="New name for the account")

    disconnect = actions.add_parser("disconnect", help="Remove an account")
    disconnect.add_argument("app", help="App the account belongs to")
    disconnect.add_argument("--alias", help="Name or label of the account to remove")
    disconnect.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")

    call = actions.add_parser(
        "call", help="Run one connector tool and print its JSON result (for scripts)",
        description=(
            "Run one tool of a connected app and print its response as one JSON document on stdout. "
            "Read-only by default: a tool the Nous Portal does not classify as read-only (its 'read' facet) "
            "is refused unless --allow-write is given. Exit codes: 0 ok, 2 bad arguments or unknown tool, "
            "3 connectors unavailable or signed out, 4 the tool returned an error (error JSON on stderr), "
            "5 refused: the tool may change data and needs --allow-write."))
    call.add_argument("tool", help="Tool to run: <app>.<tool> (for example gmail.fetch_emails) "
                                   "or connectors__<app>__<tool>")
    call.add_argument("--args", help="Tool arguments as a JSON object")
    call.add_argument("--account", help="Name of the account to use when the app has several")
    call.add_argument("--allow-write", action="store_true",
                      help="Allow a tool that may send, change or delete data")

    parser.set_defaults(func=cmd_connectors)

"""Parser setup for ``hermes sessions list``."""

from hermes_cli.config_defaults import SESSION_LIST_SORT_VALUES


def add_sessions_list_parser(sessions_subparsers) -> None:
    """Attach the list action and its display/filter options."""
    sessions_list = sessions_subparsers.add_parser("list", help="List recent sessions")
    sessions_list.add_argument("--source", help="Filter by source (cli, telegram, discord, etc.)")
    sessions_list.add_argument("--limit", type=int, default=20, help="Max sessions to show")
    sessions_list.add_argument("--workspace", metavar="NEEDLE",
        help="Only sessions in one workspace: a git repo root or project dir "
        "(matched by path substring or basename).")
    sessions_list.add_argument(
        "--sort",
        choices=SESSION_LIST_SORT_VALUES,
        default=None,
        help="Sort order: 'started' by session creation time, "
        "'last-active' by most recent message "
        "(default: sessions.list_sort from config.yaml)",
    )

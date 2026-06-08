"""``hermes marker`` subcommand parser.

Provides CLI commands for managing memory locations (markers).
"""

from __future__ import annotations

from typing import Callable


def build_marker_parser(subparsers, *, cmd_marker: Callable) -> None:
    """Attach the ``marker`` subcommand to ``subparsers``."""
    marker_parser = subparsers.add_parser(
        "marker",
        help="Manage memory locations (markers)",
        description="Create, list, navigate, and delete memory locations (markers) in Hermes sessions.",
    )
    marker_subparsers = marker_parser.add_subparsers(dest="marker_action")

    # marker create
    marker_create = marker_subparsers.add_parser(
        "create",
        help="Create a new memory location at the current position",
    )
    marker_create.add_argument(
        "label",
        help="Descriptive label for the memory location"
    )
    marker_create.add_argument(
        "--tags",
        help="Comma-separated list of tags (e.g., bug,frontend)",
    )
    marker_create.add_argument(
        "--persistent",
        action="store_true",
        help="Make this location persistent across sessions",
    )
    marker_create.add_argument(
        "--session",
        help="Session ID to attach to (defaults to current session)",
    )

    # marker list
    marker_list = marker_subparsers.add_parser(
        "list",
        help="List memory locations",
    )
    marker_list.add_argument(
        "--persistent",
        action="store_true",
        help="List only persistent locations",
    )
    marker_list.add_argument(
        "--session",
        help="Filter by session ID",
    )
    marker_list.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # marker goto
    marker_goto = marker_subparsers.add_parser(
        "goto",
        help="Jump to a memory location",
    )
    marker_goto.add_argument(
        "location_id",
        type=int,
        help="ID of the memory location to jump to",
    )

    # marker delete
    marker_delete = marker_subparsers.add_parser(
        "delete",
        help="Delete a memory location",
    )
    marker_delete.add_argument(
        "location_id",
        type=int,
        help="ID of the memory location to delete",
    )

    marker_parser.set_defaults(func=cmd_marker)

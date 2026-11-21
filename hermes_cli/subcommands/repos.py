"""``hermes repos`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag


def build_repos_parser(subparsers, *, cmd_repos: Callable) -> None:
    """Attach the ``repos`` subcommand to ``subparsers``."""
    repos_parser = subparsers.add_parser(
        "repos",
        help="Inventory repo-backed installs (plugins, skills, MCP, clones)",
        description=(
            "One inventory of every repo-backed install under the Hermes home, "
            "with its source URL, pinned revision and on/off state. Read-only: "
            "the lifecycle verbs live on the surfaces that own them."
        ),
    )
    repos_subparsers = repos_parser.add_subparsers(dest="repos_action")

    repos_list = repos_subparsers.add_parser(
        "list", aliases=["ls"], help="List repo-backed installs"
    )
    repos_list.add_argument(
        "--surface",
        choices=["plugin", "skill", "mcp", "clone"],
        help="Only one surface (default: all of them)",
    )
    repos_list.add_argument(
        "--enabled", action="store_true", help="Only on/installed rows"
    )
    repos_list.add_argument(
        "--disabled", action="store_true", help="Only off/uninstalled rows"
    )
    repos_list.add_argument(
        "--plain", action="store_true", help="Compact plain-text output"
    )
    add_json_flag(repos_list, "Print machine-readable JSON")

    repos_show = repos_subparsers.add_parser(
        "show", aliases=["info"], help="Show one install by name or id"
    )
    repos_show.add_argument("name", help="Install name or id (e.g. plugin:holographic)")

    repos_subparsers.add_parser(
        "sync", help="Rebuild the registry cache from disk (no network)"
    )

    repos_subparsers.add_parser(
        "audit", help="Report drift between the registry cache and disk"
    )

    repos_parser.set_defaults(func=cmd_repos)

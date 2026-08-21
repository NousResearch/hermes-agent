"""``hermes memory`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_memory_parser(subparsers, *, cmd_memory: Callable) -> None:
    """Attach the ``memory`` subcommand to ``subparsers``."""
    memory_parser = subparsers.add_parser(
        "memory",
        help="Configure external memory provider",
        description=(
            "Set up and manage external memory provider plugins.\n\n"
            "Available providers: honcho, openviking, mem0, hindsight,\n"
            "holographic, retaindb, byterover.\n\n"
            "Only one external provider can be active at a time.\n"
            "Built-in memory (MEMORY.md/USER.md) is always active."
        ),
    )
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    _setup_parser = memory_sub.add_parser(
        "setup", help="Interactive provider selection and configuration"
    )
    _setup_parser.add_argument(
        "provider",
        nargs="?",
        default=None,
        help="Provider to configure directly (e.g. honcho), skipping the picker",
    )
    memory_sub.add_parser("status", help="Show current memory provider config")
    _review_parser = memory_sub.add_parser(
        "review",
        help="List confidence-memory items for review when the active provider supports it",
    )
    _review_parser.add_argument("--include-inactive", action="store_true")
    _review_parser.add_argument("--limit", type=int, default=50)
    _search_parser = memory_sub.add_parser(
        "search",
        help="Search confidence-memory items when the active provider supports it",
    )
    _search_parser.add_argument("query")
    _search_parser.add_argument("--include-inactive", action="store_true")
    _search_parser.add_argument("--limit", type=int, default=10)
    _confirm_parser = memory_sub.add_parser(
        "confirm",
        help="Confirm a confidence-memory item by id",
    )
    _confirm_parser.add_argument("id")
    _confirm_parser.add_argument("--source-excerpt", default="user confirmed via CLI")
    _delete_parser = memory_sub.add_parser(
        "delete",
        help="Delete a confidence-memory item by id",
    )
    _delete_parser.add_argument("id")
    memory_sub.add_parser("off", help="Disable external provider (built-in only)")
    _reset_parser = memory_sub.add_parser(
        "reset",
        help="Erase all built-in memory (MEMORY.md and USER.md)",
    )
    _reset_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    _reset_parser.add_argument(
        "--target",
        choices=["all", "memory", "user"],
        default="all",
        help="Which store to reset: 'all' (default), 'memory', or 'user'",
    )
    memory_parser.set_defaults(func=cmd_memory)

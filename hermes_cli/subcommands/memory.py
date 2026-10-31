"""``hermes memory`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.memory_cmd import memory_command as memory_review_command
from hermes_cli.subcommands._shared import add_yes_flag


def build_memory_parser(subparsers, *, cmd_memory: Callable) -> None:
    """Attach the ``memory`` subcommand to ``subparsers``."""
    memory_parser = subparsers.add_parser(
        "memory", help="Configure external memory provider",
        description="Set up and manage external memory provider plugins.\n\n"
            "Bundled providers: retaindb, byterover.\n"
            "Catalog providers (e.g. honcho, hindsight, supermemory,\n"
            "mem0, openviking, holographic): hermes plugins install <name>.\n\n"
            "Only one external provider can be active at a time.\n"
            "Built-in memory (MEMORY.md/USER.md) is always active.")
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    _setup_parser = memory_sub.add_parser(
        "setup", help="Interactive provider selection and configuration")
    _setup_parser.add_argument(
        "provider", nargs="?", default=None,
        help="Provider to configure directly (e.g. honcho), skipping the picker")
    memory_sub.add_parser("status", help="Show current memory provider config")
    memory_sub.add_parser("off", help="Disable external provider (built-in only)")
    _reset_parser = memory_sub.add_parser(
        "reset", help="Erase all built-in memory (MEMORY.md and USER.md)")
    add_yes_flag(_reset_parser)
    _reset_parser.add_argument(
        "--target", choices=["all", "memory", "user"], default="all",
        help="Which store to reset: 'all' (default), 'memory', or 'user'")

    _list_parser = memory_sub.add_parser(
        "list", help="List entries in MEMORY.md and USER.md")
    _list_parser.set_defaults(func=memory_review_command)

    _show_parser = memory_sub.add_parser(
        "show", help="Show all entries in one memory store")
    _show_parser.add_argument("target", choices=["memory", "user"])
    _show_parser.set_defaults(func=memory_review_command)

    _delete_parser = memory_sub.add_parser(
        "delete", help="Delete an entry matching a substring")
    _delete_parser.add_argument("target", choices=["memory", "user"])
    _delete_parser.add_argument("substring")
    add_yes_flag(_delete_parser)
    _delete_parser.set_defaults(func=memory_review_command)

    memory_parser.set_defaults(func=cmd_memory)

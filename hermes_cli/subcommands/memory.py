"""``hermes memory`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable


_LEGACY_RESET_TARGETS = frozenset({"all", "memory", "user"})


def _dispatch_memory_reset(args, *, legacy_handler: Callable):
    """Route existing targets to the legacy handler and conversation reset."""
    target = getattr(args, "target", "all")
    if target in _LEGACY_RESET_TARGETS:
        return legacy_handler(args)

    # Lazy import keeps the ordinary CLI startup path lightweight.
    from hermes_cli.memory_reset import cmd_memory_reset

    return cmd_memory_reset(args)


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
    memory_parser.set_defaults(func=cmd_memory)
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
    memory_sub.add_parser("off", help="Disable external provider (built-in only)")
    _reset_parser = memory_sub.add_parser(
        "reset",
        help="Erase built-in memory or persisted conversation history",
    )
    _reset_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    _reset_parser.add_argument(
        "--target",
        choices=["all", "memory", "user", "conversations"],
        default="all",
        help=(
            "Which store to reset: 'all' (MEMORY.md + USER.md, default), "
            "'memory', 'user', or 'conversations'"
        ),
    )
    _reset_parser.set_defaults(
        func=partial(_dispatch_memory_reset, legacy_handler=cmd_memory)
    )

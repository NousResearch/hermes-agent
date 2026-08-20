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
    memory_sub.add_parser("off", help="Disable external provider (built-in only)")
    _audit_parser = memory_sub.add_parser(
        "audit",
        help="Audit built-in MEMORY.md/USER.md quality and usage",
    )
    _audit_parser.add_argument(
        "--target",
        choices=["all", "memory", "user"],
        default="all",
        help="Which built-in store to show: all (default), memory, or user",
    )
    _audit_parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Refresh memories/metadata.json sidecar without changing prompt memory",
    )
    _audit_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report",
    )
    memory_sub.add_parser("pending", help="List staged built-in memory writes")
    _approve_parser = memory_sub.add_parser(
        "approve",
        help="Approve a staged built-in memory write by id, or 'all'",
    )
    _approve_parser.add_argument("pending_id", help="Pending write id, or 'all'")
    _reject_parser = memory_sub.add_parser(
        "reject",
        aliases=["deny", "drop"],
        help="Reject a staged built-in memory write by id, or 'all'",
    )
    _reject_parser.add_argument("pending_id", help="Pending write id, or 'all'")
    _approval_parser = memory_sub.add_parser(
        "approval",
        aliases=["mode"],
        help="Show or set memory.write_approval",
    )
    _approval_parser.add_argument(
        "mode",
        nargs="?",
        choices=["on", "off", "true", "false", "yes", "no", "1", "0"],
        help="Turn write approval on or off",
    )
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

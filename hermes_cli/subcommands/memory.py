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
    service_parser = memory_sub.add_parser(
        "service",
        help="Manage an external memory provider service",
    )
    service_sub = service_parser.add_subparsers(
        dest="service_command",
        required=True,
    )
    for action, help_text in (
        ("install", "Install and start the external provider service"),
        ("status", "Show external provider service status"),
        ("restart", "Restart the external provider service"),
        ("logs", "Show external provider service log paths"),
    ):
        action_parser = service_sub.add_parser(action, help=help_text)
        action_parser.add_argument(
            "provider",
            choices=["hindsight"],
            help="Provider service to manage",
        )
        action_parser.add_argument(
            "--executable",
            help="Path to the provider daemon executable (defaults to discovered hindsight-api)",
        )
        action_parser.add_argument(
            "--env-file",
            help="Optional daemon environment file to source before launch",
        )
        if action == "install":
            action_parser.add_argument(
                "--force",
                action="store_true",
                help="Replace an existing launchd registration before installing",
            )
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

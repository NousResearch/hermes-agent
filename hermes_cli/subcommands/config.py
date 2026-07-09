"""``hermes config`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_config_parser(subparsers, *, cmd_config: Callable) -> None:
    """Attach the ``config`` subcommand to ``subparsers``."""
    # =========================================================================
    # config command
    # =========================================================================
    config_parser = subparsers.add_parser(
        "config",
        help="View and edit configuration",
        description="Manage Hermes Agent configuration",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    # config show (default)
    config_subparsers.add_parser("show", help="Show current configuration")

    # config edit
    config_subparsers.add_parser("edit", help="Open config file in editor")

    # config set
    config_set = config_subparsers.add_parser("set", help="Set a configuration value")
    config_set.add_argument(
        "key", nargs="?", help="Configuration key (e.g., model, terminal.backend)"
    )
    config_set.add_argument("value", nargs="?", help="Value to set")

    # config path
    config_subparsers.add_parser("path", help="Print config file path")

    # config env-path
    config_subparsers.add_parser("env-path", help="Print .env file path")

    # config check
    config_check = config_subparsers.add_parser(
        "check", help="Check for missing/outdated config"
    )
    config_check.add_argument(
        "--all-profiles",
        action="store_true",
        help="Include every named profile under HERMES_HOME/profiles",
    )
    config_check.add_argument(
        "--kanban-workers",
        action="store_true",
        help="Validate Kanban worker profile execution budgets",
    )

    # config repair
    config_repair = config_subparsers.add_parser(
        "repair", help="Repair bounded configuration issues"
    )
    repair_subparsers = config_repair.add_subparsers(dest="repair_command")
    worker_budgets = repair_subparsers.add_parser(
        "worker-budgets",
        help="Repair zero execution budgets in Kanban worker profiles",
    )
    worker_budgets.add_argument(
        "--all-profiles",
        action="store_true",
        help="Repair every named profile under HERMES_HOME/profiles",
    )
    worker_budgets.add_argument("--profile", help="Repair one named profile")
    worker_budgets.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing config files",
    )
    worker_budgets.add_argument(
        "--set",
        dest="set_value",
        type=int,
        default=120,
        help="Positive budget value to write for zero budgets",
    )

    # config migrate
    config_subparsers.add_parser("migrate", help="Update config with new options")

    config_parser.set_defaults(func=cmd_config)

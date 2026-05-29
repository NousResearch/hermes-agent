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

    # config get
    config_get = config_subparsers.add_parser(
        "get", help="Print a resolved configuration value"
    )
    config_get.add_argument("key", nargs="?", help="Configuration key (e.g., model)")
    config_get.add_argument("--json", action="store_true", help="Print value as JSON")

    # config set
    config_set = config_subparsers.add_parser("set", help="Set a configuration value")
    config_set.add_argument(
        "key", nargs="?", help="Configuration key (e.g., model, terminal.backend)"
    )
    config_set.add_argument("value", nargs="?", help="Value to set")
    config_set.add_argument(
        "--force",
        action="store_true",
        help="Bypass schema validation (write unknown keys without warning). "
        "Use this when setting a key that a newer Hermes version supports "
        "but the running version doesn't recognize yet.",
    )

    # config unset
    config_unset = config_subparsers.add_parser(
        "unset", help="Remove a configuration value"
    )
    config_unset.add_argument("key", nargs="?", help="Configuration key to remove")

    # config path
    config_subparsers.add_parser("path", help="Print config file path")

    # config env-path
    config_subparsers.add_parser("env-path", help="Print .env file path")

    # config check
    config_subparsers.add_parser("check", help="Check for missing/outdated config")

    # config migrate
    config_subparsers.add_parser("migrate", help="Update config with new options")

    config_parser.set_defaults(func=cmd_config)

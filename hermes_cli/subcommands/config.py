"""``hermes config`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag


def build_config_parser(subparsers, *, cmd_config: Callable) -> None:
    """Attach the ``config`` subcommand to ``subparsers``."""
    config_parser = subparsers.add_parser(
        "config", help="View and edit configuration",
        description="Manage Hermes Agent configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    config_subparsers.add_parser("show", help="Show current configuration")
    config_subparsers.add_parser("edit", help="Open config file in editor")

    from hermes_cli.config_inventory import config_keys_command

    config_keys = config_subparsers.add_parser(
        "keys", help="List registered configuration paths (--values: resolved key=value)",
        description="List registered roots and defaulted nested paths, without reading user "
        "configuration. Open mappings allow additional user-defined keys; this is not an "
        "exhaustive runtime schema. --values lists every resolved leaf as key=value instead.")
    config_keys.add_argument(
        "--values", action="store_true",
        help="List every resolved leaf (defaults + config.yaml) as key=value; credentials masked")
    add_json_flag(config_keys, "Print paths as a JSON array (with --values: a key -> value object)")
    config_keys.set_defaults(func=config_keys_command)

    config_get = config_subparsers.add_parser("get", help="Print a resolved configuration value")
    config_get.add_argument("key", nargs="?", help="Configuration key (e.g., model)")
    add_json_flag(config_get, "Print value as JSON")
    config_get.add_argument(
        "--raw", action="store_true",
        help="Print credential values unmasked (default masks api_key/token/secret-shaped values)")

    config_set = config_subparsers.add_parser("set", help="Set a configuration value")
    config_set.add_argument(
        "key", nargs="?", help="Configuration key (e.g., model, terminal.backend)")
    config_set.add_argument("value", nargs="?", help="Value to set")
    config_set.add_argument(
        "--force", action="store_true",
        help="Write a key the running version doesn't recognize: an unknown path under a known "
        "section is otherwise refused, and an unknown top-level key is written with a notice.")

    config_unset = config_subparsers.add_parser("unset", help="Remove a configuration value")
    config_unset.add_argument("key", nargs="?", help="Configuration key to remove")

    config_subparsers.add_parser("path", help="Print config file path")
    config_subparsers.add_parser("env-path", help="Print .env file path")
    config_subparsers.add_parser("check", help="Check for missing/outdated config")
    config_subparsers.add_parser("migrate", help="Update config with new options")

    config_parser.set_defaults(func=cmd_config)

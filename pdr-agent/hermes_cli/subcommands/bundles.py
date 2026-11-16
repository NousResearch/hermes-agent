"""``hermes bundles`` subcommand parser."""

from __future__ import annotations


def build_bundles_parser(subparsers) -> None:
    """Attach the ``bundles`` subcommand to ``subparsers``."""
    bundles_parser = subparsers.add_parser(
        "bundles", help="Create, list, and manage skill bundles (aliases for multiple skills)",
        description="Skill bundles let you load several skills under one slash "
            "command. `/<bundle>` from the CLI or gateway loads every "
            "referenced skill at once.")
    from hermes_cli.bundles import register_cli as _bundles_register, bundles_command
    _bundles_register(bundles_parser)
    bundles_parser.set_defaults(func=bundles_command)

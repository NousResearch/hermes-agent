"""``hermes migrate`` subcommand parser."""

from __future__ import annotations


def build_migrate_parser(subparsers) -> None:
    """Attach the ``migrate`` subcommand to ``subparsers``."""
    from hermes_cli.migrate import cmd_migrate, cmd_migrate_xai

    migrate_parser = subparsers.add_parser(
        "migrate", help="Migrate configuration for retired models or deprecated settings",
        description="Diagnose and (optionally) rewrite the active config.yaml to "
            "replace references to retired models or deprecated settings.")
    migrate_subparsers = migrate_parser.add_subparsers(dest="migrate_type")

    migrate_xai = migrate_subparsers.add_parser(
        "xai", help="Migrate xAI models scheduled for retirement on May 15, 2026",
        description="Scan config.yaml for references to xAI models retiring on "
            "May 15, 2026 and, with --apply, rewrite them in-place to the "
            "official replacements per the xAI migration guide. The original "
            "config.yaml is backed up before any rewrite.")
    migrate_xai.add_argument(
        "--apply", action="store_true",
        help="Rewrite config.yaml in-place (default: dry-run, no writes)")
    migrate_xai.add_argument(
        "--no-backup", action="store_true",
        help="Skip the timestamped backup of config.yaml when applying")
    migrate_xai.set_defaults(func=cmd_migrate_xai)
    migrate_parser.set_defaults(func=cmd_migrate)

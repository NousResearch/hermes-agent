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

    migrate_relay = migrate_subparsers.add_parser(
        "relay", help="Convert legacy HERMES_NEMO_RELAY_ATIF_*/ATOF_* exporter vars into relay-plugins.toml",
        description="The NeMo Relay cutover stopped reading the legacy exporter variables; a .env that still "
            "carries them (and no HERMES_NEMO_RELAY_PLUGINS_TOML) exports nothing. Generate "
            "<hermes home>/relay-plugins.toml from them, point HERMES_NEMO_RELAY_PLUGINS_TOML at it, "
            "and comment the legacy lines out. `hermes update` runs this for every profile automatically.")
    migrate_relay.add_argument(
        "--all-profiles", action="store_true",
        help="Migrate the default home and every named profile (what `hermes update` does)")
    migrate_relay.add_argument(
        "--no-validate", action="store_true",
        help="Skip activating the generated file through Relay's validator before writing it")
    migrate_relay.set_defaults(func=_cmd_migrate_relay)
    migrate_parser.set_defaults(func=cmd_migrate)


def _cmd_migrate_relay(args) -> None:
    from hermes_cli.relay_plugin_migrate import cmd_migrate_relay
    cmd_migrate_relay(args)

"""``hermes migrate`` subcommand parser."""

from __future__ import annotations


def build_migrate_parser(subparsers) -> None:
    """Attach the ``migrate`` subcommand to ``subparsers``."""
    from hermes_cli.migrate import cmd_migrate, cmd_migrate_xai

    migrate_parser = subparsers.add_parser(
        "migrate", help="Move over from another agent, or migrate retired models and deprecated settings",
        description="With no type: find another agent's setup on this computer (OpenClaw, Claude Code, "
            "Codex CLI, Cursor, Grok Bot, OpenBot) and offer to import it. Name an agent to import from it; "
            "xai and relay rewrite deprecated settings in the active config.yaml.")
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
        description="The NeMo Relay cutover stopped reading the legacy exporter variables. Generate "
            "<hermes home>/relay-plugins.toml from them, point HERMES_NEMO_RELAY_PLUGINS_TOML at it, "
            "and comment the legacy lines out. `hermes update` runs this for every profile automatically.")
    migrate_relay.add_argument(
        "--all-profiles", action="store_true",
        help="Migrate the default home and every named profile (what `hermes update` does)")
    migrate_relay.add_argument(
        "--no-validate", action="store_true",
        help="Skip activating the generated file through Relay's validator before writing it")
    migrate_relay.set_defaults(func=_cmd_migrate_relay)
    _add_agent_import_parsers(migrate_subparsers)
    migrate_parser.set_defaults(func=cmd_migrate)


def _add_agent_import_parsers(migrate_subparsers) -> None:
    """``hermes migrate <agent>``: one per importer setup offers; runs the agent-importers plugin's
    ``hermes import-agent <agent>`` (installing the plugin first when it is missing)."""
    from hermes_cli.migrate import cmd_migrate_agent
    from hermes_cli.setup_agent_import import IMPORT_SOURCES

    for source in IMPORT_SOURCES:
        sub = migrate_subparsers.add_parser(
            source.name, help=f"Import your {source.label} setup into Hermes",
            description=f"Import your {source.label} setup into the active profile through the "
                "agent-importers plugin (installed from the plugin catalog when missing). Always "
                "previews first and asks before changing anything.")
        sub.add_argument("--source", help="Where the agent's setup (or, for Grok Bot, its export file) is")
        sub.add_argument("--dry-run", action="store_true", help="Preview only")
        sub.add_argument("--overwrite", action="store_true", help="Replace existing items on name conflicts")
        sub.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt")
        if source.name == "openclaw":
            sub.add_argument("--migrate-secrets", action="store_true",
                             help="Also import OpenClaw's allowlisted API keys and bot tokens")
        if source.name in ("grok-bot", "openbot"):
            sub.add_argument("--activate-routines", action="store_true",
                             help="Create the imported cron jobs running instead of paused")
            sub.add_argument("--bot", action="append", metavar="NAME",
                             help="Only this Bot or agent, by name (repeatable)")
        sub.set_defaults(func=cmd_migrate_agent, migrate_agent=source.name)


def _cmd_migrate_relay(args) -> None:
    from hermes_cli.relay_plugin_migrate import cmd_migrate_relay
    cmd_migrate_relay(args)

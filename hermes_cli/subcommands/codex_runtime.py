"""``hermes codex-runtime`` — noninteractive counterpart of the ``/codex-runtime`` slash command.

``hermes codex-runtime migrate [--dry-run] [--json]`` runs the same ``~/.codex/config.toml``
migration the slash command triggers when the codex app-server runtime is enabled, on the
selected profile home (``hermes -p NAME codex-runtime migrate``), so automation no longer has to
import ``hermes_cli.codex_runtime_plugin_migration`` directly (issue #79023).
"""

from __future__ import annotations

import argparse
import dataclasses
import json


def cmd_codex_runtime_migrate(args: argparse.Namespace) -> int:
    """Project Hermes MCP servers (+ codex plugins) into ~/.codex/config.toml; 1 on any error."""
    from hermes_cli.codex_runtime_plugin_migration import migrate
    from hermes_cli.config import load_config

    report = migrate(load_config(), dry_run=args.dry_run)
    if args.json:
        payload = dataclasses.asdict(report)
        payload["target_path"] = str(report.target_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(report.summary())
    return 1 if report.errors else 0


def build_codex_runtime_parser(subparsers) -> None:
    """Attach the ``codex-runtime`` subcommand (``migrate`` action) to ``subparsers``."""
    parser = subparsers.add_parser(
        "codex-runtime", help="Manage the optional codex app-server runtime (migrate MCP config)",
        description="Noninteractive counterpart of the /codex-runtime slash command. Toggling the "
            "runtime itself stays in the chat command (`/codex-runtime on|off`); `migrate` "
            "re-projects Hermes' mcp_servers + installed codex plugins into the managed block of "
            "~/.codex/config.toml for the selected profile.")
    actions = parser.add_subparsers(dest="codex_runtime_action")
    migrate_parser = actions.add_parser(
        "migrate", help="Regenerate the hermes-managed block in codex's config.toml",
        description="Idempotent: replaces the managed block, keeps user text verbatim, skips Hermes "
            "servers whose name the user already declares outside the block, validates the result "
            "as TOML before writing atomically.")
    migrate_parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be written without touching config.toml")
    migrate_parser.add_argument(
        "--json", action="store_true", help="Print the migration report as JSON (for automation)")
    migrate_parser.set_defaults(func=cmd_codex_runtime_migrate)

    def _print_help(args):  # noqa: ANN001 — bare `hermes codex-runtime` lists the actions
        parser.print_help()
        return 0

    parser.set_defaults(func=_print_help)

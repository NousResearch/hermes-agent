"""Explicit identity migration for the existing built-in memory files."""
from __future__ import annotations

import json

from hermes_cli.subcommands._shared import add_yes_flag


def add_identity_migration_parser(subparsers):
    parser = subparsers.add_parser("migrate-identities", help="Preview or migrate built-in memory to stable UUIDs")
    parser.add_argument("--target", choices=["all", "memory", "user"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changing memory files")
    add_yes_flag(parser)
    parser.set_defaults(func=cmd_memory_identity_migration)


def cmd_memory_identity_migration(args):
    from tools.memory_identity_store import migrate_target
    from tools.memory_tool import load_on_disk_store
    store = load_on_disk_store()
    commit = bool(args.yes) and not bool(args.dry_run)
    targets = ("memory", "user") if args.target == "all" else (args.target,)
    outcomes = [migrate_target(store, target, commit=commit) for target in targets]
    for result in outcomes:
        print(json.dumps(result, ensure_ascii=False))
    if not commit:
        print("Preview only. Run again with --yes to create recoverable backups and apply the migration.")
    return 0 if all(result.get("success") for result in outcomes) else 1

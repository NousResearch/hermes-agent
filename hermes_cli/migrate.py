"""CLI handlers for ``hermes migrate ...``.

Exposes configuration retirement migrations and the safety-first
``state-postgres`` migration.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

from hermes_cli.colors import Colors, color
from hermes_cli.config import load_config


def cmd_migrate(args: Any) -> int:
    """Dispatcher for ``hermes migrate <subtype>``."""
    sub = getattr(args, "migrate_type", None)
    if sub == "xai":
        return cmd_migrate_xai(args)
    if sub == "state-postgres":
        return cmd_migrate_state_postgres(args)

    print(
        "usage: hermes migrate {xai|state-postgres}",
        file=sys.stderr,
    )
    return 2


def cmd_migrate_xai(args: Any) -> int:
    """Run xAI May-15 model migration in dry-run or apply mode."""
    from hermes_cli.xai_retirement import (
        MIGRATION_GUIDE_URL,
        RETIREMENT_DATE,
        apply_migration,
        find_retired_xai_refs,
        format_issue,
    )

    apply = bool(getattr(args, "apply", False))
    no_backup = bool(getattr(args, "no_backup", False))

    config = load_config()
    issues = find_retired_xai_refs(config)

    print()
    print(color(
        f"◆ xAI Model Retirement Migration ({RETIREMENT_DATE})",
        Colors.CYAN, Colors.BOLD,
    ))
    print()

    if not issues:
        print(f"  {color('✓', Colors.GREEN)} No retired xAI models in config — nothing to migrate.")
        return 0

    print(f"  Found {len(issues)} retired xAI model reference(s):")
    print()
    for issue in issues:
        print(f"    {color('⚠', Colors.YELLOW)} {format_issue(issue)}")
    print()
    print(f"    {color('→', Colors.CYAN)} Migration guide: {MIGRATION_GUIDE_URL}")
    print()

    config_path = _resolve_config_path()

    if not apply:
        print(color("Dry-run mode — no changes written.", Colors.DIM))
        print(color(
            "Re-run with `hermes migrate xai --apply` to rewrite "
            f"{config_path} in-place (backup created automatically).",
            Colors.DIM,
        ))
        return 0

    if not config_path or not config_path.exists():
        print(
            f"  {color('✗', Colors.RED)} Could not locate config.yaml "
            f"(looked at: {config_path})",
            file=sys.stderr,
        )
        return 1

    try:
        result = apply_migration(
            config_path=config_path,
            issues=issues,
            backup=not no_backup,
        )
    except Exception as exc:
        print(
            f"  {color('✗', Colors.RED)} Migration failed: {exc}",
            file=sys.stderr,
        )
        return 1

    if not result.config_changed:
        print(f"  {color('⚠', Colors.YELLOW)} No changes written.")
        return 0

    if result.backup_path is not None:
        print(f"  {color('✓', Colors.GREEN)} Backup: {result.backup_path}")
    print(
        f"  {color('✓', Colors.GREEN)} Updated {len(result.issues_resolved)} "
        f"slot(s) in {result.file_path}"
    )
    print()
    print(color(
        "Run `hermes doctor` to confirm no retired xAI models remain.",
        Colors.DIM,
    ))
    return 0


def _resolve_config_path() -> Path:
    """Best-effort: locate the active config.yaml on disk."""
    from hermes_cli.config import get_hermes_home

    return get_hermes_home() / "config.yaml"


def cmd_migrate_state_postgres(args: Any) -> int:
    """Run the adapter-driven SQLite-to-PostgreSQL state migration.

    PostgreSQL credentials stay in the environment.  The SQLite adapter fails
    closed until the runtime installs an enforceable SessionDB writer fence.
    """

    from hermes_cli.config import (
        atomic_switch_state_to_postgres,
        get_hermes_home,
        load_config,
    )
    from hermes_cli.state_postgres_migration import (
        MigrationRequest,
        StatePostgresMigration,
    )
    from state_store.postgres.migration_adapter import PostgresMigrationTargetAdapter
    from state_store.sqlite.migration_adapter import SQLiteMigrationSourceAdapter

    apply = bool(getattr(args, "apply", False))
    dsn_env = str(getattr(args, "dsn_env", "HERMES_STATE_POSTGRES_DSN"))
    schema = str(getattr(args, "schema", ""))
    batch_size = int(getattr(args, "batch_size", 1_000))
    run_id = getattr(args, "run_id", None)
    home = get_hermes_home()
    try:
        sqlite_path = _configured_sqlite_state_path(home, load_config())
        source = SQLiteMigrationSourceAdapter(sqlite_path)
        target = PostgresMigrationTargetAdapter(
            dsn_env=dsn_env,
            schema=schema,
            home=home,
        )

        def cutover(_report: Any) -> None:
            atomic_switch_state_to_postgres(dsn_env=dsn_env, schema=schema)

        report = StatePostgresMigration(
            source,
            target,
            cutover=cutover if apply else None,
        ).run(
            MigrationRequest(
                apply=apply,
                run_id=run_id,
                batch_size=batch_size,
            )
        )
    except Exception as exc:
        # Inputs contain only paths, identifiers, and environment-variable
        # names.  Do not render driver exception chains that could include a DSN.
        print(f"state-postgres migration could not start: {type(exc).__name__}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0 if report.succeeded else 1


def _configured_sqlite_state_path(home: Path, config: Any) -> Path:
    """Resolve only the legacy SQLite path without reading a PostgreSQL DSN."""

    state: Any = {}
    if isinstance(config, dict):
        sessions = config.get("sessions")
        if isinstance(sessions, dict):
            candidate = sessions.get("state")
            if isinstance(candidate, dict):
                state = candidate
    configured = state.get("sqlite_path", "state.db")
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("sessions.state.sqlite_path must be a non-empty string")
    path = Path(configured).expanduser()
    return path if path.is_absolute() else home / path

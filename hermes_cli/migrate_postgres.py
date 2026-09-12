"""CLI confirmation and reporting for SQLite-to-PostgreSQL state migration."""
from __future__ import annotations

import sys
from typing import Any

from hermes_cli.colors import Colors, color
from hermes_cli.config import load_config


def cmd_migrate_state_to_postgres(args: Any) -> int:
    """Copy the SQLite state database into a PostgreSQL backend.

    Wraps :func:`migrate_state_to_postgres.migrate` — the migration logic
    lives entirely in that standalone module; this handler only handles
    argument resolution, user confirmation, and result reporting.
    """
    import os

    # Lazy import keeps the postgres extra optional for unrelated migrate
    # subcommands (e.g. xai).  Import the whole module so mocks can target
    # the module-level names via ``migrate_state_to_postgres.<name>``.
    try:
        import migrate_state_to_postgres as _m2pg
    except ImportError as exc:
        print(
            f"  {color('✗', Colors.RED)} Could not import migration module: {exc}",
            file=sys.stderr,
        )
        return 1

    # --- Resolve SQLite source path ---
    sqlite_path = _m2pg._resolve_sqlite_path(getattr(args, "sqlite_path", None))

    # --- Resolve PostgreSQL DSN ---
    explicit_dsn: str | None = getattr(args, "dsn", None)
    if explicit_dsn:
        dsn: str | None = explicit_dsn
    else:
        # Check env vars first (same order as the standalone script).
        dsn = None
        for key in ("HERMES_STATE_DATABASE_URL", "HERMES_STATE_POSTGRES_DSN"):
            val = (os.environ.get(key) or "").strip()
            if val:
                dsn = val
                break

        if not dsn:
            # Fall back to the config-based resolver so a user who already set
            # sessions.state_backend: postgres can run the command bare.
            try:
                from hermes_state_postgres import resolve_postgres_dsn

                config = load_config()
                dsn = resolve_postgres_dsn(config)
            except Exception:
                dsn = None

    if not dsn:
        print(
            f"  {color('✗', Colors.RED)} No PostgreSQL DSN found.\n"
            "  Provide one with --dsn, set the HERMES_STATE_DATABASE_URL or\n"
            "  HERMES_STATE_POSTGRES_DSN environment variable, or set\n"
            "  sessions.state_backend: postgres and sessions.postgres_dsn\n"
            "  in config.yaml.",
            file=sys.stderr,
        )
        return 1

    yes: bool = bool(getattr(args, "yes", False))
    is_tty: bool = sys.stdin.isatty()

    # Non-interactive without --yes: refuse rather than hang.
    if not is_tty and not yes:
        print(
            f"  {color('✗', Colors.RED)} stdin is not a TTY and --yes was not passed.\n"
            "  Re-run with -y / --yes to confirm the migration non-interactively.",
            file=sys.stderr,
        )
        return 1

    # --- Confirmation prompt ---
    if not yes:
        from hermes_state_postgres import _redact_dsn

        print()
        print(color("◆ SQLite → PostgreSQL State Migration", Colors.CYAN, Colors.BOLD))
        print()
        print(f"  Source : {sqlite_path}")
        print(f"  Target : {_redact_dsn(dsn)}")
        print()
        print(
            color(
                "  This will copy all sessions and messages into the target database.\n"
                "  The SQLite source is opened read-only and is never modified.",
                Colors.DIM,
            )
        )
        print()
        try:
            answer = input("  Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            print("  Aborted.", file=sys.stderr)
            return 1
        if answer not in ("y", "yes"):
            print("  Aborted.")
            return 1
        print()

    # --- Run the migration ---
    try:
        summary = _m2pg.migrate(sqlite_path, dsn)
    except SystemExit as exc:
        # migrate() raises SystemExit for user-facing errors (missing file,
        # missing postgres extra, etc.).
        print(f"  {color('✗', Colors.RED)} {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Catch psycopg / connection errors so no raw traceback reaches the user.
        exc_type = type(exc).__name__
        print(
            f"  {color('✗', Colors.RED)} Migration failed ({exc_type}): {exc}\n"
            "  Check that the DSN is correct and the PostgreSQL server is reachable.",
            file=sys.stderr,
        )
        return 1

    # --- Report results ---
    # Compare against the counts scoped to THIS migration, not the target's
    # whole-table totals: a target that already holds rows would otherwise
    # satisfy any >= check no matter how much of the source was dropped.
    src_s = summary["source_sessions"]
    src_m = summary["source_messages"]
    got_s = summary["migrated_sessions"]
    got_m = summary["migrated_messages"]
    dst_s = summary["target_sessions"]
    dst_m = summary["target_messages"]

    sessions_ok = got_s == src_s
    messages_ok = got_m == src_m
    # complete includes field verification, which can fail with matching counts.
    ok = summary["complete"] and summary["nul_rows"] == 0
    field_check = summary["field_check"]
    mismatch_count = field_check.get("mismatch_count", len(field_check["field_mismatches"]))

    if ok:
        print(
            f"  {color('✓', Colors.GREEN)} Migration complete.\n"
            f"    Sessions : {got_s}/{src_s} migrated\n"
            f"    Messages : {got_m}/{src_m} migrated\n"
            f"    Target now holds {dst_s} sessions / {dst_m} messages in total\n"
            f"    SQLite source left untouched: {summary['sqlite_path']}"
        )
    else:
        print(
            f"  {color('⚠', Colors.YELLOW)} Migration INCOMPLETE — verification failed. "
            "Do not switch backends yet.\n"
            f"    Sessions : {got_s}/{src_s} migrated"
            + (f"  {color('← MISSING', Colors.YELLOW)}" if not sessions_ok else "")
            + f"\n    Messages : {got_m}/{src_m} migrated"
            + (f"  {color('← MISSING', Colors.YELLOW)}" if not messages_ok else "")
            + f"\n    nul_rows : {summary['nul_rows']}"
            + f"\n    Field value mismatches : "
            f"{mismatch_count}"
            + f"\n    SQLite source left untouched: {summary['sqlite_path']}\n"
            + "\n  Rows keep their original SQLite ids and are inserted with "
            "ON CONFLICT DO NOTHING,\n  so this usually means the target already "
            "contains rows with the same ids.\n  Migrate into an empty database."
        )
        return 1

    return 0

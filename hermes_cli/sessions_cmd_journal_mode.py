"""`hermes sessions set-journal-mode delete|wal` — the offline self-service path for #100896.

``database.journal_mode: delete`` can never self-apply on a store that is already WAL:
``apply_wal_with_fallback`` deliberately never live-downgrades (#68545 — other gateway/cron/worker connections
may hold uncheckpointed WAL commits). Before this command the only escape hatch was an undocumented hand-run
``PRAGMA journal_mode=DELETE``. This runs it through the store admin's ``set_journal_mode``, under the same
admission the maintenance paths use: refuse while ANY foreign process holds the file or a sidecar, flip without
waiting out openers, then verify the header bytes 18/19 SQLite writes for the mode.
"""
from __future__ import annotations

from pathlib import Path


def cmd_set_journal_mode(args) -> int:
    from persistence import PersistenceError
    from persistence.core.admin import AdminUnsupported
    from persistence_assembly import open_store_admin, profile_scope, store_location
    from sqlite_support import resolve_journal_mode

    target = args.mode
    db_path = getattr(args, "db", None) or store_location(profile_scope())
    if db_path is None:
        print("✗ Setting the journal mode needs a file-backed session store; this profile's store has no file.")
        return 1
    db_path = Path(db_path)
    admin = open_store_admin(db_path, read_only=False) if db_path.exists() else None
    if admin is None or isinstance(admin, AdminUnsupported):
        print(f"No database at {db_path} (nothing to convert).")
        return 1
    try:
        current, after = admin.set_journal_mode(target, force=getattr(args, "force", False))
    except PersistenceError as exc:
        print(f"✗ {exc}")
        return 1
    if current == after:
        print(f"✓ {db_path} is already journal_mode={target}.")
        return 0
    print(f"✓ {db_path}: journal_mode {current} → {after} (header verified).")
    configured = resolve_journal_mode()
    if configured != target:
        print(f"  Note: config.yaml has database.journal_mode: {configured} — the next open would switch the file "
              f"back. Set `database.journal_mode: {target}` in config.yaml to keep it.")
    return 0

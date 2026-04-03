"""CLI helpers for exporting/importing local persistent memory snapshots."""

from __future__ import annotations

from pathlib import Path

from hermes_constants import get_hermes_home
from tools.persistent_memory_store import PersistentMemoryStore


def _store() -> PersistentMemoryStore:
    hermes_home = get_hermes_home()
    return PersistentMemoryStore(
        db_path=hermes_home / "memory.db",
        memory_dir=hermes_home / "memories",
    )


def memory_command(args) -> None:
    action = getattr(args, "memory_action", None) or "status"
    store = _store()

    if action == "export":
        output = Path(getattr(args, "output", None) or (get_hermes_home() / "memory-snapshot.json"))
        path = store.export_snapshot_to_file(output)
        print(f"Exported memory snapshot: {path}")
        return

    if action == "import":
        input_path = getattr(args, "input", None)
        if not input_path:
            raise SystemExit("Error: memory import requires --input <file>")
        result = store.import_snapshot_from_file(input_path)
        if not result.get("success"):
            raise SystemExit(result.get("error", "Memory import failed."))
        print(
            f"Imported memory snapshot: {input_path} "
            f"({result['imported']} new, {result['updated']} updated, {result['entry_count']} rows in snapshot)"
        )
        return

    if action == "status":
        entries = store.export_snapshot()
        memory_md = store.memory_dir / "MEMORY.md"
        user_md = store.memory_dir / "USER.md"
        print(f"memory.db: {store.db_path}")
        print(f"active entries: {len(store.list_entries('memory')) + len(store.list_entries('user'))}")
        print(f"snapshot rows: {entries['entry_count']}")
        print(f"MEMORY.md: {memory_md}")
        print(f"USER.md: {user_md}")
        return

    raise SystemExit(f"Unknown memory action: {action}")

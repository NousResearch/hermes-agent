"""CLI helpers for exporting/importing local persistent memory snapshots."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.persistent_memory_store import PersistentMemoryStore


def _store() -> PersistentMemoryStore:
    hermes_home = get_hermes_home()
    return PersistentMemoryStore(
        db_path=hermes_home / "memory.db",
        memory_dir=hermes_home / "memories",
    )


def _format_type_breakdown(store: PersistentMemoryStore) -> list[str]:
    lines: list[str] = []
    for target in ("memory", "user"):
        rows = store.list_entries(target, include_inactive=False)
        counts = Counter((row.get("entry_type") or row.get("kind") or "lesson") for row in rows)
        if counts:
            parts = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
            lines.append(f"{target}: {parts}")
    return lines


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
        print("Recall paths:")
        print("- hot memory: persistent steering facts")
        print("- session_search: transcript/session history")

        type_lines = _format_type_breakdown(store)
        if type_lines:
            print("Type breakdown:")
            for line in type_lines:
                print(f"- {line}")

        memory_selection = store.explain_prompt_selection("memory")
        if memory_selection.get("selected"):
            print("Hot memory selection:")
            for item in memory_selection["selected"]:
                print(f"- {item['content']}")
                print(f"  path: {item['path']}")
                print(f"  why: {item['reason']}")

        user_selection = store.explain_prompt_selection("user")
        if user_selection.get("selected"):
            print("Hot user memory selection:")
            for item in user_selection["selected"]:
                print(f"- {item['content']}")
                print(f"  path: {item['path']}")
                print(f"  why: {item['reason']}")
        return

    raise SystemExit(f"Unknown memory action: {action}")

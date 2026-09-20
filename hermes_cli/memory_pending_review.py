"""Read-only, current-disk review of native pending memory proposals."""

from __future__ import annotations

import difflib
import json

from tools import write_approval as wa


def memory_pending_diff(rest, *, memory_store=None) -> str:
    """Project a proposal for review, never replay it or mutate the caller's store."""
    if not rest:
        return "Usage: /memory diff <id>"
    record = wa.get_pending(wa.MEMORY, rest[0])
    if not record:
        return f"No pending memory write with id '{rest[0]}'."
    payload = record.get("payload", {})
    header = f"# Pending memory write {record['id']}: {record.get('summary', '')}\n\n"
    try:
        from tools.memory_tool import ENTRY_DELIMITER, MemoryStore, load_on_disk_store

        target = payload.get("target", "memory")
        if payload.get("provider") or target not in ("memory", "user"):
            raise ValueError("Not a native memory target")
        store = memory_store if memory_store is not None else load_on_disk_store()
        path = MemoryStore._path_for(target)
        raw, readable = MemoryStore._read_raw_checked(path)
        if not readable:
            raise ValueError("Current memory file could not be read")
        before = list(dict.fromkeys(MemoryStore._parse_entries(raw)))
        after = list(before)
        operations = payload.get("operations") if payload.get("action") == "batch" else [payload]
        if not isinstance(operations, list) or not operations:
            raise ValueError("No operations to preview")
        for i, op in enumerate(operations, 1):
            if not isinstance(op, dict):
                raise ValueError(f"Operation {i} is not an object")
            error = MemoryStore._apply_batch_op(
                after, op.get("action"), (op.get("content") or op.get("new_text") or "").strip(),
                (op.get("old_text") or "").strip(), f"Operation {i}")
            if error:
                raise ValueError(error)
        current = len(ENTRY_DELIMITER.join(before))
        projected = len(ENTRY_DELIMITER.join(after))
        limit = store._char_limit(target)
        before_lines = ENTRY_DELIMITER.join(before).splitlines()
        after_lines = ENTRY_DELIMITER.join(after).splitlines()
        diff = "\n".join(difflib.unified_diff(
            before_lines, after_lines, fromfile=f"current/{path.name}", tofile=f"proposed/{path.name}",
            n=max(len(before_lines), len(after_lines)), lineterm=""))
        return (header + "Current-disk preview only; approval revalidates all safety checks.\n"
                + f"Characters: {current:,} -> {projected:,} / {limit:,} (net {projected-current:+,}).\n"
                + ("Projected size exceeds the configured limit.\n" if projected > limit else "")
                + "\n" + (diff or "No text changes."))
    except Exception as exc:
        # Stale/malformed records remain reviewable, never show a partial batch as the result.
        return (header + f"Cannot project against current memory: {exc}\n\n"
                + "Full staged proposal (unchanged):\n" + json.dumps(payload, ensure_ascii=False, indent=2))

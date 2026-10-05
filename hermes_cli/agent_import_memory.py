"""Merge imported notes into identity-managed memory through its existing file lock."""
from __future__ import annotations

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.memory_entry_identity import IdentityError, PREFIX, parse_entries
from tools.memory_identity_store import _backup_bytes
from tools.memory_tool import MemoryStore


def _details(stats):
    return {f"{name}_entries": stats[key] for name, key in (
        ("existing", "existing"), ("added", "added"), ("duplicate", "duplicates"), ("overflowed", "overflowed"))}


def _write(importer, destination, incoming, details):
    from hermes_cli.agent_import import MEMORY_CHAR_LIMIT, merge_entries
    store = MemoryStore(memory_char_limit=MEMORY_CHAR_LIMIT)

    def merge(entries, limit):
        if not store._identity_enabled["memory"]:
            return {"success": False, "error": "Memory identity metadata changed after import preview; retry."}
        updated, stats = merge_entries(entries, incoming, limit)
        details.update(_details(stats))
        if not stats["added"]:
            return {"success": True}
        details["backup"] = str(_backup_bytes(destination, destination.read_bytes()))
        return updated, "Imported memory entries."

    token = set_hermes_home_override(importer.target_root)
    try:
        result = store._mutate("memory", merge)
        return None if result.get("success") else result.get("error", "Memory import failed.")
    except (OSError, RuntimeError) as exc:
        return f"Could not persist imported memory: {exc}"
    finally:
        reset_hermes_home_override(token)


def import_managed_memory(importer, kind, source, destination, incoming):
    """Return True when the native identity path handled this import, including refusals."""
    from hermes_cli.agent_import import MEMORY_CHAR_LIMIT, merge_entries
    raw, read_ok = MemoryStore._read_raw_checked(destination)
    if not read_ok:
        importer.record(kind, source, destination, "error", "Memory file could not be read; nothing was imported.")
        return True
    if PREFIX not in raw and "hermes-memory-identities:" not in raw:
        return False
    try:
        existing = parse_entries(raw, target="memory")
    except IdentityError as exc:
        importer.record(kind, source, destination, "error", str(exc))
        return True
    _, stats = merge_entries(existing, incoming, MEMORY_CHAR_LIMIT)
    details = _details(stats)
    if not stats["added"]:
        importer.record(kind, source, destination, "skipped", "No new entries to import", **details)
        return True
    importer.apply(kind, source, destination, "Would merge entries", lambda: _write(importer, destination, incoming, details), details)
    return True

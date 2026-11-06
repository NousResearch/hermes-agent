"""``hermes memory show`` / ``hermes memory forget`` — inspect and prune one entry.

The built-in stores (``MEMORY.md`` / ``USER.md``) are readable in file form but not
through any product surface: without opening a chat and asking the agent, there is
no way to see what the agent remembers or remove one wrong fact. ``hermes memory
reset`` wipes everything, which is not what anyone wants for a single stale line.

Both commands run without an agent: ``load_on_disk_store()`` builds the same
configured store the agent uses, and a removal is mirrored to the active external
provider through ``MemoryManager.on_memory_write`` so a mirror never drifts from
the source file.
"""
from __future__ import annotations

import shutil
import sys

# Targets the store understands -> display name.
_TARGET_LABELS = (("memory", "MEMORY.md"), ("user", "USER.md"))


def _print_err(message: str) -> None:
    print(f"\n  ✗ {message}\n", file=sys.stderr)


def _load_store(force_fresh: bool = False):
    """Configured on-disk store, loaded. Raises nothing: reports and returns None on failure."""
    try:
        from tools.memory_tool import load_on_disk_store
        store = load_on_disk_store()
        if force_fresh:  # a fresh disk read, not whatever the caller's module state holds
            store.load_from_disk()
        return store
    except (OSError, ValueError, RuntimeError) as e:  # unreadable/invalid memory dir
        _print_err(f"Could not read built-in memory: {e}")
        return None


def _entries_for(store, target: str) -> list:
    getter = getattr(store, "_entries_for", None)
    if callable(getter):
        return list(getter(target))
    return list(getattr(store, "user_entries" if target == "user" else "memory_entries", []))


def _char_limit(store, target: str) -> int:
    getter = getattr(store, "_char_limit", None)
    return int(getter(target)) if callable(getter) else (2200 if target == "memory" else 1375)


def _resolve_target(args) -> str:
    """``--target`` (default: show both, forget requires one explicit store)."""
    return (getattr(args, "target", None) or "memory").strip().lower()


def _mirror_removal(target: str, entry: str) -> None:
    """Tell the active external provider an entry was removed, so mirrors stay in step.

    Best-effort: a provider failure is logged, never fatal — the source file is
    already updated at this point.
    """
    try:
        from agent.memory_manager import MemoryManager
        MemoryManager().on_memory_write(
            "remove", target, "", metadata={"previous_content": entry, "tool_name": "hermes memory forget"})
    except (ImportError, OSError, RuntimeError, ValueError) as e:  # mirror failure must not undo a committed remove
        print(f"  ! Could not notify the memory provider: {e}", file=sys.stderr)


def cmd_memory_show(args) -> None:
    """Print built-in memory entries (and their size against the cap)."""
    store = _load_store()
    if store is None:
        return
    only = (getattr(args, "target", None) or "").strip().lower()
    total = 0
    width = shutil.get_terminal_size((100, 24)).columns
    for target, filename in _TARGET_LABELS:
        if only and target != only:
            continue
        entries = _entries_for(store, target)
        limit = _char_limit(store, target)
        used = sum(len(e) + 2 for e in entries)
        print(f"\n  {filename} — {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}, "
              f"{used:,}/{limit:,} chars{'  (OVER LIMIT)' if used > limit else ''}")
        if not entries:
            print("    (empty)")
            continue
        for i, entry in enumerate(entries, 1):
            text = " ".join(entry.split())
            if len(text) > max(40, width - 8):
                text = text[: max(40, width - 11)] + "..."
            print(f"    {i:>2}. {text}")
        total += len(entries)
    if total == 0 and only:
        print(f"\n  {only} store is empty.\n")
        return
    print(f"\n  {total} total. Forget one with: hermes memory forget <text> --target <memory|user>\n")


def cmd_memory_forget(args) -> None:
    """Remove one entry, by an unambiguous substring, from the chosen store."""
    old_text = (getattr(args, "entry", None) or "").strip()
    if not old_text:
        _print_err("Provide the text of the entry to forget: hermes memory forget <text>")
        return
    target = _resolve_target(args)
    store = _load_store()
    if store is None:
        return
    if not list(_entries_for(store, target)):
        _print_err(f"{target} store is empty.")
        return
    try:
        result = store.remove(target, old_text)
    except (OSError, RuntimeError, ValueError) as e:  # lock/write failure on the store
        _print_err(f"Could not remove entry: {e}")
        return
    if isinstance(result, dict) and not result.get("success"):
        _print_err(result.get("error") or "Remove failed.")
        matches = result.get("matches")
        if matches:
            print("  Matching entries:")
            for m in matches[:10]:
                print(f"    - {m}")
            print("  Be more specific.\n")
        return
    removed = (result or {}).get("removed_entry", "")
    print(f"\n  ✓ Removed from {target}: {removed[:120]}{'...' if len(removed) > 120 else ''}")
    _mirror_removal(target, removed or old_text)
    remaining = len(_entries_for(store, target))
    print(f"  {remaining} entr{'y' if remaining == 1 else 'ies'} left.\n")


def cmd_memory(args) -> None:
    sub = getattr(args, "memory_command", None)
    if sub == "show":
        cmd_memory_show(args)
    elif sub == "forget":
        cmd_memory_forget(args)

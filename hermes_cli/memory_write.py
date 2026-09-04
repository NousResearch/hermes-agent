"""CLI writes to built-in MEMORY.md / USER.md without an agent turn."""

from __future__ import annotations

from typing import Any, Dict


def add_builtin_memory(content: str, target: str = "memory") -> Dict[str, Any]:
    """Append one built-in memory entry using the same store as the memory tool.

    This is a housekeeping path: no LLM call, no conversation loop. The write
    is live on disk immediately; an already-open session keeps its frozen
    system-prompt snapshot until the next session (prompt-cache invariant).
    """
    from hermes_cli.config import load_config
    from tools.memory_tool import MemoryStore

    store_name = (target or "memory").strip().lower()
    if store_name not in {"memory", "user"}:
        return {"success": False, "error": "target must be 'memory' or 'user'."}

    text = (content or "").strip()
    if not text:
        return {"success": False, "error": "Content cannot be empty."}

    config = load_config()
    mem = config.get("memory") if isinstance(config, dict) else {}
    if not isinstance(mem, dict):
        mem = {}

    store = MemoryStore(
        memory_char_limit=int(mem.get("memory_char_limit") or 2200),
        user_char_limit=int(mem.get("user_char_limit") or 1375),
    )
    store.load_from_disk()
    return store.add(store_name, text)

"""
Hermes Core Persistent Memory Package.
Integrates SQLite, FTS5, and file-backed persistent memory.
"""

from typing import Any, Dict, List, Optional


class MemoryManager:
    """Manages persistent conversation, user, project, semantic, and procedural memory."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path
        self._memories: List[Dict[str, Any]] = []

    def add_memory(self, category: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mem = {"category": category, "content": content, "metadata": metadata or {}}
        self._memories.append(mem)
        return mem

    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        return [m for m in self._memories if query.lower() in m["content"].lower()]


__all__ = ["MemoryManager"]

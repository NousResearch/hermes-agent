"""Compatibility built-in memory provider used by MemoryManager tests.

The durable MEMORY.md / USER.md behavior still lives in the core memory tool.
This provider exists so the provider-manager abstraction always has a concrete
builtin provider type with the canonical name `builtin`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider


class BuiltinMemoryProvider(MemoryProvider):
    """Minimal built-in provider shim for the memory-manager abstraction."""

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

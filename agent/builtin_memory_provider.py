"""Compatibility shim for the always-on built-in memory provider.

The real persistent memory storage for MEMORY.md / USER.md lives in
``tools.memory_tool.MemoryStore`` and is already managed by the agent's
memory tool path.  Some callers and tests still import
``agent.builtin_memory_provider.BuiltinMemoryProvider`` though, so this
module provides a lightweight provider that satisfies the expected
``MemoryProvider`` interface without introducing duplicate tool schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider


class BuiltinMemoryProvider(MemoryProvider):
    """No-op provider representing the always-on built-in memory layer."""

    def __init__(self) -> None:
        self._session_id: Optional[str] = None
        self._init_kwargs: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._init_kwargs = dict(kwargs)

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        return None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        return None

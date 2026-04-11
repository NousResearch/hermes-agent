"""No-op built-in memory provider shim.

The built-in memory store is implemented by ``tools.memory_tool.MemoryStore``.
This provider exists so ``MemoryManager`` callers can represent that built-in
slot explicitly alongside one external memory plugin.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider


class BuiltinMemoryProvider(MemoryProvider):
    """Compatibility provider for the always-present built-in memory slot."""

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self.session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

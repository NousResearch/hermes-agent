"""BuiltinMemoryProvider — wraps MEMORY.md / USER.md as a MemoryProvider.

Always registered as the first provider. Cannot be disabled or removed.
This is the existing Hermes memory system exposed through the provider
interface for compatibility with the MemoryManager.

The actual storage logic lives in tools/memory_tool.py (MemoryStore).
This provider is a thin adapter that delegates to MemoryStore and
exposes the memory tool schema.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


class BuiltinMemoryProvider(MemoryProvider):
    """Built-in file-backed memory (MEMORY.md + USER.md).

    Always active, never disabled by other providers. The `memory` tool
    is handled by run_agent.py's agent-level tool interception (not through
    the normal registry), so get_tool_schemas() returns an empty list —
    the memory tool is already wired separately.
    """

    def __init__(
        self,
        memory_store=None,
        memory_enabled: bool = False,
        user_profile_enabled: bool = False,
    ):
        self._store = memory_store
        self._memory_enabled = memory_enabled
        self._user_profile_enabled = user_profile_enabled

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        """Built-in memory is always available."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """Load memory from disk if not already loaded."""
        if self._store is not None:
            self._store.load_from_disk()

    def system_prompt_block(self) -> str:
        """Return MEMORY.md and USER.md content for the system prompt.

        Uses the frozen snapshot captured at load time. This ensures the
        system prompt stays stable throughout a session (preserving the
        prompt cache), even though the live entries may change via tool calls.
        """
        if not self._store:
            return ""

        parts = []
        if self._memory_enabled:
            mem_block = self._store.format_for_system_prompt("memory")
            if mem_block:
                parts.append(mem_block)
        if self._user_profile_enabled:
            user_block = self._store.format_for_system_prompt("user")
            if user_block:
                parts.append(user_block)

        return "\n\n".join(parts)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Built-in memory doesn't do query-based recall — it's injected via system_prompt_block."""
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Built-in memory doesn't auto-sync turns — writes happen via the memory tool."""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return empty list.

        The `memory` tool is an agent-level intercepted tool, handled
        specially in run_agent.py before normal tool dispatch. It's not
        part of the standard tool registry. We don't duplicate it here.
        """
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Not used — the memory tool is intercepted in run_agent.py."""
        return json.dumps({"error": "Built-in memory tool is handled by the agent loop"})

    def shutdown(self) -> None:
        """No cleanup needed — files are saved on every write."""

    # -- Property access for backward compatibility --------------------------

    @property
    def store(self):
        """Access the underlying MemoryStore for legacy code paths."""
        return self._store

    @property
    def memory_enabled(self) -> bool:
        return self._memory_enabled

    @property
    def user_profile_enabled(self) -> bool:
        return self._user_profile_enabled

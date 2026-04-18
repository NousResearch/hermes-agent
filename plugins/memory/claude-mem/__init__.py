"""claude-mem memory provider for Hermes Agent."""
from __future__ import annotations

import logging

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


class ClaudeMemMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "claude-mem"

    def is_available(self) -> bool:
        return False  # filled in Phase 3

    def initialize(self, session_id: str, **kwargs) -> None:
        pass  # filled in Phase 3

    def get_tool_schemas(self):
        return []  # filled in Phase 4


def register(ctx) -> None:
    """Register claude-mem as a memory provider plugin."""
    ctx.register_memory_provider(ClaudeMemMemoryProvider())

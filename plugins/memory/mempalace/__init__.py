"""MemPalace memory plugin — MemoryProvider interface.

Local-first AI memory with semantic search, knowledge graphs, and spatial
memory palace. Stores memories in ChromaDB with entity extraction via
knowledge graph.
"""

from __future__ import annotations

import logging

from .provider import MemPalaceMemoryProvider

__all__ = ["MemPalaceMemoryProvider", "register"]

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Register MemPalace as a memory provider plugin."""
    provider = MemPalaceMemoryProvider()
    if not provider.is_available():
        logger.warning("MemPalace plugin skipped: mempalace package is not installed")
        return
    ctx.register_memory_provider(provider)

"""Codex app-server compaction cooldown lookup.

Split from ``agent/conversation_compression.py`` (facade size cap), which imports it at module level;
this module must never import the facade (import cycle). It logs under the facade's logger name so log
consumers keep one source.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent.conversation_compression")


def _codex_compaction_cooldown_remaining(agent: Any) -> float:
    """Seconds left on this session's compaction-failure cooldown (0 = clear)."""
    compressor = getattr(agent, "context_compressor", None)
    getter = getattr(compressor, "get_active_compression_failure_cooldown", None)
    if not callable(getter):
        return 0.0
    try:
        state = getter(refresh=True)
    except Exception:
        logger.debug("codex compaction cooldown lookup failed", exc_info=True)
        return 0.0
    try:
        return max(0.0, float(state.get("remaining_seconds") or 0.0)) if state else 0.0
    except (TypeError, ValueError):
        return 0.0

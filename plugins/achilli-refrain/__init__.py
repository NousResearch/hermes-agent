"""
Achilli Refrain -- Memory contradiction detection and cognitive harmonization.

Hooks into on_session_end to trigger periodic YantrikDB think/consolidation
scans. Implements time-based debouncing since on_session_end fires every turn.

Provides refrain_status and resolve_conflict tools.
Requires YantrikDB MCP server for consolidation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_LAST_CONSOLIDATION: Optional[float] = None
_LAST_CONSOLIDATION_RESULT: Optional[Dict[str, Any]] = None
_CONSOLIDATION_IN_PROGRESS: bool = False


def _get_debounce_seconds() -> float:
    import os
    try:
        return float(os.environ.get("ACHILLI_REFRAIN_DEBOUNCE", "3600"))
    except (ValueError, TypeError):
        return 3600.0


def _pattern_mining_enabled() -> bool:
    import os
    return os.environ.get("ACHILLI_REFRAIN_RUN_PATTERN_MINING", "").lower() in {
        "1", "true", "yes"
    }


def _disabled() -> bool:
    import os
    return os.environ.get("ACHILLI_REFRAIN_DISABLE", "").lower() in {
        "1", "true", "yes"
    }


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _on_session_end(**_: Any) -> None:
    """Trigger debounced consolidation scan."""
    global _LAST_CONSOLIDATION, _LAST_CONSOLIDATION_RESULT, _CONSOLIDATION_IN_PROGRESS

    if _disabled():
        return

    if _CONSOLIDATION_IN_PROGRESS:
        logger.debug("refrain: consolidation already in progress, skipping")
        return

    now = time.time()
    debounce = _get_debounce_seconds()

    if _LAST_CONSOLIDATION is not None and (now - _LAST_CONSOLIDATION) < debounce:
        logger.debug(
            "refrain: debounce active (%.0fs / %.0fs), skipping",
            now - _LAST_CONSOLIDATION, debounce,
        )
        return

    _CONSOLIDATION_IN_PROGRESS = True
    try:
        # Note: At this point we're inside a hook. We can't directly call
        # yantrikdb tools ourselves -- we log the intent and rely on the
        # agent to call refrain_status on its next turn, which will show
        # "consolidation needed". The actual think() call should be agent-driven.
        _LAST_CONSOLIDATION = now
        logger.info(
            "refrain: debounce window elapsed (debounce=%.0fs), consolidation due",
            debounce,
        )
        # Set a flag that refrain_status will detect, prompting the agent
        # to run the actual think() call
        _LAST_CONSOLIDATION_RESULT = {
            "status": "consolidation_due",
            "timestamp": now,
            "message": "Run yantrikdb think to perform consolidation scan.",
        }
    finally:
        _CONSOLIDATION_IN_PROGRESS = False


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _refrain_status_tool(*_, **__) -> dict:
    """Return memory health dashboard."""
    global _LAST_CONSOLIDATION_RESULT

    now = time.time()
    debounce = _get_debounce_seconds()
    elapsed = (now - _LAST_CONSOLIDATION) if _LAST_CONSOLIDATION else None
    due = elapsed is None or elapsed >= debounce

    # Build result
    result = {
        "enabled": not _disabled(),
        "debounce_s": debounce,
        "last_consolidation": _LAST_CONSOLIDATION,
        "elapsed_since_last_s": round(elapsed, 1) if elapsed else None,
        "consolidation_due": due,
        "pattern_mining_enabled": _pattern_mining_enabled(),
    }

    if _LAST_CONSOLIDATION_RESULT:
        result["last_result"] = _LAST_CONSOLIDATION_RESULT

    # Clear the "consolidation due" flag after reporting
    if due and _LAST_CONSOLIDATION_RESULT:
        if _LAST_CONSOLIDATION_RESULT.get("status") == "consolidation_due":
            _LAST_CONSOLIDATION_RESULT["status"] = "reported"

    return result


def _resolve_conflict_tool(
    strategy: str = "present",
    conflict_id: str = "",
    new_text: str = "",
    resolution_note: str = "",
    **_: Any,
) -> str:
    """
    Interactive conflict resolution.

    In a real invocation, this would call yantrikdb's conflict(action=resolve).
    Since we're a plugin hook (not an MCP client), we return a structured
    prompt for the agent to execute.

    Strategies: keep_a, keep_b, keep_both, merge, dismiss
    """
    if not conflict_id:
        return (
            "No conflict_id provided. Use yantrikdb conflict(action=list) to "
            "find open conflicts, then call resolve_conflict with the conflict_id."
        )

    strategies = ["keep_a", "keep_b", "keep_both", "merge", "dismiss"]
    if strategy not in strategies:
        return "Invalid strategy '{}'. Use one of: {}".format(strategy, strategies)

    if strategy == "present":
        return (
            "Conflict '{}' needs resolution. "
            "Use yantrikdb conflict(action=get, conflict_id='{}') to view it, "
            "then call resolve_conflict again with strategy=keep_a|keep_b|keep_both|merge|dismiss."
        ).format(conflict_id, conflict_id)

    note_part = " Note: {}".format(resolution_note) if resolution_note else ""
    merge_part = " Merged text: {}".format(new_text) if strategy == "merge" and new_text else ""

    return (
        "To resolve conflict '{}' with strategy '{}':\n"
        "Call yantrikdb conflict(action=resolve, conflict_id='{}', strategy='{}'{merge}{note})"
    ).format(
        conflict_id, strategy, conflict_id, strategy,
        merge=", new_text='{}'".format(new_text) if merge_part else "",
        note=", resolution_note='{}'".format(resolution_note) if note_part else "",
    )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_tool(
        name="refrain_status",
        description=(
            "Return memory health dashboard: open conflicts, last consolidation "
            "scan, debounce status. (achilli-refrain)"
        ),
        parameters={"type": "object", "properties": {}},
        handler=_refrain_status_tool,
    )
    ctx.register_tool(
        name="resolve_conflict",
        description=(
            "Interactive conflict resolution for YantrikDB open conflicts. "
            "Call with conflict_id and strategy (keep_a, keep_b, keep_both, merge, dismiss). "
            "Leave strategy='present' to get resolution instructions. (achilli-refrain)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "conflict_id": {
                    "type": "string",
                    "description": "YantrikDB conflict ID to resolve",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["present", "keep_a", "keep_b", "keep_both", "merge", "dismiss"],
                    "description": "Resolution strategy. 'present' shows instructions.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Merged text (only used with strategy=merge)",
                },
                "resolution_note": {
                    "type": "string",
                    "description": "Note explaining why this resolution was chosen",
                },
            },
        },
        handler=_resolve_conflict_tool,
    )

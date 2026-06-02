"""Bridge built-in flat memory writes to external memory providers.

The flat MEMORY.md/USER.md stores are deliberately small and prompt-cache
friendly.  External providers can mirror successful writes and, for capacity
failures, preserve the attempted fact as a durable fallback without mutating the
live system prompt snapshot.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CAPACITY_ERROR_MARKERS = (
    "would exceed the limit",
    "Replacement would put memory at",
)


def _parse_memory_result(result: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(result)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_capacity_failure(parsed: Dict[str, Any]) -> bool:
    if parsed.get("success") is not False:
        return False
    error = str(parsed.get("error", ""))
    return any(marker in error for marker in _CAPACITY_ERROR_MARKERS)


def should_bridge_memory_result(action: str, result: str) -> tuple[bool, Dict[str, Any]]:
    """Return whether a built-in memory result should notify providers.

    Successful add/replace writes are mirrored normally.  Capacity failures are
    also mirrored so high-value attempted facts are not lost when MEMORY.md or
    USER.md is full.  Other failures (blocked injection, missing args, ambiguous
    replace, etc.) are not bridged to avoid persisting rejected content.
    """

    if action not in {"add", "replace"}:
        return False, {}
    parsed = _parse_memory_result(result)
    if parsed.get("success") is True:
        return True, parsed
    if _is_capacity_failure(parsed):
        return True, parsed
    return False, parsed


def notify_external_memory_write(
    agent: Any,
    function_args: Dict[str, Any],
    function_result: str,
    *,
    effective_task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> None:
    """Notify agent._memory_manager when a memory result should be mirrored."""

    manager = getattr(agent, "_memory_manager", None)
    if not manager:
        return

    action = function_args.get("action", "")
    should_bridge, parsed = should_bridge_memory_result(action, function_result)
    if not should_bridge:
        return

    content = function_args.get("content", "")
    if not content:
        return

    target = function_args.get("target", "memory")
    try:
        metadata = agent._build_memory_write_metadata(
            task_id=effective_task_id,
            tool_call_id=tool_call_id,
        )
    except Exception:
        metadata = {}

    metadata = dict(metadata or {})
    metadata["flat_memory_success"] = parsed.get("success") is True
    if _is_capacity_failure(parsed):
        metadata.update(
            {
                "fallback_reason": "flat_memory_capacity",
                "flat_memory_error": parsed.get("error", ""),
                "flat_memory_usage": parsed.get("usage", ""),
            }
        )

    try:
        manager.on_memory_write(action, target, content, metadata=metadata)
    except Exception as exc:
        logger.debug("external memory write bridge failed: %s", exc)

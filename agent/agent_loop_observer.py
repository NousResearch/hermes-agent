"""Observer helpers for tools handled inside the agent loop.

The normal registry-dispatched tools flow through ``model_tools`` and emit
generic plugin hooks there. Agent-owned tools need local session state, so they
are intercepted before registry dispatch. This module gives plugins and memory
providers one shared observation point for those intercepted tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_OBSERVED_AGENT_LOOP_TOOLS = frozenset(
    {"todo", "memory", "delegate_task", "session_search"}
)


def notify_agent_loop_tool(
    agent: Any,
    tool_name: str,
    args: Dict[str, Any],
    result: Any,
    *,
    task_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
) -> List[Dict[str, Any]]:
    """Notify generic observers that an agent-loop tool completed."""
    if tool_name not in _OBSERVED_AGENT_LOOP_TOOLS:
        return []

    metadata = {
        "task_id": task_id or "",
        "session_id": getattr(agent, "session_id", "") or "",
        "parent_session_id": getattr(agent, "_parent_session_id", "") or "",
        "platform": getattr(agent, "platform", "") or "",
        "tool_name": tool_name,
        "tool_call_id": tool_call_id or "",
        "duration_ms": int(duration_ms or 0),
    }
    try:
        if hasattr(agent, "_build_memory_write_metadata"):
            metadata.update(
                agent._build_memory_write_metadata(
                    task_id=task_id,
                    tool_call_id=tool_call_id,
                )
            )
    except Exception:
        logger.debug("agent-loop observer metadata build failed", exc_info=True)

    annotations: List[Dict[str, Any]] = []
    memory_manager = getattr(agent, "_memory_manager", None)
    if memory_manager:
        try:
            annotations.extend(
                memory_manager.on_tool_observed(
                    tool_name,
                    dict(args or {}),
                    result,
                    metadata=metadata,
                )
            )
        except Exception:
            logger.debug("memory manager agent-loop observer failed", exc_info=True)

    try:
        from hermes_cli.plugins import invoke_hook

        for item in invoke_hook(
            "agent_loop_tool_observed",
            tool_name=tool_name,
            args=dict(args or {}),
            result=result,
            **metadata,
        ):
            if isinstance(item, dict):
                annotations.append(item)
    except Exception:
        logger.debug("plugin agent-loop observer failed", exc_info=True)

    return annotations


def append_observer_metadata(result: Any, annotations: List[Dict[str, Any]]) -> Any:
    """Attach observer metadata to JSON object tool results when possible."""
    if not annotations:
        return result
    if not isinstance(result, str):
        return result
    try:
        payload = json.loads(result)
    except Exception:
        return result
    if not isinstance(payload, dict):
        return result
    existing = payload.get("observer_metadata")
    if isinstance(existing, list):
        existing.extend(annotations)
    else:
        payload["observer_metadata"] = annotations
    return json.dumps(payload, ensure_ascii=False)

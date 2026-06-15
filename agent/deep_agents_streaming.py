"""Bridge between DeepAgents/LangGraph streaming and Hermes callbacks.

Hermes uses callback-based progress notifications:
  callback("tool.started", tool_name=..., preview=...)
  callback("tool.progress", tool_name=..., output=...)
  callback("tool.completed", tool_name=..., ...)
  callback("message.delta", text=...)
  callback("message.complete", ...)

LangGraph streams events differently. This module translates between them.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stream event bridge
# ---------------------------------------------------------------------------


class _HermesStreamingBridge:
    """Converts LangGraph stream events to Hermes callback format."""

    def __init__(self, callback, agent, task_id):
        self._callback = callback
        self._agent = agent
        self._task_id = task_id
        self._result = None
        self._events: list[Tuple[str, dict]] = []

    def iter_events(self, stream):
        """Iterate over LangGraph stream events, yielding (event_type, data).

        stream_mode="updates" produces dict updates per node.
        """
        for event in stream:
            if isinstance(event, dict):
                if "events" in event:
                    for sub_event in event["events"]:
                        yield from self._process_sub_event(sub_event)
                elif "output" in event:
                    yield from self._process_output(event["output"])
                else:
                    yield "unknown", event
            else:
                yield "raw_event", {"data": str(event)[:200]}

    def _process_sub_event(self, sub_event):
        """Process a LangGraph sub-event (message, tool, etc.)."""
        if not isinstance(sub_event, dict):
            return

        event_type = sub_event.get("type", "")
        data = sub_event.get("data", {})

        if event_type == "AIMessageChunk":
            content = data.get("content", "")
            if content:
                yield "message_delta", {"text": content}
        elif event_type == "ToolCall":
            name = data.get("name", "")
            args = data.get("args", {})
            yield "tool_start", {"tool_name": name, "args": str(args)}
        elif event_type == "ToolResult":
            name = data.get("name", "")
            result = data.get("content", "")
            yield (
                "tool_complete",
                {
                    "tool_name": name,
                    "result_preview": _shorten(result, 200),
                },
            )
        else:
            yield "raw_event", {"type": event_type, "data": str(data)[:200]}

    def _process_output(self, output):
        """Process node output (the hermes_engine result)."""
        if isinstance(output, dict):
            self._result = output
            yield "complete", {"result": output}
        elif isinstance(output, str):
            try:
                self._result = json.loads(output)
            except json.JSONDecodeError:
                self._result = {"final_response": output, "errors": [output]}
            yield "complete", {"result": self._result}

    def _get_latest_tool_event(self):
        """Get the most recent tool event to determine next events."""
        for event_type, data in reversed(self._events):
            if event_type.startswith("tool_"):
                return event_type, data
        return None, None

    def final_result(self) -> dict:
        """Return the final parsed result dict."""
        if self._result and isinstance(self._result, dict):
            return self._result
        return {
            "final_response": "",
            "messages": [],
            "api_calls": 0,
            "completed": False,
            "failed": False,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _shorten(text: str, max_len: int = 200) -> str:
    """Truncate text for preview."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

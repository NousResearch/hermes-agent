"""Turn-visibility relays of ``ClaudeAgentSdkSession``: stream deltas, interim
assistant text, tool-iteration and tool-started notifications, plus the tool
preview. Extracted from ``claude_agent_sdk_session.py``; every method resolves
through ``ClaudeAgentSdkSession``'s MRO unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


def _tool_preview(name: str, args: dict) -> str:
    """Short human preview of a tool call for progress breadcrumbs."""
    for key in ("command", "file_path", "path", "url", "query", "prompt"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value[:120]
    return name


class ClaudeSdkNotifyMixin:
    """Per-turn visibility callbacks and their relays (see module docstring)."""

    def set_turn_visibility_callbacks(
        self,
        *,
        on_interim_assistant: Optional[Callable[[str], None]],
        on_tool_iteration: Optional[Callable[[], None]],
    ) -> None:
        """Atomically install current-turn, runtime-owned visibility hooks."""
        with self._turn_callback_lock:
            self._on_interim_assistant = on_interim_assistant
            self._on_tool_iteration = on_tool_iteration

    def _forward_stream_delta(self, message: Any) -> None:
        """Relay a top-level text delta to the display callback (never the
        transcript). Subagent streams (parent_tool_use_id set) stay quiet."""
        if self._on_stream_delta is None:
            return
        if getattr(message, "parent_tool_use_id", None):
            return
        event = getattr(message, "event", None) or {}
        if event.get("type") != "content_block_delta":
            return
        delta = event.get("delta") or {}
        if delta.get("type") != "text_delta":
            return
        text = delta.get("text")
        if not text:
            return
        try:
            self._on_stream_delta(text)
        except Exception:  # pragma: no cover - display callback
            logger.debug("stream delta callback raised", exc_info=True)

    def _notify_interim_assistant(self, message: Any) -> None:
        """Relay completed tool-adjacent assistant prose as commentary."""
        if getattr(message, "parent_tool_use_id", None):
            return
        with self._turn_callback_lock:
            callback = self._on_interim_assistant
        if callback is None:
            return
        if type(message).__name__ != "AssistantMessage":
            return
        blocks = list(getattr(message, "content", None) or [])
        if not any(type(block).__name__ == "ToolUseBlock" for block in blocks):
            return
        text = "\n".join(
            str(getattr(block, "text", "") or "")
            for block in blocks
            if type(block).__name__ == "TextBlock" and getattr(block, "text", "")
        ).strip()
        if not text:
            return
        try:
            callback(text)
        except Exception:  # pragma: no cover - display callback
            logger.debug("interim assistant callback raised", exc_info=True)

    def _notify_tool_iteration(self) -> None:
        with self._turn_callback_lock:
            callback = self._on_tool_iteration
        if callback is None:
            return
        try:
            callback()
        except Exception:  # pragma: no cover - display callback
            logger.debug("tool-iteration callback raised", exc_info=True)

    def _notify_tool_started(self, message: Any) -> None:
        """Bridge ToolUseBlocks to Hermes tool-progress (gateway breadcrumbs),
        mirroring codex_runtime._codex_note_to_tool_progress (#38835)."""
        if self._on_tool_started is None:
            return
        if type(message).__name__ != "AssistantMessage":
            return
        for block in getattr(message, "content", None) or []:
            if type(block).__name__ != "ToolUseBlock":
                continue
            name = getattr(block, "name", "") or "unknown"
            args = getattr(block, "input", None) or {}
            if not isinstance(args, dict):
                args = {"input": args}
            preview = _tool_preview(name, args)
            try:
                self._on_tool_started(name, preview, args)
            except Exception:  # pragma: no cover - display callback
                logger.debug("tool-progress callback raised", exc_info=True)

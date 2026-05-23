"""Projects Cursor SDK stream messages into Hermes' messages list.

Converts ``SDKMessage`` events from ``cursor-sdk`` into the standard
OpenAI-shaped ``{role, content, tool_calls, tool_call_id}`` entries that
``agent/curator.py`` and the sessions DB already understand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


def _deterministic_call_id(tool_name: str, call_id: str) -> str:
    if call_id:
        return f"cursor_{tool_name}_{call_id}"
    digest = hashlib.sha256(tool_name.encode()).hexdigest()[:16]
    return f"cursor_{tool_name}_{digest}"


def _format_tool_args(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def _as_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "__dict__"):
        return {k: v for k, v in vars(message).items() if not k.startswith("_")}
    return {}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class ProjectionResult:
    messages: list[dict] = field(default_factory=list)
    is_tool_iteration: bool = False
    final_text: Optional[str] = None


class CursorEventProjector:
    """Stateful projector for Cursor SDK message stream events."""

    def __init__(self) -> None:
        self._pending_reasoning: list[str] = []
        self._pending_tool_calls: dict[str, dict] = {}

    def project(self, message: Any) -> ProjectionResult:
        msg = _as_dict(message)
        msg_type = str(_get(msg, "type") or "").strip().lower()

        if msg_type == "thinking":
            text = _get(msg, "text") or ""
            if text:
                self._pending_reasoning.append(str(text))
            return ProjectionResult()

        if msg_type == "assistant":
            return self._project_assistant(msg)

        if msg_type == "tool_call":
            return self._project_tool_call(msg)

        if msg_type == "user":
            return self._project_user(msg)

        return ProjectionResult()

    def _project_assistant(self, msg: dict) -> ProjectionResult:
        inner = _get(msg, "message") or {}
        content_blocks = _get(inner, "content") or []
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in content_blocks:
            block_dict = _as_dict(block)
            block_type = str(_get(block_dict, "type") or "").lower()
            if block_type == "text":
                text_parts.append(str(_get(block_dict, "text") or ""))
            elif block_type == "tool_use":
                call_id = str(_get(block_dict, "id") or _get(block_dict, "call_id") or "")
                name = str(_get(block_dict, "name") or "tool")
                args = _get(block_dict, "input") or _get(block_dict, "args") or {}
                if not isinstance(args, dict):
                    args = {"input": args}
                tool_calls.append(
                    {
                        "id": _deterministic_call_id(name, call_id),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _format_tool_args(args),
                        },
                    }
                )
        content = "".join(text_parts).strip()
        if tool_calls:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            }
            if self._pending_reasoning:
                assistant_msg["reasoning"] = "\n".join(self._pending_reasoning)
                self._pending_reasoning = []
            return ProjectionResult(messages=[assistant_msg], is_tool_iteration=True)

        if not content:
            return ProjectionResult()
        out: dict[str, Any] = {"role": "assistant", "content": content}
        if self._pending_reasoning:
            out["reasoning"] = "\n".join(self._pending_reasoning)
            self._pending_reasoning = []
        return ProjectionResult(messages=[out], final_text=content)

    def _project_tool_call(self, msg: dict) -> ProjectionResult:
        status = str(_get(msg, "status") or "").strip().lower()
        call_id = str(_get(msg, "call_id") or "")
        name = str(_get(msg, "name") or "tool")
        stable_id = _deterministic_call_id(name, call_id)

        if status == "running":
            args = _get(msg, "args") or {}
            if not isinstance(args, dict):
                args = {"args": args}
            self._pending_tool_calls[stable_id] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": stable_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _format_tool_args(args),
                        },
                    }
                ],
            }
            if self._pending_reasoning:
                self._pending_tool_calls[stable_id]["reasoning"] = "\n".join(
                    self._pending_reasoning
                )
                self._pending_reasoning = []
            return ProjectionResult()

        if status not in {"completed", "error"}:
            return ProjectionResult()

        assistant_msg = self._pending_tool_calls.pop(
            stable_id,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": stable_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _format_tool_args(_get(msg, "args") or {}),
                        },
                    }
                ],
            },
        )
        result = _get(msg, "result")
        if status == "error":
            content = json.dumps({"error": result or "tool failed"}, ensure_ascii=False)[:4000]
        elif result is not None:
            if isinstance(result, str):
                content = result[:4000]
            else:
                content = json.dumps(result, ensure_ascii=False)[:4000]
        else:
            content = ""
        tool_msg = {
            "role": "tool",
            "tool_call_id": stable_id,
            "content": content,
        }
        return ProjectionResult(
            messages=[assistant_msg, tool_msg], is_tool_iteration=True
        )

    def _project_user(self, msg: dict) -> ProjectionResult:
        inner = _get(msg, "message") or {}
        content_blocks = _get(inner, "content") or []
        text_parts: list[str] = []
        for block in content_blocks:
            block_dict = _as_dict(block)
            if str(_get(block_dict, "type") or "").lower() == "text":
                text_parts.append(str(_get(block_dict, "text") or ""))
        if not text_parts:
            return ProjectionResult()
        return ProjectionResult(
            messages=[{"role": "user", "content": "\n".join(text_parts)}]
        )

"""Projects codex app-server ``item/*`` notifications into OpenAI-shaped messages.

userMessage → user; agentMessage → assistant; reasoning → stashed onto the next
assistant entry; commandExecution / fileChange / mcpToolCall / dynamicToolCall →
assistant tool_call + tool result; anything else → opaque assistant note.
Each item yields AT MOST one assistant + one tool entry (message-alternation
invariant). ``is_tool_iteration`` ticks once per completed tool-shaped item.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def _deterministic_call_id(item_type: str, item_id: str) -> str:
    """Stable tool_call id: the codex item id, else a hash so replay keeps prefix caches valid."""
    return f"codex_{item_type}_{item_id or hashlib.sha256(f'{item_type}'.encode()).hexdigest()[:16]}"


def _format_tool_args(d: dict) -> str:
    """Format a dict as JSON the way Hermes' existing tool_calls path does."""
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def _dict_args(raw: Any) -> dict:
    args = raw or {}
    return args if isinstance(args, dict) else {"arguments": args}


@dataclass
class ProjectionResult:
    """Output of projecting one Codex item; empty ``messages`` = ignored (e.g. a streaming delta)."""

    messages: list[dict] = field(default_factory=list)
    is_tool_iteration: bool = False
    final_text: Optional[str] = None  # Set when an agentMessage completes


class CodexEventProjector:
    """Stateful projector; stashes codex's separate reasoning items onto the next assistant message."""

    def __init__(self) -> None:
        self._pending_reasoning: list[str] = []

    def project(self, notification: dict) -> ProjectionResult:
        """Project one notification; only ``item/completed`` materializes messages (deltas are display-only)."""
        if notification.get("method", "") != "item/completed":
            return ProjectionResult()
        item = (notification.get("params", {}) or {}).get("item") or {}
        item_type = item.get("type") or ""
        item_id = item.get("id") or ""
        if item_type == "agentMessage":
            return self._project_agent_message(item)
        if item_type == "reasoning":
            self._pending_reasoning.extend(item.get("summary") or [])
            self._pending_reasoning.extend(item.get("content") or [])
            return ProjectionResult()
        if item_type == "userMessage":
            return self._project_user_message(item)
        tool_projection = self._TOOL_PROJECTIONS.get(item_type)
        if tool_projection is not None:
            return self._project_tool_item(item, item_id, tool_projection)
        # Unknown / rare items (plan, hookPrompt, ...): opaque note, no fabricated tool_call structure.
        return self._project_opaque(item, item_type)

    def _assistant_message(self, content: Optional[str], **extra: Any) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": content, **extra}
        if self._pending_reasoning:
            msg["reasoning"] = "\n".join(self._pending_reasoning)
            self._pending_reasoning = []
        return msg

    def _project_agent_message(self, item: dict) -> ProjectionResult:
        text = item.get("text") or ""
        return ProjectionResult(messages=[self._assistant_message(text)], final_text=text)

    @staticmethod
    def _project_user_message(item: dict) -> ProjectionResult:
        # userMessage content is a list of UserInput variants; flatten text
        # fragments and drop non-text parts (Hermes' messages store text only).
        text_parts = [
            (fragment.get("text") or "") if fragment.get("type") == "text" else str(fragment["text"])
            for fragment in item.get("content") or []
            if isinstance(fragment, dict) and (fragment.get("type") == "text" or "text" in fragment)
        ]
        return ProjectionResult(messages=[{"role": "user", "content": "\n".join(text_parts)}])

    def _project_tool_item(
        self, item: dict, item_id: str, spec: Callable[[dict], tuple[str, str, dict, str]]
    ) -> ProjectionResult:
        """Emit the (assistant tool_call, tool result) pair; ``spec(item)`` -> ``(id_type, name, args, content)``."""
        id_type, name, args, content = spec(item)
        call_id = _deterministic_call_id(id_type, item_id)
        assistant_msg = self._assistant_message(
            None,
            tool_calls=[{"id": call_id, "type": "function", "function": {"name": name, "arguments": _format_tool_args(args)}}],
        )
        tool_msg = {"role": "tool", "tool_call_id": call_id, "content": content}
        return ProjectionResult(messages=[assistant_msg, tool_msg], is_tool_iteration=True)

    @staticmethod
    def _command_spec(item: dict) -> tuple[str, str, dict, str]:
        args = {"command": item.get("command") or "", "cwd": item.get("cwd") or ""}
        output = item.get("aggregatedOutput") or ""
        exit_code = item.get("exitCode")
        if exit_code is not None and exit_code != 0:
            output = f"[exit {exit_code}]\n{output}"
        return "exec", "exec_command", args, output

    @staticmethod
    def _file_change_spec(item: dict) -> tuple[str, str, dict, str]:
        # Per-file change kinds only — full file contents can be huge.
        changes_summary = [
            {"kind": (change.get("kind") or {}).get("type") or "update", "path": change.get("path") or ""}
            for change in item.get("changes") or []
        ]
        status = item.get("status") or "unknown"
        content = f"apply_patch status={status}, {len(changes_summary)} change(s)"
        return "apply_patch", "apply_patch", {"changes": changes_summary}, content

    @staticmethod
    def _mcp_tool_call_spec(item: dict) -> tuple[str, str, dict, str]:
        server = item.get("server") or "mcp"
        tool = item.get("tool") or "unknown"
        result, error = item.get("result"), item.get("error")
        if error:
            content = f"[error] {json.dumps(error, ensure_ascii=False)[:1000]}"
        else:
            content = json.dumps(result, ensure_ascii=False)[:4000] if result is not None else ""
        # Call id mirrors the native MCP name convention (mcp__server__tool).
        return f"mcp__{server}__{tool}", f"mcp.{server}.{tool}", _dict_args(item.get("arguments")), content

    @staticmethod
    def _dynamic_tool_call_spec(item: dict) -> tuple[str, str, dict, str]:
        tool = item.get("tool") or "unknown"
        content_items = item.get("contentItems") or []
        content = (
            json.dumps(content_items, ensure_ascii=False)[:4000] if isinstance(content_items, list) and content_items
            else f"success={item.get('success')}"
        )
        return f"dyn_{tool}", tool, _dict_args(item.get("arguments")), content

    _TOOL_PROJECTIONS: dict[str, Callable[[dict], tuple[str, str, dict, str]]] = {
        "commandExecution": _command_spec,
        "fileChange": _file_change_spec,
        "mcpToolCall": _mcp_tool_call_spec,
        "dynamicToolCall": _dynamic_tool_call_spec,
    }

    @staticmethod
    def _project_opaque(item: dict, item_type: str) -> ProjectionResult:
        try:
            payload = json.dumps(item, ensure_ascii=False)[:1500]
        except (TypeError, ValueError):
            payload = repr(item)[:1500]
        return ProjectionResult(messages=[{"role": "assistant", "content": f"[codex {item_type}] {payload}"}])

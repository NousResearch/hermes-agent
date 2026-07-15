"""Projects codex app-server events into Hermes' messages list.

The translator that lets Hermes' memory/skill review keep working under the
Codex runtime: it converts Codex `item/*` notifications into the standard
OpenAI-shaped `{role, content, tool_calls, tool_call_id}` entries that
`agent/curator.py` already knows how to read.

Codex emits items with a discriminator field `type`:
  - userMessage         → {role: "user", content}
  - agentMessage        → {role: "assistant", content}
  - reasoning           → stashed in the assistant's "reasoning" field
  - commandExecution    → assistant tool_call(name="exec") + tool result
  - fileChange          → assistant tool_call(name="apply_patch") + tool result
  - mcpToolCall         → assistant tool_call(name=f"mcp.{server}.{tool}") + tool result
  - dynamicToolCall     → assistant tool_call(name=tool) + tool result
  - plan/hookPrompt/collabAgentToolCall → recorded as opaque assistant notes

Each item maps to AT MOST one assistant entry + one tool entry, preserving
Hermes' message-alternation invariants (system → user → assistant → user/tool
→ assistant → ...). Multiple Codex tool calls within one Codex turn produce
multiple consecutive (assistant, tool) pairs, which is the same shape Hermes
already produces for parallel tool calls.

Counters tracked alongside projection:
  - tool_iterations: ticks once per completed tool-shaped item. Used by
    AIAgent._iters_since_skill (skill nudge gate, default threshold 10).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


def _deterministic_call_id(item_type: str, item_id: str) -> str:
    """Stable id for tool_call message correlation.

    Codex item ids are the only protocol identity stable across replay. Never
    fabricate an id for malformed/id-less tool items: doing so conflates
    distinct calls and corrupts tool-call/result correlation.
    """
    if not isinstance(item_type, str) or not item_type.strip():
        raise ValueError("item_type must be a non-empty string")
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError("item_id must be a non-empty string")
    # Length-prefix the first component so underscore placement cannot make
    # distinct (type, id) pairs collapse onto the same opaque correlation id.
    return f"codex_{len(item_type)}_{item_type}_{item_id}"


def _format_tool_args(d: dict) -> str:
    """Format a dict as JSON the way Hermes' existing tool_calls path does."""
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def _provider_tool_identity(item_type: str, item: dict) -> Optional[tuple[str, ...]]:
    """Return the provider-visible tool identity when the item supplies one."""
    if item_type == "commandExecution":
        return ("exec",)
    if item_type == "fileChange":
        return ("apply_patch",)
    if item_type == "dynamicToolCall":
        tool = item.get("tool")
        if isinstance(tool, str) and tool.strip():
            return (tool,)
    if item_type == "mcpToolCall":
        server = item.get("server")
        tool = item.get("tool")
        if (
            isinstance(server, str)
            and server.strip()
            and isinstance(tool, str)
            and tool.strip()
        ):
            return (server, tool)
    return None


@dataclass
class ProjectionResult:
    """Output of projecting one Codex item.

    `messages` is a list because some Codex items produce two messages
    (assistant tool_call + tool result). Empty list = item ignored (e.g. a
    streaming `outputDelta` that doesn't materialize into messages until the
    `item/completed` event)."""

    messages: list[dict] = field(default_factory=list)
    is_tool_iteration: bool = False
    final_text: Optional[str] = None  # Set when an agentMessage completes


class CodexEventProjector:
    """Stateful projector consuming Codex notifications in arrival order.

    Owns the in-progress reasoning content (codex emits reasoning as separate
    items but Hermes stashes it on the next assistant message)."""

    def __init__(self) -> None:
        self._pending_reasoning: list[str] = []
        self._completed_item_keys: set[str] = set()
        self._item_types_by_id: dict[str, str] = {}
        self._item_tool_identities_by_id: dict[str, tuple[str, ...]] = {}

    def project(self, notification: dict) -> ProjectionResult:
        """Project a single notification. Idempotent for non-completion events;
        only `item/completed` and `turn/completed` materialize messages."""
        method = notification.get("method", "")
        params = notification.get("params", {}) or {}

        # Bind every stable provider item id to its start-time type. A delayed
        # completion with the same id but a different type must not create a
        # second canonical tool occurrence or consume the valid completion.
        if method == "item/started":
            item = params.get("item") or {}
            if not isinstance(item, dict):
                return ProjectionResult()
            item_type = item.get("type")
            item_id = item.get("id")
            if (
                isinstance(item_type, str)
                and item_type.strip()
                and isinstance(item_id, str)
                and item_id.strip()
            ):
                self._item_types_by_id.setdefault(item_id, item_type)
                tool_identity = _provider_tool_identity(item_type, item)
                if tool_identity is not None:
                    self._item_tool_identities_by_id.setdefault(item_id, tool_identity)
            return ProjectionResult()

        # We only materialize messages on `item/completed`. Streaming deltas
        # (`item/<type>/outputDelta`, `item/<type>/delta`) are display-only and
        # don't enter the messages list — same way Hermes already only writes
        # the assistant message after the streaming completion event.
        if method != "item/completed":
            return ProjectionResult()

        item = params.get("item") or {}
        if not isinstance(item, dict):
            return ProjectionResult()
        item_type = item.get("type") or ""
        item_id = item.get("id") or ""

        # Codex may replay terminal notifications after reconnects. Persisting
        # the same completed item twice creates duplicate tool-call/result
        # pairs with an identical deterministic call id. Item ids are also the
        # only stable occurrence identity the protocol gives us. Drop malformed
        # id-less terminal events rather than collapsing distinct reasoning by
        # content or fabricating colliding tool ids.
        if not isinstance(item_type, str) or not item_type.strip():
            return ProjectionResult()
        if not isinstance(item_id, str) or not item_id.strip():
            return ProjectionResult()
        expected_item_type = self._item_types_by_id.get(item_id)
        if expected_item_type is not None and expected_item_type != item_type:
            return ProjectionResult()
        tool_identity = _provider_tool_identity(item_type, item)
        if item_type in {"mcpToolCall", "dynamicToolCall"} and tool_identity is None:
            return ProjectionResult()
        expected_tool_identity = self._item_tool_identities_by_id.get(item_id)
        if (
            expected_tool_identity is not None
            and expected_tool_identity != tool_identity
        ):
            return ProjectionResult()
        self._item_types_by_id.setdefault(item_id, item_type)
        if tool_identity is not None:
            self._item_tool_identities_by_id.setdefault(item_id, tool_identity)
        completed_key = f"id:{item_id}"
        if completed_key in self._completed_item_keys:
            return ProjectionResult()
        self._completed_item_keys.add(completed_key)

        if item_type == "agentMessage":
            return self._project_agent_message(item)
        if item_type == "reasoning":
            self._pending_reasoning.extend(item.get("summary") or [])
            self._pending_reasoning.extend(item.get("content") or [])
            return ProjectionResult()
        if item_type == "commandExecution":
            return self._project_command(item, item_id)
        if item_type == "fileChange":
            return self._project_file_change(item, item_id)
        if item_type == "mcpToolCall":
            return self._project_mcp_tool_call(item, item_id)
        if item_type == "dynamicToolCall":
            return self._project_dynamic_tool_call(item, item_id)
        if item_type == "userMessage":
            return self._project_user_message(item)

        # Unknown / rare items (plan, hookPrompt, collabAgentToolCall, etc.)
        # — record as opaque assistant note so memory review can still see
        # *something* happened, but don't fabricate tool_call structure.
        return self._project_opaque(item, item_type)

    # ---------- per-type projections ----------

    def _project_agent_message(self, item: dict) -> ProjectionResult:
        text = item.get("text") or ""
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if self._pending_reasoning:
            msg["reasoning"] = "\n".join(self._pending_reasoning)
            self._pending_reasoning = []
        return ProjectionResult(messages=[msg], final_text=text)

    def _project_user_message(self, item: dict) -> ProjectionResult:
        # codex's userMessage content is a list of UserInput variants. For
        # projection purposes we flatten any text fragments and ignore
        # non-text parts (images, etc.) — Hermes' messages store text only.
        text_parts: list[str] = []
        for fragment in item.get("content") or []:
            if isinstance(fragment, dict):
                if fragment.get("type") == "text":
                    text_parts.append(fragment.get("text") or "")
                elif "text" in fragment:
                    text_parts.append(str(fragment["text"]))
        return ProjectionResult(
            messages=[{"role": "user", "content": "\n".join(text_parts)}]
        )

    def _project_command(self, item: dict, item_id: str) -> ProjectionResult:
        call_id = _deterministic_call_id("exec", item_id)
        args = {
            "command": item.get("command") or "",
            "cwd": item.get("cwd") or "",
        }
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": _format_tool_args(args),
                    },
                }
            ],
        }
        if self._pending_reasoning:
            assistant_msg["reasoning"] = "\n".join(self._pending_reasoning)
            self._pending_reasoning = []
        output = item.get("aggregatedOutput") or ""
        exit_code = item.get("exitCode")
        if exit_code is not None and exit_code != 0:
            output = f"[exit {exit_code}]\n{output}"
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": output,
        }
        return ProjectionResult(
            messages=[assistant_msg, tool_msg], is_tool_iteration=True
        )

    def _project_file_change(self, item: dict, item_id: str) -> ProjectionResult:
        call_id = _deterministic_call_id("apply_patch", item_id)
        # Reduce the codex changes array to a digest the agent loop will
        # find readable. We record per-file change kinds (Add/Update/Delete)
        # without inlining full file contents — those can be huge.
        changes_summary = []
        for change in item.get("changes") or []:
            kind = (change.get("kind") or {}).get("type") or "update"
            path = change.get("path") or ""
            changes_summary.append({"kind": kind, "path": path})
        args = {"changes": changes_summary}
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": _format_tool_args(args),
                    },
                }
            ],
        }
        if self._pending_reasoning:
            assistant_msg["reasoning"] = "\n".join(self._pending_reasoning)
            self._pending_reasoning = []
        status = item.get("status") or "unknown"
        n = len(changes_summary)
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"apply_patch status={status}, {n} change(s)",
        }
        return ProjectionResult(
            messages=[assistant_msg, tool_msg], is_tool_iteration=True
        )

    def _project_mcp_tool_call(self, item: dict, item_id: str) -> ProjectionResult:
        tool_identity = _provider_tool_identity("mcpToolCall", item)
        if tool_identity is None:
            return ProjectionResult()
        server, tool = tool_identity
        # Mirror the native MCP tool-name convention (mcp__server__tool) so the
        # deterministic call_id input stays consistent with registration names.
        call_id = _deterministic_call_id(f"mcp__{server}__{tool}", item_id)
        args = item.get("arguments") or {}
        if not isinstance(args, dict):
            args = {"arguments": args}
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": f"mcp.{server}.{tool}",
                        "arguments": _format_tool_args(args),
                    },
                }
            ],
        }
        if self._pending_reasoning:
            assistant_msg["reasoning"] = "\n".join(self._pending_reasoning)
            self._pending_reasoning = []
        result = item.get("result")
        error = item.get("error")
        if error:
            content = f"[error] {json.dumps(error, ensure_ascii=False)[:1000]}"
        elif result is not None:
            content = json.dumps(result, ensure_ascii=False)[:4000]
        else:
            content = ""
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }
        return ProjectionResult(
            messages=[assistant_msg, tool_msg], is_tool_iteration=True
        )

    def _project_dynamic_tool_call(
        self, item: dict, item_id: str
    ) -> ProjectionResult:
        tool_identity = _provider_tool_identity("dynamicToolCall", item)
        if tool_identity is None:
            return ProjectionResult()
        (tool,) = tool_identity
        call_id = _deterministic_call_id(f"dyn_{tool}", item_id)
        args = item.get("arguments") or {}
        if not isinstance(args, dict):
            args = {"arguments": args}
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool,
                        "arguments": _format_tool_args(args),
                    },
                }
            ],
        }
        if self._pending_reasoning:
            assistant_msg["reasoning"] = "\n".join(self._pending_reasoning)
            self._pending_reasoning = []
        content_items = item.get("contentItems") or []
        if isinstance(content_items, list) and content_items:
            text_items = [
                entry.get("text")
                for entry in content_items
                if isinstance(entry, dict) and isinstance(entry.get("text"), str)
            ]
            content = "\n".join(text_items)[:4000] if text_items else json.dumps(
                content_items, ensure_ascii=False
            )[:4000]
        else:
            success = item.get("success")
            content = f"success={success}"
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }
        return ProjectionResult(
            messages=[assistant_msg, tool_msg], is_tool_iteration=True
        )

    def _project_opaque(self, item: dict, item_type: str) -> ProjectionResult:
        # Record the existence of the item without inventing tool_calls.
        # Memory review will see this and may or may not save anything.
        try:
            payload = json.dumps(item, ensure_ascii=False)[:1500]
        except (TypeError, ValueError):
            payload = repr(item)[:1500]
        return ProjectionResult(
            messages=[
                {
                    "role": "assistant",
                    "content": f"[codex {item_type}] {payload}",
                }
            ]
        )

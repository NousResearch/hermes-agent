"""Project observed Antigravity ``stream-json`` events into Hermes callbacks/history.

Only the documented stream envelopes (``init``, ``step_update`` and ``result``)
are interpreted.  Unknown payloads remain opaque: in particular, a permission or
approval-shaped event is reported as a fail-closed protocol error instead of being
mistaken for assistant text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _correlation(event: dict) -> tuple[Optional[int], Optional[int]]:
    """Keep the wire's ordering coordinates verbatim when they are scalar values."""
    sequence = event.get("sequence")
    step_index = event.get("step_index")
    return sequence if isinstance(sequence, int) else None, step_index if isinstance(step_index, int) else None


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _event_part(event: dict) -> dict:
    """Return the one observed update payload without guessing from arbitrary fields."""
    for key in ("delta", "step", "update"):
        part = event.get(key)
        if isinstance(part, dict):
            return part
    return {}


def _tool_id(name: str, sequence: Optional[int], step_index: Optional[int], item_id: Any) -> str:
    stable = str(item_id) if item_id is not None else hashlib.sha256(
        f"{name}:{sequence!r}:{step_index!r}".encode()
    ).hexdigest()[:16]
    return f"antigravity_{name}_{stable}"


def _tool_args(part: dict) -> dict[str, Any]:
    """Carry only source metadata that was actually present on the tool update."""
    args = part.get("arguments") or part.get("input") or part.get("parameters")
    if isinstance(args, dict):
        return dict(args)
    result: dict[str, Any] = {}
    for key in ("command", "cwd", "path", "paths", "file", "files"):
        if key in part:
            result[key] = part[key]
    return result


def _tool_result(part: dict) -> tuple[str, bool]:
    for key in ("output", "result", "text", "message", "error"):
        if key in part:
            value = part[key]
            if isinstance(value, str):
                return value, key == "error"
            return json.dumps(value, ensure_ascii=False, sort_keys=True), key == "error"
    return "", False


@dataclass
class ProjectionResult:
    """One source envelope's history/display projection and its exact correlation."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    assistant_delta: Optional[str] = None
    final_text: Optional[str] = None
    system_progress: Optional[str] = None
    tool_event: Optional[dict[str, Any]] = None
    is_tool_iteration: bool = False
    sequence: Optional[int] = None
    step_index: Optional[int] = None
    error: Optional[str] = None


class AntigravityEventProjector:
    """Stateful, fail-closed translator for the small observed stream-json contract."""

    def __init__(self) -> None:
        self._active_tools: dict[str, tuple[str, dict[str, Any], Optional[int], Optional[int]]] = {}
        self.projected_messages: list[dict[str, Any]] = []
        self.final_text = ""
        self._conversation_id: str | None = None

    def feed(self, event: dict[str, Any]) -> ProjectionResult:
        """Project one wire event and retain its durable transcript rows."""
        projected = self.project(event)
        self.projected_messages.extend(projected.messages)
        if projected.final_text is not None:
            self.final_text = projected.final_text
        return projected

    def project(self, event: dict[str, Any]) -> ProjectionResult:
        if not isinstance(event, dict):
            return ProjectionResult()
        wire_step = event.get("step_update")
        correlation_source = wire_step if isinstance(wire_step, dict) else event
        sequence, step_index = _correlation(correlation_source)
        result = ProjectionResult(sequence=sequence, step_index=step_index)
        event_type = event.get("event") or event.get("type")
        if event_type == "init":
            conversation_id = event.get("conversation_id")
            if isinstance(conversation_id, str) and conversation_id:
                self._conversation_id = conversation_id
            return result
        if event_type == "result":
            return self._project_result(event, result)
        if event_type != "step_update":
            return self._unknown(event, result)
        return self._project_step(event, result)

    def _project_result(self, event: dict, result: ProjectionResult) -> ProjectionResult:
        raw_payload = event.get("result")
        payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else event
        text = (_text(payload.get("response")) or _text(payload.get("text"))
                or _text(payload.get("final_text")) or _text(payload.get("final_response")))
        if text is None:
            return result
        result.final_text = text
        message = self._assistant(text, result)
        conversation_id = (payload.get("conversation_id") or payload.get("conversationId")
                           or event.get("conversation_id") or self._conversation_id)
        if isinstance(conversation_id, str) and conversation_id:
            self._conversation_id = conversation_id
            message["_antigravity"] = {"conversation_id": conversation_id}
        result.messages.append(message)
        return result

    def _project_step(self, event: dict, result: ProjectionResult) -> ProjectionResult:
        wire = event.get("step_update")
        part = wire if isinstance(wire, dict) else _event_part(event)
        kind = part.get("step_type") or part.get("type") or part.get("kind")
        if kind in {"agent_response", "assistant", "assistant_delta"}:
            result.assistant_delta = (_text(part.get("text_delta")) or _text(part.get("text"))
                                      or _text(part.get("delta")) or _text(part.get("content")))
            return result
        if kind in {"system_message", "system", "progress", "system_progress"}:
            result.system_progress = (_text(part.get("text_delta")) or _text(part.get("text"))
                                      or _text(part.get("message")) or _text(part.get("content")))
            return result
        if kind in {"tool", "tool_use", "tool_call"}:
            normalized = dict(part)
            normalized.setdefault("status", part.get("state"))
            normalized.setdefault("name", part.get("tool_name"))
            tool_info = part.get("tool_info")
            if isinstance(tool_info, dict):
                for key, value in tool_info.items():
                    normalized.setdefault(key, value)
            return self._project_tool(normalized, result)
        if kind in {"approval", "permission"}:
            return self._unknown(part, result)
        return self._unknown(event, result)

    def _project_tool(self, part: dict, result: ProjectionResult) -> ProjectionResult:
        status = part.get("status")
        name = _text(part.get("name")) or _text(part.get("tool_name")) or _text(part.get("tool"))
        if status not in {"ACTIVE", "DONE", "ERROR"} or not name:
            return self._unknown(part, result)
        call_id = _tool_id(name, result.sequence, result.step_index, part.get("id") or part.get("tool_id"))
        args = _tool_args(part)
        if status == "ACTIVE":
            self._active_tools[call_id] = (name, args, result.sequence, result.step_index)
        else:
            name, args, _, _ = self._active_tools.pop(call_id, (name, args, result.sequence, result.step_index))
        output, output_error = _tool_result(part)
        result.tool_event = {"status": status, "id": call_id, "name": name, "args": args, "output": output,
                             "is_error": status == "ERROR" or output_error}
        if status != "ACTIVE":
            result.messages.extend([
                self._assistant(None, result, tool_call={"id": call_id, "name": name, "args": args}),
                {"role": "tool", "tool_call_id": call_id, "content": output,
                 **self._wire_fields(result)},
            ])
            result.is_tool_iteration = True
        return result

    @staticmethod
    def _wire_fields(result: ProjectionResult) -> dict[str, int]:
        fields: dict[str, int] = {}
        if result.sequence is not None:
            fields["sequence"] = result.sequence
        if result.step_index is not None:
            fields["step_index"] = result.step_index
        return fields

    def _assistant(self, content: Optional[str], result: ProjectionResult, tool_call: Optional[dict] = None) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content, **self._wire_fields(result)}
        if tool_call is not None:
            message["tool_calls"] = [{"id": tool_call["id"], "type": "function", "function": {
                "name": tool_call["name"], "arguments": json.dumps(tool_call["args"], ensure_ascii=False, sort_keys=True),
            }}]
        return message

    def _unknown(self, event: dict, result: ProjectionResult) -> ProjectionResult:
        # Only protocol discriminators can request approval. Never inspect model/tool text:
        # untrusted output containing "permission" must not be able to abort a turn.
        raw_step = event.get("step_update")
        step: dict[str, Any] = raw_step if isinstance(raw_step, dict) else {}
        discriminators = {
            str(event.get("event") or "").lower(),
            str(event.get("type") or "").lower(),
            str(step.get("step_type") or "").lower(),
            str(step.get("tool_name") or "").lower(),
        }
        if any(label in {"approval", "approval_requested", "permission", "permission_requested"}
               for label in discriminators):
            result.error = "unsupported Antigravity approval/permission event; refusing by default"
            logger.warning("%s", result.error)
        else:
            logger.debug("ignoring unknown Antigravity stream-json event type=%r", event.get("type"))
        return result


def make_antigravity_event_bridge(agent: Any, *, on_protocol_error: Optional[Callable[[str], None]] = None) -> Callable[..., None]:
    """Build a guarded live-display bridge with the canonical Hermes callback contract."""
    active: dict[str, tuple[str, dict[str, Any], float]] = {}
    projector = AntigravityEventProjector()

    def guarded(attr: str, *args: Any, **kwargs: Any) -> None:
        callback = getattr(agent, attr, None)
        if callable(callback):
            try:
                callback(*args, **kwargs)
            except Exception:
                logger.debug("Antigravity %s callback raised", attr, exc_info=True)

    def on_event(event: dict, projected: Optional[ProjectionResult] = None) -> None:
        if projected is None:
            projected = projector.feed(event)
        if projected.error:
            if on_protocol_error is not None:
                try:
                    on_protocol_error(projected.error)
                except Exception:
                    logger.debug("Antigravity protocol-error callback raised", exc_info=True)
            return
        if projected.assistant_delta:
            guarded("_fire_stream_delta", projected.assistant_delta)
        if projected.system_progress:
            guarded("_fire_reasoning_delta", projected.system_progress)
        tool = projected.tool_event
        if tool is None:
            if projected.final_text and getattr(agent, "show_commentary", True):
                guarded("_emit_interim_assistant_message", {"role": "assistant", "content": projected.final_text})
            return
        status, call_id, name, args = tool["status"], tool["id"], tool["name"], tool["args"]
        if status == "ACTIVE":
            active[call_id] = (name, args, time.monotonic())
            guarded("tool_progress_callback", "tool.started", name, None, args, tool_call_id=call_id)
            guarded("tool_start_callback", call_id, name, args)
        else:
            prior = active.pop(call_id, None)
            duration = time.monotonic() - prior[2] if prior else None
            guarded("tool_progress_callback", "tool.completed", name, None, None, duration=duration,
                    is_error=tool["is_error"], result=tool["output"], tool_call_id=call_id)
            guarded("tool_complete_callback", call_id, name, args, tool["output"])

    setattr(on_event, "_accepts_projected", True)
    return on_event

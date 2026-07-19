"""Project Claude Code stream-json events into Hermes text + tool activity.

Phase 2a: text deltas, final result, and MCP tool_use / tool_result events.

Claude Code stream-json (observed + documented shapes):

* ``{"type":"stream_event","event":{"type":"content_block_delta",
  "delta":{"type":"text_delta","text":"..."}}}`` — partial assistant text
  (requires ``--include-partial-messages``).
* ``{"type":"assistant","message":{"content":[{"type":"text","text":"..."},
  {"type":"tool_use","id":"...","name":"mcp__hermes-tools__terminal",
   "input":{...}}]}}`` — assistant snapshot, may include tool_use blocks.
* ``{"type":"user","message":{"content":[{"type":"tool_result",
  "tool_use_id":"...","content":"..."}]}}`` — tool results (MCP returns
  these after Hermes executes the tool).
* ``{"type":"stream_event","event":{"type":"content_block_start",
  "content_block":{"type":"tool_use",...}}}`` — streaming tool_use start.
* ``{"type":"result","subtype":"success"|"error", "is_error":bool,
  "result":"...", "usage":{...}, "session_id":"..."}`` — terminal event.
* ``{"type":"system", ...}`` / other — ignored for projection (except session_id).

Text projection rules (anti-garble):
  * Only ``text_delta`` partials feed the live stream (not thinking_delta,
    signature_delta, input_json_delta, content_block_start seeds, or
    message_delta envelopes — those caused mis-joins / duplication).
  * Each new ``message_start`` resets the per-message stream buffer so
    intermediate tool-loop narration is not concatenated onto the final
    answer.
  * ``final_text`` prefers the terminal ``result`` field, then the latest
    assistant snapshot, then the current message's streamed text — never
    a stitched multi-message partial buffer.

Tool round-trip is internal to ``claude -p`` + the hermes-tools MCP server.
This projector only *observes* tool activity for Hermes UI / counters; it
does not execute or proxy tools.

Mirrors ``codex_event_projector.CodexEventProjector`` in role (stateful
accumulator + delta extraction) without inventing a shared abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent.transports.claude_cli import (
    HERMES_MCP_TOOL_PREFIX,
    strip_hermes_mcp_tool_prefix,
)


@dataclass
class ClaudeToolCallRecord:
    """One observed tool_use (+ optional later tool_result) in the turn."""

    id: str
    name: str  # bare Hermes name when mcp__hermes-tools__* stripped
    raw_name: str  # full mcp__… name as Claude emitted it
    input: dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    is_error: bool = False
    started: bool = True
    completed: bool = False


@dataclass
class ClaudeProjectionState:
    """Accumulated state after consuming zero or more stream-json events."""

    # Streaming partial text for the *current* assistant message only
    # (reset on each message_start). Used for live deltas + final fallback.
    streamed_text: str = ""
    # Last complete assistant message text (from type=assistant snapshots).
    assistant_text: str = ""
    # Terminal result fields.
    result_text: str = ""
    is_error: bool = False
    result_event: Optional[dict[str, Any]] = None
    usage: Optional[dict[str, Any]] = None
    session_id: Optional[str] = None
    total_cost_usd: Optional[float] = None
    finished: bool = False
    # Text deltas emitted during the last ``consume`` call (for stream bridge).
    last_text_deltas: list[str] = field(default_factory=list)
    # Tool activity observed this consume() (for stream bridge).
    last_tool_started: list[ClaudeToolCallRecord] = field(default_factory=list)
    last_tool_completed: list[ClaudeToolCallRecord] = field(default_factory=list)
    # Full turn tool ledger (order of first observation).
    tool_calls: list[ClaudeToolCallRecord] = field(default_factory=list)
    _tool_by_id: dict[str, ClaudeToolCallRecord] = field(default_factory=dict, repr=False)
    # True once any text_delta landed for the current assistant message —
    # prevents seeding streamed_text from a full assistant snapshot that
    # would double-emit relative to already-streamed partials.
    _message_has_streamed_deltas: bool = field(default=False, repr=False)

    @property
    def final_text(self) -> str:
        """Best available assistant text for the completed turn.

        Prefer the terminal ``result`` string (Claude's authoritative final
        answer), then the latest full assistant snapshot, then the current
        message's streamed partials. This avoids concatenating intermediate
        tool-loop narration with the final reply.
        """
        for candidate in (self.result_text, self.assistant_text, self.streamed_text):
            if candidate and str(candidate).strip():
                return str(candidate)
        return self.streamed_text or self.assistant_text or self.result_text or ""

    @property
    def tool_iterations(self) -> int:
        """Count of completed tool calls (Hermes skill-nudge counter)."""
        return sum(1 for t in self.tool_calls if t.completed)


def _extract_text_from_content_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        # Strict: only real text blocks. Never thinking / tool_use / redacted.
        if btype == "text" and "text" in block:
            parts.append(str(block.get("text") or ""))
        elif btype is None and "text" in block and "name" not in block:
            # Defensive legacy shape without type, but not a tool_use.
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


def extract_text_delta(event: dict[str, Any]) -> Optional[str]:
    """Return a text delta string if this event carries partial assistant text.

    Only ``text_delta`` content is accepted. thinking_delta, signature_delta,
    input_json_delta, content_block_start seeds, and message_delta envelopes
    are intentionally ignored — joining them produces garbled projections
    (duplicate seeds, HTML-ish fragments, binary signature junk).
    """
    etype = event.get("type")

    # Nested stream_event envelope used by --include-partial-messages.
    if etype == "stream_event":
        inner = event.get("event") or {}
        if not isinstance(inner, dict):
            return None
        return extract_text_delta(inner)

    # Anthropic Messages streaming shapes — text only.
    if etype == "content_block_delta":
        delta = event.get("delta") or {}
        if isinstance(delta, dict):
            dtype = delta.get("type")
            if dtype == "text_delta" and "text" in delta:
                text = delta.get("text")
                # Preserve empty string? No — empty contributes nothing.
                return str(text) if text else None
        return None

    # Direct text_delta only (no message_delta — that carries stop_reason/usage).
    if etype == "text_delta":
        delta = event.get("delta") if "delta" in event else event.get("text")
        if isinstance(delta, dict) and "text" in delta:
            text = delta.get("text")
            return str(text) if text else None
        if isinstance(delta, str) and delta:
            return delta
    return None


def extract_assistant_message_text(event: dict[str, Any]) -> Optional[str]:
    """Pull full assistant text from a type=assistant snapshot event."""
    if event.get("type") != "assistant":
        return None
    message = event.get("message")
    if isinstance(message, dict):
        text = _extract_text_from_content_blocks(message.get("content"))
        return text if text else None
    # Some shapes put content at the top level.
    text = _extract_text_from_content_blocks(event.get("content"))
    return text if text else None


def extract_usage(event: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Normalize usage from a result (or message) event into a flat dict."""
    raw = event.get("usage")
    if not isinstance(raw, dict):
        message = event.get("message")
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            raw = message["usage"]
        else:
            return None
    # Claude result usage keys: input_tokens, output_tokens,
    # cache_creation_input_tokens, cache_read_input_tokens, ...
    return dict(raw)


def _tool_input_as_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None:
        return {}
    if isinstance(raw, str):
        # Streaming partial JSON sometimes lands as a string; keep opaque.
        return {"_raw": raw} if raw else {}
    return {"_raw": raw}


def _tool_result_as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {None, "text"} and "text" in item:
                    parts.append(str(item.get("text") or ""))
                elif "content" in item:
                    parts.append(str(item.get("content") or ""))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        return str(content)
    return str(content)


def extract_tool_use_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return tool_use block dicts from assistant / stream_event shapes."""
    blocks: list[dict[str, Any]] = []
    etype = event.get("type")

    if etype == "stream_event":
        inner = event.get("event") or {}
        if isinstance(inner, dict):
            return extract_tool_use_blocks(inner)
        return blocks

    if etype == "content_block_start":
        block = event.get("content_block") or {}
        if isinstance(block, dict) and block.get("type") == "tool_use":
            blocks.append(block)
        return blocks

    if etype == "assistant":
        message = event.get("message") if isinstance(event.get("message"), dict) else event
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    blocks.append(block)
        return blocks

    # Top-level tool_use (defensive).
    if etype == "tool_use" or (
        event.get("type") is None and event.get("name") and event.get("id")
    ):
        blocks.append(event)
    return blocks


def extract_tool_result_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return tool_result block dicts from user / stream shapes."""
    blocks: list[dict[str, Any]] = []
    etype = event.get("type")

    if etype == "stream_event":
        inner = event.get("event") or {}
        if isinstance(inner, dict):
            return extract_tool_result_blocks(inner)
        return blocks

    if etype in {"user", "tool_result"}:
        if etype == "tool_result":
            blocks.append(event)
            return blocks
        message = event.get("message") if isinstance(event.get("message"), dict) else event
        content = message.get("content") if isinstance(message, dict) else event.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    blocks.append(block)
        elif isinstance(content, dict) and content.get("type") == "tool_result":
            blocks.append(content)
        return blocks

    return blocks


def is_hermes_mcp_tool_name(name: str) -> bool:
    """True when the tool name is under the Hermes MCP server prefix."""
    if not isinstance(name, str) or not name:
        return False
    return name.startswith(HERMES_MCP_TOOL_PREFIX) or name.startswith("mcp__hermes-tools__")


def _is_message_start(event: dict[str, Any]) -> bool:
    """True for stream_event / bare message_start (new assistant generation)."""
    etype = event.get("type")
    if etype == "message_start":
        return True
    if etype == "stream_event":
        inner = event.get("event")
        if isinstance(inner, dict) and inner.get("type") == "message_start":
            return True
    return False


class ClaudeEventProjector:
    """Stateful projector: feed stream-json events, accumulate text + tools + result."""

    def __init__(self) -> None:
        self.state = ClaudeProjectionState()

    def _note_tool_use(self, block: dict[str, Any]) -> Optional[ClaudeToolCallRecord]:
        raw_name = str(block.get("name") or "unknown")
        tool_id = str(block.get("id") or block.get("tool_use_id") or "")
        if not tool_id:
            # Synthesize a stable-ish id so we can still track completions.
            tool_id = f"anon-{len(self.state.tool_calls)}-{raw_name}"
        existing = self.state._tool_by_id.get(tool_id)
        if existing is not None:
            # Update input if a fuller snapshot arrives later.
            inp = _tool_input_as_dict(block.get("input") or block.get("arguments"))
            if inp and (not existing.input or existing.input == {"_raw": ""}):
                existing.input = inp
            return None  # already started
        rec = ClaudeToolCallRecord(
            id=tool_id,
            name=strip_hermes_mcp_tool_prefix(raw_name),
            raw_name=raw_name,
            input=_tool_input_as_dict(block.get("input") or block.get("arguments")),
            started=True,
            completed=False,
        )
        self.state._tool_by_id[tool_id] = rec
        self.state.tool_calls.append(rec)
        self.state.last_tool_started.append(rec)
        return rec

    def _note_tool_result(self, block: dict[str, Any]) -> Optional[ClaudeToolCallRecord]:
        tool_id = str(block.get("tool_use_id") or block.get("id") or "")
        if not tool_id:
            return None
        rec = self.state._tool_by_id.get(tool_id)
        if rec is None:
            # Result arrived without a prior tool_use (resume / race). Create stub.
            raw_name = str(block.get("name") or "unknown")
            rec = ClaudeToolCallRecord(
                id=tool_id,
                name=strip_hermes_mcp_tool_prefix(raw_name),
                raw_name=raw_name,
                started=True,
                completed=False,
            )
            self.state._tool_by_id[tool_id] = rec
            self.state.tool_calls.append(rec)
        rec.result = _tool_result_as_text(block.get("content") if "content" in block else block.get("result"))
        rec.is_error = bool(block.get("is_error"))
        # tool_use_error strings from Claude (race before MCP connects) count as error.
        if rec.result and "<tool_use_error>" in rec.result:
            rec.is_error = True
        if not rec.completed:
            rec.completed = True
            self.state.last_tool_completed.append(rec)
        return rec

    def consume(self, event: dict[str, Any]) -> ClaudeProjectionState:
        """Incorporate one event. Returns the shared state (mutated in place)."""
        self.state.last_text_deltas = []
        self.state.last_tool_started = []
        self.state.last_tool_completed = []
        if not isinstance(event, dict):
            return self.state

        etype = event.get("type")

        # New assistant generation → reset per-message stream buffer so
        # intermediate tool-loop text is not concatenated into final_text.
        if _is_message_start(event):
            self.state.streamed_text = ""
            self.state._message_has_streamed_deltas = False

        # Streaming partials (strict text_delta only).
        delta = extract_text_delta(event)
        if delta:
            self.state.streamed_text += delta
            self.state.last_text_deltas.append(delta)
            self.state._message_has_streamed_deltas = True

        # Full assistant snapshots (overwrite with latest complete text).
        assistant = extract_assistant_message_text(event)
        if assistant is not None:
            self.state.assistant_text = assistant
            # Seed stream only when this message never saw partials (e.g.
            # partials disabled). Never overwrite/duplicate after deltas.
            if not self.state._message_has_streamed_deltas and not self.state.streamed_text:
                self.state.streamed_text = assistant

        # Tool use / result (MCP hermes-tools + any other tools Claude emits).
        for block in extract_tool_use_blocks(event):
            self._note_tool_use(block)
        for block in extract_tool_result_blocks(event):
            self._note_tool_result(block)

        # Session id may appear on many event types.
        sid = event.get("session_id")
        if isinstance(sid, str) and sid:
            self.state.session_id = sid

        # Terminal result.
        if etype == "result":
            self.state.finished = True
            self.state.result_event = event
            self.state.is_error = bool(event.get("is_error"))
            result_text = event.get("result")
            if isinstance(result_text, str):
                self.state.result_text = result_text
            usage = extract_usage(event)
            if usage is not None:
                self.state.usage = usage
            cost = event.get("total_cost_usd")
            if isinstance(cost, (int, float)):
                self.state.total_cost_usd = float(cost)
            # subtype=error without is_error flag still counts as error.
            subtype = str(event.get("subtype") or "").lower()
            if subtype in {"error", "failure", "failed", "error_max_turns"}:
                self.state.is_error = True

        return self.state

    def consume_line(self, line: str) -> ClaudeProjectionState:
        from agent.transports.claude_cli import parse_stream_json_line

        event = parse_stream_json_line(line)
        if event is None:
            self.state.last_text_deltas = []
            self.state.last_tool_started = []
            self.state.last_tool_completed = []
            return self.state
        return self.consume(event)

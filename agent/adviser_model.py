"""Adviser model loop: one private conversation per adviser, tools included.

Port of oh-my-pi's ``advisor/advise-tool.ts`` plus the prompt-cycle half of
``advisor/runtime.ts``. The loop is deliberately small and stateful:

* every review appends its delta batch as *separate* user messages, so a provider's
  prefix cache can hit every earlier message (one ever-growing message would
  invalidate the whole batch on every turn);
* the only side channel out is the ``advise`` tool, admitted by
  :class:`EmissionGuard`; read-only tools come from the Hermes registry and their
  output is returned as ordinary tool results;
* a response is quarantined before it enters the adviser transcript when it
  contains output-only destructive directives, and a repeated identical tool call
  is corrected once and then aborts the update.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent.adviser_emission import (
    DEFAULT_BUDGET_PER_UPDATE,
    EmissionGuard,
    severity_rank,
)
from agent.adviser_prompts import ADVISE_TOOL_DESCRIPTION

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
LOOP_GUARD_THRESHOLD = 3
MAX_QUARANTINE_RETRIES = 2

ACK_SENT = "Delivered."
ACK_DEFERRED = "Queued for the end of the turn. Do not re-raise."
ACK_SUPPRESSED = {
    "empty": "Dropped: empty note.",
    "noise": "Dropped: nothing actionable.",
    "duplicate": "Dropped: already raised.",
    "rate-limit": "Dropped: this update's advice budget is spent.",
}

ADVISE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "advise",
        "description": ADVISE_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "One concrete piece of advice for the agent you are watching. "
                                   "Terse, specific, actionable.",
                },
                "severity": {
                    "type": "string",
                    "enum": ["nit", "concern", "blocker"],
                    "description": "How strongly to weigh this. Omit for a plain nit.",
                },
            },
            "required": ["note"],
        },
    },
}


class AdviserTurnError(RuntimeError):
    """The adviser turn produced no assistant response (not a provider failure)."""


class AdviserOutputQuarantined(RuntimeError):
    """Generated output carried output-only destructive directives; the turn was discarded."""


class AdviserContextOverflow(RuntimeError):
    """The adviser's own context no longer fits its model window."""


class AdviserUsageLimit(RuntimeError):
    """Provider quota/rate-limit: latch the runtime until an explicit reset."""


class AdviserPermanentError(RuntimeError):
    """A request shape/model the provider will never accept for this configuration."""


class AdviserLoopAborted(RuntimeError):
    """The adviser ignored a tool-call loop correction."""


_PERMANENT_RE = re.compile(
    r"invalid_request_error|model[_ ]not[_ ]found|is not supported when|does not exist", re.IGNORECASE,
)
_QUOTA_RE = re.compile(
    r"quota|rate ?limit|usage limit|insufficient_quota|too many requests|429\b", re.IGNORECASE,
)
_OVERFLOW_RE = re.compile(
    r"context (?:length|window)|maximum context|too many tokens|prompt is too long|"
    r"input is too long|exceeds the context|context_length_exceeded", re.IGNORECASE,
)
_REFUSAL_RE = re.compile(r"content[_ ]policy|content filter|refus(?:ed|al)|blocked by.*safety", re.IGNORECASE)

_HAZARDS = (
    ("account-deletion claim", re.compile(r"\buser\b.{0,80}\b(?:deleted|erased)\b.{0,80}\baccount\b", re.IGNORECASE | re.DOTALL)),
    ("instruction override", re.compile(r"\bignore\s+(?:all\s+)?(?:prior|previous|earlier)\s+(?:user\s+)?instructions\b", re.IGNORECASE)),
    ("destructive shell command", re.compile(r"\brm\s+(?=(?:-[a-z]+\s*)*-[a-z]*r[a-z]*)(?=(?:-[a-z]+\s*)*-[a-z]*f[a-z]*)(?:-[a-z]+\s*)+", re.IGNORECASE)),
    ("denial instruction", re.compile(r"\bdeny\s+(?:this|it|the\s+request)\s+if\s+(?:asked|questioned)\b", re.IGNORECASE)),
)


def classify_error(exc: BaseException) -> str:
    """``quota`` | ``overflow`` | ``refusal`` | ``permanent`` | ``transient``."""
    message = str(exc or "")
    if isinstance(exc, AdviserUsageLimit):
        return "quota"
    if isinstance(exc, AdviserContextOverflow):
        return "overflow"
    if _QUOTA_RE.search(message):
        return "quota"
    if _OVERFLOW_RE.search(message):
        return "overflow"
    if _REFUSAL_RE.search(message):
        return "refusal"
    if _PERMANENT_RE.search(message):
        return "permanent"
    return "transient"


def quarantine_generated(generated: str, source_text: str) -> Optional[str]:
    """Return the quarantine reason when *generated* carries a hazard its source does not."""
    labels: list[str] = []
    for label, pattern in _HAZARDS:
        if pattern.search(generated) and not pattern.search(source_text):
            labels.append(label)
    if "destructive shell command" in labels and "instruction override" not in labels:
        if re.search(r"\bignore\s+(?:all\s+)?(?:prior|previous|earlier)\s+(?:user\s+)?instructions\b", generated, re.IGNORECASE):
            labels.append("instruction override")
    if "destructive shell command" in labels or len(labels) >= 3:
        return ", ".join(labels)
    return None


@dataclass
class _DeferredNote:
    key: str
    note: str
    severity: Optional[str]
    turn: Optional[int]


def _normalize_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_call_parts(call: Any, index: int) -> tuple[str, str, dict[str, Any]]:
    """``(id, name, args)`` from an SDK object or an already-normalized dict."""
    if isinstance(call, dict):
        call_id = str(call.get("id") or f"call_{index}")
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = str(function.get("name") or "")
        raw = function.get("arguments")
    else:
        call_id = str(getattr(call, "id", "") or f"call_{index}")
        function = getattr(call, "function", None)
        name = str(getattr(function, "name", "") or "")
        raw = getattr(function, "arguments", None)
    return call_id, name, _normalize_arguments(raw)


class AdviseTool:
    """The adviser's only outward channel: admit, defer, and acknowledge every note."""

    def __init__(self, on_advice: Callable[[str, Optional[str], Optional[int], bool], None],
                 guard: Optional[EmissionGuard] = None) -> None:
        self._on_advice = on_advice
        self._guard = guard or EmissionGuard()
        self._in_progress = False
        self._covered_turn: Optional[int] = None
        self._terminal = False
        self._deferred: list[_DeferredNote] = []

    @property
    def guard(self) -> EmissionGuard:
        return self._guard

    def begin_update(self, in_progress: bool, covered_turn: Optional[int] = None,
                     terminal: bool = False) -> None:
        was_in_progress = self._in_progress
        self._in_progress = bool(in_progress)
        self._covered_turn = covered_turn
        self._terminal = bool(terminal)
        self._guard.begin_update()
        if was_in_progress and not in_progress:
            self._flush_deferred()

    def flush_deferred_notes(self) -> None:
        self._in_progress = False
        self._flush_deferred()

    def reset(self) -> None:
        self._guard.reset()
        self._in_progress = False
        self._deferred = []

    def execute(self, args: dict[str, Any]) -> str:
        note = str(args.get("note") or "")
        severity = str(args.get("severity") or "nit")
        if severity not in ("nit", "concern", "blocker"):
            severity = "nit"
        rank = severity_rank(severity)
        key = _note_key(note)
        if self._in_progress and severity != "blocker":
            return self._defer(key, note, severity, rank)
        reserved = next((item for item in self._deferred if item.key == key), None)
        if reserved is not None:
            self._deferred.remove(reserved)
        decision = self._guard.admit(note, rank=rank, pending=False)
        if not decision.accepted:
            return ACK_SUPPRESSED.get(decision.reason, ACK_SUPPRESSED["duplicate"])
        self._on_advice(note, severity, self._covered_turn, self._terminal)
        return ACK_SENT

    def _defer(self, key: str, note: str, severity: str, rank: int) -> str:
        pending = next((item for item in self._deferred if item.key == key), None)
        if pending is not None:
            if rank > severity_rank(pending.severity):
                pending.severity = severity
                self._guard.escalate_pending(note, rank)
            return ACK_DEFERRED
        decision = self._guard.admit(note, rank=rank, pending=True)
        if not decision.accepted:
            return ACK_SUPPRESSED.get(decision.reason, ACK_SUPPRESSED["duplicate"])
        if decision.displaced_key is not None:
            self._deferred = [item for item in self._deferred if item.key != decision.displaced_key]
        self._deferred.append(_DeferredNote(key, note, severity, self._covered_turn))
        return ACK_DEFERRED

    def _flush_deferred(self) -> None:
        if not self._deferred:
            return
        pending, self._deferred = self._deferred, []
        for item in pending:
            self._guard.mark_routed(item.note)
            self._on_advice(item.note, item.severity,
                            item.turn if item.turn is not None else self._covered_turn, self._terminal)


def _note_key(note: str) -> str:
    from agent.adviser_emission import normalize_note

    return normalize_note(note)


class AdviserModel:
    """Stateful adviser conversation, one instance per configured adviser."""

    def __init__(self, *, system_blocks: list[str], tools: list[dict[str, Any]],
                 advise: AdviseTool, run_tool: Callable[[str, dict[str, Any]], str],
                 call_model: Callable[..., Any], on_record: Callable[[dict[str, Any]], None],
                 include_reasoning: bool = True) -> None:
        self._system = "\n\n".join(blocks for blocks in system_blocks if blocks)
        self._tools = [ADVISE_SCHEMA, *tools]
        self._advise = advise
        self._run_tool = run_tool
        self._call_model = call_model
        self._on_record = on_record
        self._include_reasoning = include_reasoning
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": self._system}]
        self._loop_counts: dict[str, int] = {}
        self._loop_corrected = False

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": self._system}]
        self._loop_counts.clear()
        self._loop_corrected = False

    def run(self, deltas: list[str], *, wip: bool, covered_turn: Optional[int] = None,
            terminal: bool = False) -> None:
        """Run one review cycle over already-rendered delta batches."""
        if not deltas:
            return
        self._advise.begin_update(wip, covered_turn=covered_turn, terminal=terminal)
        for text in deltas:
            self._append({"role": "user", "content": text})
        for _round in range(MAX_TOOL_ROUNDS):
            response = self._call_model(messages=self._messages, tools=self._tools)
            content, calls = _read_response(response)
            source_text = _source_text(self._messages, deltas)
            reason = quarantine_generated(content, source_text)
            if reason:
                raise AdviserOutputQuarantined(f"generated output-only destructive directives: {reason}")
            assistant: dict[str, Any] = {"role": "assistant", "content": content or ""}
            if calls:
                assistant["tool_calls"] = calls
            self._append(assistant)
            if not calls:
                return
            for call in calls:
                call_id, name, args = _tool_call_parts(call, 0)
                result = self._dispatch(name, args)
                self._append({
                    "role": "tool", "tool_call_id": call_id, "name": name or "tool", "content": result,
                })
        raise AdviserLoopAborted(f"adviser exceeded {MAX_TOOL_ROUNDS} tool rounds")

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        if name == "advise":
            return self._advise.execute(args)
        signature = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        self._loop_counts[signature] = self._loop_counts.get(signature, 0) + 1
        if self._loop_counts[signature] >= LOOP_GUARD_THRESHOLD:
            if self._loop_corrected:
                raise AdviserLoopAborted(f"adviser repeated {name} after a loop redirect")
            self._loop_corrected = True
            self._loop_counts[signature] = 0
            self._append({
                "role": "user",
                "content": f"You have called {name} repeatedly with identical arguments. "
                           "Change approach or finish; identical calls will not produce new information.",
            })
        try:
            return str(self._run_tool(name, args))
        except Exception as exc:  # health: allow BLE001 -- a tool failure is a result, never a loop failure
            logger.debug("adviser tool %s failed", name, exc_info=True)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def _append(self, message: dict[str, Any]) -> None:
        self._messages.append(message)
        try:
            self._on_record(message)
        except Exception:
            logger.debug("adviser recorder failed", exc_info=True)

    def clear_loop_guard(self) -> None:
        self._loop_counts.clear()
        self._loop_corrected = False


def _read_response(response: Any) -> tuple[str, list[dict[str, Any]]]:
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, TypeError):
        raise AdviserTurnError("adviser provider returned no choices") from None
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(message, dict):
        content = message.get("content")
    raw_calls = getattr(message, "tool_calls", None) if message is not None else None
    if raw_calls is None and isinstance(message, dict):
        raw_calls = message.get("tool_calls")
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls or ()):
        call_id, name, _args = _tool_call_parts(call, index)
        function = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
        raw = function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", None)
        calls.append({"id": call_id, "type": "function",
                      "function": {"name": name, "arguments": raw if isinstance(raw, str) else json.dumps(raw or {})}})
    return str(content or ""), calls


def _source_text(messages: list[dict[str, Any]], deltas: list[str]) -> str:
    parts = list(deltas)
    for message in messages:
        if message.get("role") == "tool":
            parts.append(str(message.get("content") or ""))
    return "\n".join(parts)


__all__ = [
    "ACK_DEFERRED",
    "ACK_SENT",
    "ACK_SUPPRESSED",
    "ADVISE_SCHEMA",
    "MAX_QUARANTINE_RETRIES",
    "AdviseTool",
    "AdviserContextOverflow",
    "AdviserLoopAborted",
    "AdviserModel",
    "AdviserOutputQuarantined",
    "AdviserPermanentError",
    "AdviserTurnError",
    "AdviserUsageLimit",
    "classify_error",
    "quarantine_generated",
]


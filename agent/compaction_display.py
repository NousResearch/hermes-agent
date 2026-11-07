"""Client-facing projection helpers for model-only compaction carriers."""

from __future__ import annotations

import re

from typing import Any, Dict, Optional

from agent.context_compressor import (
    _LEAN_RECOVERY_HEADING, _LEAN_USER_MESSAGES_HEADING, ContextCompressor, is_compaction_summary_message,
)
from agent.message_content import flatten_message_text


_COMPACTION_INTERNAL_FIELDS = (
    "tool_calls",
    "finish_reason",
    "reasoning",
    # Provider replay/metadata fields that ride the wire on every request but are invisible to
    # ``msg["content"]``/``msg["tool_calls"]`` accounting. Codex Responses sessions in particular carry
    # ``codex_reasoning_items`` blobs of ``encrypted_content`` that can dominate the serialized session (a
    # measured 214-turn session held ~115K tokens / 27% of its payload there — #55572).
    # ``reasoning_details`` is handled separately (see ``_reasoning_details_text_chars``): its signed/base64
    # envelope is excluded from the budget, mirroring the preflight estimator's exclusion in
    # ``model_metadata._estimate_message_tokens_without_images`` (#73298).
    # An assistant turn may carry only reasoning/thinking content with no visible text (extended-thinking
    # turns, thinking-only recovery responses). Such a turn is persisted with its reasoning fields and is
    # recallable from the transcript, but dropping it here as "empty" makes it vanish from the
    # resumed/reloaded session view while the desktop's reasoning disclosure has nothing to render. Keep it
    # when it carries reasoning so the "Thinking…" block still shows. (#44022)
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
)


def project_compaction_message_for_display(message: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return authentic transcript content, or ``None`` for a pure handoff.

    Model-facing recovery history retains the complete carrier. Display
    projections instead remove the handoff, inherited tool state, and internal
    reasoning while preserving any real prior-tail content or live user ask
    embedded in the carrier.
    """
    if not isinstance(message, dict):
        return None
    if not is_compaction_summary_message(message):
        return message.copy()

    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    if projected is None:
        return None

    projected = projected.copy()
    for key in _COMPACTION_INTERNAL_FIELDS:
        projected.pop(key, None)
    # A safety-refused ask embedded in a composite carrier keeps its ``content_refused`` tag through
    # the unwrap, or a reopened chat would redraw it as an ordinary user bubble with no refused notice
    # (#149708 review): the model-facing handoff is gone, the user's live ask survives as a greyed
    # refused block. A carrier that already names its producer notice kind keeps it too.
    from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND
    from gateway.response_filters import PRODUCER_NOTICE_DISPLAY_KINDS
    kind = projected.get("display_kind")
    if kind == CONTENT_REFUSED_DISPLAY_KIND or kind in PRODUCER_NOTICE_DISPLAY_KINDS:
        return projected
    projected.pop("display_kind", None)
    # The compressor merges the summary into the tail row when no standalone summary row can
    # alternate. When that tail was a producer notice, its ``display_kind`` sidecar never reached the
    # carrier, and the unwrapped envelope would paint as a user message: name its kind again.
    notice = producer_notice_display(flatten_message_text(projected.get("content")))
    if notice is not None:
        projected["display_kind"], label = notice
        metadata = projected.get("display_metadata")
        projected["display_metadata"] = {**(metadata if isinstance(metadata, dict) else {}), **label}
    return projected


# The envelopes ``tools.process_registry_notifications`` writes. ``hermes_state_timeline._SYNTHETIC_PROMPT``
# and the Desktop's ``asyncResultBody`` key on the same leading tags.
_DELEGATION_TASK_HEADER = re.compile(
    r"^--- [✓✗⚠] TASK (\d+)/\d+(?:: ([\s\S]*?))?  \(status=([^,)\n]*)([^\n]*)\) ---$", re.MULTILINE)
_DELEGATION_GOAL = re.compile(r"^(?:Original goal|Task): (.*)$", re.MULTILINE)
_DELEGATION_STATUS = re.compile(r"^Status: (\S+)", re.MULTILINE)
_PROCESS_COMPLETION_HEADER = re.compile(
    r"^\[IMPORTANT: Background process \S+ (.+?) \(exit code (-?\d+|\?)", re.MULTILINE)
_PROCESS_BATCH_HEADER = re.compile(r"\[IMPORTANT: \d+ background processes completed\.")
_PROCESS_COMMAND = re.compile(r"^Command: (.*)$", re.MULTILINE)


def producer_notice_display(text: str) -> Optional[tuple[str, dict[str, Any]]]:
    """``(display_kind, display_metadata)`` for a producer notice envelope, else ``None``.

    The labels come from the producers' own helpers, rebuilt from the envelope's headers, so a
    recovered row reads as the original notice did. Anything else, a user's ask included, is ``None``.
    """
    from tools.process_registry_notifications import (
        PROCESS_COMPLETE_DISPLAY_KIND, async_delegation_display_text, process_completion_display_text)
    if text.startswith("[ASYNC DELEGATION "):
        event = _delegation_event(text)
        return "async_delegation_complete", {
            "display_text": async_delegation_display_text(event), "task_count": len(event.get("results") or [event])}
    if text.startswith("[IMPORTANT: Background process ") or _PROCESS_BATCH_HEADER.match(text):
        events = _process_events(text)
        if events:
            return PROCESS_COMPLETE_DISPLAY_KIND, {"display_text": process_completion_display_text(events)}
    return None


def _delegation_event(text: str) -> dict[str, Any]:
    tasks = _DELEGATION_TASK_HEADER.findall(text)
    if tasks:
        return {"results": [
            {"task_index": int(index) - 1, "goal": goal, "status": status, "truncated": "TRUNCATED" in rest}
            for index, goal, status, rest in tasks]}
    goal, status = _DELEGATION_GOAL.search(text), _DELEGATION_STATUS.search(text)
    return {"goal": goal.group(1) if goal else "", "status": status.group(1) if status else "completed",
            "truncated": "[TRUNCATED: hit max_iterations" in text}


def _process_events(text: str) -> list[dict[str, Any]]:
    from tools.process_registry_notifications import _REASON_STATUS
    events = []
    for header in _PROCESS_COMPLETION_HEADER.finditer(text):
        status, exit_code = header.groups()
        reason = "killed" if status.startswith("terminated by ") else next(
            (key for key, value in _REASON_STATUS.items() if value == status), "exited")
        command = _PROCESS_COMMAND.search(text, header.end())
        events.append({"completion_reason": reason, "exit_code": int(exit_code) if exit_code != "?" else "?",
                       "command": command.group(1) if command else ""})
    return events



# Lean-mode sections written for the model, not for a reader: the compacted turns' user messages
# quoted again, and the footer naming ``session_search`` (``ContextCompressor._augment_summary_lean``).
_MODEL_ONLY_SECTIONS = (_LEAN_USER_MESSAGES_HEADING, _LEAN_RECOVERY_HEADING)
_HISTORICAL_ONLY_LINE = "Historical only; newer protected-tail messages after this summary win."
_COMPACTED_COUNT = re.compile(r"The (\d+) compacted message\(s\) remain fully preserved")


def compaction_summary(message: Any) -> Optional[dict[str, Any]]:
    """What a compaction handoff tells a reader: ``{summary, compacted_messages}``, or ``None`` for any
    other message. ``summary`` is the summary as the compressor wrote it, without the model-facing
    preamble and end marker, prior-tail content or live ask merged into the carrier, the lean
    sections above and the snapshot's ranking line; ``compacted_messages`` is the count the recovery
    footer names (``None`` when it has none)."""
    if not is_compaction_summary_message(message):
        return None
    text = ContextCompressor._strip_summary_prefix(flatten_message_text(message.get("content")))
    sections = {}
    for heading in _MODEL_ONLY_SECTIONS:
        start = text.find(heading)
        if start >= 0:
            end = text.find("\n## ", start + len(heading))
            sections[heading] = text[start:end if end >= 0 else len(text)]
            text = text[:start] + (text[end + 1:] if end >= 0 else "")
    # The footer's count, never one a quoted user message happens to spell out (the quotes are cut first).
    counts = _COMPACTED_COUNT.findall(sections.get(_LEAN_RECOVERY_HEADING, text))
    text = "\n".join(line for line in text.splitlines() if line.strip() != _HISTORICAL_ONLY_LINE)
    return {"summary": re.sub(r"\n{3,}", "\n\n", text).strip(),
            "compacted_messages": int(counts[-1]) if counts else None}

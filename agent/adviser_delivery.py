"""How an adviser note reaches the primary agent — natively, in this process.

Port of oh-my-pi's ``resolveAdvisorDeliveryChannel`` + ``#deliverAdvisorBatch``
adapted to Hermes' own machinery:

* **steer** — a standalone ``role:"user"`` row with ``display_kind="adviser"``,
  queued by :meth:`agent.interrupt_control.InterruptControlMixin.steer` and drained
  after the current tool batch. A queued nit is Hermes' aside: it does not
  interrupt, and it lands at the next safe boundary.
* **interrupt** — a ``blocker`` also aborts the running tool batch.
* **continuation** — at a terminal boundary the note is handed to the stop gate,
  which appends the adviser row and lets the model answer once more.
* **card** — after the run (or when the surface must not be woken) the row is
  appended to the durable transcript and announced to the surface.

The wire format is the fork's adviser contract, parsed by
``hermes_cli.cli_agent_setup_mixin._adviser_items``::

    [adviser — advisory only]
    - [concern · reviewer] re-read the failing test
    [end adviser]
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from agent.adviser_emission import is_interrupting, meets_severity
from agent.adviser_prompts import BOUNDARY_GUIDANCE

logger = logging.getLogger(__name__)

FRAME_OPEN = "[adviser — advisory only]"
FRAME_CLOSE = "[end adviser]"


def format_batch(notes: list[tuple[str, Optional[str], Optional[str]]], *,
                 include_guidance: bool = False) -> str:
    """Render ``(note, severity, adviser name)`` rows as the agent-facing message."""
    lines: list[str] = []
    if include_guidance:
        lines.append(BOUNDARY_GUIDANCE)
    lines.append(FRAME_OPEN)
    for note, severity, name in notes:
        label = severity or "nit"
        if name:
            label = f"{label} · {name}"
        lines.append(f"- [{label}] {note.strip()}")
    lines.append(FRAME_CLOSE)
    return "\n".join(lines)


def resolve_channel(*, severity: Optional[str], continue_severity: str = "concern",
                    auto_resume_suppressed: bool = False, streaming: bool = False,
                    aborting: bool = False, terminal_idle: bool = False,
                    interrupt_immune: bool = False, preserve_only: bool = False) -> str:
    """``preserve`` | ``aside`` | ``steer`` — first matching rule wins (omp order)."""
    if preserve_only and not streaming:
        return "preserve"
    if terminal_idle and not streaming and not aborting:
        if not meets_severity(severity, continue_severity):
            return "preserve"
    elif not is_interrupting(severity):
        return "aside"
    if auto_resume_suppressed and (aborting or not streaming):
        return "preserve"
    if interrupt_immune and severity != "blocker":
        return "preserve" if terminal_idle else "aside"
    return "steer"


def _live_facts(agent: Any) -> dict[str, Any]:
    request_active = getattr(agent, "_model_request_active", None)
    return {
        "streaming": bool(request_active.is_set()) if request_active is not None else False,
        "aborting": bool(getattr(agent, "_interrupt_requested", False)),
        "tool_batch": bool(getattr(agent, "_executing_tools", False)),
    }


def deliver(adviser: Any, notes: list[tuple[str, Optional[str], Optional[str]]], *,
            turn: Optional[int] = None, terminal: bool = False) -> str:
    """Route one batch and return the channel actually used (``steer``/``preserve``/``continuation``)."""
    agent = adviser.agent
    text = format_batch(notes, include_guidance=terminal and len(notes) > 1)
    severity = notes[0][1] if notes else None
    facts = _live_facts(agent)
    terminal_idle = terminal and not facts["streaming"] and not facts["tool_batch"] and not facts["aborting"]
    channel = resolve_channel(
        severity=severity,
        continue_severity=getattr(adviser, "continue_severity", "concern"),
        auto_resume_suppressed=bool(getattr(agent, "_adviser_auto_resume_suppressed", False)),
        streaming=facts["streaming"],
        aborting=facts["aborting"],
        terminal_idle=terminal_idle,
        interrupt_immune=adviser.interrupt_immune_active(),
        preserve_only=not adviser.turn_live(),
    )
    if channel == "preserve":
        adviser.preserve_card(text, severity=severity, turn=turn)
        return "preserve"
    if channel == "aside":
        agent.steer(text, kind="adviser")
        return "steer"
    # A terminal note at/above continueSeverity wakes the model once more instead of
    # parking advice the idle primary would never read.
    if terminal and meets_severity(severity, getattr(adviser, "continue_severity", "concern")):
        if adviser.set_continuation(text, source=notes[0][2] if notes else ""):
            return "continuation"
        adviser.preserve_card(text, severity=severity, turn=turn)
        return "preserve"
    agent.steer(text, kind="adviser")
    if severity == "blocker":
        agent.interrupt(tool_reason="adviser")
    return "steer"


def append_card(agent: Any, text: str, *, severity: Optional[str] = None,
                metadata: Optional[dict[str, Any]] = None) -> bool:
    """Append a durable adviser row to the live transcript and announce it to the surface."""
    messages = getattr(agent, "_session_messages", None)
    if not isinstance(messages, list):
        messages = getattr(agent, "messages", None)
    if not isinstance(messages, list):
        return False
    from agent.plugin_events import notify_plugin_message
    from agent.prompt_builder import advisory_user_row

    row = advisory_user_row(text)
    payload = dict(metadata or {})
    if severity:
        payload.setdefault("severity", severity)
    if payload:
        row["display_metadata"] = payload
    row["plugin_message_id"] = row.get("plugin_message_id") or uuid.uuid4().hex
    messages.append(row)
    try:
        agent._flush_messages_to_session_db(messages, None)
    except Exception:
        logger.debug("adviser card flush failed", exc_info=True)
    notify_plugin_message(agent, row)
    return True


__all__ = [
    "FRAME_CLOSE",
    "FRAME_OPEN",
    "append_card",
    "deliver",
    "format_batch",
    "resolve_channel",
]

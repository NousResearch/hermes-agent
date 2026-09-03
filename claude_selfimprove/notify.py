"""Bridge into the existing Hermes self-improvement notification pipe.

George already runs ``~/.hermes/scripts/self_improvement_watch.sh`` on a
schedule: it drains ``~/.hermes/state/self-improvement-notify/queue.jsonl``,
batches, deduplicates, redacts, and posts one Slack message to
``#self-improvement`` — the SAME channel his skill curator and background
memory review already report through. That script's own comment states its
design goal plainly: "This script is the ONLY thing that posts to
#self-improvement. One delivery path means one dedupe ledger, so an event
can never be sent twice."

Rather than building a second Slack sender, this module writes JSON lines
into that same queue file, in the exact schema
``self-improvement-notify/__init__.py`` already produces
(``v``, ``ts``, ``trigger``, ``kind``, ``action``, ``artifact``, ``label``,
``improvement``, ``status``, ``reason``, ``session_id``, ``id``). The
watcher script does not care which process appended a line — it only reads
the file. This is what requirement 12 (batched, non-noisy notifications)
and requirement 13 (route approved lessons into existing Hermes
mechanisms) come down to.

Every event's ``improvement`` field is the already-redacted, LLM- or
heuristic-written lesson *summary* — never a raw transcript excerpt. There
is no code path in this module that accepts raw transcript text as an
argument.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from . import paths, redact

_MAX_LABEL = 80
_MAX_IMPROVEMENT = 160
_MAX_REASON = 300

TRIGGER = "claude-selfimprove"
KIND = "claude_pipeline"


def _event_id(fields: dict[str, Any]) -> str:
    raw = "|".join(
        str(fields.get(k, ""))
        for k in ("trigger", "kind", "action", "artifact", "improvement", "status", "reason")
    )
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _make_event(
    *,
    action: str,
    artifact: str,
    label: str,
    improvement: str,
    status: str,
    session_id: str = "",
    reason: str = "",
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "v": 1,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trigger": TRIGGER,
        "kind": KIND,
        "action": action,
        "artifact": artifact,
        "label": redact.truncate(redact.redact_text(label), _MAX_LABEL),
        "improvement": redact.truncate(redact.redact_text(improvement), _MAX_IMPROVEMENT),
        "status": status,
        "reason": redact.truncate(redact.redact_text(reason), _MAX_REASON) if status == "failure" else "",
        "session_id": str(session_id or "")[:64],
    }
    event["id"] = _event_id(event)
    return event


def enqueue(event: dict[str, Any]) -> bool:
    """Append one event. Never raises — a notification failure must not
    abort the pipeline run that triggered it."""
    try:
        path = paths.self_improvement_queue_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, payload.encode("utf-8", "replace"))
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


def notify_applied(*, artifact: str, label: str, improvement: str, session_id: str = "") -> bool:
    return enqueue(
        _make_event(
            action="apply", artifact=artifact, label=label,
            improvement=improvement, status="success", session_id=session_id,
        )
    )


def notify_conflict_blocked(*, artifact: str, label: str, reason: str, session_id: str = "") -> bool:
    return enqueue(
        _make_event(
            action="conflict_blocked", artifact=artifact, label=label,
            improvement="", status="failure", reason=reason, session_id=session_id,
        )
    )


def notify_rollback(*, artifact: str, label: str, reason: str) -> bool:
    return enqueue(
        _make_event(
            action="rollback", artifact=artifact, label=label,
            improvement="", status="failure", reason=reason,
        )
    )


def notify_failure(*, artifact: str, label: str, reason: str) -> bool:
    return enqueue(
        _make_event(
            action="failure", artifact=artifact, label=label,
            improvement="", status="failure", reason=reason,
        )
    )

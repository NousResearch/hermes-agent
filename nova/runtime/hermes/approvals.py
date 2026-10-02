"""Approval requests from task work, as the control plane reads and answers them.

A task worker runs unattended, so the runtime's own approval prompt never reaches a person
there. The policy plugin files the call instead — see "approval in task work" in
``enforcement.py`` — and holds the task. This module is the other half: it lists what is
waiting and records a person's answer, through the same store and the same helpers the
plugin uses, so there is one file format and one implementation of it.

A person answers through the existing work decisions (``decide.py``): **release** approves
the held call, **reject** refuses it with a reason. Either way the task goes back on the
board and the worker runs again — with the call let through once, or with its refusal.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from nova.runtime.hermes.enforcement import (
    APPROVAL_BLOCK_PREFIX,
    APPROVED,
    PENDING,
    REFUSED,
    approvals_root,
    request_path,
    summarize_call,
    write_request,
)

__all__ = [
    "APPROVAL_BLOCK_PREFIX", "APPROVED", "PENDING", "REFUSED",
    "pending", "first_pending", "answer", "reopen", "view",
]


def pending(home: Path, task_id: str) -> list[dict[str, Any]]:
    """The task's unanswered requests, newest first. Empty for an id that is not a task's."""
    # Any well-formed id gives the task's folder; it also refuses a task id that is a path.
    probe = request_path(approvals_root(home), task_id, "ap_" + "0" * 20)
    if probe is None or not probe.parent.is_dir():
        return []
    found = []
    for path in probe.parent.glob("ap_*.json"):
        if ".used-" in path.name:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("status") == PENDING:
            found.append(record)
    return sorted(found, key=lambda r: str(r.get("requested_at", "")), reverse=True)


def answer(home: Path, record: dict[str, Any], *, approve: bool, actor: str, reason: str = "") -> dict[str, Any]:
    """Record a person's answer on a request and return the updated record."""
    updated = {
        **record,
        "status": APPROVED if approve else REFUSED,
        "decided_by": actor,
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "decision_reason": reason,
    }
    write_request(approvals_root(home), updated)
    return updated


def reopen(home: Path, record: dict[str, Any]) -> None:
    """Put an answered request back to waiting — used when the task could not be moved."""
    write_request(approvals_root(home), {
        key: value for key, value in record.items()
        if key not in ("decided_by", "decided_at", "decision_reason")
    } | {"status": PENDING})


#: How much of the arguments the Work screen is sent. Enough to review a message or a
#: record update; the full call stays in the request file.
_VIEW_ARGS_CHARS = 4000


def view(record: dict[str, Any]) -> dict[str, Any]:
    """What the Work screen shows a person deciding: what, why, and the exact call."""
    arguments = json.dumps(record.get("args") or {}, indent=2, sort_keys=True, ensure_ascii=False)
    if len(arguments) > _VIEW_ARGS_CHARS:
        arguments = arguments[:_VIEW_ARGS_CHARS] + f"\n… ({len(arguments)} characters in all)"
    return {
        "request_id": record.get("request_id", ""),
        "tool": record.get("tool", ""),
        "action": record.get("action", ""),
        "reason": record.get("reason", ""),
        "requested_at": record.get("requested_at", ""),
        "agent_id": record.get("agent_id", ""),
        "call": summarize_call(str(record.get("tool", "")), record.get("args")),
        "arguments": arguments,
    }


def first_pending(home: Path, task_id: str) -> Optional[dict[str, Any]]:
    waiting = pending(home, task_id)
    return waiting[0] if waiting else None

"""Safe in-process evidence records for delegate_task runs.

This module intentionally keeps records process-local. It gives gateway
reporting code proof that delegation happened without writing private task
prompts, session keys, or result bodies to persistent state.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import re
import threading
from typing import Any


DELEGATE_EVIDENCE_LANES = (
    "implementation",
    "review",
    "verification",
    "safety",
    "deployment",
    "domain",
)

_MAX_RECORDS = 200
_MAX_SUMMARY_CHARS = 500
_records: deque[dict[str, Any]] = deque(maxlen=_MAX_RECORDS)
_records_lock = threading.Lock()

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9._-]{8,}\b"),
    re.compile(r"\b[A-Za-z0-9_]*(?:token|secret|password|api[_-]?key)[A-Za-z0-9_]*\s*[:=]\s*\S+", re.I),
    re.compile(r"private prompt[^.\n,;]*", re.I),
    re.compile(r"confidential prompt[^.\n,;]*", re.I),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_ref(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return f"sha256:{digest[:16]}"


def redact_delegate_text(value: Any, *, max_chars: int = _MAX_SUMMARY_CHARS) -> str:
    """Return a short, secret-scrubbed text summary."""
    text = str(value or "").strip()
    if not text:
        return ""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 15].rstrip() + " ...[truncated]"
    return text


def normalize_delegate_lane(lane: Any = None, *, task_goal: Any = None, context: Any = None) -> str:
    explicit = str(lane or "").strip().lower()
    if explicit in DELEGATE_EVIDENCE_LANES:
        return explicit

    folded = f"{task_goal or ''} {context or ''}".lower()
    if any(term in folded for term in ("review", "audit", "critique")):
        return "review"
    if any(term in folded for term in ("verify", "test", "validation", "check")):
        return "verification"
    if any(term in folded for term in ("safe", "security", "secret", "risk")):
        return "safety"
    if any(term in folded for term in ("deploy", "restart", "runtime", "production")):
        return "deployment"
    if any(term in folded for term in ("research", "domain", "analyze")):
        return "domain"
    return "implementation"


def normalize_delegate_status(status: Any) -> str:
    folded = str(status or "").strip().lower()
    if folded in {"completed", "complete", "succeeded", "success", "ok"}:
        return "succeeded"
    if folded in {"pending"}:
        return "pending"
    if folded in {"interrupted", "timeout", "failed", "failure", "error"}:
        return "failed"
    if folded == "skipped":
        return "skipped"
    return "failed"


def record_delegate_evidence(
    *,
    lane: Any = None,
    task_goal: Any = None,
    context: Any = None,
    delegate_name: str = "delegate_task",
    delegate_type: str = "subagent",
    invoked_at: str | None = None,
    completed_at: str | None = None,
    status: Any = "pending",
    result_summary: Any = "",
    evidence_pointer: Any = None,
    session_key: Any = None,
    child_session_id: Any = None,
) -> dict[str, Any]:
    """Append and return a redacted delegate evidence record."""
    normalized_lane = normalize_delegate_lane(lane, task_goal=task_goal, context=context)
    task_ref = _hash_ref(session_key) or _hash_ref(task_goal)
    pointer = _hash_ref(evidence_pointer) or _hash_ref(child_session_id)
    record = {
        "task_ref": task_ref,
        "lane": normalized_lane,
        "delegate_name": str(delegate_name or "delegate_task"),
        "delegate_type": str(delegate_type or "subagent"),
        "invoked_at": invoked_at or _now_iso(),
        "completed_at": completed_at,
        "status": normalize_delegate_status(status),
        "safe_result_summary": redact_delegate_text(result_summary),
        "evidence_pointer": pointer,
    }
    with _records_lock:
        _records.append(dict(record))
    return record


def get_recent_delegate_evidence(
    limit: int = 20,
    *,
    session_key: Any = None,
    session_id: Any = None,
) -> list[dict[str, Any]]:
    with _records_lock:
        records = list(_records)
    ref_filter = _hash_ref(session_key) or _hash_ref(session_id)
    if ref_filter:
        records = [record for record in records if record.get("task_ref") == ref_filter]
    records = records[-max(0, int(limit)) :]
    return [dict(record) for record in records]


def clear_delegate_evidence_records() -> None:
    """Test helper; production callers should not need this."""
    with _records_lock:
        _records.clear()


def summarize_delegate_evidence(records: list[dict[str, Any]] | None = None) -> str:
    items = records if records is not None else get_recent_delegate_evidence()
    safe_items = [item for item in items if isinstance(item, dict)]
    if not safe_items:
        return ""
    parts = []
    for item in safe_items[:5]:
        lane = item.get("lane") or "unknown"
        status = item.get("status") or "unknown"
        summary = item.get("safe_result_summary") or "no summary"
        parts.append(f"lane={lane}; status={status}; summary={summary}")
    return " | ".join(parts)

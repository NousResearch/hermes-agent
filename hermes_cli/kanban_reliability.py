"""Read-only Kanban burn-down reliability reporting.

The report deliberately opens an existing board database with SQLite
``mode=ro`` and ``query_only``.  It is safe to run from cron or the gateway
while the dispatcher is active: no schema initialization, migrations, or
board mutations are performed.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


_SUCCESS_OUTCOMES = {"completed"}
_STALE_LOCK_RE = re.compile(r"\bstale_lock=([^\s,;]+)", re.IGNORECASE)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _metadata(run: Mapping[str, Any]) -> dict[str, Any]:
    value = run.get("metadata")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return _as_dict(value)


def categorize_failure(run: Mapping[str, Any]) -> str:
    """Map a failed run to a stable operator-facing failure category."""
    metadata = _metadata(run)
    failure_category = str(metadata.get("failure_category") or "").lower()
    outcome = str(run.get("outcome") or "").lower()
    text = " ".join(
        str(value or "")
        for value in (
            failure_category,
            metadata.get("error"),
            metadata.get("exit_kind"),
            run.get("error"),
            run.get("summary"),
        )
    ).lower()

    if "heartbeat" in text and ("reclaim" in text or outcome == "reclaimed"):
        return "heartbeat_reclaim"
    if "completion" in text and any(
        marker in text for marker in ("gate", "evidence", "contract", "rejected", "blocked")
    ):
        return "completion_gate"
    if "protocol violation" in text or "protocol_violation" in text:
        return "protocol_violation"
    if any(marker in text for marker in ("iteration budget", "iteration limit", "max iterations")):
        return "iteration_exhaustion"
    if any(marker in text for marker in ("pid not alive", "pid is not alive", "worker pid", "process not alive")):
        return "pid_not_alive"
    if any(marker in text for marker in ("workspace", "artifact guard", "missing artifact", "worktree")):
        return "workspace_or_artifact"
    if any(
        marker in text
        for marker in (
            "auth",
            "credential",
            "missing token",
            "token is missing",
            "command not found",
            "no such executable",
            "tooling missing",
        )
    ):
        return "auth_or_tooling"
    if outcome == "blocked":
        return "blocked"
    return failure_category or outcome or "unknown"


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _reason_from_payload(value: Any) -> str:
    payload = _parse_json_object(value)
    reason = payload.get("reason") or payload.get("error") or payload.get("message") or "unspecified"
    return " ".join(str(reason).split())[:240]


def _stale_lock(run: Mapping[str, Any]) -> str | None:
    error = str(run.get("error") or "")
    match = _STALE_LOCK_RE.search(error)
    if match:
        return match.group(1)
    metadata = _metadata(run)
    value = metadata.get("stale_lock")
    return str(value) if value else None


@contextmanager
def open_readonly(path: Path | str) -> Iterator[sqlite3.Connection]:
    """Open an existing SQLite database without permitting writes."""
    resolved = Path(path).expanduser().resolve(strict=True)
    conn = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    try:
        yield conn
    finally:
        conn.close()


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def build_reliability_report(
    conn: sqlite3.Connection,
    *,
    window_seconds: int,
    now: int | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Build a bounded reliability summary from a read-only connection."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")
    now_ts = int(time.time()) if now is None else int(now)
    since = now_ts - int(window_seconds)

    rows = [
        dict(row)
        for row in conn.execute(
            "SELECT id, task_id, profile, status, outcome, summary, metadata, error, "
            "started_at, ended_at FROM task_runs "
            "WHERE COALESCE(ended_at, started_at) >= ? "
            "ORDER BY COALESCE(ended_at, started_at), id",
            (since,),
        )
    ]
    outcome_counts = Counter(str(row.get("outcome") or "running") for row in rows)
    profile_counts = Counter(str(row.get("profile") or "(unassigned)") for row in rows)
    failures = [
        row for row in rows
        if row.get("outcome") and str(row["outcome"]) not in _SUCCESS_OUTCOMES
    ]
    category_counts = Counter(categorize_failure(row) for row in failures)

    bursts: dict[str, dict[str, Any]] = {}
    for row in failures:
        lock = _stale_lock(row)
        if not lock:
            continue
        timestamp = int(row.get("ended_at") or row.get("started_at") or 0)
        burst = bursts.setdefault(
            lock,
            {"stale_lock": lock, "count": 0, "first_at": timestamp, "last_at": timestamp},
        )
        burst["count"] += 1
        burst["first_at"] = min(burst["first_at"], timestamp)
        burst["last_at"] = max(burst["last_at"], timestamp)
    recurring_bursts = [item for item in bursts.values() if item["count"] > 1]
    stale_lock_bursts = sorted(
        recurring_bursts,
        key=lambda item: (-item["count"], -item["last_at"], item["stale_lock"]),
    )[:limit]

    blocked_reasons = Counter(
        _reason_from_payload(row["payload"])
        for row in conn.execute(
            "SELECT payload FROM task_events WHERE kind IN ('blocked', 'spawn_auto_blocked') "
            "AND created_at >= ?",
            (since,),
        )
    )
    top_blocked_reasons = [
        {"reason": reason, "count": count}
        for reason, count in sorted(blocked_reasons.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]

    blocker_rows = list(
        conn.execute(
            "SELECT t.id, t.title, t.assignee, t.status, t.block_kind, "
            "(SELECT e.payload FROM task_events e "
            " WHERE e.task_id=t.id AND e.kind IN ('blocked', 'spawn_auto_blocked') "
            " ORDER BY e.id DESC LIMIT 1) AS block_payload, "
            "t.last_failure_error "
            "FROM tasks t WHERE t.status='blocked' OR "
            "(t.status='triage' AND (t.block_kind IS NOT NULL OR t.last_failure_error IS NOT NULL)) "
            "ORDER BY COALESCE(t.started_at, t.created_at) ASC, t.id ASC LIMIT ?",
            (limit + 1,),
        )
    )
    active_blockers = []
    for row in blocker_rows[:limit]:
        reason = _reason_from_payload(row["block_payload"])
        if reason == "unspecified" and row["last_failure_error"]:
            reason = " ".join(str(row["last_failure_error"]).split())[:240]
        active_blockers.append(
            {
                "task_id": row["id"],
                "title": row["title"],
                "assignee": row["assignee"],
                "status": row["status"],
                "block_kind": row["block_kind"],
                "reason": reason,
            }
        )

    return {
        "generated_at": now_ts,
        "window": {"seconds": int(window_seconds), "since": since},
        "runs": {
            "total": len(rows),
            "by_outcome": _sorted_counts(outcome_counts),
            "by_profile": _sorted_counts(profile_counts),
        },
        "failures": {
            "total": len(failures),
            "by_category": _sorted_counts(category_counts),
        },
        "stale_lock_bursts": stale_lock_bursts,
        "top_blocked_reasons": top_blocked_reasons,
        "active_actionable_blockers": active_blockers,
        "truncated": {
            "stale_lock_bursts": len(recurring_bursts) > limit,
            "top_blocked_reasons": len(blocked_reasons) > limit,
            "active_actionable_blockers": len(blocker_rows) > limit,
        },
    }


def render_reliability_report(report: Mapping[str, Any]) -> str:
    """Render a compact terminal-friendly report."""
    window = _as_dict(report.get("window"))
    runs = _as_dict(report.get("runs"))
    failures = _as_dict(report.get("failures"))
    lines = [
        f"Kanban reliability ({window.get('label') or window.get('seconds', '?')}): "
        f"{runs.get('total', 0)} runs, {failures.get('total', 0)} non-success",
        "",
        "By outcome: " + ", ".join(f"{k}={v}" for k, v in _as_dict(runs.get("by_outcome")).items()),
        "By profile: " + ", ".join(f"{k}={v}" for k, v in _as_dict(runs.get("by_profile")).items()),
        "By failure category: "
        + (", ".join(f"{k}={v}" for k, v in _as_dict(failures.get("by_category")).items()) or "none"),
        "",
        "Top stale_lock bursts:",
    ]
    bursts = report.get("stale_lock_bursts") or []
    lines.extend(
        f"  {item['stale_lock']}: {item['count']} runs "
        f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(item['first_at']))} → "
        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(item['last_at']))})"
        for item in bursts
    )
    if not bursts:
        lines.append("  none")
    lines.append("\nTop blocked reasons:")
    reasons = report.get("top_blocked_reasons") or []
    lines.extend(f"  {item['count']}× {item['reason']}" for item in reasons)
    if not reasons:
        lines.append("  none")
    lines.append("\nActive actionable blockers:")
    blockers = report.get("active_actionable_blockers") or []
    lines.extend(
        f"  {item['task_id']} [{item['status']}] @{item['assignee'] or '-'} "
        f"{item['title']} — {item['reason']}"
        for item in blockers
    )
    if not blockers:
        lines.append("  none")
    return "\n".join(lines) + "\n"

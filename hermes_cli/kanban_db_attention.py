"""Current native review attention and handoff-scoped display annotations."""
import json
import sqlite3
from typing import Optional


def _valid_badge(badge) -> bool:
    return (isinstance(badge, dict) and isinstance(badge.get("label"), str)
            and 0 < len(badge["label"].strip()) <= 80
            and badge.get("tone") in ("success", "info", "warning", "danger"))


def current_attention(task, events, runs) -> Optional[dict]:
    """Project the current native handoff/failure; never infer it from prose.

    Inputs are already ordered by id by the diagnostic bulk read. Event ids,
    not wall-clock timestamps, distinguish reopen/review in the same second.
    """
    if task["status"] in {"running", "done", "archived"} or task["claim_lock"] or not runs:
        return None
    run = runs[-1]
    if run["ended_at"] is None or task["current_run_id"] is not None:
        return None
    kind = None
    event = None
    if task["status"] == "review" and run["outcome"] == "review_requested":
        event = next((e for e in reversed(events) if e["kind"] == "review_requested"), None)
        try:
            metadata = json.loads(run["metadata"] or "null")
        except (TypeError, ValueError):
            return None
        reason = metadata.get("wait_reason") if isinstance(metadata, dict) else None
        if isinstance(reason, str) and reason in {"awaiting-answer", "awaiting-test"}:
            kind = reason
    elif run["outcome"] in {"crashed", "timed_out", "spawn_failed", "gave_up"}:
        kind = "run-failed"
        event = next((e for e in reversed(events) if e["run_id"] == run["id"]
                      and e["kind"] in {"crashed", "timed_out", "spawn_failed", "protocol_violation", "gave_up"}), None)
    if not kind or event is None or event["run_id"] != run["id"]:
        return None
    # Harmless comments, assignment and priority edits do not answer a question.
    # Explicit lifecycle changes do, even when old metadata stays in history.
    resets = {"claimed", "spawned", "worker_registered", "reclaimed", "reconciled", "review_reopened",
              "changes_requested", "review_requested", "completed", "blocked", "unblocked", "specified",
              "promoted", "promoted_manual", "archived", "scheduled", "status", "descendant_invalidated"}
    if any(e["id"] > event["id"] and e["kind"] in resets for e in events):
        return None
    summary = (run["error"] or run["summary"]) if kind == "run-failed" else run["summary"]
    attention = {"kind": kind, "run_id": run["id"], "summary": summary or None}
    if kind in {"awaiting-answer", "awaiting-test"}:
        attention["handoff_event_id"] = event["id"]
    marker = next((e for e in reversed(events) if e["kind"] == "review_attention" and e["id"] > event["id"]), None)
    if kind == "awaiting-test" and marker is not None and marker["run_id"] == run["id"]:
        try:
            payload = json.loads(marker["payload"] or "null")
        except (TypeError, ValueError):
            payload = None
        if (isinstance(payload, dict) and type(payload.get("handoff_event_id")) is int
                and payload.get("handoff_event_id") == event["id"]
                and _valid_badge(payload.get("badge"))):
            attention["badge"] = payload["badge"]
    return attention

def set_review_attention(conn: sqlite3.Connection, task_id: str, *,
                         expected_handoff_event_id: int, badge: Optional[dict]) -> bool:
    """Annotate only the current unclaimed testing handoff, without changing its lifecycle.

    The producer owns selection policy. Event/run identity makes stale updates harmless;
    explicit null badges clear earlier assertions without rewriting review provenance.
    True includes an identical no-op; False means the handoff is no longer eligible.
    """
    from hermes_cli import kanban_db as kb

    if type(expected_handoff_event_id) is not int or (badge is not None and not _valid_badge(badge)):
        return False
    if badge is not None:
        badge = {"label": badge["label"].strip(), "tone": badge["tone"]}
    with kb.write_txn(conn):
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None:
            return False
        events = conn.execute("SELECT * FROM task_events WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
        runs = conn.execute("SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
        attention = current_attention(task, events, runs)
        if (not attention or attention["kind"] != "awaiting-test"
                or attention["handoff_event_id"] != expected_handoff_event_id):
            return False
        if attention.get("badge") != badge:
            kb._append_event(conn, task_id, "review_attention", {
                "handoff_event_id": expected_handoff_event_id, "badge": badge}, run_id=attention["run_id"])
    return True

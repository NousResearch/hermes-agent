"""Completion-focused helpers split from :mod:`hermes_cli.kanban_db`."""

from __future__ import annotations

from typing import Any, Optional


if "EmptyCompletionError" not in globals():
    # ``importlib.reload`` retains the module dictionary. Keep the original
    # exception identity so callers holding ``kanban_db.EmptyCompletionError``
    # continue to catch errors raised by this module after a plugin reload.
    class EmptyCompletionError(ValueError):
        """Completion was refused because no substantive evidence was supplied."""

        def __init__(self, task_id: str):
            self.task_id = task_id
            super().__init__(
                f"completion blocked: {task_id} has no result or summary evidence"
            )


def completion_state_snapshot(conn, task_id: str) -> Optional[tuple]:
    """Return state/evidence generation used to fence operator completion."""
    row = conn.execute(
        "SELECT t.current_run_id, t.status, t.result, "
        "COALESCE((SELECT MAX(e.id) FROM task_events e WHERE e.task_id=t.id), 0) "
        "FROM tasks t WHERE t.id=?",
        (task_id,),
    ).fetchone()
    return tuple(row) if row else None


def triage_resolution_allowed(conn, task_id: str) -> bool:
    """Allow normal states, or the current loop-breaker Triage generation only."""
    from hermes_cli import kanban_db as _kb

    row = conn.execute(
        "SELECT t.status, t.block_recurrences, "
        "(SELECT e.kind FROM task_events e WHERE e.task_id=t.id "
        "AND e.kind IN ('block_loop_detected', 'specified', 'decomposed', 'status', 'imported') "
        "ORDER BY e.id DESC LIMIT 1) AS triage_generation_event "
        "FROM tasks t WHERE t.id=?",
        (task_id,),
    ).fetchone()
    if row is None:
        return False
    if row["status"] != "triage":
        return True
    return bool(
        row["block_recurrences"] >= _kb.BLOCK_RECURRENCE_LIMIT
        and row["triage_generation_event"] == "block_loop_detected"
    )


def gate_empty_completion(
    conn, task_id: str, *, result: Optional[str], summary: Optional[str],
    require_current_evidence: bool = False,
) -> None:
    """Reject evidence-free completion, with stricter Triage freshness."""
    from hermes_cli import kanban_db as _kb

    row = conn.execute(
        "SELECT status, result FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None or row["status"] == "review":
        return
    fresh_required = require_current_evidence and row["status"] == "triage"
    if (_kb._substantive_text(result) or _kb._substantive_text(summary)
            or (not fresh_required and _kb._substantive_text(row["result"]))):
        return
    with _kb.write_txn(conn):
        _kb._append_event(
            conn,
            task_id,
            "completion_blocked_empty_result",
            {
                "result_preview": _kb._first_line(result, 200) or None,
                "summary_preview": _kb._first_line(summary, 200) or None,
            },
        )
    raise EmptyCompletionError(task_id)


def _cleaned_artifact_paths(metadata: Any) -> list[str]:
    """Return non-blank paths declared in ``metadata["artifacts"]``."""
    if not isinstance(metadata, dict):
        return []
    raw = metadata.get("artifacts")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(path).strip() for path in raw if isinstance(path, str) and str(path).strip()]


def completed_event_payload(
    result: Optional[str], event_summary: Optional[str], verified_cards: list[str], metadata: Any,
) -> dict:
    """Build the compact completion payload consumed by notifiers and the dashboard."""
    from hermes_cli.kanban_db import _first_line

    payload: dict = {
        "result_len": len(result) if result else 0,
        "summary": _first_line(event_summary, 400) or None,
    }
    if verified_cards:
        payload["verified_cards"] = verified_cards
    if isinstance(metadata, dict):
        cleaned = _cleaned_artifact_paths(metadata)
        if cleaned:
            payload["artifacts"] = cleaned
    return payload

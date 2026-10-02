"""Review handoffs for the Kanban board."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


def request_review(
    conn: sqlite3.Connection, task_id: str, *, summary: Optional[str] = None,
    metadata: Optional[dict] = None, reviewer: Optional[str] = None,
    expected_run_id: Optional[int] = None, force: bool = False, with_reason: bool = False,
):
    """``running``/``ready`` -> ``review``; never touches block recurrence accounting.

    Implementer and reviewer are recorded on the event so requested changes
    route back to the right profile; ``reviewer`` reassigns the task, and on
    re-review defaults to the latest ``changes_requested`` provenance. A live
    claim is only cleared with proof of ownership (``expected_run_id``) or
    ``force=True``. Returns ``bool``, or ``(ok, reason)`` with ``with_reason``.

    ``metadata["artifacts"]`` names the handoff's deliverable
    files; a review handoff is the last implementer transition, and the
    *reviewer's* completion is what cleans the managed scratch workspace up, so
    the files are staged into the task's durable attachments dir here and the
    staged paths ride the ``review_requested`` payload for the notifier to
    upload. A declared artifact that cannot be preserved raises
    :class:`ArtifactPreservationError`, rolling the whole transition back: the
    task stays ``running`` and retryable, with no attachments and no event.
    """

    from hermes_cli.kanban_db import (
        _append_event, _canonical_assignee, _claim_is_live, _cleaned_artifact_paths,
        _discard_staged_copies, _end_or_synthesize_run, _first_line,
        _merge_completion_prose_artifacts, _parents_satisfied,
        _stage_completion_artifacts, redact_review_value, write_txn,
    )

    def _ret(ok: bool, reason: Optional[str] = None):
        return (ok, reason) if with_reason else ok

    summary = redact_review_value(summary)
    metadata = redact_review_value(metadata)
    # Declared (metadata["artifacts"]) and prose-referenced files
    # must be durable BEFORE anything can clean the scratch workspace up: for a
    # review-bound card the reviewer's completion is the cleanup trigger.
    metadata = _merge_completion_prose_artifacts(conn, task_id, metadata, summary=summary, result=None)
    now = int(time.time())
    # Staged copies live outside the txn: a rollback after staging must not
    # leave orphans that make the retry stage ``name_1.ext`` beside them.
    staged_copies: list[Path] = []
    try:
        with write_txn(conn):
            if not _parents_satisfied(conn, task_id):
                return _ret(False, "parent dependencies are not satisfied")
            trow = conn.execute(
                "SELECT assignee, status, claim_lock, current_run_id, worker_pid, "
                "worker_started_at FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            if trow is None:
                return _ret(False, "task not found")
            # Refuse to clear a live worker's claim without proof of ownership
            # (expected_run_id) or an explicit human override (force=True);
            # the same fence as complete_task (_claim_is_live).
            if expected_run_id is None and not force and _claim_is_live(trow):
                return _ret(
                    False, "task is running under a live claim; pass expected_run_id "
                    "(worker ownership) or force=True (explicit operator "
                    "override) instead of clearing the live run's claim",
                )
            if reviewer is None:
                reviewer = _prior_reviewer(conn, task_id)
                if reviewer is False:
                    return _ret(
                        False, "re-review has no durable reviewer provenance (the "
                        "latest changes_requested event is missing or "
                        "malformed); pass reviewer= explicitly",
                    )
            reviewer = _canonical_assignee(reviewer)
            # The actor is the run that did the work. ``assignee`` is the actor
            # only while a worker holds the card; on a never-claimed card it is
            # whoever the operator assigned -- possibly the reviewer itself,
            # which is what ``kanban create --assignee <reviewer>`` followed by
            # ``request-review`` produces. Recording the reviewer as its own
            # implementer is worse than recording nothing: request_changes()
            # routes on this field, and it already refuses a handoff that
            # carries no implementer provenance.
            implementer = None
            if trow["current_run_id"] is not None:
                arow = conn.execute(
                    "SELECT profile FROM task_runs WHERE id = ?",
                    (trow["current_run_id"],),
                ).fetchone()
                implementer = arow["profile"] if arow else None
            if implementer is None and trow["assignee"] != reviewer:
                implementer = trow["assignee"]
            assignee_sql = ", assignee = ?" if reviewer is not None else ""
            run_guard = "" if expected_run_id is None else " AND current_run_id = ?"
            params: tuple[Any, ...] = (
                *(() if reviewer is None else (reviewer,)), task_id,
                *(() if expected_run_id is None else (int(expected_run_id),)),
            )
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status        = 'review',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL
                """ + assignee_sql + """
                 WHERE id = ?
                   AND status IN ('running', 'ready')
                """ + run_guard,
                params,
            )
            if cur.rowcount != 1:
                return _ret(
                    False, "task is not in running/ready (or expected_run_id did not match the current run)",
                )
            if isinstance(metadata, dict):
                staged_copies = _stage_completion_artifacts(
                    conn, task_id, metadata, now, uploaded_by="kanban_request_review",
                )
            run_id = _end_or_synthesize_run(
                conn, task_id, outcome="review_requested", status="review",
                summary=summary, metadata=metadata, synthesize=bool(summary or metadata),
                profile=implementer,
            )
            payload: dict = {
                "summary": _first_line(summary, 400) or None,
                "implementer": implementer,
                "reviewer": reviewer,
            }
            staged = _cleaned_artifact_paths(metadata)
            if staged:
                payload["artifacts"] = staged
            _append_event(conn, task_id, "review_requested", payload, run_id=run_id)
    except Exception:
        if staged_copies:
            _discard_staged_copies(staged_copies, staged_copies[0].parent)
        raise
    return _ret(True)


def _prior_reviewer(conn: sqlite3.Connection, task_id: str):
    """Reviewer recorded by the latest ``changes_requested`` run's event.
    ``None`` = first review (no such run); ``False`` = a run exists but its
    provenance is missing/malformed."""
    from hermes_cli.kanban_db import _json_dict, _latest_event, _row_get

    changes_run = conn.execute(
        "SELECT id FROM task_runs "
        "WHERE task_id = ? AND outcome = 'changes_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if changes_run is None:
        return None
    changes_event = _latest_event(conn, task_id, "changes_requested", changes_run["id"])
    reviewer = _json_dict(_row_get(changes_event, "payload")).get("reviewer")
    return reviewer if isinstance(reviewer, str) and reviewer.strip() else False



"""Machine-acceptance submissions: the DB half of ``hermes_cli.kanban_acceptance``.

A public completion of a card with ``tasks.acceptance`` does not finish it. It creates one
``task_acceptance_submissions`` row (``state='open'``) and parks the card in ``review``. While that
row is open the card is a *machine wait*: no reviewer lane, claim or capacity reservation sees it,
and public completion refuses it. The dispatcher pass leases the row, evaluates outside any
transaction, and applies the verdict here under a compare-and-swap. Only that internal finalizer
can move a machine wait to ``done``.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached late-bound via ``_kb``.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import hermes_cli.kanban_db as _kb
from hermes_cli import kanban_acceptance as _ka
from hermes_cli.kanban_pr_acceptance import _PR


class AcceptanceSubmissionError(ValueError):
    """``complete_task`` refused to create a machine-acceptance submission (nothing changed)."""


def task_acceptance(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    row = conn.execute("SELECT acceptance FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return row["acceptance"] if row and row["acceptance"] else None


def open_submission(conn: sqlite3.Connection, task_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM task_acceptance_submissions WHERE task_id = ? AND state = 'open'", (task_id,),
    ).fetchone()


def is_machine_wait(conn: sqlite3.Connection, task_id: str) -> bool:
    """True while the card sits in ``review`` with an open submission (the evaluator owns it)."""
    return conn.execute(
        "SELECT 1 FROM tasks t JOIN task_acceptance_submissions s ON s.task_id = t.id "
        "WHERE t.id = ? AND t.status = 'review' AND s.state = 'open'", (task_id,),
    ).fetchone() is not None


def submission_summary(conn: sqlite3.Connection, task_id: str) -> Optional[dict]:
    """Latest submission for read surfaces (``kanban show`` / ``kanban_show``)."""
    row = conn.execute(
        "SELECT * FROM task_acceptance_submissions WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if row is None:
        return None
    receipt = _kb._json_dict(row["last_receipt"])
    return {
        "id": row["id"], "state": row["state"], "verdict": row["verdict"], "reason": receipt.get("reason"),
        "attempts": row["attempts"], "implementer": row["implementer"], "origin_run_id": row["origin_run_id"],
        "created_at": row["created_at"], "next_check_at": row["next_check_at"], "decided_at": row["decided_at"],
    }


def _verdict_event(conn, task_id: str, run_id, payload: dict) -> None:
    _kb._append_event(conn, task_id, "acceptance_verdict", payload, run_id=run_id)


def _cancel(conn: sqlite3.Connection, sub: sqlite3.Row, reason: str, now: int, *,
            route: bool = False, keep_assignee: bool = False) -> None:
    """Close an open submission without a verdict. A card left in ``review`` is then not an ordinary
    review card (no ``request_review`` put it there), so ``route`` sends it to ready/todo for the
    submission's implementer instead of leaving it for the reviewer lane. Status changes out of
    ``review`` by any other path are cancelled by the ``trg_acceptance_left_review`` trigger."""
    cur = conn.execute(
        "UPDATE task_acceptance_submissions SET state = 'cancelled', verdict = 'cancelled', decided_at = ?, "
        "lease_owner = NULL, lease_until = NULL, next_check_at = NULL, last_receipt = ? "
        "WHERE id = ? AND state = 'open'",
        (now, json.dumps({"verdict": "cancelled", "reason": reason}), sub["id"]),
    )
    if cur.rowcount != 1:
        return
    payload = {"verdict": "cancelled", "reason": reason, "submission_id": sub["id"]}
    if route and _kb._task_status(conn, sub["task_id"]) == "review":
        status = _kb._landing_status_after_parents(conn, sub["task_id"])
        conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL"
            + ("" if keep_assignee else ", assignee = COALESCE(?, assignee)") + " WHERE id = ? AND status = 'review'",
            (status, *(() if keep_assignee else (sub["implementer"],)), sub["task_id"]),
        )
        payload["status"] = status
    _verdict_event(conn, sub["task_id"], sub["origin_run_id"], payload)


def leave_machine_wait(conn: sqlite3.Connection, task_id: str, reason: str, *, keep_assignee: bool = False) -> bool:
    """Inside the caller's txn: end a machine wait whose card stays put otherwise (reassign, spec
    removal): cancel the submission and route the card to ready/todo. True when it was one."""
    if not is_machine_wait(conn, task_id):
        return False
    _cancel(conn, open_submission(conn, task_id), reason, int(time.time()), route=True, keep_assignee=keep_assignee)
    return True


def cancel_open_submission(conn: sqlite3.Connection, task_id: str, reason: str) -> Optional[sqlite3.Row]:
    """Cancel ``task_id``'s open submission inside the caller's txn; returns the cancelled row."""
    sub = open_submission(conn, task_id)
    if sub is not None:
        _cancel(conn, sub, reason, int(time.time()))
    return sub


# --------------------------------------------------------------------------- submission

def _bind_contract(contract: Optional[str], metadata: Any) -> Optional[str]:
    """``completion_contract`` after publication binding, exactly as ``prepare_acceptance`` binds it
    (OWNER/REPO + matching ``metadata.published_pr`` -> that PR), but without collecting evidence."""
    if not contract or contract == "local-only":
        return contract
    published = metadata.get("published_pr") if isinstance(metadata, dict) else None
    match = _PR.fullmatch(published) if isinstance(published, str) else None
    if _PR.fullmatch(contract):
        if published is not None and published != contract:
            raise AcceptanceSubmissionError(
                "metadata.published_pr does not match the card's completion_contract PR; nothing changed")
        return contract
    if match and match[1] == contract:
        return published
    raise AcceptanceSubmissionError(
        f"completion_contract {contract} needs metadata.published_pr naming the published PR in that "
        "repository before the card can be submitted for machine acceptance; nothing changed")


def _implementer(conn, task_id: str, trow, run_row, reviewing: bool) -> Optional[str]:
    """Who gets the work back on a machine fail — never the approver. An approval (complete on a
    ``review`` card, or a reviewer run claimed from ``review``) takes the implementer recorded by the
    latest ``review_requested`` / ``acceptance_submitted`` event and refuses without one."""
    if reviewing:
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind IN ('review_requested', 'acceptance_submitted') ORDER BY id DESC LIMIT 1", (task_id,),
        ).fetchone()
        implementer = _kb._nonblank_str(_kb._json_dict(_kb._row_get(event, "payload")).get("implementer"))
        if implementer is None:
            raise AcceptanceSubmissionError(
                "cannot submit an approved review for machine acceptance: the review handoff has no "
                "implementer provenance (latest review_requested event missing or malformed); nothing changed")
        return implementer
    profile = run_row["profile"] if run_row is not None else None
    return _kb._nonblank_str(profile) or _kb._nonblank_str(trow["assignee"])


def _run_worker(conn, task_id: str, run_id: int) -> dict:
    """Exit provenance of one claimed run: pid/fingerprint from the run row, else from its
    ``spawned`` / ``worker_registered`` event (the reaper clears the row once the pid is gone); host
    from the run's claim lock, else its ``claimed`` event. A claimed run with no pid stays an unknown
    worker (pending), never "no worker"."""
    row = conn.execute(
        "SELECT worker_pid, worker_started_at, claim_lock FROM task_runs WHERE id = ?", (run_id,),
    ).fetchone()
    pid = row["worker_pid"] if row is not None else None
    fingerprint = row["worker_started_at"] if pid else None
    if not pid:
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
            "AND kind IN ('spawned', 'worker_registered') ORDER BY id DESC LIMIT 1", (task_id, run_id),
        ).fetchone()
        payload = _kb._json_dict(_kb._row_get(event, "payload"))
        pid, fingerprint = payload.get("pid"), payload.get("started_at")
    lock = row["claim_lock"] if row is not None else None
    if not lock:
        claimed = _kb._latest_event(conn, task_id, "claimed", run_id)
        lock = _kb._json_dict(_kb._row_get(claimed, "payload")).get("lock")
    return {"kind": "worker", "host": str(lock).split(":", 1)[0] if lock else None,
            "pid": int(pid) if pid else None, "fingerprint": fingerprint if pid else None}


_NO_WORKER = {"kind": None, "host": None, "pid": None, "fingerprint": None}


def _origin_workers(conn, task_id: str, run_id: Optional[int]) -> tuple[dict, dict]:
    """``(implementation, approver)`` exit provenance for a submission, from durable ``claimed``
    events. The implementation worker is the newest claim not taken from ``review``; a reviewer's
    exit never stands in for it. The approver is the newest review claim after that implementation
    claim (a reviewer approving now, or one whose approval an operator re-completes). No
    implementation claim at all = the work was never claimed (``synthesized``, no worker); an
    approver that exists must also be proven gone."""
    claims = [
        (r["id"], r["run_id"], _kb._json_dict(r["payload"]).get("source_status"))
        for r in conn.execute(
            "SELECT id, run_id, payload FROM task_events WHERE task_id = ? AND kind = 'claimed' "
            "AND run_id IS NOT NULL ORDER BY id", (task_id,),
        )
    ]
    if run_id is not None and all(c[1] != run_id for c in claims):
        claims.append((float("inf"), run_id, None))  # an active run without its event: treat as implementation
    impl = next((c for c in reversed(claims) if c[2] != "review"), None)
    approver = next((c for c in reversed(claims) if c[2] == "review" and (impl is None or c[0] > impl[0])), None)
    impl_worker = _run_worker(conn, task_id, impl[1]) if impl else {**_NO_WORKER, "kind": "synthesized"}
    approver_worker = _run_worker(conn, task_id, approver[1]) if approver else dict(_NO_WORKER)
    return impl_worker, approver_worker


def submit_completion(
    conn: sqlite3.Connection, task_id: str, *, result: Optional[str], summary: Optional[str],
    handoff_summary: Optional[str], metadata: Optional[dict], verified_cards: list[str],
    expected_run_id: Optional[int], force: bool,
) -> bool:
    """``complete_task`` for a card that declares ``acceptance``: persist the handoff, preserve
    artifacts, close the worker run (``acceptance_submitted``), park the card in ``review`` and open
    one submission bound to the origin run, implementer, worker identity and spec/contract
    fingerprints. Not done: no ``completed`` event, hook or cleanup until the evaluator passes it.
    False when refused (stale run, not completable, or an open machine wait already exists)."""
    now = int(time.time())
    staged_copies: list[Path] = []
    try:
        with _kb.write_txn(conn):
            if not _kb._parents_satisfied(conn, task_id):
                return False
            trow = conn.execute(
                "SELECT status, assignee, claim_lock, current_run_id, worker_pid, worker_started_at, "
                "acceptance, completion_contract FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            if trow is None or trow["status"] not in ("running", "ready", "blocked", "review"):
                return False
            if not trow["acceptance"]:
                # Removed between routing and this txn: never wait on a spec that no longer exists.
                raise AcceptanceSubmissionError(
                    "the card's acceptance was removed while completing; nothing changed — retry the completion")
            prior_status = trow["status"]
            stale = open_submission(conn, task_id)
            if stale is not None:
                if prior_status == "review":
                    return False  # a machine wait: only the evaluator decides it
                _cancel(conn, stale, "superseded", now)
            if expected_run_id is None and not force and _kb._claim_is_live(trow):
                raise _kb.LiveClaimError(task_id)
            if expected_run_id is not None and trow["current_run_id"] != int(expected_run_id):
                return False
            contract = _bind_contract(trow["completion_contract"], metadata)
            run_id = trow["current_run_id"]
            reviewing = prior_status == "review"
            if run_id is not None:
                claimed = _kb._json_dict(_kb._row_get(_kb._latest_event(conn, task_id, "claimed", run_id), "payload"))
                reviewing = reviewing or claimed.get("source_status") == "review"
            run_row = conn.execute(
                "SELECT id, profile FROM task_runs WHERE id = ?", (run_id,),
            ).fetchone() if run_id is not None else None
            implementer = _implementer(conn, task_id, trow, run_row, reviewing)
            worker, approver = _origin_workers(conn, task_id, run_id)
            cur = conn.execute(
                "UPDATE tasks SET status = 'review', result = ?, completion_contract = ?, "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status IN ('running', 'ready', 'blocked', 'review')"
                + ("" if expected_run_id is None else " AND current_run_id = ?"),
                (result, contract, task_id, *(() if expected_run_id is None else (int(expected_run_id),))),
            )
            if cur.rowcount != 1:
                return False
            if isinstance(metadata, dict):
                staged_copies = _kb._stage_completion_artifacts(conn, task_id, metadata, now)
            origin_run_id = _kb._end_run(
                conn, task_id, outcome="acceptance_submitted", status="review",
                summary=handoff_summary, metadata=metadata,
            )
            event_summary = handoff_summary
            if origin_run_id is None:
                synth_summary, synth_metadata = handoff_summary, metadata
                if prior_status == "review" and not synth_summary and not synth_metadata:
                    synth_summary = _kb._REVIEW_APPROVED_NOTE
                    synth_metadata = {"source_status": "review", "approval": "manual"}
                origin_run_id = _kb._synthesize_ended_run(
                    conn, task_id, outcome="acceptance_submitted", summary=synth_summary, metadata=synth_metadata,
                )
            if prior_status == "review" and not event_summary:
                event_summary = _kb._REVIEW_APPROVED_NOTE
            sub_cur = conn.execute(
                """
                INSERT INTO task_acceptance_submissions (
                    task_id, origin_run_id, implementer, spec_fp, contract_fp,
                    worker_host, worker_pid, worker_fingerprint, origin_kind,
                    approver_host, approver_pid, approver_fingerprint, approver_kind,
                    state, attempts, next_check_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', 0, ?, ?)
                """,
                (
                    task_id, origin_run_id, implementer, _ka.fingerprint(trow["acceptance"]),
                    _ka.fingerprint(contract),
                    worker["host"], worker["pid"], worker["fingerprint"], worker["kind"],
                    approver["host"], approver["pid"], approver["fingerprint"], approver["kind"],
                    None if (_ka.load_spec(trow["acceptance"]) or {}).get("kind") == "tbd" else now, now,
                ),
            )
            completed_payload = _kb._completed_event_payload(result, event_summary, verified_cards, metadata)
            _kb._append_event(conn, task_id, "acceptance_submitted", {
                "submission_id": int(sub_cur.lastrowid), "implementer": implementer,
                "acceptance": _ka.load_spec(trow["acceptance"]), "completion_contract": contract,
                "completed": completed_payload,
            }, run_id=origin_run_id)
    except Exception:
        if staged_copies:
            _kb._discard_staged_copies(staged_copies, staged_copies[0].parent)
        raise
    _kb._flag_phantom_prose_refs(conn, task_id, origin_run_id, summary, result, verified_cards)
    return True


# --------------------------------------------------------------------------- evaluation lifecycle

def due_submission_ids(conn: sqlite3.Connection, now: int, limit: int = 100) -> list[int]:
    return [int(r["id"]) for r in conn.execute(
        "SELECT id FROM task_acceptance_submissions WHERE state = 'open' AND next_check_at IS NOT NULL "
        "AND next_check_at <= ? AND (lease_until IS NULL OR lease_until < ?) "
        "ORDER BY next_check_at, id LIMIT ?", (now, now, limit),
    )]


def _task_snapshot(conn, task_id: str):
    return conn.execute(
        "SELECT status, assignee, acceptance, completion_contract, current_run_id, "
        "(SELECT MAX(id) FROM task_runs r WHERE r.task_id = tasks.id) AS latest_run_id "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()


def _diverged(sub, task) -> Optional[str]:
    """Why ``task`` no longer matches the submission it is waiting on, or None. The origin run must
    still be the card's newest run: any later attempt is a new lifecycle generation."""
    if task is None or task["status"] != "review" or task["current_run_id"] is not None:
        return "task_left_review"
    if sub["origin_run_id"] is not None and task["latest_run_id"] != sub["origin_run_id"]:
        return "superseded_run"
    if (_ka.fingerprint(task["acceptance"]) != sub["spec_fp"]
            or _ka.fingerprint(task["completion_contract"]) != sub["contract_fp"]):
        return "spec_changed"
    return None


def lease_submission(conn: sqlite3.Connection, sub_id: int, owner: str, *, lease_seconds: int) -> Optional[dict]:
    """Short txn: claim one due submission for ``owner`` and snapshot what the verdict will be
    checked against. None when another pass holds it or it is no longer due; ``{"cancelled": …}``
    when the card already left the machine wait (nothing to evaluate)."""
    now = int(time.time())
    with _kb.write_txn(conn):
        sub = conn.execute(
            "SELECT * FROM task_acceptance_submissions WHERE id = ? AND state = 'open' "
            "AND next_check_at IS NOT NULL AND next_check_at <= ? AND (lease_until IS NULL OR lease_until < ?)",
            (sub_id, now, now),
        ).fetchone()
        if sub is None:
            return None
        task = _task_snapshot(conn, sub["task_id"])
        reason = _diverged(sub, task)
        if reason is not None:
            _cancel(conn, sub, reason, now, route=True)
            return {"cancelled": reason, "task_id": sub["task_id"]}
        conn.execute(
            "UPDATE task_acceptance_submissions SET lease_owner = ?, lease_until = ? WHERE id = ?",
            (owner, now + lease_seconds, sub_id),
        )
    return {
        "submission": sub, "submission_id": sub_id, "owner": owner, "task_id": sub["task_id"],
        "spec_fp": sub["spec_fp"], "contract_fp": sub["contract_fp"], "assignee": task["assignee"],
        "acceptance": task["acceptance"], "completion_contract": task["completion_contract"],
    }


def _release(conn, sub_id: int, *, recheck_now: Optional[int] = None) -> None:
    conn.execute(
        "UPDATE task_acceptance_submissions SET lease_owner = NULL, lease_until = NULL"
        + (", next_check_at = ?" if recheck_now is not None else "") + " WHERE id = ?",
        ((recheck_now, sub_id) if recheck_now is not None else (sub_id,)),
    )


def apply_verdict(conn: sqlite3.Connection, lease: dict, results: list[dict], *,
                  failure_limit: Optional[int] = None) -> Optional[dict]:
    """CAS txn: apply the evaluated ``results`` only if the submission is still open under this
    lease with the leased spec/contract, and the card is still the same machine wait (status,
    assignee, spec, contract, parents). A lost lease or re-bound spec applies nothing; a card that
    moved on cancels the submission. Returns ``{verdict, reason}`` when something was recorded."""
    now = int(time.time())
    sub_id, task_id = lease["submission_id"], lease["task_id"]
    verdict, reason = _ka.combine(results)
    receipt = _ka.bounded_receipt(results)
    passed = False
    with _kb.write_txn(conn):
        sub = conn.execute("SELECT * FROM task_acceptance_submissions WHERE id = ?", (sub_id,)).fetchone()
        if sub is None or sub["state"] != "open" or sub["lease_owner"] != lease["owner"]:
            return None
        if sub["spec_fp"] != lease["spec_fp"] or sub["contract_fp"] != lease["contract_fp"]:
            _release(conn, sub_id)  # spec re-bound mid-evaluation: the next pass judges the new one
            return None
        task = _task_snapshot(conn, task_id)
        diverged = _diverged(sub, task)
        if diverged is None and task["assignee"] != lease["assignee"]:
            _release(conn, sub_id, recheck_now=now)  # identity changed: re-evaluate as the new assignee
            return None
        if diverged is None and not _kb._parents_satisfied(conn, task_id):
            diverged = "parents_unsatisfied"
        if diverged is not None:
            _cancel(conn, sub, diverged, now, route=True)
            return {"verdict": "cancelled", "reason": diverged}
        attempts = int(sub["attempts"] or 0) + 1
        last_reason = _kb._json_dict(sub["last_receipt"]).get("reason")
        stored = json.dumps({"verdict": verdict, "reason": reason, "results": json.loads(receipt)},
                            ensure_ascii=False, sort_keys=True)
        payload = {"verdict": verdict, "reason": reason, "submission_id": sub_id,
                   "results": json.loads(receipt)}
        if verdict == "pending":
            next_check = None if reason == "tbd" else now + _ka.backoff_seconds(attempts)
            conn.execute(
                "UPDATE task_acceptance_submissions SET attempts = ?, verdict = 'pending', last_receipt = ?, "
                "next_check_at = ?, lease_owner = NULL, lease_until = NULL WHERE id = ?",
                (attempts, stored, next_check, sub_id),
            )
            if (sub["verdict"], last_reason) != ("pending", reason):
                _verdict_event(conn, task_id, sub["origin_run_id"], {**payload, "next_check_at": next_check})
            return {"verdict": verdict, "reason": reason}
        decided = conn.execute(
            "UPDATE task_acceptance_submissions SET state = ?, verdict = ?, attempts = ?, last_receipt = ?, "
            "decided_at = ?, next_check_at = NULL, lease_owner = NULL, lease_until = NULL "
            "WHERE id = ? AND state = 'open'",
            ("passed" if verdict == "pass" else "failed", verdict, attempts, stored, now, sub_id),
        )
        if decided.rowcount != 1:
            return None
        if verdict == "fail":
            _apply_fail(conn, sub, task_id, reason, results, payload, failure_limit)
        else:
            _apply_pass(conn, sub, task_id, payload, now)
            passed = True
    if passed:
        _finalize_done(conn, task_id, sub["origin_run_id"])
    return {"verdict": verdict, "reason": reason}


def _apply_fail(conn, sub, task_id: str, reason: str, results: list[dict], payload: dict,
                failure_limit: Optional[int]) -> None:
    """Back to the implementer (dependency-aware ready/todo), failure on the origin run's ``error``
    so the next worker context shows it, workspace kept, breaker charged exactly once."""
    from hermes_cli.kanban_db_dispatch import _record_task_failure
    new_status = _kb._landing_status_after_parents(conn, task_id)
    conn.execute(
        "UPDATE tasks SET status = ?, assignee = COALESCE(?, assignee), claim_lock = NULL, "
        "claim_expires = NULL, worker_pid = NULL WHERE id = ? AND status = 'review'",
        (new_status, sub["implementer"], task_id),
    )
    text = _ka.failure_text(results)
    if sub["origin_run_id"] is not None:
        conn.execute(
            "UPDATE task_runs SET outcome = 'acceptance_failed', status = 'acceptance_failed', error = ? "
            "WHERE id = ?", (text, sub["origin_run_id"]),
        )
    _verdict_event(conn, task_id, sub["origin_run_id"], {
        **payload, "implementer": sub["implementer"], "status": new_status, "error": text[:400]})
    # Fixed text from an enumerated reason: evaluator output never reaches last_failure_error,
    # where the respawn guard's auth/quota pattern would park the card.
    code = reason if reason in _ka.FAIL_REASONS else "unspecified"
    _record_task_failure(
        conn, task_id, f"machine acceptance failed ({code}); see the latest attempt's error",
        outcome="acceptance_failed", failure_limit=failure_limit, _nested=True,
    )


def _apply_pass(conn, sub, task_id: str, payload: dict, now: int) -> None:
    """The one terminal write: same task columns as ``complete_task``'s done branch, the origin run
    recorded ``completed``, and the ``completed`` event built at submission time."""
    cur = conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = ?, claim_lock = NULL, claim_expires = NULL, "
        "worker_pid = NULL, block_kind = NULL, block_recurrences = 0 "
        "WHERE id = ? AND status = 'review' AND current_run_id IS NULL",
        (now, task_id),
    )
    if cur.rowcount != 1:
        raise RuntimeError(f"acceptance finalizer lost the terminal CAS on {task_id}")
    if sub["origin_run_id"] is not None:
        conn.execute("UPDATE task_runs SET outcome = 'completed', status = 'done' WHERE id = ?",
                     (sub["origin_run_id"],))
    _verdict_event(conn, task_id, sub["origin_run_id"], payload)
    completed = {}
    for row in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'acceptance_submitted' ORDER BY id DESC",
        (task_id,),
    ):
        submitted = _kb._json_dict(row["payload"])
        if submitted.get("submission_id") == sub["id"]:
            completed = submitted.get("completed") or {}
            break
    _kb._append_event(conn, task_id, "completed", completed, run_id=sub["origin_run_id"])


def _finalize_done(conn, task_id: str, run_id: Optional[int]) -> None:
    """Post-commit side effects of a pass, each once: breaker reset, dependants released,
    workspace cleaned, ``kanban_task_completed`` hook."""
    _kb._clear_failure_counter(conn, task_id)
    _kb.recompute_ready(conn)
    _kb._cleanup_workspace(conn, task_id)
    run = _kb.get_run(conn, run_id) if run_id is not None else None
    _kb._fire_task_hook("kanban_task_completed", _kb.get_task(conn, task_id), task_id, run_id,
                        summary=run.summary if run else None)


# --------------------------------------------------------------------------- operator edit

def set_acceptance(conn: sqlite3.Connection, task_id: str, acceptance: Optional[dict]) -> bool:
    """Operator edit of ``tasks.acceptance`` (validated; None clears). An open submission is
    re-bound to the new spec and due now, so an in-flight evaluation of the old spec loses its CAS;
    clearing the spec cancels the submission. False for an unknown, done or archived card."""
    canonical = _ka.validate_acceptance(acceptance)
    now = int(time.time())
    with _kb.write_txn(conn):
        status = _kb._task_status(conn, task_id)
        if status is None or status in ("done", "archived"):
            return False
        conn.execute("UPDATE tasks SET acceptance = ? WHERE id = ?", (canonical, task_id))
        sub = open_submission(conn, task_id)
        rebound = None
        if sub is not None and canonical is None:
            _cancel(conn, sub, "acceptance_removed", now, route=True)
        elif sub is not None:
            due = None if (_ka.load_spec(canonical) or {}).get("kind") == "tbd" else now
            conn.execute(
                "UPDATE task_acceptance_submissions SET spec_fp = ?, next_check_at = ?, lease_owner = NULL, "
                "lease_until = NULL WHERE id = ?", (_ka.fingerprint(canonical), due, sub["id"]),
            )
            rebound = sub["id"]
        _kb._append_event(conn, task_id, "acceptance_updated", {
            "acceptance": _ka.load_spec(canonical), "submission_id": rebound})
    _kb.notify_task_updated(conn, task_id, ["acceptance"])
    return True

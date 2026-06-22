"""Producer + consumer for the code-craftsman ↔ wags-reviewer review loop.

Why this module exists
----------------------
The kanban has all the primitives for a review loop — ``status='review'``
(``claim_review_task``), reassignment (``reassign_task``), and a stranded
reaper (``reap_stranded_in_ready``) — but nothing closes the producer
half. When a worker calls ``kanban_block(reason="review-required: ...")``
the task lands in ``status='blocked'`` under the worker's own assignee.
``claim_review_task`` only matches ``status='review'``, so the card sits
in the worker's blocked queue forever and the reviewer's queue stays
empty.

Two functions close that gap:

* :func:`handoff_blocked_to_review` — producer. Find tasks whose latest
  ``blocked`` event payload contains ``"review-required:"``, reassign
  them to ``wags-reviewer``, and flip status to ``'review'``. Idempotent:
  already-routed cards (a ``review_handoff`` event newer than the latest
  ``blocked`` event) are skipped.
* :func:`completion_route` — consumer. After wags-reviewer marks a card
  ``done``, parse PASS / FAIL / WAIVER from the last comment. PASS
  leaves the card closed (an orchestrator or operator closes the loop).
  FAIL / WAIVER reassign back to the original assignee recorded in the
  matching ``review_handoff`` event and set status to ``'ready'``.

Wiring
------
Both functions are called from :func:`dispatch_once` in ``kanban_db``
next to ``reap_stranded_in_ready`` so the loop runs every dispatcher
tick (the daemon ticks once a minute; this is the de-facto safety net).
The producer is intentionally idempotent so cron-style callers may
invoke it as often as they like without duplicating events.

Verdict contract for wags-reviewer
----------------------------------
The reviewer is expected to leave a comment on the card whose body
contains one of these tokens (case-insensitive):

* ``Verdict: PASS`` — ship as-is
* ``Verdict: FAIL`` — needs rework; original assignee picks it up
* ``Verdict: WAIVER`` — accept the risk, but route back anyway

A bare ``PASS`` at the very start of the last comment is also accepted
as a weak signal, but the explicit ``Verdict:`` form is preferred so
reviews can describe FAIL cases in prose without the parser grabbing
the wrong word.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Optional

from hermes_cli.kanban_db import (
    _append_event,
    assign_task,
    reassign_task,
    write_txn,
)


# Public constants — exported so tests + the CLI can import the same
# values rather than re-declaring string literals.
REVIEW_REQUIRED_PREFIX = "review-required:"
REVIEWER = "wags-reviewer"
HANDOFF_EVENT = "review_handoff"
COMPLETION_EVENT = "review_completion"

# Verdict strings. The parser accepts "PASS" / "FAIL" / "WAIVER" only
# when prefixed with "Verdict:" / "Decision:" so the bare tokens don't
# match every comment that mentions failure.
VERDICT_RE = re.compile(
    r"\b(?:verdict|decision)\s*[:=]\s*(PASS|FAIL|WAIVER)\b",
    re.IGNORECASE,
)


def parse_verdict(comment_body: Optional[str]) -> Optional[str]:
    """Return ``"PASS"`` / ``"FAIL"`` / ``"WAIVER"`` or ``None``.

    Strong signal: a ``Verdict: PASS|FAIL|WAIVER`` (or ``Decision:``)
    substring anywhere in the comment. Weak signal: a bare ``PASS`` at
    the very start of the comment, after stripping markdown emphasis
    (``*``, ``_``, ``#``). FAIL / WAIVER without the ``Verdict:`` prefix
    are intentionally NOT matched — too many review comments mention
    "failure" in prose for the bare match to be reliable.
    """
    if not comment_body:
        return None
    m = VERDICT_RE.search(comment_body)
    if m:
        return m.group(1).upper()
    # Weak fallback — only PASS. We deliberately don't try to detect
    # FAIL here; the strong-signal path is the contract.
    head = comment_body.strip().split("\n", 1)[0].strip().lstrip("*_# ")
    if head.startswith("PASS"):
        return "PASS"
    return None


def _latest_blocked_subquery() -> str:
    """SQL fragment: the ``created_at`` of the latest ``blocked`` event
    for a given task id, used as a correlated anchor in the handoff
    SELECT. Pulled out so the producer and any future callers share the
    exact same expression (and so the test fixture can compare against
    it without copy-paste)."""
    return (
        "(SELECT MAX(e.created_at) FROM task_events e "
        " WHERE e.task_id = t.id AND e.kind = 'blocked')"
    )


def _select_blocked_review_candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Tasks eligible for hand-off right now.

    Conditions, all on the same row:

    * ``status = 'blocked'`` and assignee is not already the reviewer
      (idempotent re-runs skip cards already in flight).
    * The latest ``blocked`` event's payload contains the
      ``review-required:`` marker. Earlier blocked events without the
      marker do NOT qualify — only the most recent reason drives the
      transition.
    * No ``review_handoff`` event newer than that latest ``blocked``
      event has been emitted yet. After a hand-off, a later
      ``kanban_block`` produces a new blocked event with a fresh
      timestamp; the next tick will see the gap and re-hand-off if
      still relevant (e.g. FAIL→ready→claim→block cycles).
    """
    latest_blocked = _latest_blocked_subquery()
    rows = conn.execute(
        f"""
        SELECT t.id, t.assignee
        FROM tasks t
        WHERE t.status = 'blocked'
          AND t.assignee != ?
          AND EXISTS (
            SELECT 1 FROM task_events e
            WHERE e.task_id = t.id
              AND e.kind = 'blocked'
              AND e.payload LIKE '%' || ? || '%'
              AND e.created_at = {latest_blocked}
          )
          AND NOT EXISTS (
            SELECT 1 FROM task_events h
            WHERE h.task_id = t.id
              AND h.kind = ?
              AND h.created_at > {latest_blocked}
          )
        """,
        (REVIEWER, REVIEW_REQUIRED_PREFIX, HANDOFF_EVENT),
    ).fetchall()
    return rows


def handoff_blocked_to_review(conn: sqlite3.Connection) -> list[str]:
    """Move ``blocked(review-required)`` cards into the review queue.

    For each eligible card:

    1. Reassign to ``wags-reviewer`` (resets the per-profile failure
       counter via :func:`reassign_task` — the reviewer is a different
       profile from the original worker and should not inherit the
       worker's streak). ``reassign_task`` opens its own write
       transaction.
    2. Flip ``status`` from ``'blocked'`` to ``'review'``. The
       ``AND status = 'blocked'`` guard prevents racing an unblock.
    3. Emit a ``review_handoff`` event with the previous assignee so
       the consumer can route a FAIL back to the right worker.

    Steps 2+3 share a single :func:`write_txn` so the row state and
    the audit log can never disagree. The window between step 1 and
    step 2+3 is intentional: a rare crash between them leaves the
    card ``blocked`` but ``assignee='wags-reviewer'`` — the
    ``assignee != wags-reviewer`` precondition in
    :func:`_select_blocked_review_candidates` will then skip it on the
    next tick, and the operator can manually flip status. Returns the
    list of successfully handed-off task ids (in the order they were
    processed); an empty list means "nothing to do".
    """
    candidates = _select_blocked_review_candidates(conn)
    handed: list[str] = []
    for row in candidates:
        task_id = row["id"]
        from_assignee = row["assignee"]
        # Blocked tasks always have claim_lock=NULL (set by block_task)
        # so reassign_task cannot hit the still-running guard.
        if not reassign_task(conn, task_id, REVIEWER):
            continue
        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'review' "
                "WHERE id = ? AND status = 'blocked'",
                (task_id,),
            )
            if cur.rowcount != 1:
                # Someone else moved the task out of blocked between
                # the SELECT and now (e.g. an operator unblock).
                # reassign_task already mutated the assignee; that's a
                # no-op side effect (the operator unblocked
                # intentionally) so we just skip the status flip.
                continue
            _append_event(
                conn, task_id, HANDOFF_EVENT,
                {"from_assignee": from_assignee, "to_assignee": REVIEWER},
            )
        handed.append(task_id)
    return handed


def _select_review_done(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Tasks the reviewer marked ``done`` and that still need routing.

    The reviewer is expected to ``complete_task`` (or have the
    orchestrator do it on their behalf); either way the row lands in
    ``status='done'`` with ``assignee='wags-reviewer'``. We additionally
    require that a ``review_handoff`` event exists so we don't try to
    re-route a card that was always reviewer-owned.
    """
    return conn.execute(
        """
        SELECT t.id
        FROM tasks t
        WHERE t.assignee = ?
          AND t.status = 'done'
          AND EXISTS (
            SELECT 1 FROM task_events e
            WHERE e.task_id = t.id AND e.kind = ?
          )
        """,
        (REVIEWER, HANDOFF_EVENT),
    ).fetchall()


def _latest_comment_body(
    conn: sqlite3.Connection, task_id: str,
) -> Optional[str]:
    row = conn.execute(
        "SELECT body FROM task_comments "
        "WHERE task_id = ? ORDER BY created_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return row["body"] if row else None


def _original_assignee_from_handoff(
    conn: sqlite3.Connection, task_id: str,
) -> Optional[str]:
    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (task_id, HANDOFF_EVENT),
    ).fetchone()
    if not row or not row["payload"]:
        return None
    try:
        payload = json.loads(row["payload"])
    except (TypeError, ValueError):
        return None
    return payload.get("from_assignee")


def completion_route(conn: sqlite3.Connection) -> list[str]:
    """Route wags-reviewer ``done`` cards back per their verdict.

    For each reviewer-done card with a ``review_handoff`` history:

    * Verdict ``PASS`` — emit a ``review_completion`` event with
      ``action='closed'`` and leave the card as ``done``. The
      orchestrator (or the operator) closes the loop; we do not
      unclaim or unassign.
    * Verdict ``FAIL`` or ``WAIVER`` — look up the
      ``from_assignee`` recorded in the most recent
      ``review_handoff`` event and reassign. If we can resolve the
      original assignee, flip status to ``'ready'`` so the worker
      picks it up on the next dispatch tick; if we can't, log a
      ``review_completion`` event with ``action='no_original_assignee'``
      and leave the row alone (operator action required).
    * No parseable verdict — skip silently. The reviewer may still be
      drafting, or the comment may not be the final verdict. The card
      stays ``done`` under the reviewer; the next tick re-checks.

    Returns the list of task ids touched by this pass (both PASS
    audits and FAIL re-routes).
    """
    rows = _select_review_done(conn)
    routed: list[str] = []
    for row in rows:
        task_id = row["id"]
        verdict = parse_verdict(_latest_comment_body(conn, task_id))
        if verdict is None:
            continue
        if verdict == "PASS":
            _append_event(
                conn, task_id, COMPLETION_EVENT,
                {"verdict": "PASS", "action": "closed"},
            )
            routed.append(task_id)
            continue
        # FAIL or WAIVER — reassign to the original worker.
        from_assignee = _original_assignee_from_handoff(conn, task_id)
        if not from_assignee:
            _append_event(
                conn, task_id, COMPLETION_EVENT,
                {"verdict": verdict, "action": "no_original_assignee"},
            )
            routed.append(task_id)
            continue
        # reassign_task opens its own write txn. We then flip status
        # + emit the audit event in a separate write_txn so the row
        # state and the audit log can never disagree.
        if not reassign_task(conn, task_id, from_assignee):
            _append_event(
                conn, task_id, COMPLETION_EVENT,
                {"verdict": verdict, "action": "reassign_failed"},
            )
            routed.append(task_id)
            continue
        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready' "
                "WHERE id = ? AND status = 'done'",
                (task_id,),
            )
            if cur.rowcount != 1:
                _append_event(
                    conn, task_id, COMPLETION_EVENT,
                    {"verdict": verdict,
                     "action": "status_change_raced"},
                )
                routed.append(task_id)
                continue
            _append_event(
                conn, task_id, COMPLETION_EVENT,
                {"verdict": verdict,
                 "action": "reassigned",
                 "to_assignee": from_assignee},
            )
        routed.append(task_id)
    return routed


__all__ = [
    "REVIEW_REQUIRED_PREFIX",
    "REVIEWER",
    "HANDOFF_EVENT",
    "COMPLETION_EVENT",
    "parse_verdict",
    "handoff_blocked_to_review",
    "completion_route",
]

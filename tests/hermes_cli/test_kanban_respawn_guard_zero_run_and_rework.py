"""Regression tests for the two ``active_pr`` respawn-guard false positives.

Both shapes are reported upstream in
`NousResearch/hermes-agent#62418 <https://github.com/NousResearch/hermes-agent/issues/62418>`_.

**(A) A never-run card referencing a PR is held unconditionally.** Step 4 of
``check_respawn_guard`` greps ``task_comments`` for a PR URL and, absent a
handoff event, returns ``active_pr`` — with no ``task_runs`` query at all. A
card whose *contract* names a PR (a merge-gate or review card, an operator's
create-time note) records no run at all, so a spawn cannot duplicate anything;
the guard must yield before the comment-regex gate.

**(B) A worker's own rework push note re-arms the guard.** The step-4 relief
valve searches handoff events strictly AFTER the newest PR-bearing comment. A
``changes_requested`` verdict followed by the worker's own "rework pushed"
note leaves no later handoff event, so ``active_pr`` re-fires on the very card
whose only remaining work is finishing that same PR. The rule must be
*state-based* — the latest review-trail event is a rework demand — not
ordering-dependent.

The intended duplicate-work case must still be guarded: a completed run, a
PR URL in a comment, no requeue and no further handoff.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


PR_URL = "https://github.com/NousResearch/hermes-agent/pull/42"
PR_COMMENT = f"Opened {PR_URL} for review."


def _backdate_comments(conn, tid, seconds: int = 60) -> None:
    """Second-granularity timestamps: make a comment older than what follows it."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = created_at - ? WHERE task_id = ?",
            (seconds, tid),
        )


def _insert_event(
    conn, tid, kind: str, payload: dict | None = None, *, created_at: int | None = None,
) -> None:
    """Insert a task event with a pinned timestamp.

    A direct INSERT (not ``kb._append_event``) so a test can pin a same-second
    tie with the PR comment or backdate a housekeeping event.
    """
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, ?, ?, ?)",
            (
                tid, kind,
                None if payload is None else json.dumps(payload),
                int(time.time()) if created_at is None else int(created_at),
            ),
        )


def _prior_completed_run(conn, tid, *, summary: str = "opened the PR") -> int:
    """A worker run of THIS task that ended two hours ago.

    Two hours back so the one-hour ``recent_success`` window never masks what
    the ``active_pr`` assertions are testing.
    """
    old = int(time.time()) - 7200
    run_id = kb._synthesize_ended_run(
        conn, tid, outcome="completed", summary=summary, metadata={"_ended_at": old},
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET ended_at = ? WHERE id = ?", (old, run_id),
        )
    return run_id


# ---------------------------------------------------------------------------
# Shape (A): a card that never ran cannot have opened a PR
# ---------------------------------------------------------------------------


def test_active_pr_guard_yields_for_a_never_run_card(kanban_home: Path) -> None:
    """A zero-run card whose note links a PR must still spawn.

    The PR URL is contract, not evidence: no worker ever ran on this card, so
    re-spawning cannot duplicate a PR. Reproduces the issue's reported 24
    consecutive refusals over ~20 minutes with no runs.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="merge reviewed PR #42 into epic", assignee="dev",
        )
        kb.add_comment(conn, tid, author="default", body=f"Merge {PR_URL}")

        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?", (tid,),
        ).fetchone()["n"] == 0
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


def test_active_pr_guard_yields_after_a_failed_first_spawn(kanban_home: Path) -> None:
    """A ``spawn_failed`` run is not a worker run — the card must not be held.

    The worker never executed, so it never posted a PR. A failed first spawn
    must not start the 24h guard on the operator's own contract comment.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="merge reviewed PR #42", assignee="dev")
        kb.add_comment(conn, tid, author="default", body=f"Merge {PR_URL}")
        kb._synthesize_ended_run(
            conn, tid, outcome="spawn_failed", summary="no restart-safe scope",
        )
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


# ---------------------------------------------------------------------------
# Shape (B): a state-based rework rule, independent of comment ordering
# ---------------------------------------------------------------------------


def test_active_pr_guard_yields_when_the_worker_pushes_rework(kanban_home: Path) -> None:
    """changes_requested -> worker push note -> the card must be spawnable.

    The reviewer's verdict is the latest *review-trail* event; the worker's own
    push note merely reports progress on that same PR and must not re-arm the
    guard. Reproduces the issue's 133 refusals over ~2.2 hours.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rework the PR", assignee="dev")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, tid)

        assert kb.request_review(
            conn, tid, summary="PR ready", reviewer="reviewer",
            expected_run_id=run_id,
        )
        rclaim = kb.claim_review_task(conn, tid)
        ok, implementer = kb.request_changes(
            conn, tid, reason="fix the tests", expected_run_id=rclaim.current_run_id,
        )
        assert (ok, implementer) == (True, "dev")
        assert kb.get_task(conn, tid).status == "ready"

        # The worker's own note about the SAME PR, posted after the verdict.
        kb.add_comment(
            conn, tid, author="dev",
            body=f"Rework pushed on {PR_URL} — all findings addressed.",
        )

        events = [
            e.kind for e in kb.list_events(conn, tid)
        ]
        assert "changes_requested" in events
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


def test_active_pr_guard_yields_after_a_review_reopen(kanban_home: Path) -> None:
    """A review reopen is a rework demand too: the implementer must re-run.

    Same state-based rule as ``changes_requested`` — the latest review-trail
    event names the work, so a stale PR comment must not hold the card.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reopen review", assignee="dev")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, tid)
        assert kb.request_review(
            conn, tid, summary="PR ready", reviewer="reviewer",
            expected_run_id=run_id,
        )
        assert kb.reopen_review_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


# ---------------------------------------------------------------------------
# Shape (C): the deliberate operator unblock after the PR comment
# ---------------------------------------------------------------------------


def test_active_pr_guard_yields_after_an_operator_unblock(kanban_home: Path) -> None:
    """An explicit unblock after the PR comment is a deliberate re-run."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unblocked after pr", assignee="dev")
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, tid)
        assert kb.block_task(conn, tid, reason="hold", kind="needs_input") is True
        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


# ---------------------------------------------------------------------------
# Shape (D): the guard's actual job is preserved
# ---------------------------------------------------------------------------


def test_active_pr_guard_still_holds_the_duplicate_work_case(kanban_home: Path) -> None:
    """Completed run + PR comment + no requeue and no handoff => still guarded.

    This is the case the guard exists for: the implementer that opened the PR
    must not be re-spawned against it.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="already shipped", assignee="dev")
        _prior_completed_run(conn, tid)
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


def test_active_pr_guard_still_holds_after_a_crashed_run(kanban_home: Path) -> None:
    """A CRASH is execution, not a fresh start — the guard still holds.

    A crash/reclaim is not a handoff, so the worker that opened the PR is still
    not re-spawned against it. This is the boundary that distinguishes "no run
    ever executed" (shape A, guard yields) from "a run executed and died".
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="crashed after opening the pr", assignee="dev")
        old = int(time.time()) - 7200
        run_id = kb._synthesize_ended_run(
            conn, tid, outcome="crashed", summary="worker died", metadata={"_ended_at": old},
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET ended_at = ?, status = 'failed' WHERE id = ?",
                (old, run_id),
            )
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


def test_active_pr_guard_still_holds_after_a_push_note_with_no_rework_demand(
    kanban_home: Path,
) -> None:
    """A push note with NO review-trail event at all leaves the guard armed.

    The state-based rule in shape (B) must key on an actual rework demand, not
    on the note's text — otherwise any worker comment would lift the guard.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no verdcit yet", assignee="dev")
        _prior_completed_run(conn, tid)
        kb.add_comment(
            conn, tid, author="dev",
            body=f"Rework pushed on {PR_URL}",
        )
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


def test_active_pr_guard_still_holds_after_the_worker_ships_a_new_pr(
    kanban_home: Path,
) -> None:
    """A PR comment NEWER than the rework demand re-arms the guard.

    The rework demand names work on the OLD PR; a fresh PR URL posted after it
    is the duplicate-work signal the guard exists for.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="second pr", assignee="dev")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, tid, seconds=120)
        assert kb.request_review(
            conn, tid, summary="PR ready", reviewer="reviewer",
            expected_run_id=run_id,
        )
        rclaim = kb.claim_review_task(conn, tid)
        assert kb.request_changes(
            conn, tid, reason="fix", expected_run_id=rclaim.current_run_id,
        )[0]
        # A NEW PR opened after the verdict — genuinely different work.
        kb.add_comment(
            conn, tid, author="dev",
            body="Opened a separate https://github.com/NousResearch/hermes-agent/pull/99",
        )
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


# ---------------------------------------------------------------------------
# Shape (E): only an OPERATOR re-queue lifts the guard
# ---------------------------------------------------------------------------


def test_active_pr_guard_holds_after_a_stale_claim_reclaim(kanban_home: Path) -> None:
    """A ``reclaimed`` event is the stale-claim sweep, not an operator re-run.

    ``_record_reclaim`` fires uncommanded whenever a claim TTL expires, so
    honouring it respawns a worker against the PR it already opened — the
    duplicate-PR path this guard exists to close.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="claim expired after the pr", assignee="dev")
        _prior_completed_run(conn, tid)
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, tid)
        _insert_event(conn, tid, "reclaimed", {"reason": "claim_expired"})
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


def test_active_pr_guard_holds_after_a_recompute_ready_promotion(kanban_home: Path) -> None:
    """``promoted`` is ``recompute_ready`` housekeeping, not an operator verb.

    The dispatcher promotes ``todo``/``blocked`` cards on every tick once their
    parents are terminal; that is not a human asking for this card to run again.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="auto promoted after the pr", assignee="dev")
        _prior_completed_run(conn, tid)
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, tid)
        _insert_event(conn, tid, "promoted", {"status": "ready"})
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


def test_active_pr_guard_holds_on_a_same_second_operator_requeue(kanban_home: Path) -> None:
    """A same-second tie stays guarded (fail closed).

    Timestamps are second-granularity, so a tie between the PR comment and a
    re-queue event is routine. The handoff check above uses ``>`` for exactly
    this reason; the re-queue leg must match it.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="tie with the pr comment", assignee="dev")
        _prior_completed_run(conn, tid)
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        comment_at = int(
            conn.execute(
                "SELECT created_at FROM task_comments WHERE task_id = ? "
                "ORDER BY id DESC LIMIT 1", (tid,),
            ).fetchone()["created_at"]
        )
        _insert_event(conn, tid, "unblocked", None, created_at=comment_at)
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


def test_active_pr_guard_holds_on_an_ancestor_reopen_status_cascade(kanban_home: Path) -> None:
    """The descendant-invalidation ``status`` event names no operator: guarded.

    ``invalidate_descendants_for_parent_reopen`` writes a bare ``status`` event
    on a *descendant* when an ancestor is reopened. The operator acted on the
    ancestor, not on this card, so the event carries no ``actor`` — fail closed.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="retracted by an ancestor reopen", assignee="dev")
        _prior_completed_run(conn, tid)
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, tid)
        _insert_event(
            conn, tid, "status",
            {"status": "todo", "reason": "ancestor_reopened", "parent": "t_parent"},
        )
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


def test_active_pr_guard_yields_after_a_dashboard_status_move(kanban_home: Path) -> None:
    """The dashboard's direct status write (drag-drop) IS operator-initiated.

    A manual done→ready drag emits a ``status`` event carrying the acting
    surface; honouring it is the relief the re-queue leg exists for.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="dragged back to ready", assignee="dev")
        _prior_completed_run(conn, tid)
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, tid)
        _insert_event(
            conn, tid, "status",
            {"status": "ready", "requested_status": "ready", "actor": "dashboard"},
        )
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


def test_active_pr_guard_yields_after_an_operator_promote(kanban_home: Path) -> None:
    """``promoted_manual`` is the structured operator verb: the relief stays."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="operator promoted after the pr", assignee="dev")
        _prior_completed_run(conn, tid)
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, tid)
        assert kb.block_task(conn, tid, reason="hold", kind="needs_input") is True
        ok, err = kb.promote_task(conn, tid, actor="default", reason="run it again")
        assert (ok, err) == (True, None)
        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


# ---------------------------------------------------------------------------
# Dispatch integration: the card actually spawns
# ---------------------------------------------------------------------------


def test_never_run_pr_card_is_dispatched_not_guarded(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the held card reaches ``spawned`` with no guard reason."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="merge PR", assignee="dev")
        kb.add_comment(conn, tid, author="default", body=f"Merge {PR_URL}")
        res = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res.spawned]
        assert dict(res.respawn_guarded).get(tid) is None


def test_rework_card_is_dispatched_not_guarded(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: changes_requested + push note spawns its implementer."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rework", assignee="dev")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        kb.add_comment(conn, tid, author="dev", body=PR_COMMENT)
        _backdate_comments(conn, tid)
        kb.request_review(
            conn, tid, summary="ready", reviewer="reviewer", expected_run_id=run_id,
        )
        rclaim = kb.claim_review_task(conn, tid)
        kb.request_changes(
            conn, tid, reason="fix", expected_run_id=rclaim.current_run_id,
        )
        kb.add_comment(conn, tid, author="dev", body=f"Rework pushed on {PR_URL}")

        res = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res.spawned]
        assert dict(res.respawn_guarded).get(tid) is None

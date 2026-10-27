"""Review-lifecycle tests: the first-class ``running -> review`` transition.

``request_review`` is the "implementation complete, awaiting review"
transition used by executor workers instead of encoding ``review-required:``
prose into a ``kanban_block`` call. The critical contract these tests pin
down:

* It transitions ``running``/``ready`` -> ``review`` and closes the active
  run with ``outcome="review_requested"``.
* It emits exactly one ``review_requested`` event carrying the handoff
  summary + implementer.
* Crucially, it is NOT a blocker: repeated review requests on the same task
  (a review -> rerun -> review follow-up cycle) never touch
  ``block_recurrences`` and never route to ``triage`` — the false
  ``block_loop_detected`` escalation that plagued the block-reason approach
  cannot happen.
* ``expected_run_id`` is honoured as a CAS guard so a stale/superseded
  worker cannot move the task.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_ops

# The real PR-state resolver, captured before the autouse fixture swaps it out —
# the cache test needs the production function.
_REAL_RESOLVE_PR_STATE = kbd._resolve_pr_state


@pytest.fixture(autouse=True)
def _no_gh_pr_lookup(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No test in this file may shell out to ``gh``.

    CI has no GitHub auth and each lookup carries a 10s timeout, so the default
    PR state is "unknown" (fail closed). Tests that need a state patch
    ``kbd._resolve_pr_state`` with their own lambda; the cache is cleared on
    both sides so module state cannot leak between tests.
    """
    kbd._reset_pr_state_cache()
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: None)
    yield
    kbd._reset_pr_state_cache()


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _row(conn, tid):
    return conn.execute(
        "SELECT status, block_kind, block_recurrences, current_run_id "
        "FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    out = [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else None)
        for r in rows
    ]
    if kind is not None:
        out = [e for e in out if e[0] == kind]
    return out


def _last_run(conn, tid):
    return conn.execute(
        "SELECT status, outcome, summary FROM task_runs "
        "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()


# ---------------------------------------------------------------------------
# Happy path: running -> review
# ---------------------------------------------------------------------------


def test_request_review_transitions_running_to_review(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl a feature", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert run_id is not None

        ok = kb.request_review(
            conn, tid,
            summary="Implementation complete\nfull details below",
            reviewer="reviewer",
            expected_run_id=run_id,
        )
        assert ok is True

        row = _row(conn, tid)
        assert row["status"] == "review"
        # The active run is closed and the pointer cleared.
        assert row["current_run_id"] is None
        # Not a block: recurrence machinery is untouched.
        assert (row["block_recurrences"] or 0) == 0
        assert row["block_kind"] is None

        run = _last_run(conn, tid)
        assert run["outcome"] == "review_requested"
        assert run["status"] == "review"

        # Exactly one review_requested event, with the handoff payload.
        rr = _events(conn, tid, kind="review_requested")
        assert len(rr) == 1
        payload = rr[0][1]
        assert payload["implementer"] == "worker"
        assert payload["reviewer"] == "reviewer"
        # First line of the summary rides the event payload.
        assert payload["summary"] == "Implementation complete"
        # No block / triage events were emitted.
        assert _events(conn, tid, kind="blocked") == []
        assert _events(conn, tid, kind="block_loop_detected") == []


# ---------------------------------------------------------------------------
# Core regression: repeated review requests never escalate to triage
# ---------------------------------------------------------------------------


def test_repeated_review_requests_never_triage(kanban_home: Path) -> None:
    """A task that goes review -> rerun -> review again (the executor
    follow-up cycle) must stay in ``review`` every time. Under the old
    ``kanban_block(review-required:)`` approach the second pass hit
    ``block_recurrences >= 2`` and was wrongly routed to ``triage`` with a
    ``block_loop_detected`` event. ``request_review`` must never do that."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cycle me", assignee="worker")

        for _ in range(4):
            # Executor claims (ready->running or review->running) and finishes
            # with a review request. claim_review_task handles review->running.
            task = kb.get_task(conn, tid)
            if task.status == "ready":
                kb.claim_task(conn, tid)
            else:
                assert task.status == "review"
                claimed = kb.claim_review_task(conn, tid)
                assert claimed is not None

            run_id = kb.get_task(conn, tid).current_run_id
            ok = kb.request_review(
                conn, tid,
                summary="pass complete",
                expected_run_id=run_id,
            )
            assert ok is True
            row = _row(conn, tid)
            assert row["status"] == "review", "must never leave the review lane"
            assert (row["block_recurrences"] or 0) == 0

        # After several cycles: never triaged, never a false loop.
        assert _row(conn, tid)["status"] == "review"
        assert _events(conn, tid, kind="block_loop_detected") == []
        assert len(_events(conn, tid, kind="review_requested")) == 4


# ---------------------------------------------------------------------------
# CAS guard + bad-input behaviour
# ---------------------------------------------------------------------------


def test_request_review_expected_run_id_mismatch_is_noop(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale worker", assignee="worker")
        kb.claim_task(conn, tid)
        real_run = kb.get_task(conn, tid).current_run_id

        # A superseded worker passes a run id that is not the current one.
        ok = kb.request_review(conn, tid, expected_run_id=(real_run or 0) + 999)
        assert ok is False
        # Task is untouched — still running under the real run.
        row = _row(conn, tid)
        assert row["status"] == "running"
        assert row["current_run_id"] == real_run
        assert _events(conn, tid, kind="review_requested") == []


def test_request_review_unknown_task_returns_false(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        assert kb.request_review(conn, "t_deadbeefcafe") is False


def test_request_review_refuses_to_clear_live_claim_without_ownership(
    kanban_home: Path,
) -> None:
    """M1 regression: a run-id-less caller must not steal a live worker's claim.

    ``request_review`` on a running+claimed task without ``expected_run_id``
    fails with a distinct reason instead of silently NULLing claim_lock /
    worker_pid. ``force=True`` (explicit human override) and the worker path
    (``expected_run_id=<own run>``) both still work.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="live claim", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        # This process stands in for the spawned worker: alive, fingerprinted.
        kbd._set_worker_pid(conn, tid, os.getpid())

        # 1) No run id, no force -> refused with a distinct reason.
        ok, reason = kb.request_review(conn, tid, with_reason=True)
        assert ok is False
        assert reason is not None and "live claim" in reason
        row = conn.execute(
            "SELECT status, claim_lock, current_run_id FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["claim_lock"] is not None  # live claim untouched
        # bool-mode caller sees plain False.
        assert kb.request_review(conn, tid) is False

        # 2) Worker path: proving ownership via expected_run_id works.
        assert kb.request_review(
            conn, tid, summary="done", expected_run_id=claimed.current_run_id,
        ) is True
        assert kb.get_task(conn, tid).status == "review"

    # 3) force=True: explicit human override on a fresh live-claimed task.
    with kbc.connect() as conn:
        tid2 = kb.create_task(conn, title="forced", assignee="worker")
        assert kb.claim_task(conn, tid2) is not None
        assert kb.request_review(conn, tid2, summary="override", force=True) is True
        assert kb.get_task(conn, tid2).status == "review"


def test_request_review_malformed_provenance_gets_distinct_reason(
    kanban_home: Path,
) -> None:
    """M1 regression: malformed re-review provenance is a named failure, not
    the generic 'unknown id or not in running/ready'."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="provenance", assignee="builder")
        claimed = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert kb.request_changes(
            conn, tid, reason="fix", expected_run_id=review.current_run_id,
        ) == (True, "builder")
        # Corrupt the changes_requested payload so re-review cannot recover
        # the prior reviewer.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET payload = '{\"reviewer\": 42}' "
                "WHERE task_id = ? AND kind = 'changes_requested'",
                (tid,),
            )
        retry = kb.claim_task(conn, tid, claimer="builder:retry")
        assert retry is not None
        ok, reason = kb.request_review(
            conn, tid, summary="v2",
            expected_run_id=retry.current_run_id, with_reason=True,
        )
        assert ok is False
        assert reason is not None and "provenance" in reason
        # Passing reviewer explicitly recovers, as the reason instructs.
        assert kb.request_review(
            conn, tid, summary="v2", reviewer="reviewer",
            expected_run_id=retry.current_run_id,
        ) is True


@pytest.mark.parametrize("blank", ["   ", "\n", "\t\n  "])
def test_request_review_whitespace_only_summary_does_not_crash(
    kanban_home: Path, blank: str
) -> None:
    """A whitespace-only handoff summary must not crash the review transition.

    Regression: the event-summary extraction tested the truthiness of the
    *pre-strip* value while indexing the *post-strip* (empty) list, so a
    summary like ``"   "`` is truthy, ``.strip()`` collapses it to ``""``,
    ``"".splitlines()`` is ``[]`` and ``[][0]`` raised ``IndexError`` inside
    ``write_txn`` — a 500 on the dashboard PATCH/bulk path, which forwards
    ``summary`` unstripped (the tool/CLI paths pre-strip to ``None`` and were
    never exposed). The transition must still succeed and the event must
    carry ``summary=None`` (whitespace collapses to no summary).
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="blank summary", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id

        ok = kb.request_review(conn, tid, summary=blank, expected_run_id=run_id)
        assert ok is True
        assert kb.get_task(conn, tid).status == "review"

        rr = _events(conn, tid, kind="review_requested")
        assert len(rr) == 1
        # Whitespace collapses to no summary on the event payload.
        assert rr[0][1]["summary"] is None


# ---------------------------------------------------------------------------
# review -> done: a human can approve/close a task parked in review
# ---------------------------------------------------------------------------


def test_complete_task_closes_review_to_done(kanban_home: Path) -> None:
    """A task parked in ``review`` (with no active run — request_review
    closed it, so ``current_run_id IS NULL``, the #54823 shape) must be
    completable by a human approval via ``complete_task``."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="approve me", assignee="worker")
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="ready",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"
        # The review lane has no active run — the exact state that used to
        # make `hermes kanban complete` a no-op (#54823).
        assert kb.get_task(conn, tid).current_run_id is None

        ok = kb.complete_task(conn, tid, summary="LGTM — merged", result="approved")
        assert ok is True
        assert kb.get_task(conn, tid).status == "done"
        assert _events(conn, tid, kind="completed")


# ---------------------------------------------------------------------------
# Wake plumbing: review_requested is a claimable terminal event for a sub
# ---------------------------------------------------------------------------


def test_review_requested_event_is_claimable_for_wake(kanban_home: Path) -> None:
    """The gateway kanban-notifier wakes an origin subscription by claiming
    unseen events whose kind is in its terminal set. ``review_requested`` is
    now in that set, so a wake subscription must see the event — and the
    subscription is NOT torn down (task is in ``review``, not done/archived),
    so later review cycles keep notifying."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wake me", assignee="worker")
        kbn.add_notify_sub(
            conn,
            task_id=tid,
            platform="slack",
            chat_id="C123",
            thread_id="T1",
        )
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="please review",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )

        # Same terminal set the notifier now uses (incl. review_requested).
        terminal_kinds = (
            "completed", "blocked", "gave_up", "crashed", "timed_out",
            "review_requested",
        )
        _old, _new, events = kbn.claim_unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="slack",
            chat_id="C123",
            thread_id="T1",
            kinds=terminal_kinds,
        )
        kinds_seen = [e.kind for e in events]
        assert "review_requested" in kinds_seen
        # Task is parked in review — the subscription must survive (only
        # done/archived tears it down), so subsequent cycles still wake.
        assert kb.get_task(conn, tid).status == "review"


# ---------------------------------------------------------------------------
# Dispatcher gate: operators may opt out of autonomous review dispatch
# ---------------------------------------------------------------------------


def test_review_dispatch_gate_prevents_phantom_reviewer(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``kanban.review_dispatch=false`` the dispatcher must NOT claim a
    task parked in ``review`` (this deployment explicitly waits for a human).
    Flipping the knob back on proves the gate, not
    something else, is what suppressed the claim."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="park", assignee="worker")
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="done",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"

        # The assignee profile is spawnable — so ONLY the gate can stop the
        # review-column dispatch from claiming it.
        monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

        # Gate OFF -> review task is left alone.
        monkeypatch.setattr(
            cfgmod, "load_config",
            lambda *a, **k: {"kanban": {"review_dispatch": False}},
        )
        res_off = kbd.dispatch_once(conn, dry_run=True)
        assert tid not in [s[0] for s in res_off.spawned]
        assert kb.get_task(conn, tid).status == "review"

        # Gate ON (the default; sdlc-review is bundled) -> the review task is
        # picked up by the dispatcher.
        monkeypatch.setattr(
            cfgmod, "load_config",
            lambda *a, **k: {"kanban": {"review_dispatch": True}},
        )
        res_on = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res_on.spawned]


def test_active_pr_guard_skipped_for_review_lane_but_defers_ready_lane(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2 regression: a fresh PR-URL comment must not block reviewer spawns.

    A task parked in ``review`` with a PR link younger than 24h is the
    CANONICAL review handoff (worker opened a PR then requested review) —
    the review-lane dispatch must still claim/spawn it. The same comment on
    a ready-lane task is a duplicate-work signal and stays deferred.
    Rate-limit cooldown still applies in the review lane.
    """
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    pr_comment = "Opened https://github.com/example/repo/pull/123 for review."

    with kbc.connect() as conn:
        # Review-lane task with a fresh PR comment.
        review_id = kb.create_task(conn, title="review me", assignee="reviewer")
        claimed = kb.claim_task(conn, review_id)
        assert claimed is not None
        kb.add_comment(conn, review_id, author="worker", body=pr_comment)
        assert kb.request_review(
            conn, review_id, summary="PR ready",
            expected_run_id=claimed.current_run_id,
        )
        # Ready-lane task with the same fresh PR comment.
        ready_id = kb.create_task(conn, title="already PRed", assignee="worker",
                                  completion_contract=_IMPLEMENTER_CONTRACT)
        kb.add_comment(conn, ready_id, author="worker", body=pr_comment)

        assert kbd.check_respawn_guard(conn, ready_id) == "active_pr"
        assert kbd.check_respawn_guard(conn, review_id, lane="review") is None

        res = kbd.dispatch_once(conn, dry_run=True)
        spawned_ids = [s[0] for s in res.spawned]
        guarded = dict(res.respawn_guarded)
        assert review_id in spawned_ids
        assert ready_id not in spawned_ids
        assert guarded.get(ready_id) == "active_pr"

        # Rate-limit cooldown still defers the review lane.
        _now = int(__import__("time").time())
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at) VALUES (?, 'reviewer', 'rate_limited', "
                "'rate_limited', ?, ?)",
                # ended_at strictly after the review-handoff run so the
                # "latest run" query deterministically picks this one.
                (review_id, _now, _now + 5),
            )
        assert kbd.check_respawn_guard(
            conn, review_id, lane="review"
        ) == "rate_limit_cooldown"


def _backdate_comments(conn, tid, seconds=60):
    """Second-granularity timestamps: make the PR comment older than the
    handoff that follows it in the same test."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = created_at - ? WHERE task_id = ?",
            (seconds, tid),
        )


# ``create_task`` normalises an omitted contract to ``local-only``, which the
# ``active_pr`` guard treats as a report/notify card (no PR to duplicate). These
# tests are about implementer cards, so they must declare a PR-bearing contract
# explicitly — a repo slug is the canonical form for "this card publishes a PR".
_IMPLEMENTER_CONTRACT = "example/repo"


def test_active_pr_guard_lifts_for_profile_handed_the_card_after_the_pr(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ready card whose PR is open spawns the profile it was handed to.

    #111910: ``active_pr`` exists to stop the implementer from opening a
    duplicate PR; it must not stop the closer/recovery profile an operator
    assigned AFTER the PR comment — that handoff is why the PR must be worked.
    The un-reassigned implementer stays guarded; a newer PR comment posted
    after the handoff (the closer's own run) guards again.
    """
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    pr_comment = "Opened https://github.com/example/repo/pull/44 for review."

    with kbc.connect() as conn:
        dev_id = kb.create_task(conn, title="dev own pr", assignee="dev",
                                completion_contract=_IMPLEMENTER_CONTRACT)
        kb.add_comment(conn, dev_id, author="dev", body=pr_comment)
        closer_id = kb.create_task(conn, title="closer recovery", assignee="dev",
                                   completion_contract=_IMPLEMENTER_CONTRACT)
        kb.add_comment(conn, closer_id, author="dev", body=pr_comment)
        _backdate_comments(conn, closer_id)
        assert kb.assign_task(conn, closer_id, "closer") is True

        assert kbd.check_respawn_guard(conn, dev_id) == "active_pr"
        assert kbd.check_respawn_guard(conn, closer_id) is None

        res = kbd.dispatch_once(conn, dry_run=True)
        assert closer_id in [s[0] for s in res.spawned]
        assert dict(res.respawn_guarded).get(dev_id) == "active_pr"

        kb.add_comment(
            conn, closer_id, author="closer",
            body="Pushed to https://github.com/example/repo/pull/44",
        )
        assert kbd.check_respawn_guard(conn, closer_id) == "active_pr"


def test_active_pr_guard_holds_through_same_profile_reassign_and_unassign(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a handoff to a DIFFERENT profile lifts ``active_pr``.

    A no-op ``assign dev -> dev`` (CLI, dashboard PATCH, ``reassign --reclaim``)
    and an unassign both record an ``assigned`` event but change no owner; if
    they counted as handoffs the implementer would be re-spawned against its own
    PR — the duplicate-work protection #111910 says must survive.
    """
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
    pr_comment = "Opened https://github.com/example/repo/pull/44 for review."
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="same assign", assignee="dev",
                             completion_contract=_IMPLEMENTER_CONTRACT)
        kb.add_comment(conn, tid, author="dev", body=pr_comment)
        _backdate_comments(conn, tid)
        assert kb.assign_task(conn, tid, "dev") is True
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        assert kb.reassign_task(conn, tid, "dev", reclaim_first=True) is True
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"

        assert kb.assign_task(conn, tid, None) is True
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        # The dispatcher's own default_assignee write is not an operator handoff.
        res = kbd.dispatch_once(conn, dry_run=False, default_assignee="dev")
        assert tid in res.auto_assigned_default
        assert dict(res.respawn_guarded).get(tid) == "active_pr"
        assert tid not in [s[0] for s in res.spawned]

        # A real handoff after all of that still lifts the guard.
        assert kb.assign_task(conn, tid, "closer") is True
        assert kbd.check_respawn_guard(conn, tid) is None


def test_active_pr_guard_lifts_for_implementer_after_changes_requested(
    kanban_home: Path,
) -> None:
    """Reviewer CHANGES_REQUESTED routes the card back to ``ready`` for the
    implementer to fix the SAME PR; ``active_pr`` must not hold it (#111910).
    ``recent_success`` is untouched by the handoff exemption."""
    pr_comment = "Opened https://github.com/example/repo/pull/44 for review."
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="changes requested", assignee="dev",
                             completion_contract=_IMPLEMENTER_CONTRACT)
        claimed = kb.claim_task(conn, tid)
        kb.add_comment(conn, tid, author="dev", body=pr_comment)
        _backdate_comments(conn, tid)
        assert kb.request_review(
            conn, tid, summary="PR ready", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        rclaim = kb.claim_review_task(conn, tid)
        ok, implementer = kb.request_changes(
            conn, tid, reason="fix tests", expected_run_id=rclaim.current_run_id,
        )
        assert (ok, implementer) == (True, "dev")
        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.check_respawn_guard(conn, tid) is None

        done_id = kb.create_task(conn, title="recent success", assignee="dev")
        kb.claim_task(conn, done_id)
        assert kb.complete_task(conn, done_id, summary="done") is True
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (done_id,))
        assert kbd.check_respawn_guard(conn, done_id) == "recent_success"


# ---------------------------------------------------------------------------
# active_pr: PR state, lane awareness, window knob, operator escape hatch
# ---------------------------------------------------------------------------

def _dispatch_ready(conn, monkeypatch: pytest.MonkeyPatch):
    """One dry-run dispatch tick with every profile spawnable and default config."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
    return kbd.dispatch_once(conn, dry_run=True)


def _pr_card(conn, monkeypatch: pytest.MonkeyPatch, *, contract: str = _IMPLEMENTER_CONTRACT):
    tid = kb.create_task(conn, title="pr card", assignee="dev",
                         completion_contract=contract)
    kb.add_comment(conn, tid, author="dev",
                   body="Opened https://github.com/example/repo/pull/44 for review.")
    return tid


def test_merged_pr_does_not_guard(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A card whose comment reports an already-merged PR must spawn.

    The duplicate-PR protection exists to stop a second PR being opened; once
    the referenced PR is merged there is nothing left to duplicate. Before this
    fix the card's own conclusion ("merged") locked it out of ``ready`` for 24h.
    """
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: "MERGED")
    with kbc.connect() as conn:
        tid = _pr_card(conn, monkeypatch)
        kb.add_comment(conn, tid, author="dev",
                       body="Merged https://github.com/example/repo/pull/74 at 15:06Z.")

        assert kbd.check_respawn_guard(conn, tid) is None
        res = _dispatch_ready(conn, monkeypatch)
        assert tid in [row[0] for row in res.spawned]
        assert tid not in [task_id for task_id, _reason in res.respawn_guarded]


def test_closed_pr_does_not_guard(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A closed-without-merge PR leaves no duplicate risk either."""
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: "CLOSED")
    with kbc.connect() as conn:
        tid = _pr_card(conn, monkeypatch)
        assert kbd.check_respawn_guard(conn, tid) is None
        res = _dispatch_ready(conn, monkeypatch)
        assert tid in [row[0] for row in res.spawned]


def test_open_pr_still_guards(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for the real protection: an OPEN PR still holds the card."""
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: "OPEN")
    with kbc.connect() as conn:
        tid = _pr_card(conn, monkeypatch)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        res = _dispatch_ready(conn, monkeypatch)
        assert tid not in [row[0] for row in res.spawned]
        assert dict(res.respawn_guarded).get(tid) == "active_pr"


def test_unresolved_pr_state_fails_closed(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable/ambiguous lookup (no ``gh``, timeout, 404) must still guard."""
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: None)
    with kbc.connect() as conn:
        tid = _pr_card(conn, monkeypatch)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_pr_state_lookup_is_cached(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One ``gh`` call per distinct PR URL per cache window — successes and failures."""
    from hermes_cli import kanban_db_pr_guard as pr_guard

    calls: list[str] = []

    def _lookup(url: str):
        calls.append(url)
        return "OPEN" if len(calls) == 1 else None

    # The resolver reads its subprocess seam in its own module, so patch there.
    monkeypatch.setattr(kbd, "_resolve_pr_state", _REAL_RESOLVE_PR_STATE)
    monkeypatch.setattr(pr_guard, "_lookup_pr_state", _lookup)
    url = "https://github.com/example/repo/pull/44"

    assert kbd._resolve_pr_state(url) == "OPEN"
    # Cached success: the second call must not fork another lookup.
    assert kbd._resolve_pr_state(url) == "OPEN"
    assert calls == [url]

    other = "https://github.com/example/repo/pull/45"
    assert kbd._resolve_pr_state(other) is None  # a failure...
    assert kbd._resolve_pr_state(other) is None  # ...is cached too, not re-forked
    assert calls == [url, other]


def test_local_only_report_card_is_exempt_and_costs_no_lookup(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A report/notify card never opens a PR, so mentioning one must not freeze it.

    ``local-only`` is the card's own declaration that it produces no PR; the
    guard must skip it entirely — including the ``gh`` lookup, so report cards
    cost zero subprocesses per tick.
    """
    calls: list[str] = []

    def _resolve(url: str):
        calls.append(url)
        return "OPEN"

    monkeypatch.setattr(kbd, "_resolve_pr_state", _resolve)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="tell gabriel the PR is ready", assignee="chief",
                             completion_contract="local-only")
        kb.add_comment(conn, tid, author="chief",
                       body="PR #78 is ready: https://github.com/example/repo/pull/78")

        assert kbd.check_respawn_guard(conn, tid) is None
        assert calls == []
        res = _dispatch_ready(conn, monkeypatch)
        assert tid in [row[0] for row in res.spawned]
        assert calls == []


def test_local_only_card_that_published_a_pr_stays_guarded(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that recorded ``published_pr`` makes the card implementer work.

    Whatever the card declared, a card that actually published a PR can be
    re-spawned against that same PR, so the duplicate-work guard must apply.
    """
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: "OPEN")
    with kbc.connect() as conn:
        tid = _pr_card(conn, monkeypatch, contract="local-only")
        # Backdated beyond the 1h ``recent_success`` window so this test isolates
        # the active_pr step rather than the completed-run guard.
        now = int(time.time())
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at, metadata) VALUES (?, 'dev', 'completed', "
                "'completed', ?, ?, ?)",
                (tid, now - 7200, now - 7100,
                 json.dumps({"published_pr": "https://github.com/example/repo/pull/9"})),
            )
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_window_knob_disables_and_shortens_the_guard(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``HERMES_KANBAN_RESPAWN_GUARD_PR_WINDOW_SECONDS`` is the window, and 0 disables it."""
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: "OPEN")
    with kbc.connect() as conn:
        tid = _pr_card(conn, monkeypatch)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"

        monkeypatch.setenv("HERMES_KANBAN_RESPAWN_GUARD_PR_WINDOW_SECONDS", "0")
        assert kbd.check_respawn_guard(conn, tid) is None

        monkeypatch.setenv("HERMES_KANBAN_RESPAWN_GUARD_PR_WINDOW_SECONDS", "60")
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        _backdate_comments(conn, tid, seconds=120)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_unguarded_event_lifts_the_guard(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``unguard_task`` is the operator escape hatch for a card that cannot be reassigned."""
    monkeypatch.setattr(kbd, "_resolve_pr_state", lambda _url: "OPEN")
    with kbc.connect() as conn:
        tid = _pr_card(conn, monkeypatch)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        _backdate_comments(conn, tid)
        assert kb.unguard_task(conn, tid, actor="user") is True
        assert kbd.check_respawn_guard(conn, tid) is None
        assert _events(conn, tid, kind="unguarded") == [("unguarded", {"by": "user"})]
        assert kb.unguard_task(conn, "t_deadbeefcafe", actor="user") is False


def test_dispatch_json_exposes_suppression_reasons(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Operators can distinguish an active PR, quota cooldown, lock, and pressure."""
    monkeypatch.setattr(
        kanban_ops.kbd,
        "dispatch_once",
        lambda *args, **kwargs: kbd.DispatchResult(
            respawn_guarded=[("active-pr-task", "active_pr")],
            rate_limited=["quota-task"],
            skipped_locked=True,
            memory_pressure="elevated",
        ),
    )

    assert kanban_ops._cmd_dispatch(
        SimpleNamespace(dry_run=True, max=None, failure_limit=kbd.DEFAULT_FAILURE_LIMIT, json=True)
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["respawn_guarded"] == [{"task_id": "active-pr-task", "reason": "active_pr"}]
    assert payload["rate_limited"] == ["quota-task"]
    assert payload["skipped_locked"] is True
    assert payload["memory_pressure"] == "elevated"




def test_review_dispatch_preserves_task_skills_and_adds_reviewer_skill(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        cfgmod,
        "load_config",
        lambda *args, **kwargs: {"kanban": {"review_dispatch": True}},
    )
    captured: list[list[str]] = []

    def spawn(task, workspace):
        captured.append(list(task.skills or []))

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="domain review",
            assignee="reviewer",
            skills=["domain-specific-review"],
        )
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="ready",
            expected_run_id=implementation.current_run_id,
        )
        monkeypatch.setattr(
            kbd,
            "check_respawn_guard",
            lambda _conn, _task_id, **_kw: "rate_limit_cooldown",
        )
        guarded = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert guarded.respawn_guarded == [(task_id, "rate_limit_cooldown")]
        assert not guarded.spawned
        guarded_task = kb.get_task(conn, task_id)
        assert guarded_task is not None
        assert guarded_task.status == "review"

        monkeypatch.setattr(kbd, "check_respawn_guard", lambda _conn, _task_id, **_kw: None)
        result = kbd.dispatch_once(conn, spawn_fn=spawn)

    assert task_id in [task[0] for task in result.spawned]
    assert captured == [["domain-specific-review", "sdlc-review"]]


def test_review_dispatch_honors_global_and_per_profile_caps(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(
        cfgmod,
        "load_config",
        lambda *args, **kwargs: {"kanban": {"review_dispatch": True}},
    )

    with kbc.connect() as conn:
        running_id = kb.create_task(conn, title="already running", assignee="builder")
        running = kb.claim_task(conn, running_id)
        assert running is not None

        review_ids: list[str] = []
        for title in ("review one", "review two"):
            task_id = kb.create_task(conn, title=title, assignee="reviewer")
            implementation = kb.claim_task(conn, task_id)
            assert implementation is not None
            assert kb.request_review(
                conn,
                task_id,
                summary="ready",
                expected_run_id=implementation.current_run_id,
            )
            review_ids.append(task_id)

        globally_capped = kbd.dispatch_once(
            conn,
            dry_run=True,
            max_in_progress=1,
        )
        assert not [
            task for task in globally_capped.spawned if task[0] in review_ids
        ]

        assert kb.complete_task(
            conn,
            running_id,
            result="done",
            expected_run_id=running.current_run_id,
        )
        global_dry_run = kbd.dispatch_once(
            conn,
            dry_run=True,
            max_in_progress=1,
        )
        assert len([
            task for task in global_dry_run.spawned if task[0] in review_ids
        ]) == 1

        per_profile_capped = kbd.dispatch_once(
            conn,
            dry_run=True,
            max_in_progress=10,
            max_in_progress_per_profile=1,
        )
        spawned_reviews = [
            task for task in per_profile_capped.spawned if task[0] in review_ids
        ]
        assert len(spawned_reviews) == 1
        assert len(per_profile_capped.skipped_per_profile_capped) == 1
        assert per_profile_capped.skipped_per_profile_capped[0][0] in review_ids


# ---------------------------------------------------------------------------
# reopen: a follow-up sends a review task back out for another pass
# ---------------------------------------------------------------------------


def test_reopen_review_task_returns_to_ready(kanban_home: Path) -> None:
    """The "changes requested" / follow-up path: a task parked in ``review``
    goes back to ``ready`` so the dispatcher re-runs the implementer. It must
    NOT touch ``block_recurrences`` (review was never a block)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reopen me", assignee="worker")
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="v1", reviewer="reviewer",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        reviewing = kb.get_task(conn, tid)
        assert reviewing is not None
        assert reviewing.status == "review"
        assert reviewing.assignee == "reviewer"

        ok = kb.reopen_review_task(conn, tid)
        assert ok is True
        row = _row(conn, tid)
        assert row["status"] == "ready"
        reopened = kb.get_task(conn, tid)
        assert reopened is not None
        assert reopened.assignee == "worker"
        assert row["current_run_id"] is None
        assert (row["block_recurrences"] or 0) == 0
        assert _events(conn, tid, kind="review_reopened")

        # Idempotent: not in review anymore -> reopening again is a no-op.
        assert kb.reopen_review_task(conn, tid) is False


def test_review_cycle_end_to_end(kanban_home: Path) -> None:
    """Full loop: run -> review -> follow-up reopen -> re-run -> review ->
    approve -> done. Never blocks, never triages, and stays wake-subscribed
    until done."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cycle", assignee="worker")

        # Pass 1: implement -> review.
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="v1",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"

        # Human asks for changes -> reopen -> re-run.
        assert kb.reopen_review_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="v2",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"

        # Human approves.
        assert kb.complete_task(conn, tid, summary="approved") is True
        row = _row(conn, tid)
        assert row["status"] == "done"
        assert (row["block_recurrences"] or 0) == 0
        assert _events(conn, tid, kind="block_loop_detected") == []


# ---------------------------------------------------------------------------
# never-claimed 'ready' task: handoff must survive via a synthesized run
# ---------------------------------------------------------------------------


def test_request_review_on_unclaimed_ready_synthesizes_run(kanban_home: Path) -> None:
    """A manual/CLI request-review on a never-claimed ``ready`` task has no
    active run to close. The handoff summary must still be preserved on a
    synthesized run so the reviewer keeps the context."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ready then review", assignee="worker")
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.get_task(conn, tid).current_run_id is None

        ok = kb.request_review(conn, tid, summary="done without a claim")
        assert ok is True
        assert kb.get_task(conn, tid).status == "review"

        run = _last_run(conn, tid)
        assert run is not None
        assert run["outcome"] == "review_requested"
        assert run["summary"] == "done without a claim"
        # Exactly one review_requested event, carrying the handoff summary.
        evs = _events(conn, tid, kind="review_requested")
        assert len(evs) == 1
        assert evs[0][1]["summary"] == "done without a claim"


def test_reviewer_reassigns_for_autonomous_dispatch(kanban_home: Path) -> None:
    """An explicit reviewer routes the review run while preserving implementer provenance."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="route reviewer", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        ok = kb.request_review(
            conn, tid, summary="v1", reviewer="lead-reviewer",
            expected_run_id=claimed.current_run_id,
        )
        assert ok is True
        assert kb.get_task(conn, tid).assignee == "lead-reviewer"
        ev = _events(conn, tid, kind="review_requested")[0][1]
        assert ev["reviewer"] == "lead-reviewer"
        assert ev["implementer"] == "worker"


def test_review_handoff_without_live_run_attributes_run_to_implementer(kanban_home: Path) -> None:
    """#111064: ``request_review`` reassigns the card to the reviewer in the same
    UPDATE that flips the status, so the zero-duration run synthesized for a
    never-claimed card must be stamped with the implementer captured before
    the rewrite, not the reviewer read back off the mutated row."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="handoff attribution", assignee="worker")
        assert kb.request_review(conn, tid, summary="ready", reviewer="lead-reviewer") is True
        assert kb.get_task(conn, tid).assignee == "lead-reviewer"
        run = conn.execute(
            "SELECT profile, outcome, step_key FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert (run["outcome"], run["profile"]) == ("review_requested", "worker")
        assert run["step_key"] == kb.get_task(conn, tid).current_step_key
        assert _events(conn, tid, kind="review_requested")[0][1]["implementer"] == "worker"


def test_review_handoff_of_card_assigned_to_its_reviewer_records_no_implementer(
    kanban_home: Path,
) -> None:
    """A card created already assigned to its reviewer has no implementer to
    record. Stamping the assignee made the payload read
    ``implementer == reviewer``, and ``request_changes`` routes on that field —
    so a rejection went back to the profile that wrote the findings. With no
    live run and nothing but the reviewer on the row, the honest provenance is
    *none*, and the rejection must refuse rather than misroute."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="already applied", assignee="reviewer-a")
        assert kb.request_review(
            conn, tid, summary="review this", reviewer="reviewer-a",
        ) is True

        ev = _events(conn, tid, kind="review_requested")[0][1]
        assert ev["reviewer"] == "reviewer-a"
        assert ev["implementer"] is None
        run = conn.execute(
            "SELECT profile, outcome FROM task_runs WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        assert (run["outcome"], run["profile"]) == ("review_requested", None)

        claimed = kb.claim_review_task(conn, tid, claimer="reviewer-a")
        assert claimed is not None
        ok, reason = kb.request_changes(conn, tid, reason="found 3 issues")
        assert ok is False
        assert "implementer provenance" in (reason or "")


def test_synthesized_run_for_unassigned_card_keeps_null_profile(kanban_home: Path) -> None:
    """A transition that does not name an actor still reads the card: an
    unassigned card's synthesized run carries ``profile=NULL`` (the actor
    sentinel must not turn "unassigned" into a re-read of the row)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unassigned handoff")
        assert kb.block_task(conn, tid, reason="waiting on upstream") is True
        run = conn.execute(
            "SELECT profile, outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        assert (run["outcome"], run["profile"]) == ("blocked", None)

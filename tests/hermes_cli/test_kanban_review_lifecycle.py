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
from pathlib import Path
from typing import cast

import pytest

from hermes_cli import kanban_db as kb


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


def _release_event_payloads(conn, tid: str) -> list[dict]:
    """Return the parsed payloads of all `respawn_released` events on a task.

    Helper for the reviewer-feedback dedupe tests (REVIEWER FEEDBACK IN
    HERMES-AGENT#2, 2026-08-21). Filters out any None payloads so callers
    can subscript directly without type-checking the second tuple element.
    """
    out: list[dict] = []
    for kind, payload in _events(conn, tid, kind="respawn_released"):
        if payload is not None:
            out.append(cast(dict, payload))
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="live claim", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

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
    with kb.connect() as conn:
        tid2 = kb.create_task(conn, title="forced", assignee="worker")
        assert kb.claim_task(conn, tid2) is not None
        assert kb.request_review(conn, tid2, summary="override", force=True) is True
        assert kb.get_task(conn, tid2).status == "review"


def test_request_review_malformed_provenance_gets_distinct_reason(
    kanban_home: Path,
) -> None:
    """M1 regression: malformed re-review provenance is a named failure, not
    the generic 'unknown id or not in running/ready'."""
    with kb.connect() as conn:
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="wake me", assignee="worker")
        kb.add_notify_sub(
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
        _old, _new, events = kb.claim_unseen_events_for_sub(
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

    with kb.connect() as conn:
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
        res_off = kb.dispatch_once(conn, dry_run=True)
        assert tid not in [s[0] for s in res_off.spawned]
        assert kb.get_task(conn, tid).status == "review"

        # Gate ON (the default; sdlc-review is bundled) -> the review task is
        # picked up by the dispatcher.
        monkeypatch.setattr(
            cfgmod, "load_config",
            lambda *a, **k: {"kanban": {"review_dispatch": True}},
        )
        res_on = kb.dispatch_once(conn, dry_run=True)
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

    with kb.connect() as conn:
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
        ready_id = kb.create_task(conn, title="already PRed", assignee="worker")
        kb.add_comment(conn, ready_id, author="worker", body=pr_comment)

        assert kb.check_respawn_guard(conn, ready_id) == "active_pr"
        assert kb.check_respawn_guard(conn, review_id, lane="review") is None

        res = kb.dispatch_once(conn, dry_run=True)
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
        assert kb.check_respawn_guard(
            conn, review_id, lane="review"
        ) == "rate_limit_cooldown"


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
        return None

    with kb.connect() as conn:
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
            kb,
            "check_respawn_guard",
            lambda _conn, _task_id, **_kw: "rate_limit_cooldown",
        )
        guarded = kb.dispatch_once(conn, spawn_fn=spawn)
        assert guarded.respawn_guarded == [(task_id, "rate_limit_cooldown")]
        assert not guarded.spawned
        guarded_task = kb.get_task(conn, task_id)
        assert guarded_task is not None
        assert guarded_task.status == "review"

        monkeypatch.setattr(kb, "check_respawn_guard", lambda _conn, _task_id, **_kw: None)
        result = kb.dispatch_once(conn, spawn_fn=spawn)

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

    with kb.connect() as conn:
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

        globally_capped = kb.dispatch_once(
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
            expected_run_id=running.current_run_id,
        )
        global_dry_run = kb.dispatch_once(
            conn,
            dry_run=True,
            max_in_progress=1,
        )
        assert len([
            task for task in global_dry_run.spawned if task[0] in review_ids
        ]) == 1

        per_profile_capped = kb.dispatch_once(
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
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
    with kb.connect() as conn:
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


# ---------------------------------------------------------------------------
# Reviewer-feedback release (SPEC-active-pr-guard-reviewer-feedback Part B):
# 6 trigger conditions + 4 non-trigger conditions for the active_pr guard.
# ---------------------------------------------------------------------------

import time as _time


def _add_comment_at(
    conn, task_id: str, author: str, body: str, created_at: int,
) -> None:
    """Insert a comment at a specific ``created_at`` epoch. Mirrors
    ``add_comment``'s write-txn semantics but lets tests control the
    timestamp so reviewer-feedback-after-PR ordering is reproducible.
    """
    with kb.write_txn(conn, allow_nested=True):
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, author.strip(), body.strip(), int(created_at)),
        )


def _patch_gh_review_decision(monkeypatch: pytest.MonkeyPatch, decision):
    """Stub out `_query_pr_review_decision` so tests don't shell out to gh."""
    monkeypatch.setattr(kb, "_query_pr_review_decision", lambda _url: decision)


def _ensure_builder_profile(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create the `~/.hermes/profiles/builder/` directory so
    ``hermes_cli.profiles.profile_exists('builder')`` returns True for
    dispatcher tests. The dispatcher's ready loop gates each task on
    ``profile_exists(assignee)`` and bucket-skips non-spawnable
    assignments (``result.skipped_nonspawnable``) — a real Hermes profile
    dir is enough to pass the gate in unit-test context.
    """
    from hermes_cli.profiles import get_profile_dir

    monkeypatch.setattr(
        "hermes_cli.profiles.normalize_profile_name",
        lambda name: "builder" if name == "builder" else name,
        raising=False,
    )
    profile_dir = get_profile_dir("builder")
    profile_dir.mkdir(parents=True, exist_ok=True)
    # Drop a minimal profile.yaml so the gate doesn't trip on missing
    # manifest validation in stricter paths.
    (profile_dir / "profile.yaml").write_text(
        "name: builder\nversion: 1\n", encoding="utf-8"
    )


def _setup_task_with_pr_and_feedback(
    conn,
    pr_url: str = "https://github.com/example/repo/pull/178",
    feedback_author: str = "aliaadil",
    feedback_body: str = (
        "the logging is not sufficient. ALL actions performed by the user "
        "and server need to be logged properly. please update this PR."
    ),
    feedback_offset_seconds: int = 60,
    pr_offset_seconds: int = 0,
    feedback_count: int = 1,
    workspace_kind: str = "scratch",
) -> str:
    """Helper: create a task with a PR-URL breadcrumb plus N reviewer
    feedback comments. Returns the task id. Uses ``int(time.time())`` so
    the PR + feedback land inside the 24h window.
    """
    tid = kb.create_task(
        conn, title="t_de993dac repro", assignee="builder",
        workspace_kind=workspace_kind,
    )
    now = int(_time.time())
    _add_comment_at(
        conn, tid, author="builder",
        body=f"Opened {pr_url} for review",
        created_at=now - 3600 + pr_offset_seconds,
    )
    for i in range(feedback_count):
        _add_comment_at(
            conn, tid, author=feedback_author,
            body=feedback_body if i == 0 else (feedback_body + f" [round {i}]"),
            created_at=now - 3600 + pr_offset_seconds + feedback_offset_seconds + i,
        )
    return tid


def test_reviewer_feedback_release_trigger_substantive_non_default_author(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trigger 1: non-default author + body >= 80 chars releases the guard."""
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(
            conn, feedback_author="aliaadil",
            feedback_body=(
                "logging is not sufficient. ALL actions performed by the "
                "user and server need to be logged properly. please update."
            ),
        )
        assert kb.check_respawn_guard(conn, tid) is None


def test_reviewer_feedback_release_trigger_body_pattern(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trigger 3: body contains a reviewer-feedback phrase releases the guard."""
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(
            conn, feedback_author="aliaadil",
            # Short comment that still matches a pattern.
            feedback_body="needs to log all clicks and key presses in the audit trail.",
        )
        assert kb.check_respawn_guard(conn, tid) is None


def test_reviewer_feedback_release_trigger_pr_number_reference(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trigger 3: body references the PR number directly (#178)."""
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(
            conn, feedback_author="aliaadil",
            feedback_body="looks good overall. one nit on PR #178: rename _audit to _audit_log.",
        )
        assert kb.check_respawn_guard(conn, tid) is None


def test_reviewer_feedback_release_trigger_distinct_content_key(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trigger 2: distinct content_key from the PR-URL comment releases."""
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(
            conn, feedback_author="aliaadil",
            # Even if body length is short, distinct content_key + non-default
            # author + pattern match → release. Use a phrase here.
            feedback_body="needs to update the comments please update",
        )
        assert kb.check_respawn_guard(conn, tid) is None


def test_reviewer_feedback_release_trigger_review_decision_changes_requested(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trigger 4: reviewDecision == CHANGES_REQUESTED releases even with
    no new comment body match.
    """
    _patch_gh_review_decision(monkeypatch, decision="CHANGES_REQUESTED")
    with kb.connect() as conn:
        # No reviewer feedback comment — just the PR-URL breadcrumb.
        tid = kb.create_task(conn, title="changes requested", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178 for review",
            created_at=now - 60,
        )
        assert kb.check_respawn_guard(conn, tid) is None


def test_reviewer_feedback_release_trigger_short_author_aliadil_pattern_match(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trigger 3 (pattern): a short body from non-default author that
    matches a phrase still releases the guard.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(
            conn, feedback_author="aliaadil",
            feedback_body="please update the type annotations here.",
        )
        assert kb.check_respawn_guard(conn, tid) is None


# --- REGRESSION: default-authored PR-number pings must NOT release ---
# (REVIEWER FEEDBACK IN HERMES-AGENT#2, 2026-08-21, issue #1).
#
# Auto-mirrored status pings routinely contain "PR #N opened" / "PR #N
# closed" — the pre-fix `_REVIEWER_FEEDBACK_PR_NUM_RE` fired
# author-agnostic, so a default-authored ping like the one below would
# release the active_pr guard, violating the spec's "auto-mirrored
# default-authored status comments do NOT release" acceptance criterion
# AND creating a respawn loop on every tick.

def test_reviewer_feedback_no_release_default_author_pr_number_opened(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-authored 'PR #178 opened' ping does NOT release the guard.

    Pre-fix this fired trigger 3 (PR-number regex). After the fix,
    trigger 3's PR-number regex is gated on author != 'default' so
    auto-mirrored status pings can't release the guard.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="pr ping", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178 for review",
            created_at=now - 120,
        )
        _add_comment_at(
            conn, tid, author="default",
            body="📌 PR #178 opened (auto-mirrored status ping)",
            created_at=now - 30,
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


def test_reviewer_feedback_no_release_default_author_pull_slash_n(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-authored 'pull/178' reference doesn't release either."""
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="pull slash", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178 for review",
            created_at=now - 120,
        )
        _add_comment_at(
            conn, tid, author="default",
            body="→ from kanban task t_x via pull/178",
            created_at=now - 30,
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


def test_reviewer_feedback_no_release_default_author_phrase_with_pr_num(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-authored phrase + PR-number: the phrase still triggers
    release (phrases stay author-agnostic) — but ONLY because phrases
    are concrete reviewer signals. A default-authored phrase is rare
    enough that the spec doesn't gate it; this test pins the
    intended behavior so future regressions are caught.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="phrase ping", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178",
            created_at=now - 120,
        )
        # default author + phrase + PR number. The phrase still wins
        # (author-agnostic), which is correct per the rationale in
        # _has_reviewer_feedback: status pings are short emoji lines,
        # not full sentences with reviewer phrases.
        _add_comment_at(
            conn, tid, author="default",
            body="please update the PR #178 with the latest changes",
            created_at=now - 30,
        )
        assert kb.check_respawn_guard(conn, tid) is None


def test_reviewer_feedback_no_release_default_author_bare_hash_n(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare '#178' from default author doesn't release."""
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="bare hash", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178 for review",
            created_at=now - 120,
        )
        _add_comment_at(
            conn, tid, author="default",
            body="see #178 for context",
            created_at=now - 30,
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


# --- REGRESSION: stale feedback must NOT re-release on every tick ---
# (REVIEWER FEEDBACK IN HERMES-AGENT#2, 2026-08-21, issue #2).
#
# Pre-fix, once the guard released and the spawned run failed without
# producing a new PR-URL comment, the same stale feedback comment kept
# satisfying trigger 1 / 3 on every subsequent tick — a respawn loop
# bounded only by the failure circuit breaker. The fix records a
# `respawn_released` event with the feedback timestamp; subsequent
# ticks skip comments at or before that watermark.

def test_reviewer_feedback_no_double_release_after_failure(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the guard releases for a given feedback comment, subsequent
    ticks with NO new feedback do NOT re-release — the respawn-loop
    failure mode is closed.

    Simulates: a release event is on file; check_respawn_guard is
    called again on the same comments. Without the watermark, the
    same stale feedback would satisfy trigger 1 / 3 again.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(
            conn, feedback_author="aliaadil",
            feedback_body=(
                "logging is not sufficient. ALL actions performed by the "
                "user and server need to be logged properly. please update."
            ),
        )
        # First tick: releases the guard.
        assert kb.check_respawn_guard(conn, tid) is None
        # Confirm the release event was recorded with the feedback_at
        # timestamp.
        released_events = _release_event_payloads(conn, tid)
        assert len(released_events) == 1, released_events
        assert released_events[0]["reason"] == "reviewer_feedback"
        assert released_events[0]["feedback_at"] > 0
        feedback_at = released_events[0]["feedback_at"]
        # Second tick WITHOUT new feedback: must NOT release.
        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        # No new release event recorded.
        assert len(_release_event_payloads(conn, tid)) == 1
        # Third tick: still no release.
        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        assert len(_release_event_payloads(conn, tid)) == 1
        # Sanity: the feedback_at we recorded is the feedback comment's
        # created_at — that comment is now in the past and skipped.
        assert feedback_at > 0


def test_reviewer_feedback_releases_again_for_newer_feedback_comment(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a prior release, a NEW (later-created_at) reviewer comment
    DOES re-release the guard. The watermark advances only when new
    feedback arrives, so legitimate fresh feedback (a follow-up
    comment from the reviewer) still fires.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(
            conn, feedback_author="aliaadil",
            feedback_body=(
                "logging is not sufficient. ALL actions performed by the "
                "user and server need to be logged properly. please update."
            ),
        )
        # First release.
        assert kb.check_respawn_guard(conn, tid) is None
        first_release_at = _release_event_payloads(conn, tid)[0]["feedback_at"]
        # Second tick still no new feedback → no release.
        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        # Reviewer follows up with a NEW substantive comment.
        later = int(_time.time()) + 5  # strictly greater than first_release_at
        _add_comment_at(
            conn, tid, author="aliaadil",
            body=(
                "thanks for the logging update. but the audit trail still "
                "misses some key events — please update the type annotations "
                "and add the missing fields."
            ),
            created_at=later,
        )
        # Now the guard releases again.
        assert kb.check_respawn_guard(conn, tid) is None
        # Two release events on file, the second with feedback_at == later.
        events = _release_event_payloads(conn, tid)
        assert len(events) == 2, events
        assert events[1]["feedback_at"] == later
        assert events[1]["feedback_at"] > first_release_at


def test_reviewer_feedback_no_release_trigger4_only_after_first(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trigger 4 (CHANGES_REQUESTED) does not re-release after a prior
    release. The release event's feedback_at advances the watermark,
    so a stale CHANGES_REQUESTED decision that hasn't changed doesn't
    fire again on the next tick.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t4 dedupe", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178 for review",
            created_at=now - 60,
        )
        # Simulate: a prior release was already recorded. (Pretend the
        # dispatcher fired earlier with the same trigger-4 outcome.)
        with kb.write_txn(conn):
            kb._append_event(
                conn, tid, "respawn_released",
                {
                    "reason": "reviewer_feedback",
                    "feedback_at": now - 1,
                    "trigger": "d",
                },
            )
        # Now patch the gh call to return CHANGES_REQUESTED — should
        # NOT release because the watermark is already set.
        _patch_gh_review_decision(monkeypatch, decision="CHANGES_REQUESTED")
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


def test_reviewer_feedback_no_release_default_pr_num_does_not_advance_watermark(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A default-authored ping with a PR number does NOT release AND
    does NOT advance the watermark — so a subsequent legitimate
    reviewer comment at the same or later timestamp still releases.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="watermark safety", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178 for review",
            created_at=now - 300,
        )
        # Default ping that mentions the PR number (must NOT release).
        _add_comment_at(
            conn, tid, author="default",
            body="📌 PR #178 opened (auto-mirrored status ping)",
            created_at=now - 60,
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        # No release event recorded.
        assert _release_event_payloads(conn, tid) == []
        # Now a legitimate reviewer comment arrives. It should release.
        _add_comment_at(
            conn, tid, author="aliaadil",
            body=(
                "the PR #178 looks good overall but please update the "
                "type annotations to use the new typing.Literal syntax."
            ),
            created_at=now - 30,
        )
        assert kb.check_respawn_guard(conn, tid) is None
        # Exactly one release event, from the real reviewer comment.
        events = _release_event_payloads(conn, tid)
        assert len(events) == 1
        assert events[0]["trigger"] == "a"  # non-default + length


def test_reviewer_feedback_release_event_has_audit_fields(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `respawn_released` event carries reason, feedback_at, and
    trigger fields so operators have a full audit trail of why a
    spawn was allowed (REVIEWER FEEDBACK IN HERMES-AGENT#2, 2026-08-21,
    issue #2 audit-trail concern).
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(
            conn, feedback_author="aliaadil",
            feedback_body="please update the type annotations here.",
        )
        assert kb.check_respawn_guard(conn, tid) is None
        [payload] = _release_event_payloads(conn, tid)
        assert payload["reason"] == "reviewer_feedback"
        assert payload["trigger"] in ("a", "b", "c", "d")
        assert isinstance(payload["feedback_at"], int)
        assert payload["feedback_at"] > 0


def test_reviewer_feedback_no_release_remirror_same_timestamp(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-mirror of the same feedback at the same timestamp does NOT
    re-release the guard — the watermark is keyed on created_at, so a
    second mirror row with the same (or earlier) timestamp is treated
    as the same feedback, not a new round.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="remirror same ts", assignee="builder",
        )
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178 for review",
            created_at=now - 300,
        )
        # First feedback comment from the reviewer.
        _add_comment_at(
            conn, tid, author="aliaadil",
            body="please update the type annotations here.",
            created_at=now - 60,
        )
        assert kb.check_respawn_guard(conn, tid) is None
        # Mirror re-pushes the same comment body at the SAME timestamp
        # — a re-mirror race where the mirror service happens to
        # re-fetch and write a duplicate row with identical timestamp.
        _add_comment_at(
            conn, tid, author="aliaadil",
            body="please update the type annotations here.",
            created_at=now - 60,
        )
        # Should NOT release a second time — the watermark equals the
        # duplicate's created_at so the watermark check skips it.
        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        # Still only one release event.
        assert len(_release_event_payloads(conn, tid)) == 1


# --- Non-trigger conditions: guard stays "active_pr" ---

def test_reviewer_feedback_no_release_only_auto_mirrored_default_comments(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-mirrored default-authored status pings do NOT release the guard."""
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="auto mirrored", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178 for review",
            created_at=now - 120,
        )
        # Several auto-mirrored 'default'-authored status pings.
        for body in (
            "🔨 raphael status: t_x is now running (agent: builder)",
            "👀 raphael status: t_x is now ready (agent: builder)",
            "✅ raphael status: t_x is now done (agent: builder)",
        ):
            _add_comment_at(
                conn, tid, author="default", body=body,
                created_at=now - 60,
            )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


def test_reviewer_feedback_no_release_short_default_comment(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short default-authored comment doesn't release the guard."""
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="short default", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178",
            created_at=now - 120,
        )
        _add_comment_at(
            conn, tid, author="default", body="ok",
            created_at=now - 30,
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


def test_reviewer_feedback_no_release_approved_pr_no_feedback(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reviewDecision != CHANGES_REQUESTED with no comment → guard holds."""
    _patch_gh_review_decision(monkeypatch, decision="APPROVED")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="approved", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178",
            created_at=now - 60,
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


def test_reviewer_feedback_no_release_comment_before_pr_url(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reviewer comment BEFORE the PR-URL comment doesn't release."""
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="out-of-order", assignee="builder")
        now = int(_time.time())
        # Reviewer comment first
        _add_comment_at(
            conn, tid, author="aliaadil",
            body="needs to update the audit trail before opening a PR",
            created_at=now - 3600,
        )
        # PR URL posted after — reviewer feedback is BEFORE not AFTER
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178",
            created_at=now - 60,
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


# --- Pattern-substring false positives (word-boundary fix) ---

def test_reviewer_feedback_no_release_substring_match_priority(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`this pr` should NOT match inside `priority` or `private`.

    The pre-2026-08-21 regex used substring alternation, which let
    `this pr` match inside `priority` and `this private method`. The
    tightened per-word word-boundary regex
    (`_compile_word_boundary_re`) eliminates that false positive so the
    guard isn't released on incidental substring hits.

    The body here is short and `default`-authored, so trigger 1
    (non-default author + length >= 80) is already gated. With
    trigger 3 (pattern match) tightened, the only remaining release
    path is trigger 4 (reviewDecision), which is mocked to None — so
    the guard must hold.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="priority substring", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178",
            created_at=now - 120,
        )
        _add_comment_at(
            conn, tid, author="aliaadil",
            body="the priority is low; also private concerns were raised.",
            created_at=now - 30,
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


def test_reviewer_feedback_no_release_substring_match_all_actions(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`all actions` should NOT match inside `hall actions` / `small action`.

    Same word-boundary fix as the priority test above. Both bodies are
    short AND non-default-authored AND distinct-content-key — that
    combination is intentional so the ONLY release path is trigger 3
    (pattern match), which we want to prove is now tightened.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="hall actions substring", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178",
            created_at=now - 120,
        )
        _add_comment_at(
            conn, tid, author="aliaadil",
            body="hall actions of the day were notable",
            created_at=now - 30,
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"


def test_reviewer_feedback_pattern_still_matches_standalone_phrase(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: tightening the regex must NOT break the
    legitimate single-occurrence phrase match.

    "this pr" surrounded by punctuation/word-boundaries still triggers
    release (trigger 3). This is the canonical T_de993dac reviewer body
    shape and must keep working.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="phrase-match-still-works", assignee="builder",
        )
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178",
            created_at=now - 120,
        )
        _add_comment_at(
            conn, tid, author="aliaadil",
            body="please update this pr with the audit logging fix",
            created_at=now - 30,
        )
        assert kb.check_respawn_guard(conn, tid) is None


# --- Branch-routing override (SPEC Part B step #6) ---

def test_branch_override_populated_when_guard_releases(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the guard releases due to reviewer feedback, the branch
    override side-channel is populated for the dispatcher to consume.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)

    # Stub out the gh subprocess for branch lookup
    def fake_gh(*args, **kwargs):
        # kwargs.get('capture_output') won't be set; just return a fake result
        class R:
            stdout = '{"headRefName": "feat/t_de993dac-add-logging", "headRefOid": "deadbeef1234"}'
            stderr = ""
            returncode = 0
        return R()

    monkeypatch.setattr(kb.subprocess, "run", fake_gh)
    # Clear any leftover override state from earlier tests
    kb._pending_reviewer_branch_override.clear()
    kb._BRANCH_OVERRIDE_CACHE.clear()

    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(conn)
        assert kb.check_respawn_guard(conn, tid) is None
        override = kb._pending_reviewer_branch_override.get(tid)
        assert override is not None, "branch override should be populated"
        branch, sha = override
        assert branch == "feat/t_de993dac-add-logging"
        assert sha == "deadbeef1234"


def test_branch_override_not_populated_when_guard_holds(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the guard returns active_pr, no branch override is set."""
    _patch_gh_review_decision(monkeypatch, decision=None)
    kb._pending_reviewer_branch_override.clear()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no feedback", assignee="builder")
        now = int(_time.time())
        _add_comment_at(
            conn, tid, author="builder",
            body="Opened https://github.com/example/repo/pull/178",
            created_at=now - 60,
        )
        assert kb.check_respawn_guard(conn, tid) == "active_pr"
        assert tid not in kb._pending_reviewer_branch_override


def test_dispatch_consumes_branch_override(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the dispatcher sees a populated ``_pending_reviewer_branch_override``
    entry, it pops the entry on claim and forwards ``pr_head_sha`` to the
    spawn fn via kwarg.

    Uses ``workspace_kind='scratch'`` to avoid the worktree provisioning
    path (which would require a real git project + worktree-lifecycle
    setup); the scratch path still exercises the override-popping logic
    AND the spawn-kwarg forwarding. The branch-rename side of the
    override (claimed.branch_name = pr_branch) only runs on worktree
    tasks, so this scratch test does NOT assert that.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    _ensure_builder_profile(kanban_home, monkeypatch)
    # Stub the gh call for branch lookup
    def fake_gh(*args, **kwargs):
        class R:
            stdout = '{"headRefName": "feat/t_de993dac-add-logging", "headRefOid": "deadbeef1234"}'
            stderr = ""
            returncode = 0
        return R()
    monkeypatch.setattr(kb.subprocess, "run", fake_gh)
    kb._pending_reviewer_branch_override.clear()
    kb._BRANCH_OVERRIDE_CACHE.clear()

    captured: dict = {}

    def spawn(task, workspace, *, board=None, pr_head_sha=None):
        captured["branch_name"] = task.branch_name
        captured["pr_head_sha"] = pr_head_sha
        return 12345  # fake PID

    # Make sure no parent env leak (regression guard for the prior
    # os.environ-mutation design — dispatcher must NOT touch the
    # process env, only the spawn kwarg).
    monkeypatch.delenv("HERMES_KANBAN_PR_HEAD_SHA", raising=False)
    original_env = dict(os.environ)

    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(conn, workspace_kind="scratch")
        # Trigger guard release, which populates _pending_reviewer_branch_override
        assert kb.check_respawn_guard(conn, tid) is None
        override = kb._pending_reviewer_branch_override.get(tid)
        assert override is not None
        pr_branch, pr_head_sha = override
        assert pr_branch == "feat/t_de993dac-add-logging"
        assert pr_head_sha == "deadbeef1234"

        # Run a tick.
        result = kb.dispatch_once(conn, spawn_fn=spawn)

    # Override dict cleared at end of tick
    assert kb._pending_reviewer_branch_override == {}
    # The task was spawned (it passed profile_exists gate thanks to
    # _ensure_builder_profile fixture).
    assert any(s[0] == tid for s in result.spawned), (
        f"task {tid} should be in spawned, got {result.spawned}"
    )
    # For scratch tasks the entire override path (branch rename + head
    # SHA forwarding) is intentionally skipped — the gate is
    # ``workspace_kind == "worktree"`` because only worktree workers
    # need to land on a specific branch + commit. The override was
    # popped at claim time (verified via the cleared dict above); we
    # assert the spawn fn saw NO override kwargs.
    assert captured.get("pr_head_sha") is None, (
        f"scratch task should not receive pr_head_sha override, "
        f"got {captured.get('pr_head_sha')!r}"
    )
    assert captured.get("branch_name") is None, (
        f"scratch task should keep its default branch_name (None), "
        f"got {captured.get('branch_name')!r}"
    )
    # Regression guard: the dispatcher must NOT mutate the parent's
    # process env to forward the SHA (prior design leaked the SHA from
    # one task's spawn into the next).
    assert os.environ == original_env, (
        f"dispatcher mutated parent process env: {set(os.environ) ^ set(original_env)}"
    )


def test_dispatch_applies_branch_rename_for_worktree_tasks(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On worktree tasks, ``check_respawn_guard`` release must cause
    ``claimed.branch_name`` to be replaced with the PR branch, AND the
    spawn fn to receive ``pr_head_sha`` as a kwarg.

    Stubs out ``_resolve_worktree_workspace`` to capture the claimed
    task at the moment ``replace(claimed, branch_name=pr_branch)``
    lands — we don't need a real git worktree to verify the rename
    happens; we just need to assert the kwargs going into the
    worktree-resolution call carry the override.
    """
    _patch_gh_review_decision(monkeypatch, decision=None)
    _ensure_builder_profile(kanban_home, monkeypatch)

    monkeypatch.setattr(kb.subprocess, "run", lambda *a, **kw: type(
        "R", (), {"stdout": '{"headRefName": "feat/t_de993dac-add-logging", "headRefOid": "deadbeef1234"}', "stderr": "", "returncode": 0}
    )())
    kb._pending_reviewer_branch_override.clear()
    kb._BRANCH_OVERRIDE_CACHE.clear()

    # Capture what the dispatcher's worktree resolver sees.
    seen: dict = {}

    def fake_resolve(task, board=None):
        seen["branch_name"] = task.branch_name
        # Return a sentinel workspace; the test doesn't run real git
        # worktree provisioning.
        from pathlib import Path as _P
        return _P("/tmp/fake-workspace"), task.branch_name or f"wt/{task.id}"

    monkeypatch.setattr(kb, "_resolve_worktree_workspace", fake_resolve)

    captured: dict = {}

    def spawn(task, workspace, *, board=None, pr_head_sha=None):
        captured["branch_name"] = task.branch_name
        captured["pr_head_sha"] = pr_head_sha
        return 12345

    monkeypatch.delenv("HERMES_KANBAN_PR_HEAD_SHA", raising=False)

    with kb.connect() as conn:
        tid = _setup_task_with_pr_and_feedback(conn, workspace_kind="worktree")
        # Pre-populate workspace_path so resolve_workspace doesn't try to
        # bootstrap a project repo.
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            ("/tmp/fake-workspace", tid),
        )
        # Trigger guard release to populate the override.
        assert kb.check_respawn_guard(conn, tid) is None
        result = kb.dispatch_once(conn, spawn_fn=spawn)

    # The resolver saw the PR branch as the claimed task's branch_name
    # (this is the dispatcher's `replace(claimed, branch_name=pr_branch)`).
    assert seen.get("branch_name") == "feat/t_de993dac-add-logging", (
        f"worktree resolver saw branch_name={seen.get('branch_name')!r}, "
        f"expected the PR branch override"
    )
    # The spawn fn received the head SHA kwarg.
    assert captured.get("pr_head_sha") == "deadbeef1234", (
        f"worktree spawn fn got pr_head_sha={captured.get('pr_head_sha')!r}, "
        f"expected 'deadbeef1234'"
    )
    assert any(s[0] == tid for s in result.spawned)

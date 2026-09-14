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
from pathlib import Path

import pytest

from hermes_cli import goals
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
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
        ready_id = kb.create_task(conn, title="already PRed", assignee="worker")
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


def _judge_must_not_run(*_args, **_kwargs):
    raise AssertionError(
        "re-authorizing a legacy correction must never consult the goal judge: "
        "it claims nothing about the work being finished"
    )


def _make_correction_legacy(conn, task_id: str) -> None:
    """Strip the captured boundary from the card's newest correction, leaving
    exactly the event shape written before boundaries existed."""
    row = conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? "
        "AND kind IN ('review_reopened', 'changes_requested') ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    payload = json.loads(row["payload"] or "{}")
    payload.pop(kb.REVIEWED_COMMENT_BOUND_KEY, None)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET payload = ? WHERE id = ?",
            (json.dumps(payload), row["id"]),
        )


def _legacy_ready_card(conn, title: str, pr: str, **create_kw) -> str:
    """A card in the state the pilot hit: ``ready`` after a real review handoff
    and correction, with a PR on the table and a pre-boundary correction row."""
    tid = kb.create_task(conn, title=title, assignee="worker", **create_kw)
    run = kb.claim_task(conn, tid)
    assert run is not None
    kb.add_comment(conn, tid, author="worker", body=pr)
    assert kb.request_review(
        conn, tid, summary="await review", expected_run_id=run.current_run_id,
    )
    assert kb.reopen_review_task(conn, tid) is True
    _make_correction_legacy(conn, tid)
    return tid


def test_explicit_review_correction_makes_same_pr_card_dispatchable(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ownership-checked review correction is authority to re-run the
    implementer on the PR that was already under review.

    Both correction paths — the controller's ``reopen-review`` and the
    reviewer's ``request_changes`` (only writable while holding the run
    claimed from ``review``) — return the card to the implementer *because of*
    the PR in the comments. Reading that same PR link as "a worker already
    opened a PR, don't respawn" strands the correction: the card sits ``ready``
    and never dispatches (native repro: create -> claim -> PR comment ->
    request_review -> reopen_review_task left ``check_respawn_guard`` at
    ``active_pr``).
    """
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    pr = "Implemented https://github.com/example/repo/pull/1"

    with kbc.connect() as conn:
        # Controller path: reopen-review on a card whose PR is on the table.
        reopened = kb.create_task(conn, title="correct the PR", assignee="worker")
        run = kb.claim_task(conn, reopened)
        assert run is not None
        kb.add_comment(conn, reopened, author="worker", body=pr)
        assert kb.request_review(
            conn, reopened, summary="await review", expected_run_id=run.current_run_id,
        )
        assert kb.reopen_review_task(conn, reopened) is True
        assert kb.get_task(conn, reopened).status == "ready"
        assert kbd.check_respawn_guard(conn, reopened) is None

        # Reviewer path: request_changes from the claimed review run.
        changed = kb.create_task(conn, title="reviewer sends it back", assignee="worker")
        impl = kb.claim_task(conn, changed)
        assert impl is not None
        kb.add_comment(conn, changed, author="worker", body=pr)
        assert kb.request_review(
            conn, changed, summary="await review", reviewer="reviewer",
            expected_run_id=impl.current_run_id,
        )
        review_run = kb.claim_review_task(conn, changed)
        assert review_run is not None
        ok, implementer = kb.request_changes(conn, changed, reason="fix the migration")
        assert (ok, implementer) == (True, "worker")
        assert kb.get_task(conn, changed).status == "ready"
        assert kbd.check_respawn_guard(conn, changed) is None

        # Recovery path: a goal-mode card already ``ready`` from a correction
        # written before boundaries existed. The operator re-authorizes it in
        # place; nothing is completed, accepted, or shown to the goal judge —
        # consulting the judge here fails the test.
        monkeypatch.setattr(goals, "judge_goal", _judge_must_not_run)
        stranded = _legacy_ready_card(conn, "goal card, old correction", pr, goal_mode=True)
        assert kbd.check_respawn_guard(conn, stranded) == "active_pr"
        assert kb.review_reauthorization_blocker(conn, stranded) is None
        assert kb.reopen_review_task(conn, stranded, reauthorize_legacy=True) is True
        # Status is untouched — the correction already moved it; only the
        # boundary this card never recorded is now on the event log.
        assert kb.get_task(conn, stranded).status == "ready"
        assert kbd.check_respawn_guard(conn, stranded) is None

        # ...and the dispatcher actually picks all three of them up.
        res = kbd.dispatch_once(conn, dry_run=True)
        assert {reopened, changed, stranded}.issubset({s[0] for s in res.spawned})
        assert not res.respawn_guarded


def test_active_pr_guard_holds_beyond_and_after_correction_authority(
    kanban_home: Path,
) -> None:
    """A correction authorizes the comments it acted on — nothing later.

    Three cards must stay ``active_pr``:

    * an ordinary ready card carrying a fresh PR link (nobody authorized a
      re-run — the duplicate-PR case the guard exists for);
    * a corrected card whose implementer then opened a SECOND PR **in the same
      wall-clock second** as the correction. Ordering here is the comment id
      the correction captured in its own write txn, so a same-second write
      cannot slip under the boundary;
    * a card whose correction was already answered — by a completion, or by a
      fresh ``review_requested`` handoff — and that is later re-queued by an
      ordinary operator action. Spent authority is not reusable.
    """
    pr1 = "Opened https://github.com/example/repo/pull/7"
    pr2 = "Superseded by https://github.com/example/repo/pull/8"

    with kbc.connect() as conn:
        # Ordinary duplicate-PR card: ready, PR link, no correction.
        ordinary = kb.create_task(conn, title="already PRed", assignee="worker")
        kb.add_comment(conn, ordinary, author="worker", body=pr1)
        assert kbd.check_respawn_guard(conn, ordinary) == "active_pr"

        # Corrected card that then received a newer PR link, same second.
        handoff = kb.create_task(conn, title="newer PR handoff", assignee="worker")
        run = kb.claim_task(conn, handoff)
        assert run is not None
        kb.add_comment(conn, handoff, author="worker", body=pr1)
        assert kb.request_review(
            conn, handoff, summary="await review", expected_run_id=run.current_run_id,
        )
        assert kb.reopen_review_task(conn, handoff) is True
        assert kbd.check_respawn_guard(conn, handoff) is None
        kb.add_comment(conn, handoff, author="worker", body=pr2)
        assert kbd.check_respawn_guard(conn, handoff) == "active_pr"

        # In-place re-authorization is refused for a ready card that carries no
        # correction at all: recovery re-states history, it never grants a
        # first-time exemption.
        assert "no review correction" in (
            kb.review_reauthorization_blocker(conn, ordinary) or ""
        )
        assert kb.reopen_review_task(conn, ordinary, reauthorize_legacy=True) is False
        assert kbd.check_respawn_guard(conn, ordinary) == "active_pr"

        # ...and for a legacy card whose implementer is running right now:
        # re-authorizing under a live claim would race that writer.
        claimed = _legacy_ready_card(conn, "legacy but claimed", pr1)
        assert kb.claim_task(conn, claimed) is not None
        assert "'running'" in (kb.review_reauthorization_blocker(conn, claimed) or "")
        assert kb.reopen_review_task(conn, claimed, reauthorize_legacy=True) is False

        # ...and for a ready row still holding a worker's claim lock, which the
        # reclaim passes own: re-authorizing there would race that writer.
        locked = _legacy_ready_card(conn, "legacy with a live lock", pr1)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET claim_lock = 'worker-1' WHERE id = ?", (locked,))
        assert "active run/claim" in (kb.review_reauthorization_blocker(conn, locked) or "")
        assert kb.reopen_review_task(conn, locked, reauthorize_legacy=True) is False

        # Spent authority: both cards are corrected, then answer the correction
        # (one completes, one hands off for review again) before an ancestor
        # reopen re-queues them.
        ancestor = kb.create_task(conn, title="ancestor", assignee="worker")
        finished = kb.create_task(conn, title="corrected then finished", assignee="worker")
        rehanded = kb.create_task(conn, title="corrected then re-reviewed", assignee="worker")
        for child in (finished, rehanded):
            kb.link_tasks(conn, ancestor, child)
        assert kb.complete_task(conn, ancestor, summary="parent done") is True
        for child in (finished, rehanded):
            child_run = kb.claim_task(conn, child)
            assert child_run is not None
            kb.add_comment(conn, child, author="worker", body=pr1)
            assert kb.request_review(
                conn, child, summary="await review",
                expected_run_id=child_run.current_run_id,
            )
            assert kb.reopen_review_task(conn, child) is True
            assert kbd.check_respawn_guard(conn, child) is None

        answered = kb.claim_task(conn, finished)
        assert answered is not None
        assert kb.complete_task(
            conn, finished, summary="shipped", expected_run_id=answered.current_run_id,
        ) is True
        rerun = kb.claim_task(conn, rehanded)
        assert rerun is not None
        assert kb.request_review(
            conn, rehanded, summary="second pass", expected_run_id=rerun.current_run_id,
        )

        kb.invalidate_descendants_for_parent_reopen(conn, ancestor, author="operator")
        kb.recompute_ready(conn)
        # ``finished`` is back in the ready lane; ``rehanded`` resumed its review
        # phase — neither may reuse the correction it already answered.
        assert kb.get_task(conn, finished).status == "ready"
        assert kbd.check_respawn_guard(conn, finished) == "active_pr"
        assert kbd.check_respawn_guard(conn, rehanded) == "active_pr"
        # Nor may re-authorization revive spent authority. Even reduced to the
        # legacy event shape, ``finished``'s correction was answered by a
        # completion; ``rehanded`` handed off for review again, so it is not a
        # ready card at all and in-place recovery never applies to it.
        for child in (finished, rehanded):
            _make_correction_legacy(conn, child)
        assert "'completed'" in (kb.review_reauthorization_blocker(conn, finished) or "")
        assert "'review'" in (kb.review_reauthorization_blocker(conn, rehanded) or "")
        assert kb.reopen_review_task(conn, finished, reauthorize_legacy=True) is False
        assert kbd.check_respawn_guard(conn, finished) == "active_pr"


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

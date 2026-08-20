"""Self-review guard: dispatcher must not respawn the implementer onto its
own just-completed review card.

Regression coverage for #kanban-self-review-gaveup (2026-08-19, t_eee837c7):
when a worker calls ``kanban_request_review(...)`` WITHOUT an explicit
``reviewer=``, ``request_review`` deliberately leaves ``tasks.assignee``
unchanged — it still names the IMPLEMENTER, not a distinct reviewer (see
``request_review``'s docstring). Before this fix, the review-lane dispatcher
keyed spawn eligibility purely on ``status='review' AND assignee is
spawnable``, so it re-spawned the SAME profile onto its OWN just-completed
card. That worker has no legitimate transition left to make (the card
already correctly holds real, delivered work) and either:

* re-issues ``kanban_request_review``/``kanban_complete`` — both correctly
  rejected with "task is not in running/ready" — then exits rc=0 with the
  task still ``running``, OR
* exits rc=0 outright without a terminal call.

Either way ``detect_crashed_workers`` reaped that exit as a clean-exit
protocol-violation ``crashed`` outcome, ticking ``consecutive_failures``
toward the circuit breaker — a false negative (``gave_up``) on a card whose
work was already complete and handed off. Measured case: t_ea832a93, runs
5258 and 5260, both spawned within 68s of run 5256's successful
``kanban_request_review``.

Fix: the review dispatch loop now checks the latest ``review_requested``
event's ``(implementer, reviewer)`` payload before calling
``claim_review_task``. When ``reviewer`` is ``None`` and ``implementer``
still equals the task's current ``assignee``, the spawn is skipped
(bucketed as ``skipped_self_review``, NOT a failure) and a one-time
event + comment explain why — mirroring the existing
``skipped_nonspawnable`` / ``_emit_nonspawnable_alert`` pattern. A review
submitted WITH a distinct ``reviewer=`` is unaffected and still spawns
normally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns):
    def _spawn(task, workspace, **kwargs):
        spawns.append(task.id)
        return 4242

    return _spawn


def _submit_for_review(conn, tid, *, reviewer=None):
    """Claim + request_review a task, mirroring what a real worker does."""
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    run_id = kb.get_task(conn, tid).current_run_id
    assert run_id is not None
    ok = kb.request_review(
        conn, tid,
        summary="work complete",
        reviewer=reviewer,
        expected_run_id=run_id,
    )
    assert ok is True


def test_self_review_card_is_skipped_not_spawned(kanban_home, monkeypatch):
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a fix", assignee="upx-senior-dev")
        _submit_for_review(conn, tid)  # no reviewer= passed

        row = conn.execute(
            "SELECT status, assignee FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert row["status"] == "review"
        assert row["assignee"] == "upx-senior-dev"

        spawns: list[str] = []
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

        # The dispatcher must NOT hand the implementer its own card back.
        assert tid not in spawns
        assert tid in res.skipped_self_review
        assert tid not in res.crashed

        # Card stays put in review — not spawned, not blocked, not crashed.
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert row["status"] == "review"

        events = kb.list_events(conn, tid)
        alert_events = [e for e in events if e.kind == "self_review_alerted"]
        assert len(alert_events) == 1
        assert alert_events[0].payload["assignee"] == "upx-senior-dev"

        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        assert "upx-senior-dev" in comments[0].body
        assert "same profile that implemented it" in comments[0].body


def test_self_review_alert_does_not_repeat_across_ticks(kanban_home, monkeypatch):
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a fix", assignee="upx-senior-dev")
        _submit_for_review(conn, tid)

        for _ in range(3):
            kb.dispatch_once(
                conn, spawn_fn=_fake_spawn_factory([]), max_in_progress=2,
            )

        events = kb.list_events(conn, tid)
        alert_events = [e for e in events if e.kind == "self_review_alerted"]
        assert len(alert_events) == 1


def test_distinct_reviewer_still_spawns_normally(kanban_home, monkeypatch):
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a fix", assignee="upx-senior-dev")
        _submit_for_review(conn, tid, reviewer="upx-reviewer")

        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert row["assignee"] == "upx-reviewer"

        spawns: list[str] = []
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

        assert tid in spawns
        assert tid not in res.skipped_self_review


def test_reopened_review_after_changes_requested_still_self_review_guarded(
    kanban_home, monkeypatch,
):
    """A re-review cycle (changes_requested -> fixed -> request_review again
    with no reviewer=) must still resolve the ORIGINAL reviewer from
    provenance, not silently treat it as self-review just because the
    latest event lacks an explicit reviewer= on the second call. This
    guards against over-tightening the fix into a new false block."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a fix", assignee="upx-senior-dev")
        _submit_for_review(conn, tid, reviewer="upx-reviewer")

        # Reviewer claims and requests changes.
        claimed = kb.claim_review_task(conn, tid)
        assert claimed is not None
        ok, reason = kb.request_changes(
            conn, tid, reason="needs another pass",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert ok, reason

        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert row["assignee"] == "upx-senior-dev"  # restored to implementer

        # Implementer fixes and re-submits WITHOUT reviewer= — request_review
        # reuses the durable reviewer provenance from the changes_requested
        # event, so assignee flips back to upx-reviewer, not self-review.
        _submit_for_review(conn, tid)

        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert row["assignee"] == "upx-reviewer"

        spawns: list[str] = []
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )
        assert tid in spawns
        assert tid not in res.skipped_self_review

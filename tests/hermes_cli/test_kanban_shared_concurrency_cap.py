"""One live concurrency cap must govern BOTH dispatch lanes (BUI-942 item 4).

``dispatch_once`` spawns from two queues in a single tick — the ready lane and
the review lane — and both add workers to ``status='running'``. The
``max_in_progress`` ceiling was only ever *composed into* the effective cap
inside::

    if max_in_progress is not None and ready_rows:

so a tick with an empty ready queue skipped the composition entirely and the
review lane dispatched against the raw ``max_spawn``, or against no ceiling at
all when only ``max_in_progress`` was configured.

That is not a theoretical ordering: :func:`reconcile_pr_ready_to_review` (item
2 of this same PR) runs *earlier in the same tick* and moves ready cards with a
live PR into ``review``. Handing off the last ready card is precisely what
leaves ``ready_rows`` empty on the tick the review lane wants to dispatch — so
the feature that fixed one hygiene bug opened a hole in the concurrency
ceiling, and a board sitting exactly at ``max_in_progress`` could be pushed
past it by a PR handoff.

These tests pin the fixed contract: the ceiling is resolved once, before any
spawning, independent of whether there is ready work, and every spawn in the
tick — ready or review — draws down the same budget.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


PR_URL = "https://github.com/acme/widgets/pull/4242"


def _park_in_review(conn, task_id):
    """Put a card in the ``review`` column, unclaimed — the state a worker
    leaves behind after opening a PR. Same direct-status helper the existing
    review-lane tests in test_kanban_db.py use."""
    conn.execute(
        "UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,)
    )


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# the exact regression: PR handoff empties ready, review lane must still be capped
# ---------------------------------------------------------------------------


def test_pr_handoff_with_empty_ready_must_not_spawn_past_max_in_progress(
    kanban_home, all_assignees_spawnable
):
    """THE regression.

    Board state: one worker already running, ``max_in_progress=1`` (the board
    is exactly at its ceiling), and one ready card carrying a live PR comment.

    Tick: ``reconcile_pr_ready_to_review`` moves that card ready -> review, so
    ``ready_rows`` comes back EMPTY. The old code therefore never entered the
    ``max_in_progress`` block — no ``in_progress >= max_in_progress`` check, no
    cap composition — and the review lane spawned a reviewer against
    ``max_spawn=2``, taking the board to 2 running with a ceiling of 1.
    """
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        running = kb.create_task(conn, title="already-running", assignee="alice")
        kb.claim_task(conn, running)

        handoff = kb.create_task(conn, title="has-pr", assignee="bob")
        kb.add_comment(conn, handoff, "worker", f"Opened {PR_URL}")

        res = kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_spawn=2, max_in_progress=1,
        )

        # The handoff itself still happens — item 2 is not being undone.
        assert handoff in res.pr_handoff_to_review
        assert kb.get_task(conn, handoff).status == "review"

        # ...but it must not buy the review lane a free slot.
        assert spawns == [], (
            "review lane spawned past max_in_progress on a tick where the PR "
            "handoff emptied the ready queue"
        )
        assert res.spawned == []
        running_now = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
        ).fetchone()[0]
    assert running_now == 1, "board exceeded its live max_in_progress ceiling"


def test_review_lane_is_capped_when_only_max_in_progress_is_set(
    kanban_home, all_assignees_spawnable
):
    """The worse shape of the same hole: with ``max_spawn`` unset, the old code
    left the review loop's break predicate at ``max_spawn is not None`` — i.e.
    False — so an empty ready queue meant the review lane was *completely
    uncapped* and would drain every review card in one tick."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        running = kb.create_task(conn, title="already-running", assignee="alice")
        kb.claim_task(conn, running)
        for i in range(4):
            t = kb.create_task(conn, title=f"pr-{i}", assignee="bob")
            kb.add_comment(conn, t, "worker", f"Opened {PR_URL}/{i}")

        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_in_progress=2)

        assert len(res.pr_handoff_to_review) == 4
        # Ceiling 2, one already running → exactly one more worker.
        assert len(spawns) == 1
        running_now = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
        ).fetchone()[0]
    assert running_now == 2


def test_review_lane_at_ceiling_spawns_nothing_with_no_ready_work(
    kanban_home, all_assignees_spawnable
):
    """No ready cards at all (nothing to hand off, nothing queued), board
    already at ``max_in_progress`` — the review lane must spawn nothing. The
    old early-return that enforced this lived behind ``and ready_rows``, so an
    idle ready queue skipped it."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        running = kb.create_task(conn, title="already-running", assignee="alice")
        kb.claim_task(conn, running)
        pending_review = kb.create_task(conn, title="in-review", assignee="bob")
        _park_in_review(conn, pending_review)

        res = kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_spawn=5, max_in_progress=1,
        )

    assert spawns == []
    assert res.spawned == []


# ---------------------------------------------------------------------------
# the shared budget: ready and review draw from ONE pool
# ---------------------------------------------------------------------------


def test_ready_and_review_share_one_budget_in_the_same_tick(
    kanban_home, all_assignees_spawnable
):
    """Two ready cards and two review cards under a ceiling of 3 with nothing
    running: exactly 3 spawns total across both lanes, not 3 per lane."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        for i in range(2):
            kb.create_task(conn, title=f"ready-{i}", assignee="alice")
        for i in range(2):
            t = kb.create_task(conn, title=f"rev-{i}", assignee="bob")
            _park_in_review(conn, t)

        res = kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_spawn=3, max_in_progress=3,
        )

    assert len(spawns) == 3
    assert len(res.spawned) == 3


def test_running_workers_are_subtracted_exactly_once_for_the_review_lane(
    kanban_home, all_assignees_spawnable
):
    """The item-3 double-count must not reappear via the review lane: one
    worker running under ``min(3, 3) == 3`` leaves headroom for two more, and
    the review lane may use it."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        running = kb.create_task(conn, title="run", assignee="alice")
        kb.claim_task(conn, running)
        for i in range(4):
            t = kb.create_task(conn, title=f"rev-{i}", assignee="bob")
            _park_in_review(conn, t)

        res = kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_spawn=3, max_in_progress=3,
        )
        running_now = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
        ).fetchone()[0]

    assert len(res.spawned) == 2
    assert running_now == 3


def test_no_ceilings_configured_still_means_unlimited(
    kanban_home, all_assignees_spawnable
):
    """Guard the other direction: sharing one cap must not accidentally impose
    one when the operator configured neither knob."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"ready-{i}", assignee="alice")
        for i in range(3):
            t = kb.create_task(conn, title=f"rev-{i}", assignee="bob")
            _park_in_review(conn, t)

        kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 6


def test_dry_run_review_spawns_also_draw_down_the_shared_cap(
    kanban_home, all_assignees_spawnable
):
    """A dry run reports what a real tick WOULD do, so its would-be review
    spawns have to respect the same shared ceiling."""
    with kb.connect() as conn:
        for i in range(2):
            kb.create_task(conn, title=f"ready-{i}", assignee="alice")
        for i in range(3):
            t = kb.create_task(conn, title=f"rev-{i}", assignee="bob")
            _park_in_review(conn, t)

        res = kb.dispatch_once(
            conn, dry_run=True, max_spawn=3, max_in_progress=3,
        )

    assert len(res.spawned) == 3

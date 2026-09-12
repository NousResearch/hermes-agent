"""#101638: review-lane hold + stale-run reclaim.

* ``request-review --force`` over a live claim parks the card: the dispatcher
  must not re-claim it on the next tick (``review_hold`` guard in the shared
  ``claim_review_task`` path + review lane enumeration skips held cards).
* A normal worker handoff (``expected_run_id``) is NOT a park: review dispatch
  proceeds unchanged.
* ``claim_review_task`` reclaims a dangling open run in the same txn
  (mirrors ``claim_task``), so dead workers leave no stale ``running`` rows.
* ``release_stale_claims`` closes the stale run row in the same txn that
  re-queues the task (baseline pin for part (b)).
* An explicit human ``kanban claim`` on a held review card unparks it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _dispatch_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *args, **kwargs: {"kanban": {"review_dispatch": True}},
    )


def _open_runs(conn, tid: str):
    return conn.execute(
        "SELECT id, status, outcome, ended_at FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NULL",
        (tid,),
    ).fetchall()


def _events(conn, tid: str, kind: str):
    return [
        (json.loads(r["payload"]) if r["payload"] else None)
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (tid, kind),
        ).fetchall()
    ]


def test_force_request_review_parks_card_dispatcher_leaves_it(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The issue's scenario: park via force, next tick must not spawn."""
    _dispatch_stubs(monkeypatch)
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 12345

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="park me", assignee="worker")
        kb.claim_task(conn, tid)
        ok, _ = kb.request_review(conn, tid, summary="waiting on human", force=True, with_reason=True)
        assert ok is True
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "review"
        assert bool(task.review_hold) is True

        # Direct claim path also refuses a parked card.
        assert kb.claim_review_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "review"

        res = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert tid not in [s[0] for s in res.spawned]
        assert tid not in spawned
        assert kb.get_task(conn, tid).status == "review"


def test_normal_handoff_still_dispatches(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker handoff with ownership proof is not a park: reviewer spawns."""
    _dispatch_stubs(monkeypatch)

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.request_review(conn, tid, summary="done", expected_run_id=claimed.current_run_id) is True
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "review"
        assert bool(task.review_hold) is False

        res = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res.spawned]


def test_claim_review_task_reclaims_dangling_run(kanban_home: Path) -> None:
    """Dead worker's leaked open run closes in the claim txn (cf. claim_task)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="dangling", assignee="reviewer")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        leaked_run_id = int(claimed.current_run_id)
        # Simulate a host-level death mid-handoff: task sits in review while
        # the implementer run row is still open (ended_at IS NULL).
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'review', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
                (tid,),
            )
        assert len(_open_runs(conn, tid)) == 1

        got = kb.claim_review_task(conn, tid)
        assert got is not None
        assert got.status == "running"
        # The leaked row is terminal now; only the fresh reviewer run is open.
        leaked = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
            (leaked_run_id,),
        ).fetchone()
        assert leaked["ended_at"] is not None
        assert leaked["outcome"] == "reclaimed"
        still_open = _open_runs(conn, tid)
        assert len(still_open) == 1
        assert int(still_open[0]["id"]) != leaked_run_id


def test_release_stale_claims_closes_run_row(kanban_home: Path) -> None:
    """Reclaim re-queues AND stamps the stale run terminal in one txn."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale", assignee="worker")
        claimed = kb.claim_task(conn, tid, ttl_seconds=1)
        assert claimed is not None
        run_id = int(claimed.current_run_id)
        # Expire the claim and ensure the worker looks dead (no such pid).
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_expires = 1, worker_pid = ? WHERE id = ?",
                (4199999, tid),
            )

        n = kb.release_stale_claims(conn)
        assert n == 1
        task = kb.get_task(conn, tid)
        assert task is not None and task.current_run_id is None
        assert task.status in ("ready", "review")
        run = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert run["ended_at"] is not None
        assert run["outcome"] == "reclaimed"
        assert _events(conn, tid, "reclaimed")


def test_human_claim_unparks_held_card(kanban_home: Path) -> None:
    """Explicit human pull (``kanban claim`` path) clears the hold."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="held", assignee="reviewer")
        kb.claim_task(conn, tid)
        ok, _ = kb.request_review(conn, tid, summary="parked", force=True, with_reason=True)
        assert ok is True
        assert bool(kb.get_task(conn, tid).review_hold) is True

        got = kb.claim_review_task(conn, tid, force=True)
        assert got is not None
        assert got.status == "running"
        assert bool(kb.get_task(conn, tid).review_hold) is False


def test_reopen_clears_hold(kanban_home: Path) -> None:
    """Leaving review via reopen drops the hold bit with the lane."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="held", assignee="worker")
        kb.claim_task(conn, tid)
        ok, _ = kb.request_review(conn, tid, summary="parked", force=True, with_reason=True)
        assert ok is True
        assert kb.reopen_review_task(conn, tid) is True
        task = kb.get_task(conn, tid)
        assert task is not None and task.status in ("ready", "todo")
        assert bool(task.review_hold) is False

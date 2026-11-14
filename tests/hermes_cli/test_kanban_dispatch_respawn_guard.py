"""Regression tests for the respawn guard's stale-failure-stamp handling.

``check_respawn_guard`` parks a card as ``blocker_auth`` when the card's
PERSISTED ``last_failure_error`` matches the quota/auth pattern.  That text
outlives the run that wrote it: when the watchdog later reclaims the newest
attempt (broken claim bookkeeping, dead worker), that run's outcome becomes
``reclaimed`` while the card keeps the OLD stamp.  The guard then read a
long-gone quota wall as a current diagnosis and parked the card permanently.

A ``reclaimed`` run carries no diagnosis of its own, so a stamp that predates
it must not park the card -- while a card whose newest run genuinely failed
with a quota/auth error must STILL be parked, and the breaker must still trip
eventually.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _quota_spawn(*_args, **_kwargs):
    """A worker that hits the quota wall."""
    raise RuntimeError("429 quota exceeded for provider acme")


def _reclaim_after_broken_bookkeeping(conn, task_id, now):
    """Run the real watchdog reclaim for a card whose worker died mid-claim.

    ``reconcile_orphaned_running`` requeues a card left ``running`` with
    broken claim bookkeeping (the crash-mid-claim state it is built for) and
    ends that run as ``reclaimed`` -- without rewriting the card's
    ``last_failure_error``.
    """
    kb.claim_task(conn, task_id)
    conn.execute("UPDATE tasks SET claim_expires = NULL WHERE id = ?", (task_id,))
    conn.commit()
    assert kbd.reconcile_orphaned_running(conn) == [task_id]


def _latest_run(conn, task_id):
    return conn.execute(
        "SELECT outcome, ended_at FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()


def test_respawn_guard_does_not_park_a_card_on_stale_text_after_a_reclaimed_run(
    kanban_home, monkeypatch, all_assignees_spawnable,
):
    """A quota stamp that predates the newest run must not park the card.

    The quota wall is gone; the watchdog already reclaimed the attempt that
    followed it, and that reclaim carries no diagnosis.  Reading the stale
    stamp parked the card forever.
    """
    now = 5_000_000
    monkeypatch.setattr(time, "time", lambda: now)

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale quota stamp", assignee="a")

        # A real quota failure stamps the card.
        kbd.dispatch_once(conn, spawn_fn=_quota_spawn)
        card = conn.execute(
            "SELECT status, last_failure_error FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert card["status"] == "ready"
        assert "quota" in card["last_failure_error"]
        assert _latest_run(conn, tid)["outcome"] == "spawn_failed"

        # The watchdog reclaims the next attempt (its worker died mid-claim).
        # The reclaim does NOT rewrite the card's stamp.
        monkeypatch.setattr(time, "time", lambda: now + 10)
        _reclaim_after_broken_bookkeeping(conn, tid, now + 10)
        assert _latest_run(conn, tid)["outcome"] == "reclaimed"
        card = conn.execute(
            "SELECT last_failure_error FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert "quota" in card["last_failure_error"]

        # The stale text must not park the card.
        assert kbd.check_respawn_guard(conn, tid) is None


def test_respawn_guard_still_parks_a_card_whose_latest_run_failed_with_quota(
    kanban_home, monkeypatch, all_assignees_spawnable,
):
    """The guard is NOT weakened: a genuine quota/auth failure still parks.

    The newest run's own failure is a current diagnosis -- parking it is the
    whole point of the guard.
    """
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="live quota failure", assignee="a")
        kbd.dispatch_once(conn, spawn_fn=_quota_spawn)

        assert _latest_run(conn, tid)["outcome"] == "spawn_failed"
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"


def test_respawn_guard_reparks_a_reclaimed_card_once_the_quota_failure_repeats(
    kanban_home, monkeypatch, all_assignees_spawnable,
):
    """A reclaimed card is retried, and a fresh quota failure parks it again.

    Lifting the stale park must not make the guard toothless: the card is
    retried, and the retry that genuinely hits the quota wall is parked with
    ``blocker_auth`` again -- so the breaker still trips eventually.
    """
    now = 5_000_000
    monkeypatch.setattr(time, "time", lambda: now)

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota, reclaim, quota", assignee="a")

        # A generous failure limit keeps the BREAKER out of the way, so this
        # test observes the guard alone (the breaker tripping is a separate
        # contract, exercised by the card's own retry budget).
        kbd.dispatch_once(conn, spawn_fn=_quota_spawn, failure_limit=5)
        monkeypatch.setattr(time, "time", lambda: now + 10)
        _reclaim_after_broken_bookkeeping(conn, tid, now + 10)

        # The retry runs: the card is not parked on the stale stamp, so the
        # dispatcher hands it to a worker instead of skipping it.
        monkeypatch.setattr(time, "time", lambda: now + 20)
        res = kbd.dispatch_once(conn, spawn_fn=_quota_spawn, failure_limit=5)
        assert not [entry for entry in res.respawn_guarded if entry[0] == tid]

        # That retry hit the quota wall again -- a fresh, current diagnosis.
        assert _latest_run(conn, tid)["outcome"] == "spawn_failed"
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"

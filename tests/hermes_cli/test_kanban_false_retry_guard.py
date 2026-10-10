"""A failure recorded against a card that already finished must not fake a retry.

The worker-spawn failure path funnels through ``_record_task_failure`` with
``release_claim=True, end_run=True``: it restores the source phase, clears the
claim and closes the run with the ``spawn_failed`` outcome. That write is
conditional on ``status = 'running'``, so when the card left ``running`` first —
it reached a terminal success (the iteration-budget path races
``kanban_complete``, which the spawn caller cannot see), was blocked, or was
reclaimed — the UPDATE matches nothing.

The counter/claim half was already a no-op there, but the run/event half was not:
the ``spawn_failed`` event landed anyway, and that event is what the notifier
renders as "timed out; dispatcher will retry" — a false retry stamped onto a card
that finished successfully and will never be retried. ``enforce_max_runtime``
guards its twin path with ``cur.rowcount == 1``; this is the same guard.

Regression for the ``local/hermes-fixes`` commit that carried it.
"""

from __future__ import annotations

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


def _kinds(conn, task_id: str) -> list[str]:
    return [
        row["kind"] for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
    ]


def _spawn_failed(conn, task_id: str) -> bool:
    return kbd._record_task_failure(
        conn, task_id, "spawn blew up",
        outcome="spawn_failed", failure_limit=3,
        release_claim=True, end_run=True,
    )


def test_a_finished_card_is_not_given_a_false_retry(kanban_home):
    """The card completed before the spawn failure was recorded: record nothing.

    The ``spawn_failed`` event is the notifier's "timed out; dispatcher will
    retry" line. Appending it to a card that is already ``done`` invents a retry
    that cannot happen and contradicts the card's own terminal status.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="finishes mid-spawn", assignee="default")
        assert kb.claim_task(conn, tid) is not None
        assert kb.complete_task(conn, tid, summary="done", force=True) is True
        before = _kinds(conn, tid)

        tripped = _spawn_failed(conn, tid)

        after = _kinds(conn, tid)
        status = kb.get_task(conn, tid).status

    assert tripped is False
    assert after == before, "a finished card must not gain a spawn_failed retry event"
    assert status == "done"


def test_a_running_card_still_records_the_spawn_failure(kanban_home):
    """Boundary: the guard must not silence the path it exists for.

    A card that really is still ``running`` under a live claim gets the failure
    recorded and its claim released, so the next tick can retry it.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="spawn really failed", assignee="default")
        assert kb.claim_task(conn, tid) is not None

        tripped = _spawn_failed(conn, tid)

        kinds = _kinds(conn, tid)
        task = kb.get_task(conn, tid)

    assert tripped is False, "one failure under a limit of three must not auto-block"
    assert "spawn_failed" in kinds
    assert task.status == "ready", "the source phase is restored for the next tick"
    assert task.claim_lock is None, "the dead worker's claim must be released"
    assert task.consecutive_failures == 1

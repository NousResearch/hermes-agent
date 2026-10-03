"""Dead-letter lane of the kanban dispatcher tick (invariant tests).

``gave_up``/``blocked`` cards stuck past ``kanban.dead_letter_after_hours``
get an idempotent ``dead_letter`` task_event so the corpse is visible.
Visibility-only contract: no status change, no requeue, no notification;
a second tick never duplicates the mark. Both invariants below drive the
REAL ``dispatch_once`` pass against a throwaway HERMES_HOME board.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time

import pytest


HOUR = 3600

MULTI_LINE_ERROR = "provider 500: upstream exploded\nquota exhausted until tomorrow\nretry later"
BLOCKED_REASON = "waiting on credentials"


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Spin up a fresh HERMES_HOME with a clean kanban DB."""
    test_home = tempfile.mkdtemp(prefix="kanban_dead_letter_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    # Force-reimport so the fresh HERMES_HOME is picked up.
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db, test_home


def _fake_spawn(*args, **kwargs):
    """Stand-in for the real worker spawn — returns a fake PID."""
    return 12345


def _age_boundary_event(conn, task_id: str, kind: str, *, age_hours: float) -> None:
    """Backdate the card's status boundary event so the current episode ages."""
    from hermes_cli import kanban_db as kb
    old = int(time.time()) - int(age_hours * HOUR)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET created_at = ? "
            "WHERE task_id = ? AND kind = ?",
            (old, task_id, kind),
        )


def _make_gave_up_card(conn, kb, *, age_hours: float, error: str) -> str:
    """A ``gave_up`` card whose gave_up event (the status anchor) is age_hours old."""
    tid = kb.create_task(conn, title="dead-letter gave_up")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'gave_up' WHERE id = ?", (tid,))
        kb._append_event(conn, tid, "gave_up", {"error": error})
    _age_boundary_event(conn, tid, "gave_up", age_hours=age_hours)
    return tid


def _make_sticky_blocked_card(conn, kb, *, age_hours: float, reason: str) -> str:
    """A sticky-blocked card (kb.block_task — raw SQL blocked is auto-recovered
    by recompute_ready) whose blocked event (the status anchor) is age_hours old."""
    tid = kb.create_task(conn, title="dead-letter blocked")
    assert kb.block_task(conn, tid, kind="needs_input", reason=reason)
    _age_boundary_event(conn, tid, "blocked", age_hours=age_hours)
    return tid


def _dead_letter_events(conn, task_id: str) -> list:
    return list(conn.execute(
        "SELECT id, payload FROM task_events "
        "WHERE task_id = ? AND kind = 'dead_letter' ORDER BY id ASC",
        (task_id,),
    ))


def _task_status(conn, task_id: str) -> str:
    return conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()["status"]


def test_aged_gave_up_and_blocked_cards_get_one_idempotent_dead_letter_event(
    isolated_kanban_home,
):
    """Aged gave_up + aged sticky blocked cards get exactly one ``dead_letter``
    event each (reason = cause's first line), statuses untouched; the next tick
    is a no-op (idempotent). A young gave_up card is never marked."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        aged_gave_up = _make_gave_up_card(
            conn, kb, age_hours=2, error=MULTI_LINE_ERROR,
        )
        aged_blocked = _make_sticky_blocked_card(
            conn, kb, age_hours=2, reason=BLOCKED_REASON,
        )
        young_gave_up = _make_gave_up_card(
            conn, kb, age_hours=0.01, error="too fresh to bury",
        )

        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dead_letter_after_hours=1,
        )
        assert set(res.dead_lettered) == {aged_gave_up, aged_blocked}

        # Exactly one mark per aged card, payload contract, reason = first line.
        by_task = {}
        for tid in (aged_gave_up, aged_blocked, young_gave_up):
            events = _dead_letter_events(conn, tid)
            assert len(events) == (1 if tid in (aged_gave_up, aged_blocked) else 0)
            if events:
                payload = json.loads(events[0]["payload"])
                assert set(payload) == {"age_hours", "status", "block_kind", "reason"}
                assert payload["age_hours"] >= 2.0
                by_task[tid] = payload

        gave_up_payload = by_task[aged_gave_up]
        assert gave_up_payload["status"] == "gave_up"
        assert gave_up_payload["block_kind"] is None
        assert gave_up_payload["reason"] == MULTI_LINE_ERROR.splitlines()[0]

        blocked_payload = by_task[aged_blocked]
        assert blocked_payload["status"] == "blocked"
        assert blocked_payload["reason"] == BLOCKED_REASON

        # Visibility-only: statuses unchanged by the marking pass.
        assert _task_status(conn, aged_gave_up) == "gave_up"
        assert _task_status(conn, aged_blocked) == "blocked"
        assert _task_status(conn, young_gave_up) == "gave_up"

        # Second tick: idempotent — no new events, empty roster.
        res2 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dead_letter_after_hours=1,
        )
        assert res2.dead_lettered == []
        assert len(_dead_letter_events(conn, aged_gave_up)) == 1
        assert len(_dead_letter_events(conn, aged_blocked)) == 1


def test_dead_letter_lane_disabled_for_zero_threshold(isolated_kanban_home):
    """``dead_letter_after_hours=0`` disables the lane: aged corpse stays
    unmarked and the tick reports an empty dead-lettered roster."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        aged_gave_up = _make_gave_up_card(
            conn, kb, age_hours=2, error="stale failure nobody triaged",
        )

        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dead_letter_after_hours=0,
        )
        assert res.dead_lettered == []
        assert _dead_letter_events(conn, aged_gave_up) == []
        assert _task_status(conn, aged_gave_up) == "gave_up"

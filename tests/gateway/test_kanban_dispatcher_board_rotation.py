"""Board-order starvation of the host-wide dispatch budget (gateway dispatcher).

Each tick hands every board its own ``dispatch_once``, and each of those
re-reads the host-wide ``kanban.max_in_progress`` budget
(``hermes_cli.kanban_db_dispatch._tick_spawn_budget``). The board offered the
budget first therefore consumes the free slot and every board after it is
refused for that tick. With a fixed board order the early board takes every
slot that frees up, so a board later in the list is starved indefinitely while
its ready queue stays non-empty.

That was the reported symptom: ``ops`` cards sat ``ready`` for hours while
``default`` spawned normally, and the same cards drained immediately under a
manual ``hermes kanban --board ops dispatch`` — which offers only that one
board, so the ordering never came into play.

``_KanbanDispatcher.tick_once`` now rotates which board is offered first. The
budget arithmetic is untouched: rotation only decides who gets to spend it, so
it cannot raise host concurrency.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

ASSIGNEE = "worker-a"


@pytest.fixture(autouse=True)
def all_assignees_spawnable(monkeypatch):
    """Treat synthetic assignees as real profiles.

    Without this the dispatcher's profile-exists guard (PR #20105) routes every
    card below into ``skipped_nonspawnable`` instead of spawning.
    """
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with the default board initialized."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Immediate crash reclaim: the dead-pid case below must not depend on the
    # launch-window grace timer.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def spawned(monkeypatch):
    """Record ``(board, task_id)`` per spawn, reporting a LIVE pid.

    The pid has to look alive: a spawned row only keeps holding its slot if
    ``detect_crashed_workers`` cannot reap it on the next tick.
    """
    calls: list[tuple[str, str]] = []

    def fake_spawn(task, workspace, board=None):
        # ``_call_spawn_fn`` only passes ``board`` when the callable names it,
        # so this signature is load-bearing.
        calls.append((board, task.id))
        return os.getpid()

    monkeypatch.setattr(kbd, "_default_spawn", fake_spawn)
    return calls


def _settings(**overrides):
    base: dict[str, Any] = dict(
        interval=60.0,
        max_spawn=None,
        max_in_progress=1,
        failure_limit=kb.DEFAULT_FAILURE_LIMIT,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=None,
    )
    base.update(overrides)
    return _DispatcherSettings(**base)


def _add_ready_cards(slug: str, count: int) -> list[str]:
    """Create board *slug* (idempotent) and put *count* ready cards on it."""
    kb.create_board(slug)
    conn = kbc.connect(board=slug)
    try:
        return [
            kb.create_task(conn, title=f"{slug} card {i}", assignee=ASSIGNEE, board=slug)
            for i in range(count)
        ]
    finally:
        conn.close()


def _task_status(slug: str, task_id: str) -> str:
    conn = kbc.connect(board=slug)
    try:
        return conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0]
    finally:
        conn.close()


def _finish(slug: str, task_id: str) -> None:
    """Simulate the worker closing its card, freeing its slot."""
    conn = kbc.connect(board=slug)
    try:
        conn.execute(
            "UPDATE tasks SET status='done', claim_lock=NULL, claim_expires=NULL WHERE id = ?",
            (task_id,),
        )
        conn.commit()
    finally:
        conn.close()


def _dead_pid() -> int:
    """PID of a child that has already exited and been reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# ---------------------------------------------------------------------------
# 1. The rotation itself: the first-served board changes every tick
# ---------------------------------------------------------------------------

def test_tick_once_rotates_which_board_is_served_first(kanban_home, monkeypatch):
    kb.create_board("alpha")
    kb.create_board("beta")
    dispatcher = _KanbanDispatcher(kb, _settings())

    visited: list[str] = []
    monkeypatch.setattr(
        dispatcher, "tick_once_for_board", lambda slug: visited.append(slug)
    )

    dispatcher.tick_once()
    first_tick = list(visited)
    visited.clear()
    dispatcher.tick_once()
    second_tick = list(visited)

    # ``default`` leads the board list (``list_boards``); the rotation must
    # move it off the head of the queue instead of pinning it there.
    assert first_tick == ["default", "alpha", "beta"]
    assert second_tick == ["alpha", "beta", "default"]
    assert sorted(second_tick) == sorted(first_tick)


# ---------------------------------------------------------------------------
# 2. The reported defect: the last board in order is starved
# ---------------------------------------------------------------------------

def test_free_slot_reaches_the_last_board_in_order(kanban_home, spawned):
    """A board with one ready card must be served within one rotation.

    ``alpha`` has a backlog, so it wins the single host slot on any tick where
    it is offered the budget first; ``zulu`` is last in board order and, before
    the rotation, never got a slot at all.
    """
    _add_ready_cards("alpha", 3)
    _add_ready_cards("zulu", 1)

    dispatcher = _KanbanDispatcher(kb, _settings(max_in_progress=1))
    boards = dispatcher._board_slugs()
    assert boards[-1] == "zulu", "fixture assumption: zulu sorts last"

    served: set[str] = set()
    consumed = 0
    for _ in range(len(boards)):
        dispatcher.tick_once()
        for board, task_id in spawned[consumed:]:
            served.add(board)
            _finish(board, task_id)
        consumed = len(spawned)

    assert "zulu" in served, "last board in order was never offered the free slot"
    assert served == {"alpha", "zulu"}, "only boards with a backlog should spawn here"


# ---------------------------------------------------------------------------
# 3. Contract the rotation relies on: a dead worker's slot frees in-tick
# ---------------------------------------------------------------------------

def test_dead_worker_pid_does_not_hold_the_host_budget(kanban_home, spawned):
    """A ``running`` row whose worker pid is gone must not consume the budget.

    Regression guard (not a new behaviour): the tick's reclaim phase runs
    *before* ``_tick_spawn_budget`` reads the running count, so a crashed
    worker's slot is free again within the same tick. This is the half of the
    fairness contract the rotation depends on — a rotation that hands out slots
    that were never actually released would still starve the last board.
    """
    kb.create_board("alpha")
    conn = kbc.connect(board="alpha")
    try:
        victim = kb.create_task(conn, title="crashed worker", assignee=ASSIGNEE, board="alpha")
        dead_claim = kb.claim_task(conn, victim)
        assert dead_claim is not None
        # The worker was claimed, then died: the pid is gone and the claim is
        # old enough that the launch-window grace timer has expired.
        conn.execute(
            "UPDATE tasks SET worker_pid=?, started_at=? WHERE id = ?",
            (_dead_pid(), int(time.time()) - 3600, victim),
        )
        conn.commit()
    finally:
        conn.close()
    _add_ready_cards("zulu", 1)

    dispatcher = _KanbanDispatcher(kb, _settings(max_in_progress=1))
    dispatcher.tick_once()

    conn = kbc.connect(board="alpha")
    try:
        row = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (victim,)
        ).fetchone()
    finally:
        conn.close()
    # The dead claim is gone and the slot it held was spendable in the same tick
    # (the row may well be re-claimed immediately — that is the point).
    assert row[1] != dead_claim, "crashed worker's claim was not reclaimed"
    assert spawned, "the reclaimed slot was not usable in the same tick"


# ---------------------------------------------------------------------------
# 4. The same fixed-order defect, on the auto-decompose budget
# ---------------------------------------------------------------------------

def test_auto_decompose_tick_rotates_which_board_spends_the_budget(
    kanban_home, monkeypatch,
):
    """``auto_decompose_per_tick`` is host-wide, so board order must rotate too.

    One triage task per board and a per-tick cap of 1: a fixed board order
    decomposes the first board on every tick and never touches the others,
    however long their triage queue sits there.
    """
    from hermes_cli import kanban_decompose as decomp

    kb.create_board("alpha")
    kb.create_board("zulu")
    triage = {"alpha": ["t-alpha"], "zulu": ["t-zulu"]}
    monkeypatch.setattr(
        decomp,
        "list_triage_ids",
        lambda: list(triage.get(os.environ["HERMES_KANBAN_BOARD"], [])),
    )

    decomposed: list[str] = []
    dispatcher = _KanbanDispatcher(kb, _settings())
    monkeypatch.setattr(
        dispatcher,
        "_decompose_one",
        lambda _decomp, slug, tid: decomposed.append(slug) or True,
    )

    for _ in range(len(dispatcher._board_slugs())):
        dispatcher.auto_decompose_tick(1)

    assert set(decomposed) == {"alpha", "zulu"}, (
        "per-tick decompose budget stayed on the first board in board order"
    )

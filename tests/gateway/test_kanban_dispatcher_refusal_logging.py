"""A dispatch tick refused the shared spawn budget must leave a trace.

``_tick_spawn_budget`` returns its refusal before either lane is enumerated, so
a refused board produced no ``spawned`` entry, no ``skipped_*`` bucket and no
board-scoped log line: "host budget full" was indistinguishable from "board
idle" and from "dispatcher dead" — the only warning was host-wide and named no
board. These tests drive a real refusal through ``dispatch_once`` and assert the
operator-visible consequence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gateway.kanban_watchers_dispatcher import _log_spawn_results
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

# ``gateway.kanban_watchers_dispatcher`` re-uses ``gateway.kanban_watchers_common``'s
# long-standing logger (name ``"gateway.run"``) so extracted log records stay
# unchanged — that is the logger ``_log_spawn_results`` writes under.
LOGGER_NAME = "gateway.run"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Per-test kanban home so boards land in ``tmp_path``, not ``~/.hermes``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board=kb.DEFAULT_BOARD)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def _every_assignee_spawnable(monkeypatch):
    """``worker`` is a synthetic assignee with no profile dir on disk; without
    this the lanes bucket it as ``skipped_nonspawnable`` instead of spawning
    (mirror of ``tests/hermes_cli/conftest.py``'s fixture)."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


def _board_at_cap():
    """Two boards, one worker slot: ``holder`` runs a worker and ``waiter``
    queues a ready card, so a ``max_in_progress=1`` tick on ``waiter`` is
    refused the shared budget."""
    kb.create_board("holder")
    kb.create_board("waiter")
    holder = kbc.connect(board="holder")
    waiter = kbc.connect(board="waiter")
    # ``create_task`` lands the row in ``ready`` (its ``initial_status`` only
    # parks a task); claiming it is what makes it ``running`` — the worker
    # occupying the single host-wide slot.
    holder_task = kb.create_task(holder, title="occupies the only slot", assignee="worker")
    assert kb.claim_task(holder, holder_task) is not None
    kb.create_task(waiter, title="queued behind the cap", assignee="worker")
    return holder, waiter


def _tick(conn, board, **kwargs):
    # dry_run: the tick must never spawn a real worker process.
    return kbd.dispatch_once(conn, board=board, dry_run=True, **kwargs)


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]


def test_refused_tick_is_logged_once_with_its_reason(kanban_home, caplog):
    _holder, waiter = _board_at_cap()

    refused = _tick(waiter, "waiter", max_in_progress=1)

    assert refused.spawned == []
    assert refused.budget_blocked == "max_in_progress"
    assert refused.budget_blocked_cap == 1
    assert refused.budget_blocked_own == 0
    assert refused.budget_blocked_other == 1
    assert refused.budget_blocked_pending == 1  # the queued card is visible

    deferred: dict[str, str] = {}
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _log_spawn_results([("waiter", refused)], deferred)
        # A capped host stays capped for ticks on end: steady state is silent.
        _log_spawn_results([("waiter", refused)], deferred)

    lines = _lines(caplog)
    assert len(lines) == 1, lines
    assert "waiter" in lines[0]
    assert "max_in_progress" in lines[0]
    assert "total=1/1" in lines[0]
    assert "unclaimed=1" in lines[0]


def test_recovery_is_logged_and_idle_boards_stay_silent(kanban_home, caplog):
    holder, waiter = _board_at_cap()
    deferred: dict[str, str] = {}

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        refused = _tick(waiter, "waiter", max_in_progress=1)
        _log_spawn_results([("waiter", refused)], deferred)

        # Free the slot: the same board now serves the card it was refused.
        with kb.write_txn(holder):
            holder.execute("UPDATE tasks SET status = 'done'")
        served = _tick(waiter, "waiter", max_in_progress=1)
        assert served.budget_blocked is None
        assert served.spawned
        _log_spawn_results([("waiter", served)], deferred)

        # A board with nothing queued and no refusal history says nothing.
        kb.create_board("quiet")
        quiet = kbc.connect(board="quiet")
        idle = _tick(quiet, "quiet", max_in_progress=1)
        assert idle.spawned == []
        _log_spawn_results([("quiet", idle)], deferred)

    lines = _lines(caplog)
    assert any("waiter" in line and "deferred" in line for line in lines), lines
    assert any("waiter" in line and "available again" in line for line in lines), lines
    assert not [line for line in lines if "quiet" in line], lines

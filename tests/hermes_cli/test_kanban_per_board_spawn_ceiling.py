"""Per-board spawn ceilings: ``kanban.max_spawn_by_board``.

``kanban.max_spawn`` is ONE ceiling applied to every board's tick, so a board
with a deep queue spends the whole host budget (``kanban.max_in_progress``) and
leaves nothing for the boards ticked after it — one busy board starves the rest.
``kanban.max_spawn_by_board`` maps a board slug to the most concurrent workers
that board may hold. The ceilings ALLOCATE the single host budget instead of
raising it, so a capped board can no longer take another board's slots, and an
unset/empty map is today's behaviour exactly (opt-in).

These tests exercise the REAL worktree modules; spawns are captured instead of
launching a process.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_dispatcher as kwd
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


def _settings(**overrides):
    """A ``_DispatcherSettings`` with the real fields, overridable by name."""
    values: dict = dict(
        interval=60.0, max_spawn=None, max_in_progress=None, failure_limit=3,
        stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
        max_in_progress_per_profile=None, max_spawn_by_board={},
    )
    values.update(overrides)
    return kwd._DispatcherSettings(**values)


def _seed(board, n):
    with kbc.connect(board=board) as conn:
        for i in range(n):
            kb.create_task(conn, title=f"{board}-{i}", assignee="alice")


def _running(board):
    with kbc.connect(board=board) as conn:
        return kbd.count_running_tasks(conn)


@pytest.fixture
def tick(monkeypatch):
    """Capture spawns for one real dispatcher tick (never launch a process)."""
    spawns: list = []

    def fake_spawn(task, workspace, board=None):
        spawns.append((board or kb.DEFAULT_BOARD, task.id))
        return 42

    real_dispatch_once = kbd.dispatch_once

    def dispatch_once(conn, *, board=None, **kwargs):
        return real_dispatch_once(conn, board=board, spawn_fn=fake_spawn, **kwargs)

    monkeypatch.setattr(kwd, "_kbd", lambda: SimpleNamespace(dispatch_once=dispatch_once))
    return spawns


# ---------------------------------------------------------------------------
# Config resolution — opt-in, additive, defensive
# ---------------------------------------------------------------------------


def test_absent_or_empty_mapping_is_unchanged_behaviour(kanban_home):
    """No per-board ceiling -> every board keeps the global ``max_spawn``."""
    kb.create_board("alpha")
    for raw in ({"max_spawn": 4}, {"max_spawn": 4, "max_spawn_by_board": {}}):
        settings = kwd._resolve_dispatcher_settings(raw, kb)
        assert settings.max_spawn_by_board == {}
        for slug in ("alpha", kb.DEFAULT_BOARD, "never-configured"):
            assert settings.max_spawn_for_board(slug) == 4


def test_named_board_uses_its_ceiling_others_keep_global(kanban_home):
    kb.create_board("alpha")
    kb.create_board("beta")
    settings = kwd._resolve_dispatcher_settings(
        {"max_spawn": 8, "max_spawn_by_board": {"alpha": 2, "beta": 5}}, kb
    )
    assert settings.max_spawn_for_board("alpha") == 2
    assert settings.max_spawn_for_board("beta") == 5
    assert settings.max_spawn_for_board(kb.DEFAULT_BOARD) == 8
    assert settings.max_spawn_for_board("never-configured") == 8


def test_non_mapping_is_ignored_with_a_warning(kanban_home, caplog):
    with caplog.at_level("WARNING"):
        settings = kwd._resolve_dispatcher_settings(
            {"max_spawn": 4, "max_spawn_by_board": ["alpha"]}, kb
        )
    assert settings.max_spawn_by_board == {}
    assert any("not a mapping" in r.getMessage() for r in caplog.records)


def test_invalid_ceiling_values_are_ignored_and_named(kanban_home, caplog):
    """0, -1 and a non-int are dropped with a warning naming board + value, and
    the global cap they might have disabled stays intact."""
    kb.create_board("alpha")
    with caplog.at_level("WARNING"):
        settings = kwd._resolve_dispatcher_settings(
            {"max_spawn": 4, "max_spawn_by_board": {"alpha": "x", "beta": 0, "gamma": -1}}, kb
        )
    assert settings.max_spawn_by_board == {}
    assert settings.max_spawn_for_board("alpha") == 4
    for slug, value in (("alpha", "x"), ("beta", 0), ("gamma", -1)):
        assert any(
            repr(slug) in r.getMessage() and repr(value) in r.getMessage()
            for r in caplog.records
        ), (slug, value)


def test_unknown_board_slug_is_inert(kanban_home, caplog):
    kb.create_board("alpha")
    with caplog.at_level("WARNING"):
        settings = kwd._resolve_dispatcher_settings(
            {"max_spawn": 4, "max_spawn_by_board": {"alpha": 2, "ghost": 9}}, kb
        )
    # The typo is dropped (and named); the real ceiling survives untouched.
    assert settings.max_spawn_by_board == {"alpha": 2}
    assert settings.max_spawn_for_board("ghost") == 4
    assert any("ghost" in r.getMessage() for r in caplog.records)


def test_unreadable_board_inventory_keeps_the_ceiling(kanban_home, monkeypatch):
    """A read failure must never drop a real ceiling — fail open, not closed."""
    monkeypatch.setattr(
        kb, "list_boards", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    settings = kwd._resolve_dispatcher_settings(
        {"max_spawn": 4, "max_spawn_by_board": {"alpha": 2}}, kb
    )
    assert settings.max_spawn_for_board("alpha") == 2


# ---------------------------------------------------------------------------
# The ceiling is resolved per board at the tick call site
# ---------------------------------------------------------------------------


def test_tick_passes_the_board_ceiling_as_max_spawn(kanban_home, monkeypatch):
    kb.create_board("alpha")
    captured: dict = {}

    def fake_dispatch_once(conn, *, board=None, **kwargs):
        captured[board] = kwargs["max_spawn"]
        return SimpleNamespace(spawned=[])

    monkeypatch.setattr(kwd, "_kbd", lambda: SimpleNamespace(dispatch_once=fake_dispatch_once))
    dispatcher = kwd._KanbanDispatcher(kb, _settings(max_spawn=8, max_spawn_by_board={"alpha": 2}))
    dispatcher.tick_once_for_board("alpha")
    dispatcher.tick_once_for_board(kb.DEFAULT_BOARD)
    assert captured["alpha"] == 2
    assert captured[kb.DEFAULT_BOARD] == 8


# ---------------------------------------------------------------------------
# Enforcement: allocate the host budget, never exceed it
# ---------------------------------------------------------------------------


def test_ceiling_limits_that_board_while_others_still_spawn(
    kanban_home, all_assignees_spawnable, tick,
):
    kb.create_board("capped")
    kb.create_board("open")
    _seed("capped", 5)
    _seed("open", 2)
    dispatcher = kwd._KanbanDispatcher(
        kb, _settings(max_in_progress=4, max_spawn_by_board={"capped": 1})
    )
    dispatcher.tick_once_for_board("capped")
    dispatcher.tick_once_for_board("open")
    assert _running("capped") == 1   # held at its ceiling despite a deep queue
    assert _running("open") >= 1     # the rest of the host budget is still available


def test_caps_summing_above_the_host_budget_never_exceed_it(
    kanban_home, all_assignees_spawnable, tick,
):
    kb.create_board("alpha")
    kb.create_board("beta")
    _seed("alpha", 4)
    _seed("beta", 4)
    host_cap = 3
    # 2 + 2 ceilings > host cap: the host cap wins, never 4 concurrent workers.
    dispatcher = kwd._KanbanDispatcher(
        kb, _settings(max_in_progress=host_cap, max_spawn_by_board={"alpha": 2, "beta": 2})
    )
    dispatcher.tick_once_for_board("alpha")
    dispatcher.tick_once_for_board("beta")
    assert _running("alpha") + _running("beta") <= host_cap


def test_busy_board_cannot_starve_a_quiet_board(
    kanban_home, all_assignees_spawnable, tick,
):
    """The bug: the early board's deep queue spent the whole host budget, so the
    board ticked after it spawned nothing. Capping the busy board fixes it."""
    kb.create_board("busy")
    kb.create_board("quiet")
    _seed("busy", 8)
    _seed("quiet", 1)
    dispatcher = kwd._KanbanDispatcher(
        kb, _settings(max_in_progress=4, max_spawn_by_board={"busy": 1})
    )
    dispatcher.tick_once_for_board("busy")   # early board, deep queue
    dispatcher.tick_once_for_board("quiet")  # later board, starved before the ceiling
    assert _running("busy") == 1
    assert _running("quiet") == 1


def test_board_without_a_ceiling_keeps_todays_behaviour(
    kanban_home, all_assignees_spawnable, tick,
):
    """An uncapped board may take the whole host budget — unchanged, and exactly
    why the busy board has to be capped to stop it starving the others."""
    kb.create_board("busy")
    _seed("busy", 8)
    dispatcher = kwd._KanbanDispatcher(
        kb, _settings(max_in_progress=4, max_spawn_by_board={})
    )
    dispatcher.tick_once_for_board("busy")
    assert _running("busy") == 4

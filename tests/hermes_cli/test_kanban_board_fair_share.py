"""Fair-share dispatch across boards (``kanban.board_fair_share``).

Covers the opt-in policy that allocates the host-wide concurrency budget
across boards instead of spending it in board-enumeration order:

* one worker per board while more than one board has spawnable work;
* the full host cap when a single board is contended;
* round-robin rotation so enumeration order cannot starve a board;
* the per-board gate in ``dispatch_once`` that enforces the allocation;
* default-off backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    # Unknown memory pressure must not cap the tick on a busy CI host.
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


# ---------------------------------------------------------------------------
# Per-board gate in dispatch_once
# ---------------------------------------------------------------------------


def test_per_board_cap_blocks_when_board_already_running(
    kanban_home, all_assignees_spawnable,
):
    """A board at its fair-share slot spawns nothing, even if the host has room."""
    spawns: list = []
    with kbc.connect() as conn:
        running_id = kb.create_task(conn, title="already-running", assignee="alice")
        assert kb.claim_task(conn, running_id) is not None
        kb.create_task(conn, title="waiting", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress_per_board=1,
        )

    assert spawns == []
    assert res.spawned == []


def test_per_board_cap_allows_one_spawn_when_idle(
    kanban_home, all_assignees_spawnable,
):
    """An idle board gets exactly one worker under a fair-share slot of 1."""
    spawns: list = []
    with kbc.connect() as conn:
        for title in ("a", "b"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress_per_board=1,
        )

    assert len(spawns) == 1
    assert len(res.spawned) == 1


def test_per_board_cap_absent_keeps_host_semantics(
    kanban_home, all_assignees_spawnable,
):
    """Without a per-board cap the host budget behaves exactly as before."""
    spawns: list = []
    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    assert len(spawns) == 2
    assert len(res.spawned) == 2


# ---------------------------------------------------------------------------
# Gateway sweep policy
# ---------------------------------------------------------------------------


def _settings(fair_share: bool, cap: int = 2):
    from gateway import kanban_watchers_dispatcher as kwd

    return kwd._DispatcherSettings(
        60.0, None, cap, 2, 0, True, None, None, board_fair_share=fair_share,
    )


def _dispatcher(monkeypatch, slugs, eligible, settings):
    from gateway import kanban_watchers_dispatcher as kwd

    kb_stub = SimpleNamespace(
        DEFAULT_BOARD="default",
        list_boards=lambda **kwargs: [{"slug": slug} for slug in slugs],
    )
    dispatcher = kwd._KanbanDispatcher(kb_stub, settings)
    monkeypatch.setattr(
        dispatcher, "_board_has_spawnable", lambda slug: slug in eligible,
    )
    calls: list[tuple[str, int | None]] = []

    def fake_tick(slug, *, max_in_progress_per_board=None):
        calls.append((slug, max_in_progress_per_board))
        return SimpleNamespace(spawned=[])

    monkeypatch.setattr(dispatcher, "tick_once_for_board", fake_tick)
    return dispatcher, calls


def test_fair_share_off_preserves_enumeration_order_and_no_cap(monkeypatch):
    slugs = ["default", "alpha", "beta"]
    dispatcher, calls = _dispatcher(
        monkeypatch, slugs, eligible={"default", "alpha"}, settings=_settings(False),
    )
    dispatcher.tick_once()

    assert calls == [(slug, None) for slug in slugs]


def test_fair_share_contended_gives_each_board_one_slot(monkeypatch):
    slugs = ["default", "alpha", "beta", "idle"]
    dispatcher, calls = _dispatcher(
        monkeypatch, slugs, eligible={"default", "alpha"}, settings=_settings(True),
    )
    dispatcher.tick_once()

    by_slug = dict(calls)
    assert by_slug["default"] == 1
    assert by_slug["alpha"] == 1
    assert by_slug["beta"] is None
    assert by_slug["idle"] is None


def test_fair_share_sole_contended_board_gets_full_host_cap(monkeypatch):
    slugs = ["default", "alpha", "beta"]
    dispatcher, calls = _dispatcher(
        monkeypatch, slugs, eligible={"beta"}, settings=_settings(True, cap=2),
    )
    dispatcher.tick_once()

    by_slug = dict(calls)
    assert by_slug["beta"] == 2
    assert by_slug["default"] is None
    assert by_slug["alpha"] is None


def test_fair_share_rotation_serves_every_contended_board(monkeypatch):
    """Three contended boards, two host slots: rotation must serve all three."""
    slugs = ["default", "alpha", "beta", "gamma"]
    dispatcher, calls = _dispatcher(
        monkeypatch, slugs,
        eligible={"default", "alpha", "beta"},
        settings=_settings(True, cap=2),
    )

    first_served: list[str] = []
    for _ in range(3):
        calls.clear()
        dispatcher.tick_once()
        first_served.append(calls[0][0])

    assert sorted(first_served) == ["alpha", "beta", "default"]
    assert len(set(first_served)) == 3


def test_fair_share_default_is_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kanban"]["board_fair_share"] is False

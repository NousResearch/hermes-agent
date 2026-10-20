"""A paused board is idle on purpose — the dispatcher must not treat it as stuck.

The pause gate lives in ``dispatch_once`` (covered by
``tests/hermes_cli/test_kanban_pause.py``); these tests pin the two per-board
probes in the gateway loop that run BEFORE/AROUND it:
``auto_decompose_tick`` (never spend an aux LLM call on a paused board) and
``ready_nonempty`` (a paused ready queue is not the "dispatcher stuck" signal).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_pause as pause


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME"):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _dispatcher() -> _KanbanDispatcher:
    return _KanbanDispatcher(
        kb,
        _DispatcherSettings(
            interval=60.0, max_spawn=None, max_in_progress=None, failure_limit=2,
            stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
            max_in_progress_per_profile=None,
        ),
    )


def test_auto_decompose_skips_a_paused_board(fresh_home, monkeypatch):
    from hermes_cli import kanban_decompose

    calls: list[str] = []
    monkeypatch.setattr(kanban_decompose, "list_triage_ids", lambda: calls.append("x") or [])

    pause.pause("default", reason="provider swap")
    dispatcher = _dispatcher()
    monkeypatch.setattr(dispatcher, "_board_slugs", lambda: ["default"])

    assert dispatcher.auto_decompose_tick(3) == 0
    assert calls == [], "a paused board must not reach the auxiliary model"

    pause.resume("default")
    dispatcher.auto_decompose_tick(3)
    assert calls == ["x"], "resuming restores auto-decompose"


def test_ready_nonempty_ignores_a_paused_board(fresh_home, monkeypatch):
    def boom(*_a, **_kw):
        raise AssertionError("a paused board must not even open its DB to probe readiness")

    monkeypatch.setattr(kbc, "connect", boom)
    pause.pause("default")

    dispatcher = _dispatcher()
    monkeypatch.setattr(dispatcher, "_board_slugs", lambda: ["default"])
    assert dispatcher.ready_nonempty() is False

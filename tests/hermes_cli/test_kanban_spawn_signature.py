"""_call_spawn_fn must decide the call shape without running the callback (#144200).

A callback whose body raises TypeError/ValueError after a side effect must
not be invoked a second time through the signature-compatibility fallback.
Signature.bind validates the keyword call without executing the callable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Unit: body errors must not trigger a second invocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error_type", [ValueError, TypeError, RuntimeError])
def test_body_error_invokes_callback_once(error_type):
    """A callback that records an effect then raises is called exactly once."""
    effects: list[str] = []

    def spawn(task, workspace, board=None):
        effects.append("effect")
        raise error_type("after-effect")

    with pytest.raises(error_type, match="after-effect"):
        kbd._call_spawn_fn(spawn, object(), "/neutral-workspace", "default")
    assert len(effects) == 1


# ---------------------------------------------------------------------------
# Unit: accepted signature shapes
# ---------------------------------------------------------------------------


def test_two_arg_stub_receives_no_board():
    calls: list[tuple] = []

    def spawn(task, workspace):
        calls.append((task, workspace))
        return 11

    assert kbd._call_spawn_fn(spawn, "t", "/ws", "default") == 11
    assert calls == [("t", "/ws")]


def test_optional_board_receives_board_kwarg():
    calls: list[tuple] = []

    def spawn(task, workspace, board=None):
        calls.append((task, workspace, board))
        return 22

    assert kbd._call_spawn_fn(spawn, "t", "/ws", "board-a") == 22
    assert calls == [("t", "/ws", "board-a")]


def test_keyword_only_board_receives_board():
    calls: list[tuple] = []

    def spawn(task, workspace, *, board=None):
        calls.append((task, workspace, board))
        return 33

    assert kbd._call_spawn_fn(spawn, "t", "/ws", "board-b") == 33
    assert calls == [("t", "/ws", "board-b")]


def test_kwargs_callable_receives_board():
    """``**kwargs`` accepts board by keyword; the old membership test dropped it."""
    calls: list[dict] = []

    def spawn(task, workspace, **kwargs):
        calls.append({"task": task, "workspace": workspace, **kwargs})
        return 44

    assert kbd._call_spawn_fn(spawn, "t", "/ws", "board-c") == 44
    assert calls == [{"task": "t", "workspace": "/ws", "board": "board-c"}]


def test_positional_only_board_falls_back_to_two_args():
    """A positional-only ``board`` cannot take ``board=...``; use the 2-arg form."""
    calls: list[tuple] = []

    def spawn(task, workspace, board=None, /):
        calls.append((task, workspace, board))
        return 55

    assert kbd._call_spawn_fn(spawn, "t", "/ws", "board-d") == 55
    # Keyword ``board=`` is refused; the two-arg fallback leaves the default.
    assert calls == [("t", "/ws", None)]


def test_uninspectable_callable_falls_back_to_two_args(monkeypatch):
    """Callables whose signatures cannot be inspected keep the 2-arg path."""
    import inspect

    calls: list[tuple] = []

    def spawn(task, workspace):
        calls.append((task, workspace))
        return 66

    def _boom(_obj, *a, **k):
        raise ValueError("no signature")

    monkeypatch.setattr(inspect, "signature", _boom)
    assert kbd._call_spawn_fn(spawn, "t", "/ws", "board-e") == 66
    assert calls == [("t", "/ws")]


# ---------------------------------------------------------------------------
# Integration: one dispatch books one spawn failure
# ---------------------------------------------------------------------------


def test_dispatch_once_books_body_error_once(
    kanban_home, all_assignees_spawnable,
):
    """Driving real dispatch_once: a raising spawn_fn is invoked once and
    consecutive_failures advances by one (claim released)."""
    spawns: list[str] = []

    def boom(task, workspace, board=None):
        spawns.append(task.id)
        raise TypeError("after-effect")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="repro", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=boom)
        task = kb.get_task(conn, tid)

    assert spawns == [tid]
    assert res.spawned == []
    assert task is not None
    assert task.consecutive_failures == 1
    assert task.status == "ready"  # claim released after spawn_failed

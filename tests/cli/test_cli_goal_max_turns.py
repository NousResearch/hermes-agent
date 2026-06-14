"""Regression tests for classic CLI /goal max-turn parsing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _make_cli(session_id: str = "cli-goal-max-turns"):
    from cli import HermesCLI

    shell = HermesCLI.__new__(HermesCLI)
    shell.session_id = session_id
    return shell


def test_get_goal_manager_preserves_zero_max_turns(hermes_home, monkeypatch):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config", lambda: {"goals": {"max_turns": 0}})

    mgr = _make_cli()._get_goal_manager()

    assert mgr is not None
    assert mgr.default_max_turns == 0


def test_get_goal_manager_preserves_positive_max_turns(hermes_home, monkeypatch):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config", lambda: {"goals": {"max_turns": 7}})

    mgr = _make_cli()._get_goal_manager()

    assert mgr is not None
    assert mgr.default_max_turns == 7


def test_kanban_goal_loop_q_preserves_zero_task_max_turns(monkeypatch):
    import cli as cli_mod
    from hermes_cli import goals
    from hermes_cli import kanban_db as kb

    captured = {}
    task = SimpleNamespace(title="open task", body="", goal_max_turns=0)

    class _Conn:
        def close(self):
            pass

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-zero")
    monkeypatch.setattr(kb, "connect", lambda: _Conn())
    monkeypatch.setattr(kb, "get_task", lambda conn, task_id: task)
    monkeypatch.setattr(goals, "run_kanban_goal_loop", lambda **kwargs: captured.update(kwargs))

    shell = SimpleNamespace(agent=SimpleNamespace(), conversation_history=[], session_id="sid")

    cli_mod._run_kanban_goal_loop_q(shell, "first response")

    assert captured["max_turns"] == 0


def test_kanban_goal_loop_q_uses_default_for_missing_task_max_turns(monkeypatch):
    import cli as cli_mod
    from hermes_cli import goals
    from hermes_cli import kanban_db as kb

    captured = {}
    task = SimpleNamespace(title="default task", body="", goal_max_turns=None)

    class _Conn:
        def close(self):
            pass

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-default")
    monkeypatch.setattr(kb, "connect", lambda: _Conn())
    monkeypatch.setattr(kb, "get_task", lambda conn, task_id: task)
    monkeypatch.setattr(goals, "run_kanban_goal_loop", lambda **kwargs: captured.update(kwargs))

    shell = SimpleNamespace(agent=SimpleNamespace(), conversation_history=[], session_id="sid")

    cli_mod._run_kanban_goal_loop_q(shell, "first response")

    assert captured["max_turns"] == goals.DEFAULT_MAX_TURNS

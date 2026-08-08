"""Tests for _run_kanban_goal_loop_q, the kanban goal-loop leaf moved from
cli.py into hermes_cli/goals.py (cli.py god-file slice R5, C3).

Covers the env-gated leaf behavior (HERMES_KANBAN_TASK guard, task/body
resolution, run_turn session sync, run_kanban_goal_loop wiring, block path)
and the main() gate's lazy-import seam back into cli.py.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import goals

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeConn:
    def close(self):
        pass


def _make_cli(session_id="s1", agent_session_id="s1", run_result=None):
    def _run_conversation(**kwargs):
        return run_result if run_result is not None else {"final_response": "FINAL"}

    agent = SimpleNamespace(
        session_id=agent_session_id, run_conversation=_run_conversation
    )
    return SimpleNamespace(agent=agent, session_id=session_id, conversation_history=[])


def _task(title="Goal", body="Body", max_turns=5):
    return SimpleNamespace(title=title, body=body, goal_max_turns=max_turns, status="ready")


@pytest.fixture(autouse=True)
def _clean_task_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)


class TestLeafBehavior:
    def test_no_task_env_returns_without_touching_db(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("kanban_db must not be touched")

        monkeypatch.setattr("hermes_cli.kanban_db.connect", _boom)
        assert goals._run_kanban_goal_loop_q(_make_cli(), "resp") is None

    def test_missing_task_returns_without_loop(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
        monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: _FakeConn())
        monkeypatch.setattr("hermes_cli.kanban_db.get_task", lambda conn, tid: None)
        calls = []
        monkeypatch.setattr(goals, "run_kanban_goal_loop", lambda **kw: calls.append(kw))
        assert goals._run_kanban_goal_loop_q(_make_cli(), "resp") is None
        assert calls == []

    def test_empty_goal_text_returns_without_loop(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
        monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: _FakeConn())
        monkeypatch.setattr(
            "hermes_cli.kanban_db.get_task", lambda conn, tid: _task(title="", body=None)
        )
        calls = []
        monkeypatch.setattr(goals, "run_kanban_goal_loop", lambda **kw: calls.append(kw))
        assert goals._run_kanban_goal_loop_q(_make_cli(), "resp") is None
        assert calls == []

    def test_loop_wires_callbacks_and_syncs_session(self, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
        monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: _FakeConn())
        monkeypatch.setattr(
            "hermes_cli.kanban_db.get_task", lambda conn, tid: _task(max_turns=7)
        )
        cli = _make_cli(session_id="s1", agent_session_id="s2")
        seen = {}

        def fake_loop(**kwargs):
            seen.update(kwargs)
            seen["run_turn_resp"] = kwargs["run_turn"]("prompt-1")
            seen["status"] = kwargs["task_status_fn"]()
            kwargs["log"]("hello")
            return {"outcome": "done"}

        monkeypatch.setattr(goals, "run_kanban_goal_loop", fake_loop)
        monkeypatch.setattr(
            "hermes_cli.kanban_db.block_task", lambda conn, task_id, reason: None
        )

        # The leaf ends in a bare call; the caller error-swallows and ignores
        # the return value, so the function itself returns None.
        assert goals._run_kanban_goal_loop_q(cli, "first-resp") is None
        assert seen["task_id"] == "t1"
        assert seen["goal_text"] == "Goal\n\nBody"
        assert seen["max_turns"] == 7
        assert seen["first_response"] == "first-resp"
        assert seen["run_turn_resp"] == "FINAL"
        assert seen["status"] == "ready"
        # mid-run compression rotation syncs cli.session_id from the agent.
        assert cli.session_id == "s2"
        out = capsys.readouterr().out
        assert "FINAL" in out

    def test_max_turns_falls_back_to_module_default(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
        monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: _FakeConn())
        monkeypatch.setattr(
            "hermes_cli.kanban_db.get_task", lambda conn, tid: _task(max_turns=None)
        )
        seen = {}
        monkeypatch.setattr(goals, "run_kanban_goal_loop", lambda **kw: seen.update(kw))
        goals._run_kanban_goal_loop_q(_make_cli(), "x")
        assert seen["max_turns"] == goals.DEFAULT_MAX_TURNS

    def test_block_fn_calls_kanban_db_block_task(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
        monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: _FakeConn())
        monkeypatch.setattr("hermes_cli.kanban_db.get_task", lambda conn, tid: _task())
        blocked = []
        monkeypatch.setattr(
            "hermes_cli.kanban_db.block_task",
            lambda conn, task_id, reason: blocked.append((task_id, reason)),
        )

        def fake_loop(**kwargs):
            kwargs["block_fn"]("stuck forever")
            return None

        monkeypatch.setattr(goals, "run_kanban_goal_loop", fake_loop)
        goals._run_kanban_goal_loop_q(_make_cli(), "x")
        assert blocked == [("t1", "stuck forever")]


class TestEnvGatedInvocationSeam:
    def test_main_gate_lazy_imports_moved_leaf(self):
        import cli as cli_mod

        src = inspect.getsource(cli_mod.main)
        assert "HERMES_KANBAN_GOAL_MODE" in src
        assert "from hermes_cli.goals import _run_kanban_goal_loop_q" in src
        assert "def _run_kanban_goal_loop_q" not in inspect.getsource(cli_mod)
        assert not hasattr(cli_mod, "_run_kanban_goal_loop_q")
        assert callable(goals._run_kanban_goal_loop_q)

    def test_cli_imports_under_goal_mode_env(self):
        code = (
            "import os\n"
            "os.environ['HERMES_KANBAN_GOAL_MODE'] = '1'\n"
            "import cli\n"
            "import hermes_cli.goals as g\n"
            "assert callable(g._run_kanban_goal_loop_q)\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

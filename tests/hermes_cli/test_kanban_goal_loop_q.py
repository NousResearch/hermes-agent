"""Regression tests for _run_kanban_goal_loop_q (extracted verbatim from
cli.py -> hermes_cli/kanban_goal_loop.py).

Wave 1 godfile extraction, shard s5 cluster c25. The function is a
module-level driver for dispatcher-spawned kanban goal_mode workers; the
tests below drive it with a fake HermesCLI-like object, fake kanban_db
entries, and a captured goals.run_kanban_goal_loop, so no real agent or DB
is touched.
"""

from __future__ import annotations

import os

import pytest

from hermes_cli import kanban_db
from hermes_cli import goals
from hermes_cli.kanban_goal_loop import _run_kanban_goal_loop_q


class _FakeConn:
    """Stand-in for a sqlite3 connection; close() must not raise."""

    def close(self):
        pass


class _FakeTask:
    def __init__(self, title="Card title", body="Card body", status="in_progress",
                 goal_max_turns=None):
        self.title = title
        self.body = body
        self.status = status
        self.goal_max_turns = goal_max_turns


class _FakeCLI:
    """Minimal HermesCLI-shaped object: agent + conversation_history + session_id."""

    def __init__(self, response="hello from agent"):
        self.agent = _FakeAgent(response)
        self.conversation_history = []
        self.session_id = "sess-1"

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, value):
        self._agent = value


class _FakeAgent:
    def __init__(self, response):
        self.response = response
        self.session_id = "sess-1"
        self.calls = []

    def run_conversation(self, *, user_message, conversation_history):
        self.calls.append((user_message, conversation_history))
        return {"final_response": self.response}


@pytest.fixture(autouse=True)
def _clean_goal_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)


class TestNoOpPaths:
    def test_no_task_env_returns_immediately_without_touching_db(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        def _boom(*args, **kwargs):
            raise AssertionError("kanban_db must not be touched without a task id")

        monkeypatch.setattr(kanban_db, "connect", _boom)
        _run_kanban_goal_loop_q(_FakeCLI(), "first response")

    def test_empty_task_env_returns_immediately(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "   ")

        def _boom(*args, **kwargs):
            raise AssertionError("kanban_db must not be touched for a blank task id")

        monkeypatch.setattr(kanban_db, "connect", _boom)
        _run_kanban_goal_loop_q(_FakeCLI(), "first response")

    def test_missing_task_returns_without_running_loop(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "42")
        monkeypatch.setattr(kanban_db, "connect", lambda: _FakeConn())
        monkeypatch.setattr(kanban_db, "get_task", lambda conn, tid: None)

        captured = {}
        monkeypatch.setattr(
            goals,
            "run_kanban_goal_loop",
            lambda **kwargs: captured.update(kwargs),
        )
        _run_kanban_goal_loop_q(_FakeCLI(), "first response")
        assert captured == {}

    def test_task_without_goal_text_returns_without_running_loop(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "42")
        monkeypatch.setattr(kanban_db, "connect", lambda: _FakeConn())
        monkeypatch.setattr(
            kanban_db, "get_task", lambda conn, tid: _FakeTask(title="", body="   ")
        )

        captured = {}
        monkeypatch.setattr(
            goals,
            "run_kanban_goal_loop",
            lambda **kwargs: captured.update(kwargs),
        )
        _run_kanban_goal_loop_q(_FakeCLI(), "first response")
        assert captured == {}


class TestFullLoop:
    def test_drives_run_kanban_goal_loop_with_wired_callbacks(self, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "7")
        task = _FakeTask(title="Build widget", body="Must pass tests",
                         status="in_progress", goal_max_turns=5)
        monkeypatch.setattr(kanban_db, "connect", lambda: _FakeConn())
        monkeypatch.setattr(kanban_db, "get_task", lambda conn, tid: task)
        monkeypatch.setattr(kanban_db, "block_task",
                            lambda conn, tid, reason=None: None)

        cli = _FakeCLI(response="turn output")
        captured = {}

        def _fake_run_loop(**kwargs):
            captured.update(kwargs)
            # Exercise the wired callbacks inside the loop.
            assert kwargs["run_turn"]("continue") == "turn output"
            assert kwargs["task_status_fn"]() == "in_progress"
            kwargs["block_fn"]("stuck")

        monkeypatch.setattr(goals, "run_kanban_goal_loop", _fake_run_loop)

        _run_kanban_goal_loop_q(cli, "first response")

        assert captured["task_id"] == "7"
        assert captured["goal_text"] == "Build widget\n\nMust pass tests"
        assert captured["max_turns"] == 5
        assert captured["first_response"] == "first response"
        assert captured["log"]("msg") is None  # logger.info callable
        # The worker turn went through cli.agent.run_conversation with the
        # CLI's conversation history and the printed response hit stdout.
        assert cli.agent.calls == [("continue", cli.conversation_history)]
        assert "turn output" in capsys.readouterr().out

    def test_goal_max_turns_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "7")
        task = _FakeTask(title="T", body="B", status="in_progress",
                         goal_max_turns=None)
        monkeypatch.setattr(kanban_db, "connect", lambda: _FakeConn())
        monkeypatch.setattr(kanban_db, "get_task", lambda conn, tid: task)

        captured = {}
        monkeypatch.setattr(
            goals,
            "run_kanban_goal_loop",
            lambda **kwargs: captured.update(kwargs),
        )
        _run_kanban_goal_loop_q(_FakeCLI(), "")
        assert captured["max_turns"] == goals.DEFAULT_MAX_TURNS

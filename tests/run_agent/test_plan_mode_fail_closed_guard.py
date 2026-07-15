"""Regression: the plan-mode dispatch guard in the tool executor must fail
CLOSED.

Both executor call sites (``execute_tool_calls_concurrent`` and
``execute_tool_calls_sequential``) previously wrapped the guard call in a
try/except that, on ANY exception, set the block to ``None`` — i.e. it silently
ALLOWED the tool. That contradicts the stated fail-closed design: if plan-mode
enforcement state cannot even be evaluated, a mutating tool must be blocked,
not executed. Read-only tools stay allowed (they are safe regardless of plan
state). The justified DB-down carve-out lives INSIDE ``plan_mode`` (a plain
absence / unavailable DB is "not in plan mode"); it is distinct from the guard
call itself raising, which is what this guards.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names):
    return [
        {"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
        for n in names
    ]


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import plan_mode

    plan_mode._DB_CACHE.clear()
    yield home
    plan_mode._DB_CACHE.clear()


@pytest.fixture
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("terminal", "read_file"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a.session_id = "fail-closed-sid"
        return a


def _tool_call(name, arguments="{}"):
    return SimpleNamespace(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _assistant_msg(tool_calls):
    return SimpleNamespace(content="", tool_calls=tool_calls)


# ── Unit: the fail-closed wrapper ────────────────────────────────────────


class TestPlanModeBlockReasonHelper:
    def test_guard_raises_blocks_mutating_tool(self, monkeypatch):
        from agent import tool_executor
        from hermes_cli import plan_mode

        def _boom(*_a, **_k):
            raise RuntimeError("state store exploded")

        monkeypatch.setattr(plan_mode, "tool_block_reason", _boom)
        ag = SimpleNamespace(session_id="s1")
        reason = tool_executor._plan_mode_block_reason(ag, "terminal", {"command": "ls"})
        assert reason is not None
        assert "failing closed" in reason.lower()

    def test_guard_raises_still_allows_read_only_tool(self, monkeypatch):
        from agent import tool_executor
        from hermes_cli import plan_mode

        def _boom(*_a, **_k):
            raise RuntimeError("state store exploded")

        monkeypatch.setattr(plan_mode, "tool_block_reason", _boom)
        ag = SimpleNamespace(session_id="s1")
        # Read-only tools are safe regardless of plan state — never blocked.
        assert tool_executor._plan_mode_block_reason(ag, "read_file", {"path": "a"}) is None

    def test_normal_block_reason_passes_through(self, monkeypatch):
        from agent import tool_executor
        from hermes_cli import plan_mode

        monkeypatch.setattr(
            plan_mode, "tool_block_reason", lambda *_a, **_k: "Blocked: plan mode"
        )
        ag = SimpleNamespace(session_id="s1")
        assert (
            tool_executor._plan_mode_block_reason(ag, "terminal", {}) == "Blocked: plan mode"
        )

    def test_normal_allow_passes_through(self, monkeypatch):
        from agent import tool_executor
        from hermes_cli import plan_mode

        monkeypatch.setattr(plan_mode, "tool_block_reason", lambda *_a, **_k: None)
        ag = SimpleNamespace(session_id="s1")
        assert tool_executor._plan_mode_block_reason(ag, "terminal", {}) is None


# ── Integration: both executor paths block, do not execute ───────────────


def _raise_guard(monkeypatch):
    from hermes_cli import plan_mode

    def _boom(*_a, **_k):
        raise RuntimeError("enforcement state read blew up")

    monkeypatch.setattr(plan_mode, "tool_block_reason", _boom)


def test_sequential_path_blocks_when_guard_raises(agent, hermes_home, monkeypatch):
    from agent.tool_executor import execute_tool_calls_sequential

    _raise_guard(monkeypatch)
    # Sentinel proving the tool never ran: if execution WERE reached the result
    # would carry this marker (and never a real `terminal` command).
    monkeypatch.setattr(agent, "_invoke_tool", lambda *a, **k: "EXECUTED-SENTINEL")

    messages: list = []
    tc = _tool_call("terminal", json.dumps({"command": "echo should-not-run"}))
    execute_tool_calls_sequential(agent, _assistant_msg([tc]), messages, "task-seq")

    assert messages, "expected a tool-result message"
    content = messages[-1]["content"]
    assert "EXECUTED-SENTINEL" not in content, "mutating tool must NOT have executed"
    assert "failing closed" in content.lower()


def test_concurrent_path_blocks_when_guard_raises(agent, hermes_home, monkeypatch):
    from agent.tool_executor import execute_tool_calls_concurrent

    _raise_guard(monkeypatch)
    monkeypatch.setattr(agent, "_invoke_tool", lambda *a, **k: "EXECUTED-SENTINEL")

    messages: list = []
    tc = _tool_call("terminal", json.dumps({"command": "echo should-not-run"}))
    execute_tool_calls_concurrent(agent, _assistant_msg([tc]), messages, "task-conc")

    assert messages, "expected a tool-result message"
    content = messages[-1]["content"]
    assert "EXECUTED-SENTINEL" not in content, "mutating tool must NOT have executed"
    assert "failing closed" in content.lower()

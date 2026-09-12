"""Integration tests for /steer breakout in the tool-execution loop (#28172).

When a user sends /steer mid-batch, the remaining tools must be deferred so the
model can act on the guidance before running more tools. These drive the real
``execute_tool_calls_*`` functions on a real ``AIAgent`` (only
``model_tools.handle_function_call`` is faked), so the breakout wiring and the
real ``apply_pending_steer_to_tool_results`` delivery are genuinely exercised.

- Sequential path: the ``bool`` return breaks the loop and defers the rest
  (``test_steer_defers_remaining_tools``).
- Concurrent path: a pending steer cancels not-yet-started futures, which are
  then rendered via ``steer_deferred_indices``. Because cancellation only
  succeeds for unstarted futures, the deferral test saturates the worker pool
  (``_MAX_TOOL_WORKERS`` patched to 1) so two tools stay queued when the steer
  poll fires (``test_concurrent_deferral_renders_for_unstarted_tools``).

The steer itself is delivered as its own ``role:"user"`` row appended AFTER every
tool result (``steer_user_row``) — never smeared onto an already-persisted tool
row — so the assertions here check that row, not the tool contents.
"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.tool_executor as te
from agent.prompt_builder import STEER_DISPLAY_KIND, STEER_MARKER_OPEN
from agent.tool_executor import (
    execute_tool_calls_concurrent,
    execute_tool_calls_sequential,
)
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch(
            "model_tools.get_tool_definitions",
            return_value=_make_tool_defs("tool_0", "tool_1", "tool_2"),
        ),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def _tc(name, arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{name}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _batch(n=3):
    """``n`` tool calls named ``tool_i`` with ids ``c0..c{n-1}``."""
    calls = [_tc(f"tool_{i}", call_id=f"c{i}") for i in range(n)]
    return SimpleNamespace(content="", tool_calls=calls), calls


def _tool_messages(messages):
    return [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]


def _steer_rows(messages):
    """The standalone user rows a mid-turn /steer is delivered as."""
    return [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
        and m.get("display_kind") == STEER_DISPLAY_KIND
    ]


def _ok(name, args, task_id, **kwargs):
    return json.dumps({"status": "ok", "tool": name})


# ---------------------------------------------------------------------------
# Sequential path
# ---------------------------------------------------------------------------


class TestSequentialSteerBreakout:
    def test_steer_defers_remaining_tools(self, agent):
        """A steer pending when the first per-tool drain runs breaks the loop:
        tool 0 executes, tools 1 & 2 are deferred (not executed)."""
        agent.steer("switch to Python")
        msg, _ = _batch()
        messages = []
        executed = []

        def fake_handle(name, args, task_id, **kwargs):
            executed.append(kwargs["tool_call_id"])
            return _ok(name, args, task_id, **kwargs)

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            steered = execute_tool_calls_sequential(agent, msg, messages, "task")

        assert steered is True
        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3, "every tool_call must get a paired tool result"
        assert [m["tool_call_id"] for m in tool_msgs] == ["c0", "c1", "c2"]

        # Tool 0 ran; tools 1 & 2 were deferred, not executed.
        assert "ok" in tool_msgs[0]["content"]
        assert "deferred" in tool_msgs[1]["content"].lower()
        assert "deferred" in tool_msgs[2]["content"].lower()
        assert executed == ["c0"]

        # The steer lands exactly once, as its own user row after every result.
        steer_rows = _steer_rows(messages)
        assert len(steer_rows) == 1
        assert STEER_MARKER_OPEN in steer_rows[0]["content"]
        assert "switch to Python" in steer_rows[0]["content"]
        assert messages[-1] is steer_rows[0]
        assert agent._pending_steer is None

    def test_interrupt_supersedes_steer_break(self, agent):
        """If an interrupt lands together with a steer, the interrupt owns the
        breakout: the steer is still delivered after the batch, but the
        remaining tools are interrupt-skipped, not steer-deferred."""
        agent.steer("switch to Python")
        msg, _ = _batch()
        messages = []

        def fake_handle(name, args, task_id, **kwargs):
            # Interrupt arrives while tool 0 is running.
            agent._interrupt_requested = True
            return _ok(name, args, task_id, **kwargs)

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            execute_tool_calls_sequential(agent, msg, messages, "task")

        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3
        assert "ok" in tool_msgs[0]["content"]
        # Remaining tools are interrupt-skipped, NOT steer-deferred.
        for m in tool_msgs[1:]:
            assert "deferred" not in m["content"].lower()
            assert "skipped" in m["content"].lower() or "cancelled" in m["content"].lower()
        # The steer is still delivered (it outlives the interrupt).
        steer_rows = _steer_rows(messages)
        assert len(steer_rows) == 1
        assert "switch to Python" in steer_rows[0]["content"]

    def test_no_steer_runs_all_tools(self, agent):
        msg, _ = _batch()
        messages = []
        executed = []

        def fake_handle(name, args, task_id, **kwargs):
            executed.append(kwargs["tool_call_id"])
            return _ok(name, args, task_id, **kwargs)

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            steered = execute_tool_calls_sequential(agent, msg, messages, "task")

        assert steered is False
        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3
        assert all("ok" in m["content"] for m in tool_msgs)
        assert all("deferred" not in m["content"].lower() for m in tool_msgs)
        assert executed == ["c0", "c1", "c2"]
        assert _steer_rows(messages) == []

    def test_steer_defers_remaining_after_invalid_args(self, agent):
        """When a tool call has invalid JSON arguments, it records the error
        result and, if a steer is pending, breaks and defers the remaining tools."""
        agent.steer("switch course")
        calls = [_tc("tool_0", "not-valid-json", call_id="bad"), _tc("tool_1", "{}", call_id="trailing")]
        msg = SimpleNamespace(content="", tool_calls=calls)
        messages = []
        executed = []

        def fake_handle(name, args, task_id, **kwargs):
            executed.append(kwargs["tool_call_id"])
            return _ok(name, args, task_id, **kwargs)

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            steered = execute_tool_calls_sequential(agent, msg, messages, "task")

        assert steered is True
        assert executed == []
        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "bad"
        assert "invalid" in tool_msgs[0]["content"].lower()
        assert tool_msgs[1]["tool_call_id"] == "trailing"
        assert "deferred" in tool_msgs[1]["content"].lower()
        steer_rows = _steer_rows(messages)
        assert len(steer_rows) == 1
        assert "switch course" in steer_rows[0]["content"]



# ---------------------------------------------------------------------------
# Concurrent path
# ---------------------------------------------------------------------------


class TestConcurrentSteerHandling:
    def test_pending_steer_is_consumed_without_breaking_alternation(self, agent):
        """A steer pending during concurrent execution must be delivered and
        must not corrupt the tool_call_id <-> tool_result pairing. Cancellation
        of unstarted futures is timing-dependent (best-effort for large
        batches), so we assert only the invariants that always hold: every
        tool_call gets exactly one paired result, and the steer is delivered."""
        agent.steer("prefer the API over scraping")
        msg, _ = _batch()
        messages = []

        with patch("model_tools.handle_function_call", side_effect=_ok):
            execute_tool_calls_concurrent(agent, msg, messages, "task")

        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3
        # Role alternation / pairing intact: one result per emitted tool_call_id.
        assert [m["tool_call_id"] for m in tool_msgs] == ["c0", "c1", "c2"]
        # Steer was delivered (drained) — not silently dropped.
        assert agent._pending_steer is None
        steer_rows = _steer_rows(messages)
        assert len(steer_rows) == 1
        assert STEER_MARKER_OPEN in steer_rows[0]["content"]
        assert "prefer the API over scraping" in steer_rows[0]["content"]
        assert messages[-1] is steer_rows[0]

    def test_no_steer_all_tools_complete(self, agent):
        msg, _ = _batch()
        messages = []

        with patch("model_tools.handle_function_call", side_effect=_ok):
            execute_tool_calls_concurrent(agent, msg, messages, "task")

        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3
        assert [m["tool_call_id"] for m in tool_msgs] == ["c0", "c1", "c2"]
        assert all("ok" in str(m["content"]) for m in tool_msgs)
        assert _steer_rows(messages) == []

    def test_concurrent_deferral_renders_for_unstarted_tools(self, agent, monkeypatch):
        """Saturate the worker pool so two tools stay queued: when a steer is
        pending, the wait-loop cancels those unstarted futures and the
        post-processing renders them via the steer_deferred_indices branch.
        This is the concurrent analogue of the sequential break, and it
        exercises the deferred-result construction the invariant test can't."""
        monkeypatch.setattr(te, "_MAX_TOOL_WORKERS", 1)
        # Remove the batch deadline so it can never race the 5s steer poll.
        monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "0")

        agent.steer("stop and summarize")
        msg, _ = _batch()
        messages = []
        executed = []
        lock = threading.Lock()

        # The single worker runs tool_0 long enough for the 5s steer poll to
        # fire while tool_1/tool_2 are still queued (and thus cancellable).
        def slow_first(name, args, task_id, **kwargs):
            with lock:
                executed.append(kwargs["tool_call_id"])
            if name == "tool_0":
                time.sleep(5.5)
            return _ok(name, args, task_id, **kwargs)

        with patch("model_tools.handle_function_call", side_effect=slow_first):
            execute_tool_calls_concurrent(agent, msg, messages, "task")

        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3
        assert [m["tool_call_id"] for m in tool_msgs] == ["c0", "c1", "c2"]
        by_id = {m["tool_call_id"]: m for m in tool_msgs}
        # The tool that occupied the worker actually ran.
        assert "ok" in str(by_id["c0"]["content"])
        # The two queued tools were deferred, not executed or errored.
        deferred = [m for m in tool_msgs if "deferred" in str(m["content"]).lower()]
        assert len(deferred) == 2, [m["content"] for m in tool_msgs]
        assert executed == ["c0"]
        assert agent._pending_steer is None
        assert len(_steer_rows(messages)) == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

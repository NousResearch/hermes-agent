"""Integration tests for /steer breakout in the tool-execution loop (#28172).

When a user sends /steer mid-batch, the remaining tools must be deferred so the
model can act on the guidance before running more tools. These drive the real
``execute_tool_calls_*`` functions with the real
``apply_pending_steer_to_tool_results`` helper.

- Sequential path: the ``bool`` return breaks the loop and defers the rest
  (``test_steer_defers_remaining_tools``).
- Concurrent path: a pending steer cancels not-yet-started futures, which are
  then rendered via ``steer_deferred_indices``. Because cancellation only
  succeeds for unstarted futures, the deferral test saturates the worker pool
  (``_MAX_TOOL_WORKERS`` patched to 1) so two tools stay queued when the steer
  poll fires (``test_concurrent_deferral_renders_for_unstarted_tools``).
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import agent.tool_executor as te

import pytest

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from agent.prompt_builder import STEER_MARKER_OPEN
from agent.tool_executor import (
    execute_tool_calls_concurrent,
    execute_tool_calls_sequential,
)


class _StubAgent:
    """Minimal AIAgent stand-in that drives the tool_executor functions.

    Uses the REAL steer helper/drain so the breakout wiring (return bool →
    break loop → defer remaining) is genuinely exercised.
    """

    def __init__(self):
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()
        self._interrupt_requested = False
        self._interrupt_message = None
        self._current_tool = None
        self.quiet_mode = True
        self.verbose_logging = False
        self.tool_progress_mode = "off"
        self.log_prefix = ""
        self.log_prefix_chars = 200
        self.tool_delay = 0
        self.valid_tool_names = None
        self.session_id = "test-session"
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self._checkpoint_mgr = MagicMock(enabled=False)
        self._subdirectory_hints = MagicMock()
        self._subdirectory_hints.check_tool_call.return_value = None
        self._tool_guardrails = MagicMock()
        self._tool_guardrails.before_call.return_value = MagicMock(allows_execution=True)
        self._memory_manager = None
        self._context_engine_tool_names = None
        self._print_fn = print
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self._last_activity = 0
        self._turns_since_memory = 0
        self._iters_since_skill = 0

    # ── real steer plumbing ──────────────────────────────────────────
    def _drain_pending_steer(self):
        with self._pending_steer_lock:
            text = self._pending_steer
            self._pending_steer = None
            return text

    def _apply_pending_steer_to_tool_results(self, messages, num_tool_msgs):
        return apply_pending_steer_to_tool_results(self, messages, num_tool_msgs)

    # ── no-op display / bookkeeping surface ──────────────────────────
    def _touch_activity(self, desc):
        pass

    def _vprint(self, msg, force=False):
        pass

    def _safe_print(self, msg):
        pass

    def _should_emit_quiet_tool_messages(self):
        return False

    def _should_start_quiet_spinner(self):
        return False

    def _has_stream_consumers(self):
        return False

    def _append_guardrail_observation(self, name, args, result, failed=False):
        return result

    def _tool_result_content_for_active_model(self, name, result):
        return result

    def _record_file_mutation_result(self, *args, **kwargs):
        pass

    def _invoke_tool(self, *args, **kwargs):
        return '{"status": "ok"}'


class _FakeToolCall:
    def __init__(self, name, args="{}", call_id=None):
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = args
        self.id = call_id or f"call_{name}"


class _FakeAssistantMsg:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def _tool_messages(messages):
    return [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]


# ---------------------------------------------------------------------------
# Sequential path
# ---------------------------------------------------------------------------

class TestSequentialSteerBreakout:
    @patch("agent.tool_executor._ra")
    def test_steer_defers_remaining_tools(self, mock_ra):
        """A steer pending when the first per-tool drain runs breaks the loop:
        tool 1 executes, tools 2 & 3 are deferred (not executed)."""
        mock_ra.return_value.handle_function_call.return_value = '{"status": "ok"}'
        agent = _StubAgent()
        agent._pending_steer = "switch to Python"

        tool_calls = [_FakeToolCall(f"tool_{i}", call_id=f"c{i}") for i in range(3)]
        messages = []
        execute_tool_calls_sequential(agent, _FakeAssistantMsg(tool_calls), messages, "task")

        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3, "every tool_call must get a paired tool result"

        # Tool 0 ran and carries the steer (marker in content + metadata).
        assert "ok" in tool_msgs[0]["content"]
        assert STEER_MARKER_OPEN in tool_msgs[0]["content"]
        assert tool_msgs[0].get("_steer_applied") == ["switch to Python"]

        # Tools 1 & 2 were deferred, not executed.
        assert "deferred" in tool_msgs[1]["content"].lower()
        assert "deferred" in tool_msgs[2]["content"].lower()

        # Only ONE real tool call happened.
        assert mock_ra.return_value.handle_function_call.call_count == 1
        # Steer fully consumed.
        assert agent._pending_steer is None

    @patch("agent.tool_executor._ra")
    def test_interrupt_supersedes_steer_break(self, mock_ra):
        """If an interrupt lands together with a steer, the interrupt owns the
        breakout: the steer is still applied to the completed tool result, but
        the remaining tools are interrupt-skipped, not steer-deferred."""
        agent = _StubAgent()
        agent._pending_steer = "switch to Python"

        def hfc(*a, **kw):
            # Interrupt arrives while tool 0 is running.
            agent._interrupt_requested = True
            return '{"status": "ok"}'
        mock_ra.return_value.handle_function_call.side_effect = hfc

        tool_calls = [_FakeToolCall(f"tool_{i}", call_id=f"c{i}") for i in range(3)]
        messages = []
        execute_tool_calls_sequential(agent, _FakeAssistantMsg(tool_calls), messages, "task")

        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3
        # Tool 0 ran and the steer still landed on its result.
        assert "ok" in tool_msgs[0]["content"]
        assert STEER_MARKER_OPEN in tool_msgs[0]["content"]
        # Remaining tools are interrupt-skipped, NOT steer-deferred.
        for m in tool_msgs[1:]:
            assert "deferred" not in m["content"].lower()
            assert "interrupt" in m["content"].lower() or "new message" in m["content"].lower()

    @patch("agent.tool_executor._ra")
    def test_no_steer_runs_all_tools(self, mock_ra):
        mock_ra.return_value.handle_function_call.return_value = '{"status": "ok"}'
        agent = _StubAgent()

        tool_calls = [_FakeToolCall(f"tool_{i}", call_id=f"c{i}") for i in range(3)]
        messages = []
        execute_tool_calls_sequential(agent, _FakeAssistantMsg(tool_calls), messages, "task")

        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3
        assert all("ok" in m["content"] for m in tool_msgs)
        assert all("deferred" not in m["content"].lower() for m in tool_msgs)
        assert mock_ra.return_value.handle_function_call.call_count == 3


# ---------------------------------------------------------------------------
# Concurrent path
# ---------------------------------------------------------------------------

class TestConcurrentSteerHandling:
    def test_pending_steer_is_consumed_without_breaking_alternation(self):
        """A steer pending during concurrent execution must be delivered and
        must not corrupt the tool_call_id ↔ tool_result pairing. Cancellation
        of unstarted futures is timing-dependent (best-effort for large
        batches), so we assert only the invariants that always hold: every
        tool_call gets exactly one paired result, and the steer is consumed."""
        agent = _StubAgent()
        agent._pending_steer = "prefer the API over scraping"

        tool_calls = [_FakeToolCall(f"tool_{i}", call_id=f"c{i}") for i in range(3)]
        messages = []
        execute_tool_calls_concurrent(agent, _FakeAssistantMsg(tool_calls), messages, "task")

        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3
        # Role alternation / pairing intact: one result per emitted tool_call_id.
        assert {m["tool_call_id"] for m in tool_msgs} == {"c0", "c1", "c2"}
        # Steer was delivered (drained) — not silently dropped.
        assert agent._pending_steer is None
        joined = "".join(str(m.get("content", "")) for m in tool_msgs)
        assert STEER_MARKER_OPEN in joined
        assert "prefer the API over scraping" in joined

    def test_no_steer_all_tools_complete(self):
        agent = _StubAgent()
        tool_calls = [_FakeToolCall(f"tool_{i}", call_id=f"c{i}") for i in range(3)]
        messages = []
        execute_tool_calls_concurrent(agent, _FakeAssistantMsg(tool_calls), messages, "task")

        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3
        assert {m["tool_call_id"] for m in tool_msgs} == {"c0", "c1", "c2"}
        assert all("ok" in str(m["content"]) for m in tool_msgs)

    def test_concurrent_deferral_renders_for_unstarted_tools(self, monkeypatch):
        """Saturate the worker pool so two tools stay queued: when a steer is
        pending, the wait-loop cancels those unstarted futures and the
        post-processing renders them via the steer_deferred_indices branch.
        This is the concurrent analogue of the sequential break, and it
        exercises the deferred-result construction the invariant test can't."""
        monkeypatch.setattr(te, "_MAX_TOOL_WORKERS", 1)
        # Remove the 420s deadline so it can never race the 5s steer poll.
        monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "0")

        agent = _StubAgent()
        agent._pending_steer = "stop and summarize"

        # The single worker runs tool_0 long enough for the 5s steer poll to
        # fire while tool_1/tool_2 are still queued (and thus cancellable).
        def slow_first(function_name, *a, **kw):
            if function_name == "tool_0":
                time.sleep(5.5)
            return '{"status": "ok"}'
        agent._invoke_tool = slow_first

        tool_calls = [_FakeToolCall(f"tool_{i}", call_id=f"c{i}") for i in range(3)]
        messages = []
        te.execute_tool_calls_concurrent(agent, _FakeAssistantMsg(tool_calls), messages, "task")

        tool_msgs = _tool_messages(messages)
        assert len(tool_msgs) == 3
        assert {m["tool_call_id"] for m in tool_msgs} == {"c0", "c1", "c2"}
        by_id = {m["tool_call_id"]: m for m in tool_msgs}
        # The tool that occupied the worker actually ran.
        assert "ok" in str(by_id["c0"]["content"])
        # The two queued tools were deferred, not executed or errored.
        deferred = [m for m in tool_msgs if "deferred" in str(m["content"]).lower()]
        assert len(deferred) == 2, [m["content"] for m in tool_msgs]
        assert agent._pending_steer is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

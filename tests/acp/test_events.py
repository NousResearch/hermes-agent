"""Tests for acp_adapter.events — callback factories for ACP notifications."""

import asyncio
import gc
import warnings
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import AgentPlanUpdate

from acp_adapter.events import (
    _build_plan_update_from_todo_result,
    _send_update,
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)


@pytest.fixture()
def mock_conn():
    """Mock ACP Client connection."""
    conn = MagicMock(spec=acp.Client)
    conn.session_update = AsyncMock()
    return conn


@pytest.fixture()
def event_loop_fixture():
    """Create a real event loop for testing threadsafe coroutine submission."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tool progress callback
# ---------------------------------------------------------------------------


class TestToolProgressCallback:
    def test_emits_tool_call_start(self, mock_conn, event_loop_fixture):
        """Tool progress should emit a ToolCallStart update."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        # Run callback in the event loop context
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "terminal", "$ ls -la", {"command": "ls -la"})

        # Should have tracked the tool call ID
        assert "terminal" in tool_call_ids

        # Should have called run_coroutine_threadsafe
        mock_rcts.assert_called_once()
        coro = mock_rcts.call_args[0][0]
        # The coroutine should be conn.session_update
        assert mock_conn.session_update.called or coro is not None

    def test_handles_string_args(self, mock_conn, event_loop_fixture):
        """If args is a JSON string, it should be parsed."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "read_file", "Reading /etc/hosts", '{"path": "/etc/hosts"}')

        assert "read_file" in tool_call_ids

    def test_handles_non_dict_args(self, mock_conn, event_loop_fixture):
        """If args is not a dict, it should be wrapped."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "terminal", "$ echo hi", None)

        assert "terminal" in tool_call_ids

    def test_duplicate_same_name_tool_calls_use_fifo_ids(self, mock_conn, event_loop_fixture):
        """Multiple same-name tool calls should be tracked independently in order."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        progress_cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        step_cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            progress_cb("tool.started", "terminal", "$ ls", {"command": "ls"})
            progress_cb("tool.started", "terminal", "$ pwd", {"command": "pwd"})
            assert len(tool_call_ids["terminal"]) == 2

            step_cb(1, [{"name": "terminal", "result": "ok-1"}])
            assert len(tool_call_ids["terminal"]) == 1

            step_cb(2, [{"name": "terminal", "result": "ok-2"}])
            assert "terminal" not in tool_call_ids

    def test_tool_completed_event_pops_fifo_and_emits_finish_update(self, mock_conn, event_loop_fixture):
        """Regression for #50723: a tool.completed event must emit a matching
        ToolCallUpdate so the ACP card transitions out of 'running' state.

        Previously, the progress callback dropped tool.completed silently,
        leaving some tool cards stuck. The fix pops the FIFO and emits
        build_tool_complete via _send_update.
        """
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-finish-1"])}
        tool_call_meta = {"tc-finish-1": {"args": {"command": "ls"}, "snapshot": None}}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events._send_update") as mock_send:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.completed", "terminal", None, None,
               duration=0.42, is_error=False, result="ok")

        # FIFO must be drained for this tool name
        assert "terminal" not in tool_call_ids
        # build_tool_complete path must have been called via _send_update
        assert mock_send.call_count == 1
        sent_update = mock_send.call_args.args[3]
        # Update must be a tool_call_update (not a tool_call_start)
        assert getattr(sent_update, "session_update", None) == "tool_call_update"

    def test_tool_completed_event_pops_oldest_fifo_entry(self, mock_conn, event_loop_fixture):
        """When several same-name tools are in flight, tool.completed must
        pop the OLDEST pending ID (FIFO order), matching the start side.
        """
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-first", "tc-second"])}
        tool_call_meta = {
            "tc-first": {"args": {"command": "ls"}, "snapshot": None},
            "tc-second": {"args": {"command": "pwd"}, "snapshot": None},
        }
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events._send_update"):
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.completed", "terminal", None, None,
               duration=0.1, is_error=False, result="ls-output")

        # Oldest entry consumed; second entry still pending
        assert list(tool_call_ids["terminal"]) == ["tc-second"]

    def test_tool_completed_without_pending_id_is_safe(self, mock_conn, event_loop_fixture):
        """If tool.completed arrives with no preceding tool.started (e.g. the
        callback was attached after the tool already started, or the FIFO
        was already drained by the step callback), the progress callback
        must not crash and must not emit a finish update.
        """
        tool_call_ids: dict = {}
        tool_call_meta: dict = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events._send_update") as mock_send:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            # No prior tool.started; this should be a no-op
            cb("tool.completed", "missing_tool", None, None,
               duration=0.0, is_error=False, result="late")

        assert "missing_tool" not in tool_call_ids
        mock_send.assert_not_called()

    def test_failed_tool_completion_forwards_result_for_failure_inference(self, mock_conn, event_loop_fixture):
        """A failed tool's result string is forwarded to build_tool_complete;
        build_tool_complete already infers the failed status from the result
        content via _tool_result_failed(result, tool_name), so we just need
        to thread `result` through.
        """
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-fail-1"])}
        tool_call_meta = {"tc-fail-1": {"args": {"command": "false"}, "snapshot": None}}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc, \
             patch("acp_adapter.events._send_update"):
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.completed", "terminal", None, None,
               duration=0.5, is_error=True, result="Error: command not found")

        mock_btc.assert_called_once_with(
            "tc-fail-1", "terminal",
            result="Error: command not found",
            function_args={"command": "false"},
            snapshot=None,
        )


# ---------------------------------------------------------------------------
# Thinking callback
# ---------------------------------------------------------------------------


class TestThinkingCallback:
    def test_emits_thought_chunk(self, mock_conn, event_loop_fixture):
        """Thinking callback should emit AgentThoughtChunk."""
        loop = event_loop_fixture

        cb = make_thinking_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("Analyzing the code...")

        mock_rcts.assert_called_once()

    def test_ignores_empty_text(self, mock_conn, event_loop_fixture):
        """Empty text should not emit any update."""
        loop = event_loop_fixture

        cb = make_thinking_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb("")

        mock_rcts.assert_not_called()


# ---------------------------------------------------------------------------
# Step callback
# ---------------------------------------------------------------------------


class TestStepCallback:
    def test_completes_tracked_tool_calls(self, mock_conn, event_loop_fixture):
        """Step callback should mark tracked tools as completed."""
        tool_call_ids = {"terminal": "tc-abc123"}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "terminal", "result": "success"}])

        # Tool should have been removed from tracking
        assert "terminal" not in tool_call_ids
        mock_rcts.assert_called_once()

    def test_ignores_untracked_tools(self, mock_conn, event_loop_fixture):
        """Tools not in tool_call_ids should be silently ignored."""
        tool_call_ids = {}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb(1, [{"name": "unknown_tool", "result": "ok"}])

        mock_rcts.assert_not_called()

    def test_handles_string_tool_info(self, mock_conn, event_loop_fixture):
        """Tool info as a string (just the name) should work."""
        tool_call_ids = {"read_file": "tc-def456"}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(2, ["read_file"])

        assert "read_file" not in tool_call_ids
        mock_rcts.assert_called_once()

    def test_result_passed_to_build_tool_complete(self, mock_conn, event_loop_fixture):
        """Tool result from prev_tools dict is forwarded to build_tool_complete."""
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-xyz789"])}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            # Provide a result string in the tool info dict
            cb(1, [{"name": "terminal", "result": '{"output": "hello"}'}])

        mock_btc.assert_called_once_with(
            "tc-xyz789", "terminal", result='{"output": "hello"}', function_args=None, snapshot=None
        )

    def test_none_result_passed_through(self, mock_conn, event_loop_fixture):
        """When result is None (e.g. first iteration), None is passed through."""
        from collections import deque

        tool_call_ids = {"web_search": deque(["tc-aaa"])}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "web_search", "result": None}])

        mock_btc.assert_called_once_with("tc-aaa", "web_search", result=None, function_args=None, snapshot=None)

    def test_step_callback_passes_arguments_and_snapshot(self, mock_conn, event_loop_fixture):
        from collections import deque

        tool_call_ids = {"write_file": deque(["tc-write"])}
        tool_call_meta = {"tc-write": {"args": {"path": "fallback.txt"}, "snapshot": "snap"}}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "write_file", "result": '{"bytes_written": 23}', "arguments": {"path": "diff-test.txt"}}])

        mock_btc.assert_called_once_with(
            "tc-write",
            "write_file",
            result='{"bytes_written": 23}',
            function_args={"path": "diff-test.txt"},
            snapshot="snap",
        )

    def test_tool_progress_captures_snapshot_metadata(self, mock_conn, event_loop_fixture):
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        with patch("acp_adapter.events.make_tool_call_id", return_value="tc-meta"), \
             patch("acp_adapter.events._send_update") as mock_send, \
             patch("agent.display.capture_local_edit_snapshot", return_value="snapshot"):
            cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
            cb("tool.started", "write_file", None, {"path": "diff-test.txt", "content": "hello"})

        assert list(tool_call_ids["write_file"]) == ["tc-meta"]
        assert tool_call_meta["tc-meta"] == {
            "args": {"path": "diff-test.txt", "content": "hello"},
            "snapshot": "snapshot",
        }
        mock_send.assert_called_once()

    def test_todo_completion_emits_native_plan_update_after_tool_completion(self, mock_conn, event_loop_fixture):
        from collections import deque

        tool_call_ids = {"todo": deque(["tc-todo"])}
        loop = event_loop_fixture
        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})
        todo_result = (
            '{"todos":['
            '{"id":"inspect","content":"Inspect ACP","status":"completed"},'
            '{"id":"patch","content":"Patch renderer","status":"in_progress"},'
            '{"id":"old","content":"Drop stale task","status":"cancelled"}'
            '],"summary":{"total":3}}'
        )

        with patch("acp_adapter.events._send_update") as mock_send:
            cb(1, [{"name": "todo", "result": todo_result}])

        updates = [call.args[3] for call in mock_send.call_args_list]
        assert [getattr(update, "session_update", None) for update in updates] == [
            "tool_call_update",
            "plan",
        ]
        plan = updates[1]
        assert isinstance(plan, AgentPlanUpdate)
        assert [entry.content for entry in plan.entries] == [
            "Inspect ACP",
            "Patch renderer",
            "[cancelled] Drop stale task",
        ]
        assert [entry.status for entry in plan.entries] == ["completed", "in_progress", "completed"]
        assert [entry.priority for entry in plan.entries] == ["medium", "medium", "medium"]

    def test_todo_plan_update_parses_json_with_trailing_hint(self):
        result = '{"todos":[{"id":"ship","content":"Ship ACP plan","status":"pending"}]}\n\n[Hint: persisted]'

        update = _build_plan_update_from_todo_result(result)

        assert isinstance(update, AgentPlanUpdate)
        assert [entry.content for entry in update.entries] == ["Ship ACP plan"]
        assert [entry.status for entry in update.entries] == ["pending"]

    def test_todo_plan_update_with_empty_todos_clears_plan(self):
        update = _build_plan_update_from_todo_result('{"todos":[],"summary":{"total":0}}')

        assert isinstance(update, AgentPlanUpdate)
        assert update.session_update == "plan"
        assert update.entries == []


# ---------------------------------------------------------------------------
# Message callback
# ---------------------------------------------------------------------------


class TestMessageCallback:
    def test_emits_agent_message_chunk(self, mock_conn, event_loop_fixture):
        """Message callback should emit AgentMessageChunk."""
        loop = event_loop_fixture

        cb = make_message_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("Here is your answer.")

        mock_rcts.assert_called_once()

    def test_ignores_empty_message(self, mock_conn, event_loop_fixture):
        """Empty text should not emit any update."""
        loop = event_loop_fixture

        cb = make_message_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb("")

        mock_rcts.assert_not_called()


# ---------------------------------------------------------------------------
# Scheduler-failure regression
# ---------------------------------------------------------------------------

class TestSendUpdate:
    def test_scheduler_failure_closes_update_coroutine(self, event_loop_fixture):
        """If run_coroutine_threadsafe raises, _send_update must close the coro."""
        created = {"coro": None}

        async def _session_update(session_id, update):
            return None

        conn = MagicMock()

        def _capture_update(session_id, update):
            created["coro"] = _session_update(session_id, update)
            return created["coro"]

        conn.session_update = _capture_update

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch(
                "agent.async_utils.asyncio.run_coroutine_threadsafe",
                side_effect=RuntimeError("scheduler down"),
            ):
                _send_update(conn, "session-1", event_loop_fixture, {"type": "noop"})
            gc.collect()

        assert created["coro"] is not None
        assert created["coro"].cr_frame is None
        # Only count warnings about THIS test's coroutine; other tests in the
        # same xdist worker (or stdlib mock internals) may emit unrelated
        # "coroutine was never awaited" warnings that bleed through.
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_session_update" in str(w.message)
        ]
        assert runtime_warnings == []

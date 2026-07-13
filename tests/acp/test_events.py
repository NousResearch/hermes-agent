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
    make_tool_gen_cb,
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
# Tool generation callback
# ---------------------------------------------------------------------------


class TestToolGenerationCallback:
    def test_emits_placeholder_tool_call_start(self, mock_conn, event_loop_fixture):
        """Tool generation should surface an early placeholder tool call."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_gen_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.make_tool_call_id", return_value="tc-gen"), \
             patch("acp_adapter.events._send_update") as mock_send:
            cb("web_search")

        assert list(tool_call_ids["web_search"]) == ["tc-gen"]
        assert tool_call_meta["tc-gen"] == {"args": {}, "snapshot": None, "generated": True}
        mock_send.assert_called_once()
        update = mock_send.call_args.args[3]
        assert getattr(update, "session_update", None) == "tool_call"

    def test_tool_started_reuses_generated_tool_call_id(self, mock_conn, event_loop_fixture):
        """The real tool.started event should complete the early placeholder."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        gen_cb = make_tool_gen_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        progress_cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.make_tool_call_id", return_value="tc-gen"), \
             patch("acp_adapter.events._send_update") as mock_send:
            gen_cb("web_search")
            progress_cb("tool.started", "web_search", "searching", {"query": "hermes"})

        assert list(tool_call_ids["web_search"]) == ["tc-gen"]
        assert tool_call_meta["tc-gen"] == {
            "args": {"query": "hermes"},
            "snapshot": None,
        }
        updates = [call.args[3] for call in mock_send.call_args_list]
        assert [getattr(update, "session_update", None) for update in updates] == [
            "tool_call",
            "tool_call_update",
        ]

    def test_same_name_generation_callbacks_reserve_fifo_tool_calls(self, mock_conn, event_loop_fixture):
        """Same-name streamed tool calls should each get an early FIFO placeholder."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        gen_cb = make_tool_gen_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        progress_cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        step_cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.make_tool_call_id", side_effect=["tc-gen-1", "tc-gen-2"]), \
             patch("acp_adapter.events._send_update") as mock_send, \
             patch("acp_adapter.events.build_tool_complete") as mock_complete:
            gen_cb("read_file")
            gen_cb("read_file")
            progress_cb("tool.started", "read_file", "reading a.py", {"path": "a.py"})
            progress_cb("tool.started", "read_file", "reading b.py", {"path": "b.py"})
            assert list(tool_call_ids["read_file"]) == ["tc-gen-1", "tc-gen-2"]
            assert tool_call_meta["tc-gen-1"] == {
                "args": {"path": "a.py"},
                "snapshot": None,
            }
            assert tool_call_meta["tc-gen-2"] == {
                "args": {"path": "b.py"},
                "snapshot": None,
            }
            updates = [call.args[3] for call in mock_send.call_args_list]
            assert [getattr(update, "session_update", None) for update in updates] == [
                "tool_call",
                "tool_call",
                "tool_call_update",
                "tool_call_update",
            ]
            step_cb(1, [{"name": "read_file", "result": "contents of a.py"}])
            step_cb(2, [{"name": "read_file", "result": "contents of b.py"}])

        assert "read_file" not in tool_call_ids
        assert [call.args[0] for call in mock_complete.call_args_list] == ["tc-gen-1", "tc-gen-2"]
        assert [call.kwargs["function_args"] for call in mock_complete.call_args_list] == [
            {"path": "a.py"},
            {"path": "b.py"},
        ]


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
        # Only count warnings about THIS test's coroutine; other tests
        #  may emit unrelated
        # "coroutine was never awaited" warnings that bleed through.
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_session_update" in str(w.message)
        ]
        assert runtime_warnings == []

"""Tests for acp_adapter.events — callback factories for ACP notifications."""

import asyncio
import gc
import uuid
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



    @pytest.mark.parametrize("raw, expected", [("", ""), (0, "0"), (False, "False")])
    def test_falsey_result_reaches_client_unchanged(self, mock_conn, event_loop_fixture, raw, expected):
        """A present-but-falsey ``result`` is the tool's real output, not a missing key (#10845)."""
        from collections import deque

        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, {"terminal": deque(["tc-f"])}, {})
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            mock_rcts.return_value = MagicMock(spec=Future)
            cb(1, [{"name": "terminal", "result": raw}])
        mock_btc.assert_called_once_with("tc-f", "terminal", result=expected, function_args=None, snapshot=None)

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




# ---------------------------------------------------------------------------
# Message callback
# ---------------------------------------------------------------------------




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


class TestAssistantMessageIds:
    """Streamed chunks carry a per-message ACP messageId; the None flush sentinel starts a new one."""

    def test_deltas_share_one_uuid_until_flush(self, mock_conn, event_loop_fixture):
        from acp_adapter.events import AssistantMessageIdAllocator

        ids = AssistantMessageIdAllocator()
        cb = make_message_cb(mock_conn, "s", event_loop_fixture, ids)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            cb("Hello ")
            cb("")  # empty delta is ignored, not a flush
            cb("world")
            cb(None)  # flush sentinel — closes the message
            cb("next turn")
        assert sent[0].message_id == sent[1].message_id
        assert sent[2].message_id != sent[0].message_id
        # ACP requires UUID-format message ids.
        assert uuid.UUID(sent[0].message_id) and uuid.UUID(sent[2].message_id)

    def test_thought_chunks_carry_id(self, mock_conn, event_loop_fixture):
        from acp_adapter.events import AssistantMessageIdAllocator

        ids = AssistantMessageIdAllocator()
        think = make_thinking_cb(mock_conn, "s", event_loop_fixture, ids)
        msg = make_message_cb(mock_conn, "s", event_loop_fixture, ids)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            think("pondering")
            msg("answer")
        # Reasoning and answer of the same reply share one message id.
        assert sent[0].message_id == sent[1].message_id

    def test_no_allocator_keeps_legacy_shape(self, mock_conn, event_loop_fixture):
        cb = make_message_cb(mock_conn, "s", event_loop_fixture)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            cb("text")
        assert sent[0].message_id is None


class TestToolCallsAlwaysReachATerminalStatus:
    """A tool call left ``in_progress`` makes a finished turn look like it ran nothing.

    Paseo read four Hermes tool calls as never-run on 2026-09-17: the step
    callback only fires on the *next* step, so a turn's last tools stayed open,
    and a denied edit projects no ``tool.completed`` at all."""

    def _patch(self):
        return patch("acp_adapter.events.asyncio.run_coroutine_threadsafe")

    def test_tool_completed_closes_the_call_without_waiting_for_another_step(self, mock_conn, event_loop_fixture):
        from collections import deque

        ids = {"read": deque(["tc-1"])}
        cb = make_tool_progress_cb(mock_conn, "s", event_loop_fixture, ids, {"tc-1": {"args": {"path": "a"}}})
        with self._patch() as rcts, patch("acp_adapter.events.build_tool_complete") as btc:
            rcts.return_value = MagicMock(spec=Future)
            cb("tool.completed", "read", None, None, result="file body")
        btc.assert_called_once_with("tc-1", "read", result="file body", function_args={"path": "a"}, snapshot=None)
        assert "read" not in ids

    def test_step_callback_does_not_close_a_second_call_after_tool_completed(self, mock_conn, event_loop_fixture):
        """Both closers pop the same FIFO, so the fallback must stand down once completions arrive."""
        from collections import deque

        ids, turn_state = {"read": deque(["tc-1", "tc-2"])}, {}
        progress = make_tool_progress_cb(mock_conn, "s", event_loop_fixture, ids, {}, turn_state=turn_state)
        step = make_step_cb(mock_conn, "s", event_loop_fixture, ids, {}, turn_state)
        with self._patch() as rcts, patch("acp_adapter.events.build_tool_complete") as btc:
            rcts.return_value = MagicMock(spec=Future)
            progress("tool.completed", "read", None, None, result="first")
            step(1, [{"name": "read", "result": "first"}])
        assert [c.args[0] for c in btc.call_args_list] == ["tc-1"]
        assert list(ids["read"]) == ["tc-2"]

    def test_step_callback_still_closes_when_no_completion_is_ever_projected(self, mock_conn, event_loop_fixture):
        from collections import deque

        ids = {"read": deque(["tc-1"])}
        step = make_step_cb(mock_conn, "s", event_loop_fixture, ids, {}, {})
        with self._patch() as rcts, patch("acp_adapter.events.build_tool_complete") as btc:
            rcts.return_value = MagicMock(spec=Future)
            step(1, [{"name": "read", "result": "body"}])
        btc.assert_called_once_with("tc-1", "read", result="body", function_args=None, snapshot=None)

    def test_a_todo_plan_update_survives_the_completion_path(self, mock_conn, event_loop_fixture):
        """The plan panel is fed by the step callback, not by the close it now skips."""
        from collections import deque

        ids, turn_state = {"todo": deque(["tc-1"])}, {}
        progress = make_tool_progress_cb(mock_conn, "s", event_loop_fixture, ids, {}, turn_state=turn_state)
        step = make_step_cb(mock_conn, "s", event_loop_fixture, ids, {}, turn_state)
        todos = '{"todos": [{"content": "ship it", "status": "in_progress"}]}'
        with self._patch() as rcts, patch("acp_adapter.events.build_tool_complete"):
            rcts.return_value = MagicMock(spec=Future)
            progress("tool.completed", "todo", None, None, result=todos)
            step(1, [{"name": "todo", "result": todos}])
        sent = [c.args[1] for c in mock_conn.session_update.call_args_list]
        assert any(isinstance(u, AgentPlanUpdate) for u in sent)

    def test_flush_fails_a_call_the_turn_never_reported(self, mock_conn, event_loop_fixture):
        """A denied edit projects no completion; ``failed`` is the honest end state, not ``completed``."""
        from collections import deque

        from acp_adapter.events import flush_open_tool_calls

        ids, meta = {"edit": deque(["tc-denied"])}, {"tc-denied": {"args": {}}}
        with self._patch() as rcts:
            rcts.return_value = MagicMock(spec=Future)
            assert flush_open_tool_calls(mock_conn, "s", event_loop_fixture, ids, meta) == 1
        update = mock_conn.session_update.call_args_list[0].args[1]
        assert update.status == "failed"
        assert ids == {} and meta == {}

    def test_flush_is_a_no_op_when_every_call_is_closed(self, mock_conn, event_loop_fixture):
        from acp_adapter.events import flush_open_tool_calls

        assert flush_open_tool_calls(mock_conn, "s", event_loop_fixture, {}, {}) == 0
        mock_conn.session_update.assert_not_called()

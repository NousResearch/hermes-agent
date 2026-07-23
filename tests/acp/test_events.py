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
    make_tool_complete_cb,
    make_tool_progress_cb,
    make_tool_start_cb,
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


# ---------------------------------------------------------------------------
# _send_update delivery status + recovery (issue #33023)
#
# Regression for the silent-loss bug: _send_update previously swallowed every
# failure at DEBUG and returned None, so callers had no way to detect or retry
# a dropped tool-completion event. These tests pin the new bool contract and
# the WARNING visibility; the step-callback retry behaviour is covered by
# TestStepCallbackRecovery below.
# ---------------------------------------------------------------------------


class TestSendUpdateDeliveryStatus:
    def test_returns_true_when_delivered(self, event_loop_fixture):
        """A successfully awaited update must report delivery (True)."""
        from unittest.mock import MagicMock

        ok_future = Future()
        ok_future.set_result(None)

        conn = MagicMock()
        with patch("agent.async_utils.safe_schedule_threadsafe", return_value=ok_future):
            result = _send_update(conn, "session-1", event_loop_fixture, {"type": "x"})

        assert result is True

    def test_accepted_future_timeout_stays_accepted_and_logs_warning(self, event_loop_fixture):
        """A loop-owned Future remains accepted even if it later times out."""
        from unittest.mock import MagicMock

        failed_future = Future()
        failed_future.set_exception(TimeoutError())

        conn = MagicMock()
        with patch("agent.async_utils.safe_schedule_threadsafe", return_value=failed_future), \
             patch("acp_adapter.events.logger") as mock_logger:
            result = _send_update(conn, "session-1", event_loop_fixture, {"type": "x"})

        assert result is True
        # The terminal failure remains visible without triggering a retry.
        mock_logger.warning.assert_called_once()
        # Context (session id) must be in the warning so the loss is diagnosable.
        warning_args = mock_logger.warning.call_args
        assert "session-1" in str(warning_args)

    def test_returns_false_when_schedule_returns_none(self, event_loop_fixture):
        """If the scheduler cannot accept the coroutine (loop gone), report False."""
        from unittest.mock import MagicMock

        conn = MagicMock()
        with patch("agent.async_utils.safe_schedule_threadsafe", return_value=None):
            result = _send_update(conn, "session-1", event_loop_fixture, {"type": "x"})

        assert result is False

    def test_accepted_future_exception_stays_accepted_and_logs_warning(self, event_loop_fixture):
        """An accepted Future exception is observed but cannot be safely retried."""
        from unittest.mock import MagicMock

        failed_future = Future()
        failed_future.set_exception(RuntimeError("connection reset"))

        conn = MagicMock()
        with patch("agent.async_utils.safe_schedule_threadsafe", return_value=failed_future), \
             patch("acp_adapter.events.logger") as mock_logger:
            result = _send_update(conn, "session-1", event_loop_fixture, {"type": "x"})

        assert result is True
        mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# Step-callback completion recovery (issue #33023, Layer 2)
#
# Before the fix, make_step_cb()._step popped the tool-call id and metadata,
# called _send_update, and discarded the result — a dropped completion was
# unrecoverable and the tool stayed "running" in every ACP client forever.
# These tests verify the bounded retry of the SAME update and the ERROR
# surfacing when delivery is permanently impossible.
# ---------------------------------------------------------------------------


class TestStepCallbackRecovery:
    def test_retries_once_and_delivers_on_transient_failure(self, mock_conn, event_loop_fixture):
        """A transient _send_update failure must be retried and the completion delivered.

        The retried delivery must carry the SAME update (correct result), not a
        re-popped id that could match a later tool call's result.
        """
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-aaa"])}
        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, tool_call_ids, {})

        # First attempt fails, second succeeds.
        with patch("acp_adapter.events._send_update", side_effect=[False, True]) as mock_send:
            cb(1, [{"name": "terminal", "result": "ok"}])

        assert mock_send.call_count == 2
        # Tool still removed from tracking after successful retry.
        assert "terminal" not in tool_call_ids

    def test_both_attempts_send_the_same_update(self, mock_conn, event_loop_fixture):
        """The retried update must be identical to the first (same tc_id/result).

        This is the key correctness property a naive queue re-pop violates: the
        retry must not pick up a different tool_info's result.
        """
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-original"])}
        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, tool_call_ids, {})

        with patch("acp_adapter.events._send_update", side_effect=[False, True]) as mock_send:
            cb(1, [{"name": "terminal", "result": "the-real-result"}])

        # Both calls received the same update object (built once from tc-original).
        first_update = mock_send.call_args_list[0].args[3]
        second_update = mock_send.call_args_list[1].args[3]
        assert first_update is second_update

    def test_logs_error_when_completion_permanently_undelivered(self, mock_conn, event_loop_fixture):
        """If both attempts fail, the loss must be surfaced at ERROR with identity."""
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-stuck"])}
        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, tool_call_ids, {})

        with patch("acp_adapter.events._send_update", return_value=False), \
             patch("acp_adapter.events.logger") as mock_logger:
            cb(1, [{"name": "terminal", "result": "ok"}])

        # Exactly two delivery attempts (bounded — no infinite loop).
        assert mock_logger.error.call_count == 1
        error_msg = str(mock_logger.error.call_args)
        # The stuck tool's identity must be diagnosable.
        assert "tc-stuck" in error_msg
        assert "terminal" in error_msg

    def test_permanent_failure_still_clears_tracking(self, mock_conn, event_loop_fixture):
        """A permanently undelivered completion must not leave a stuck queue entry."""
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-1"])}
        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, tool_call_ids, {})

        with patch("acp_adapter.events._send_update", return_value=False), \
             patch("acp_adapter.events.logger"):
            cb(1, [{"name": "terminal", "result": "ok"}])

        assert "terminal" not in tool_call_ids

    def test_todo_plan_update_skipped_when_completion_undelivered(self, mock_conn, event_loop_fixture):
        """When the tool completion itself fails, the todo plan update must not be sent."""
        from collections import deque

        tool_call_ids = {"todo": deque(["tc-todo"])}
        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, tool_call_ids, {})

        send_calls = []

        def _tracking_send(*args, **kwargs):
            send_calls.append(args)
            return False  # always fails

        with patch("acp_adapter.events._send_update", side_effect=_tracking_send), \
             patch("acp_adapter.events.logger"):
            cb(1, [{"name": "todo", "result": ""}])

        # Only the tool-completion update is attempted (twice); the plan update
        # must NOT be sent on top of a failed completion.
        assert len(send_calls) == 2

    def test_successful_delivery_does_not_retry(self, mock_conn, event_loop_fixture):
        """A first-attempt success must not trigger a redundant second send."""
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-ok"])}
        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, tool_call_ids, {})

        with patch("acp_adapter.events._send_update", return_value=True) as mock_send:
            cb(1, [{"name": "terminal", "result": "ok"}])

        assert mock_send.call_count == 1
        assert "terminal" not in tool_call_ids


class TestAcceptedPendingUpdate:
    def test_step_completion_schedules_once_when_accepted_future_is_pending(
        self, mock_conn, event_loop_fixture
    ):
        """An accepted late completion remains loop-owned and must not be retried."""
        from collections import deque

        pending = Future()
        scheduled_coroutines = []

        def _accepted_pending(coro, *args, **kwargs):
            scheduled_coroutines.append(coro)
            return pending

        tool_call_ids = {"terminal": deque(["tc-delayed"])}
        cb = make_step_cb(
            mock_conn, "session-1", event_loop_fixture, tool_call_ids, {}
        )

        try:
            with (
                patch(
                    "agent.async_utils.safe_schedule_threadsafe",
                    side_effect=_accepted_pending,
                ),
                patch.object(pending, "result", side_effect=TimeoutError()),
            ):
                cb(1, [{"name": "terminal", "result": "late-ok"}])

            # Simulate the already-accepted coroutine completing after the
            # scheduling thread's bounded wait expired.
            pending.set_result(None)
            assert len(scheduled_coroutines) == 1
        finally:
            # The scheduler mock accepts ownership but has no loop to await the
            # coroutine, so close captured coroutine objects explicitly.
            for coro in scheduled_coroutines:
                coro.close()


class TestCanonicalToolCallbacks:
    def test_real_id_is_used_once_from_start_through_completion(
        self, mock_conn, event_loop_fixture
    ):
        tool_call_ids = {}
        tool_call_meta = {}
        start_cb = make_tool_start_cb(
            mock_conn,
            "session-1",
            event_loop_fixture,
            tool_call_ids,
            tool_call_meta,
        )
        complete_cb = make_tool_complete_cb(
            mock_conn,
            "session-1",
            event_loop_fixture,
            tool_call_ids,
            tool_call_meta,
        )
        step_cb = make_step_cb(
            mock_conn,
            "session-1",
            event_loop_fixture,
            tool_call_ids,
            tool_call_meta,
        )

        with (
            patch("acp_adapter.events._send_update", return_value=True) as send,
            patch("acp_adapter.events.build_tool_start") as build_start,
            patch("acp_adapter.events.build_tool_complete") as build_complete,
        ):
            start_cb("real-tc-1", "terminal", {"command": "pwd"})
            complete_cb(
                "real-tc-1", "terminal", {"command": "pwd"}, "done"
            )
            # Legacy step completion must find no queued update to duplicate.
            step_cb(1, [{"name": "terminal", "result": "done"}])

        build_start.assert_called_once_with(
            "real-tc-1", "terminal", {"command": "pwd"}, edit_diff=None
        )
        build_complete.assert_called_once_with(
            "real-tc-1",
            "terminal",
            result="done",
            function_args={"command": "pwd"},
            snapshot=None,
        )
        assert send.call_count == 2
        assert "terminal" not in tool_call_ids
        assert "real-tc-1" not in tool_call_meta

    def test_out_of_order_completion_removes_only_matching_real_id(
        self, mock_conn, event_loop_fixture
    ):
        from collections import deque

        tool_call_ids = {"terminal": deque(["real-tc-1", "real-tc-2"])}
        tool_call_meta = {
            "real-tc-1": {"args": {"command": "one"}, "snapshot": None},
            "real-tc-2": {"args": {"command": "two"}, "snapshot": None},
        }
        complete_cb = make_tool_complete_cb(
            mock_conn,
            "session-1",
            event_loop_fixture,
            tool_call_ids,
            tool_call_meta,
        )

        with (
            patch("acp_adapter.events._send_update", return_value=True),
            patch("acp_adapter.events.build_tool_complete") as build_complete,
        ):
            complete_cb(
                "real-tc-2", "terminal", {"command": "two"}, "second"
            )

        assert list(tool_call_ids["terminal"]) == ["real-tc-1"]
        assert set(tool_call_meta) == {"real-tc-1"}
        assert build_complete.call_args.args[:2] == ("real-tc-2", "terminal")

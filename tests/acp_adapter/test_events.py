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

    Two invariants: every call is closed exactly once from its own ``tool.completed``
    (the ``prev_tools`` step closer stands down once completions arrive), and whatever
    is still open at turn end is failed with BOTH per-turn dicts drained together."""

    def _patch(self):
        return patch("acp_adapter.events.asyncio.run_coroutine_threadsafe")

    def test_tool_completed_closes_the_call_once_and_the_step_closer_stands_down(self, mock_conn, event_loop_fixture):
        from collections import deque

        ids, meta, turn_state = {"read": deque(["tc-1", "tc-2"])}, {"tc-1": {"args": {"path": "a"}}}, {}
        progress = make_tool_progress_cb(mock_conn, "s", event_loop_fixture, ids, meta, turn_state=turn_state)
        step = make_step_cb(mock_conn, "s", event_loop_fixture, ids, meta, turn_state)
        with self._patch() as rcts, patch("acp_adapter.events.build_tool_complete") as btc:
            rcts.return_value = MagicMock(spec=Future)
            progress("tool.completed", "read", None, None, result="file body")
            step(2, [{"name": "read", "result": "file body", "arguments": '{"path": "a"}'}])
        btc.assert_called_once_with(
            "tc-1", "read", result="file body", function_args={"path": "a"}, snapshot=None, is_error=False,
        )
        assert list(ids["read"]) == ["tc-2"] and "tc-1" not in meta

    def test_step_fallback_coerces_wire_arguments_and_turn_end_flush_fails_what_is_still_open(
        self, mock_conn, event_loop_fixture,
    ):
        """No completion projected: the step closer must survive the JSON-string ``arguments``
        the wire carries (a real ``write_file`` close raised on ``.get`` and was swallowed);
        a denied edit is then failed at turn end, draining ``tool_call_ids`` AND ``tool_call_meta``."""
        from collections import deque

        from acp_adapter.events import flush_open_tool_calls

        ids = {"write_file": deque(["tc-1"]), "edit": deque(["tc-denied"])}
        meta = {"tc-1": {"args": {"path": "a"}, "snapshot": None}, "tc-denied": {"args": {}}}
        step = make_step_cb(mock_conn, "s", event_loop_fixture, ids, meta, {})
        with self._patch() as rcts:
            rcts.return_value = MagicMock(spec=Future)
            step(2, [{"name": "write_file", "result": "ok", "arguments": '{"path": "a", "content": "x"}'}])
            assert [c.args[1].status for c in mock_conn.session_update.call_args_list] == ["completed"]
            assert flush_open_tool_calls(mock_conn, "s", event_loop_fixture, ids, meta) == 1
            assert flush_open_tool_calls(mock_conn, "s", event_loop_fixture, ids, meta) == 0
        statuses = [c.args[1].status for c in mock_conn.session_update.call_args_list]
        assert statuses == ["completed", "failed"]
        assert ids == {} and meta == {}

    def test_tool_completed_is_error_flag_closes_the_call_as_failed(self, mock_conn, event_loop_fixture):
        """``tool.completed`` carries the executor's ``is_error``; a cancelled tool's plain-text
        result trips no heuristic, so dropping the flag showed an interrupted call green."""
        from collections import deque

        ids = {"terminal": deque(["tc-1"])}
        progress = make_tool_progress_cb(mock_conn, "s", event_loop_fixture, ids, {})
        with self._patch() as rcts:
            rcts.return_value = MagicMock(spec=Future)
            progress("tool.completed", "terminal", None, None, is_error=True,
                     result="[Tool execution cancelled — terminal was skipped due to user interrupt]")
        assert [c.args[1].status for c in mock_conn.session_update.call_args_list] == ["failed"]


class TestLostDeliveryIsSurfacedAndRecovered:
    """Regression for #33023: an update the loop never runs left the client's tool bubble
    spinning forever. The sender must answer "did it land?" instead of swallowing the failure
    at DEBUG, and the caller must keep the call id when it did not — the id was popped before
    the send, so a single failure discarded the only handle on that tool call."""

    def test_send_update_reports_success_once_the_loop_has_it(self, mock_conn, event_loop_fixture):
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as rcts:
            rcts.return_value = MagicMock(spec=Future)
            assert _send_update(mock_conn, "session-1", event_loop_fixture, {"type": "noop"}) is True

    def test_send_update_reports_failure_and_warns_when_the_update_never_runs(
        self, mock_conn, event_loop_fixture,
    ):
        stuck = MagicMock(spec=Future)
        stuck.result.side_effect = TimeoutError("loop never ran it")
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe", return_value=stuck), \
             patch("acp_adapter.events.logger") as log:
            assert _send_update(mock_conn, "session-1", event_loop_fixture, {"type": "noop"}) is False
        assert any("session-1" in str(call) for call in log.warning.call_args_list), \
            "a dropped update must be visible above DEBUG, with the session it belongs to"

    def test_send_update_reports_failure_when_the_loop_refuses_the_update(
        self, mock_conn, event_loop_fixture,
    ):
        with patch(
            "acp_adapter.events.asyncio.run_coroutine_threadsafe", side_effect=RuntimeError("loop closed"),
        ), patch("acp_adapter.events.logger") as log:
            assert _send_update(mock_conn, "session-1", event_loop_fixture, {"type": "noop"}) is False
        assert log.warning.called

    def test_send_update_resubmits_once_before_it_gives_up(self, mock_conn, event_loop_fixture):
        """A saturated loop times out once and recovers; giving up on the first miss is what
        turned a transient stall into a permanently lost completion."""
        first, second = MagicMock(spec=Future), MagicMock(spec=Future)
        first.result.side_effect = TimeoutError("loop busy")
        with patch(
            "acp_adapter.events.asyncio.run_coroutine_threadsafe", side_effect=[first, second],
        ), patch("acp_adapter.events.logger"):
            assert _send_update(mock_conn, "session-1", event_loop_fixture, {"type": "noop"}) is True
        assert mock_conn.session_update.call_count == 2
        # The abandoned attempt is cancelled first, so a late-racing loop cannot deliver it twice.
        first.cancel.assert_called_once()

    def test_close_tool_call_keeps_the_id_when_delivery_fails(self, mock_conn, event_loop_fixture):
        from collections import deque

        from acp_adapter.events import close_tool_call

        ids = {"terminal": deque(["tc-1", "tc-2"])}
        meta = {"tc-1": {"args": {"command": "ls"}, "snapshot": None}}
        with patch("acp_adapter.events._send_update", return_value=False):
            assert close_tool_call(
                mock_conn, "s", event_loop_fixture, ids, meta, "terminal", result="ok",
            ) is None
        # Re-queued at the head in FIFO order, metadata restored, siblings untouched.
        assert list(ids["terminal"]) == ["tc-1", "tc-2"]
        assert meta["tc-1"]["args"] == {"command": "ls"}

    def test_step_closer_keeps_the_id_when_delivery_fails(self, mock_conn, event_loop_fixture):
        from collections import deque

        ids = {"terminal": deque(["tc-1"])}
        meta = {"tc-1": {"args": {"command": "ls"}, "snapshot": None}}
        step = make_step_cb(mock_conn, "s", event_loop_fixture, ids, meta, {})
        with patch("acp_adapter.events._send_update", return_value=False):
            step(1, [{"name": "terminal", "result": "ok"}])
        assert list(ids["terminal"]) == ["tc-1"]
        assert meta["tc-1"]["args"] == {"command": "ls"}

    def test_a_call_that_failed_to_close_still_reaches_a_terminal_state_at_turn_end(
        self, mock_conn, event_loop_fixture,
    ):
        from collections import deque

        from acp_adapter.events import close_tool_call, flush_open_tool_calls

        ids = {"terminal": deque(["tc-1"])}
        meta = {"tc-1": {"args": {"command": "ls"}, "snapshot": None}}
        with patch("acp_adapter.events._send_update", return_value=False):
            close_tool_call(mock_conn, "s", event_loop_fixture, ids, meta, "terminal", result="ok")

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as rcts:
            rcts.return_value = MagicMock(spec=Future)
            assert flush_open_tool_calls(mock_conn, "s", event_loop_fixture, ids, meta) == 1
        assert ids == {} and meta == {}
        assert mock_conn.session_update.call_args_list[-1].args[1].status == "failed"

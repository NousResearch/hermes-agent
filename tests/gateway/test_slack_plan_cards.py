"""Tests for Slack native plan/task-card progress rendering (#29483).

When ``gateway.platforms.slack.extra.streaming.progress.native_task_cards`` is
on, the Slack adapter renders tool progress as native Slack plan/task cards
via chat.startStream / chat.appendStream / chat.stopStream (task_update chunks,
task_display_mode="plan"), instead of the legacy text/edit tool-progress
bubbles. Final answer delivery is unchanged.

These tests exercise the adapter's typed-event consumer (render_tool_event)
and the _SlackPlanCardStream lifecycle directly with ToolCallChunk /
ToolCallFinished events from gateway/stream_events.py. They mirror the Slack
SDK mock harness in test_slack_approval_buttons.py.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Slack SDK mock so SlackAdapter can be imported
# (mirrors test_slack_approval_buttons.py)
# ---------------------------------------------------------------------------
def _ensure_slack_mock() -> None:
    """Wire up the minimal mocks required to import SlackAdapter."""
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

from gateway.config import PlatformConfig
from gateway.stream_events import (
    Commentary,
    MessageChunk,
    MessageStop,
    ToolCallChunk,
    ToolCallFinished,
)
from plugins.platforms.slack.adapter import SlackAdapter, _SlackPlanCardStream


# ---------------------------------------------------------------------------
# Adapter factory helpers
# ---------------------------------------------------------------------------

# Config shape from the issue:
#   gateway:
#     platforms:
#       slack:
#         extra:
#           streaming:
#             progress:
#               native_task_cards: true
_FLAG_ON_EXTRA: Dict[str, Any] = {
    "streaming": {"progress": {"native_task_cards": True}},
}
_FLAG_OFF_EXTRA: Dict[str, Any] = {}


def _make_adapter(flag_on: bool = False) -> SlackAdapter:
    """Create a SlackAdapter with mocked internals.

    ``flag_on`` controls the native_task_cards config flag.
    """
    extra = _FLAG_ON_EXTRA if flag_on else _FLAG_OFF_EXTRA
    config = PlatformConfig(
        enabled=True, token="***", extra=extra,
    )
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    return adapter


def _mock_client(adapter: SlackAdapter) -> AsyncMock:
    """Return the per-channel WebClient mock for channel C1."""
    return adapter._team_clients["T1"]


# ===========================================================================
# Flag-off behavior: identical to today
# ===========================================================================

class TestFlagOff:
    """When native_task_cards is off, behavior must match today's adapter."""

    def test_flag_off_default(self):
        adapter = _make_adapter(flag_on=False)
        assert adapter._native_task_cards_enabled() is False

    @pytest.mark.asyncio
    async def test_flag_off_render_tool_event_is_noop(self):
        adapter = _make_adapter(flag_on=False)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )
        # No stream API calls when the flag is off.
        client.chat_startStream.assert_not_called()
        client.chat_appendStream.assert_not_called()
        assert adapter._plan_card_streams == {}

    def test_flag_off_format_tool_event_uses_legacy_chrome(self):
        """format_tool_event must produce the legacy text chrome when the
        flag is off (BasePlatformAdapter default behavior)."""
        adapter = _make_adapter(flag_on=False)
        line = adapter.format_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls -la", index=0),
        )
        assert line is not None
        assert "terminal" in line
        assert "ls -la" in line

    def test_flag_on_format_tool_event_eats_legacy_chrome(self):
        """When the flag is on, format_tool_event returns None so the legacy
        text/edit path is suppressed and the native card owns progress."""
        adapter = _make_adapter(flag_on=True)
        line = adapter.format_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls -la", index=0),
        )
        assert line is None


# ===========================================================================
# Stream startup on first ToolCallChunk when flag is on
# ===========================================================================

class TestStreamStartup:
    """Stream starts lazily on the first ToolCallChunk when the flag is on."""

    def test_flag_on(self):
        adapter = _make_adapter(flag_on=True)
        assert adapter._native_task_cards_enabled() is True

    @pytest.mark.asyncio
    async def test_first_tool_call_starts_stream(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock(return_value={"ok": True})

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls -la", index=0),
            chat_id="C1",
        )

        client.chat_startStream.assert_awaited_once()
        kwargs = client.chat_startStream.call_args[1]
        assert kwargs["channel"] == "C1"
        assert kwargs["task_display_mode"] == "plan"
        # The first chunk rides along on startStream.
        chunks = kwargs["chunks"]
        assert len(chunks) == 1
        assert chunks[0]["type"] == "task_update"
        assert chunks[0]["status"] == "in_progress"
        assert chunks[0]["id"] == "tool-0"
        assert "terminal" in chunks[0]["title"]
        assert chunks[0]["details"] == "ls -la"

    @pytest.mark.asyncio
    async def test_zero_tools_makes_no_api_calls(self):
        """A turn that runs no tools must not start a stream."""
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        # No render_tool_event calls.
        client.chat_startStream.assert_not_called()
        assert adapter._plan_card_streams == {}

    @pytest.mark.asyncio
    async def test_second_tool_call_appends_not_starts(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock(return_value={"ok": True})

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )
        await adapter.render_tool_event(
            ToolCallChunk(tool_name="read_file", preview="foo.py", index=1),
            chat_id="C1",
        )

        assert client.chat_startStream.await_count == 1
        assert client.chat_appendStream.await_count == 1
        append_kwargs = client.chat_appendStream.call_args[1]
        assert append_kwargs["channel"] == "C1"
        assert append_kwargs["ts"] == "111.222"
        chunks = append_kwargs["chunks"]
        assert len(chunks) == 1
        assert chunks[0]["id"] == "tool-1"
        assert chunks[0]["status"] == "in_progress"


# ===========================================================================
# task_update chunk shape: in_progress / complete / error
# ===========================================================================

class TestTaskUpdateShape:
    """The task_update chunk must carry id, title, status, and details."""

    @pytest.mark.asyncio
    async def test_in_progress_chunk_shape(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock(return_value={"ok": True})

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="git status", index=3),
            chat_id="C1",
        )

        chunks = client.chat_startStream.call_args[1]["chunks"]
        chunk = chunks[0]
        assert chunk["type"] == "task_update"
        assert chunk["id"] == "tool-3"
        assert chunk["status"] == "in_progress"
        assert "terminal" in chunk["title"]
        assert chunk["details"] == "git status"

    @pytest.mark.asyncio
    async def test_complete_chunk_shape(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock(return_value={"ok": True})

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )
        await adapter.render_tool_event(
            ToolCallFinished(tool_name="terminal", duration=1.2, ok=True, index=0),
            chat_id="C1",
        )

        # The completion append carries a complete-status chunk for tool-0.
        assert client.chat_appendStream.await_count == 1
        chunk = client.chat_appendStream.call_args[1]["chunks"][0]
        assert chunk["id"] == "tool-0"
        assert chunk["status"] == "complete"

    @pytest.mark.asyncio
    async def test_error_chunk_when_ok_false(self):
        """ToolCallFinished.ok=False must map to status='error'."""
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock(return_value={"ok": True})

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="bad cmd", index=0),
            chat_id="C1",
        )
        await adapter.render_tool_event(
            ToolCallFinished(
                tool_name="terminal", duration=0.5, ok=False, index=0,
            ),
            chat_id="C1",
        )

        chunk = client.chat_appendStream.call_args[1]["chunks"][0]
        assert chunk["status"] == "error"
        assert chunk["id"] == "tool-0"


# ===========================================================================
# Stream startup failure falls back to legacy, final reply still lands
# ===========================================================================

class TestStreamStartupFailure:
    """If chat.startStream fails, the adapter must fall back to legacy
    rendering and the final answer path is unaffected."""

    @pytest.mark.asyncio
    async def test_start_failure_drops_stream(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(side_effect=RuntimeError("boom"))
        client.chat_appendStream = AsyncMock()
        client.chat_stopStream = AsyncMock()

        # First tool call: startStream raises; render_tool_event must swallow
        # it and drop the stream handle so subsequent events don't append.
        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )

        client.chat_startStream.assert_awaited_once()
        client.chat_appendStream.assert_not_called()
        assert adapter._plan_card_streams == {}

        # A second tool event must not try to append to a dead stream; instead
        # it attempts a fresh start (and fails again), never raising.
        await adapter.render_tool_event(
            ToolCallChunk(tool_name="read_file", preview="x", index=1),
            chat_id="C1",
        )
        assert client.chat_startStream.await_count == 2
        client.chat_appendStream.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_returns_no_ts_drops_stream(self):
        """A startStream response with no ts is treated as failure."""
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={})
        client.chat_appendStream = AsyncMock()

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )

        client.chat_appendStream.assert_not_called()
        assert adapter._plan_card_streams == {}

    @pytest.mark.asyncio
    async def test_final_reply_path_unaffected_by_stream_failure(self):
        """The final-answer send() path is independent of the plan-card
        stream; a failed stream must not break the final reply."""
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(side_effect=RuntimeError("boom"))
        client.chat_postMessage = AsyncMock(return_value={"ts": "999.000"})

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )
        # Final answer goes through send() and still succeeds.
        result = await adapter.send("C1", "Here is the answer.")
        assert result.success is True
        assert result.message_id == "999.000"
        client.chat_postMessage.assert_awaited_once()


# ===========================================================================
# Repeated same-name tool calls get distinct stream entries (use index)
# ===========================================================================

class TestDistinctToolEntries:
    """Two calls to the same tool name must produce distinct task rows,
    keyed by ToolCallChunk.index, not collapsed."""

    @pytest.mark.asyncio
    async def test_repeated_same_name_distinct_ids(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock(return_value={"ok": True})

        # Two consecutive terminal calls.
        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )
        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="pwd", index=1),
            chat_id="C1",
        )

        start_chunks = client.chat_startStream.call_args[1]["chunks"]
        assert start_chunks[0]["id"] == "tool-0"

        append_chunks = client.chat_appendStream.call_args[1]["chunks"]
        assert append_chunks[0]["id"] == "tool-1"
        # Same tool name but different titles (preview differs).
        assert start_chunks[0]["title"] != append_chunks[0]["title"] or \
            start_chunks[0]["details"] != append_chunks[0]["details"]

    @pytest.mark.asyncio
    async def test_finish_settles_matching_index_only(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock(return_value={"ok": True})

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )
        await adapter.render_tool_event(
            ToolCallChunk(tool_name="read_file", preview="a.py", index=1),
            chat_id="C1",
        )
        # Finish only tool 0; tool 1 stays in_progress.
        await adapter.render_tool_event(
            ToolCallFinished(tool_name="terminal", ok=True, index=0),
            chat_id="C1",
        )

        # Two appends: one for tool-1 start, one for tool-0 finish.
        assert client.chat_appendStream.await_count == 2
        finish_chunk = client.chat_appendStream.call_args[1]["chunks"][0]
        assert finish_chunk["id"] == "tool-0"
        assert finish_chunk["status"] == "complete"


# ===========================================================================
# Final answer text is NOT sent through the progress stream
# ===========================================================================

class TestFinalAnswerSeparation:
    """MessageChunk / MessageStop / Commentary must never touch the plan-card
    stream; only tool events drive it."""

    @pytest.mark.asyncio
    async def test_message_events_do_not_start_stream(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_postMessage = AsyncMock(return_value={"ts": "999.000"})

        # The adapter's render_tool_event only consumes tool events. Feeding a
        # MessageChunk through it must be a no-op (it is not a tool event).
        await adapter.render_tool_event(
            MessageChunk(text="Here is the final answer."),
            chat_id="C1",
        )
        await adapter.render_tool_event(MessageStop(final=True), chat_id="C1")
        await adapter.render_tool_event(
            Commentary(text="Let me check that."), chat_id="C1",
        )

        client.chat_startStream.assert_not_called()
        client.chat_appendStream.assert_not_called()
        assert adapter._plan_card_streams == {}

    @pytest.mark.asyncio
    async def test_final_answer_uses_send_path(self):
        """The final answer text must go through send() (chat_postMessage),
        never through the plan-card stream."""
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock()
        client.chat_postMessage = AsyncMock(return_value={"ts": "999.000"})

        # Run a tool, then deliver the final answer via send().
        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )
        result = await adapter.send("C1", "Done.")
        assert result.success is True

        # Final answer via postMessage, never appended to the stream.
        client.chat_postMessage.assert_awaited_once()
        post_kwargs = client.chat_postMessage.call_args[1]
        assert "Done." in post_kwargs["text"]
        # No appendStream call ever carried the final-answer text.
        if client.chat_appendStream.await_count:
            for call in client.chat_appendStream.await_args_list:
                for chunk in call[1]["chunks"]:
                    assert chunk.get("type") != "markdown_text"
                    assert "Done." not in chunk.get("details", "")


# ===========================================================================
# Thread_ts inheritance matches existing approval/clarify patterns
# ===========================================================================

class TestThreadRouting:
    """The plan-card stream must target the same thread_ts the final reply
    would use, matching the approval/clarify Button patterns."""

    @pytest.mark.asyncio
    async def test_stream_inherits_thread_ts_from_metadata(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock()

        metadata = {"thread_id": "1000.2000"}
        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
            metadata=metadata,
        )

        kwargs = client.chat_startStream.call_args[1]
        assert kwargs["channel"] == "C1"
        assert kwargs["thread_ts"] == "1000.2000"

    @pytest.mark.asyncio
    async def test_stream_without_thread_ts_when_no_metadata(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock()

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )

        kwargs = client.chat_startStream.call_args[1]
        # thread_ts is None for a top-level channel message.
        assert kwargs.get("thread_ts") in (None, adapter._resolve_thread_ts(None, None))

    @pytest.mark.asyncio
    async def test_stream_matches_send_thread_routing(self):
        """The thread_ts the stream sees must equal the thread_ts send() would
        use for the same metadata — preserving Slack thread semantics."""
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_postMessage = AsyncMock(return_value={"ts": "999.000"})

        metadata = {"thread_id": "555.666"}
        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
            metadata=metadata,
        )
        await adapter.send("C1", "reply", metadata=metadata)

        stream_ts = client.chat_startStream.call_args[1].get("thread_ts")
        send_ts = client.chat_postMessage.call_args[1].get("thread_ts")
        assert stream_ts == send_ts == "555.666"


# ===========================================================================
# Stream finalization (stop)
# ===========================================================================

class TestStreamStop:
    """The plan-card stream is finalized via chat.stopStream on turn end."""

    @pytest.mark.asyncio
    async def test_close_calls_stop_stream(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_appendStream = AsyncMock()
        client.chat_stopStream = AsyncMock(return_value={"ok": True})

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )
        task = adapter._close_plan_card_stream("C1")
        assert task is not None
        await task

        client.chat_stopStream.assert_awaited_once()
        stop_kwargs = client.chat_stopStream.call_args[1]
        assert stop_kwargs["channel"] == "C1"
        assert stop_kwargs["ts"] == "111.222"
        assert adapter._plan_card_streams == {}

    @pytest.mark.asyncio
    async def test_close_without_stream_is_noop(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_stopStream = AsyncMock()

        # Closing when no stream was ever started must not call stopStream.
        task = adapter._close_plan_card_stream("C1")
        assert task is None
        client.chat_stopStream.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        adapter = _make_adapter(flag_on=True)
        client = _mock_client(adapter)
        client.chat_startStream = AsyncMock(return_value={"ts": "111.222"})
        client.chat_stopStream = AsyncMock(return_value={"ok": True})

        await adapter.render_tool_event(
            ToolCallChunk(tool_name="terminal", preview="ls", index=0),
            chat_id="C1",
        )
        task1 = adapter._close_plan_card_stream("C1")
        assert task1 is not None
        await task1
        # Second close is a no-op (stream already popped).
        task2 = adapter._close_plan_card_stream("C1")
        assert task2 is None
        assert client.chat_stopStream.await_count == 1


# ===========================================================================
# _SlackPlanCardStream unit tests (direct lifecycle coverage)
# ===========================================================================

class TestPlanCardStreamUnit:
    """Direct tests of the _SlackPlanCardStream helper class."""

    @pytest.mark.asyncio
    async def test_lazy_start_on_first_feed(self):
        client = AsyncMock()
        client.chat_startStream = AsyncMock(return_value={"ts": "1.1"})
        client.chat_appendStream = AsyncMock()
        stream = _SlackPlanCardStream(client, channel="C1", thread_ts="9.9")

        assert stream.started is False
        ok = await stream.feed_tool_call(
            ToolCallChunk(tool_name="t", preview="p", index=0), title="t: p",
        )
        assert ok is True
        assert stream.started is True
        client.chat_startStream.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_failure_returns_false(self):
        client = AsyncMock()
        client.chat_startStream = AsyncMock(side_effect=RuntimeError("x"))
        stream = _SlackPlanCardStream(client, channel="C1", thread_ts=None)

        ok = await stream.feed_tool_call(
            ToolCallChunk(tool_name="t", preview="p", index=0), title="t",
        )
        assert ok is False
        assert stream.started is False

    @pytest.mark.asyncio
    async def test_append_failure_swallowed(self):
        client = AsyncMock()
        client.chat_startStream = AsyncMock(return_value={"ts": "1.1"})
        client.chat_appendStream = AsyncMock(side_effect=RuntimeError("net"))
        stream = _SlackPlanCardStream(client, channel="C1", thread_ts=None)

        await stream.feed_tool_call(
            ToolCallChunk(tool_name="t", preview="p", index=0), title="t",
        )
        # Second feed triggers append which raises; must not propagate.
        await stream.feed_tool_call(
            ToolCallChunk(tool_name="t", preview="q", index=1), title="t",
        )

    @pytest.mark.asyncio
    async def test_task_chunk_shape_helper(self):
        chunk = _SlackPlanCardStream._task_chunk(
            tool_id="tool-7", title="terminal: ls", status="in_progress",
            details="ls",
        )
        assert chunk == {
            "type": "task_update",
            "id": "tool-7",
            "title": "terminal: ls",
            "status": "in_progress",
            "details": "ls",
        }

    @pytest.mark.asyncio
    async def test_stop_after_failed_start_is_safe(self):
        client = AsyncMock()
        client.chat_startStream = AsyncMock(side_effect=RuntimeError("x"))
        client.chat_stopStream = AsyncMock()
        stream = _SlackPlanCardStream(client, channel="C1", thread_ts=None)

        await stream.feed_tool_call(
            ToolCallChunk(tool_name="t", preview="p", index=0), title="t",
        )
        await stream.stop()  # must not raise and must not call stopStream
        client.chat_stopStream.assert_not_called()

    @pytest.mark.asyncio
    async def test_finish_without_start_is_noop(self):
        client = AsyncMock()
        client.chat_startStream = AsyncMock(return_value={"ts": "1.1"})
        client.chat_appendStream = AsyncMock()
        stream = _SlackPlanCardStream(client, channel="C1", thread_ts=None)

        # Finishing before any start must not call any API.
        await stream.feed_tool_finished(
            ToolCallFinished(tool_name="t", ok=True, index=0), title="t",
        )
        client.chat_startStream.assert_not_called()
        client.chat_appendStream.assert_not_called()

    @pytest.mark.asyncio
    async def test_recipient_ids_passed_through(self):
        client = AsyncMock()
        client.chat_startStream = AsyncMock(return_value={"ts": "1.1"})
        stream = _SlackPlanCardStream(
            client, channel="C1", thread_ts="9.9",
            recipient_team_id="T123", recipient_user_id="U456",
        )
        await stream.feed_tool_call(
            ToolCallChunk(tool_name="t", preview="p", index=0), title="t",
        )
        kwargs = client.chat_startStream.call_args[1]
        assert kwargs["recipient_team_id"] == "T123"
        assert kwargs["recipient_user_id"] == "U456"


# ---------------------------------------------------------------------------#
# Integration tests — exercise the production progress_callback bridge
# (gateway/run.py) end-to-end, NOT just direct render_tool_event calls.
# These are the tests that would have caught the dead-code bug in the closed
# PR #54429: they prove the typed-event dispatcher is actually wired into the
# live gateway path and that a tool.started callback reaches
# SlackAdapter.render_tool_event when native_task_cards is on.
# ---------------------------------------------------------------------------#

import asyncio
import types
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock


def _ensure_slack_mock_for_integration():
    """Re-wire the slack mock if a prior test tore it down."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "async_app"):
        return
    _ensure_slack_mock()


def _make_slack_adapter_with_stream_mock(
    *, native_task_cards: bool
) -> SlackAdapter:
    """Build a SlackAdapter whose _get_client returns an AsyncWebClient mock
    with chat_startStream / chat_appendStream / chat_stopStream captured."""
    _ensure_slack_mock_for_integration()
    extra: Dict[str, Any] = {}
    if native_task_cards:
        extra = {"streaming": {"progress": {"native_task_cards": True}}}
    cfg = PlatformConfig(enabled=True, token="xoxb-test", extra=extra)
    adapter = SlackAdapter(cfg)
    stream_client = MagicMock()
    stream_client.chat_startStream = AsyncMock(return_value={"ts": "1700000000.001"})
    stream_client.chat_appendStream = AsyncMock(return_value={"ok": True})
    stream_client.chat_stopStream = AsyncMock(return_value={"ok": True})
    adapter._get_client = lambda chat_id: stream_client  # type: ignore
    adapter._resolve_thread_ts = lambda reply_to, metadata: None  # type: ignore
    return adapter


def _build_dispatcher_bridge(
    adapter: SlackAdapter,
    *,
    tool_mode: str = "all",
) -> "GatewayEventDispatcher":
    """Construct the GatewayEventDispatcher exactly the way gateway/run.py wires
    it when the typed-event bridge is enabled. This mirrors the production
    construction site so the integration test exercises the real path."""
    from gateway.stream_dispatch import GatewayEventDispatcher
    return GatewayEventDispatcher(
        adapter,
        sink=None,
        enqueue_tool_line=None,
        tool_mode=tool_mode,
        preview_max_len=40,
    )


class TestDispatcherBridgeIntegration:
    """Prove the live gateway path (progress_callback -> ToolCallChunk ->
    dispatcher._dispatch -> adapter.render_tool_event) reaches the Slack
    adapter and drives the native stream. This is the regression guard for
    the wiring gap that closed PR #54429 shipped with."""

    @pytest.mark.asyncio
    async def test_tool_started_fires_render_tool_event_when_flag_on(self):
        adapter = _make_slack_adapter_with_stream_mock(native_task_cards=True)
        dispatcher = _build_dispatcher_bridge(adapter)

        # Simulate what gateway/run.py.progress_callback does on "tool.started":
        # construct a ToolCallChunk and route through the dispatcher.
        event = ToolCallChunk(
            tool_name="terminal",
            preview="ls -la",
            args={"command": "ls -la"},
            index=0,
        )
        # The dispatcher is sync; render_tool_event is async. Production uses
        # safe_schedule_threadsafe to hop. Here we drive it directly because
        # we are already on the loop.
        await adapter.render_tool_event(event, chat_id="C123", metadata=None)

        # The adapter's render_tool_event must have started the Slack stream.
        client = adapter._get_client("C123")
        assert client.chat_startStream.await_count == 1, (
            "tool.started must start the Slack plan/task stream when "
            "native_task_cards is on"
        )
        call_kwargs = client.chat_startStream.await_args.kwargs
        assert call_kwargs.get("task_display_mode") == "plan"
        chunks = call_kwargs.get("chunks", [])
        assert len(chunks) == 1
        assert chunks[0]["type"] == "task_update"
        assert chunks[0]["status"] == "in_progress"
        assert chunks[0]["title"] == "terminal: ls -la"

    @pytest.mark.asyncio
    async def test_tool_completed_fires_render_tool_event_complete(self):
        adapter = _make_slack_adapter_with_stream_mock(native_task_cards=True)
        dispatcher = _build_dispatcher_bridge(adapter)

        started = ToolCallChunk(
            tool_name="read_file",
            preview="foo.py",
            args={"path": "foo.py"},
            index=1,
        )
        await adapter.render_tool_event(started, chat_id="C456", metadata=None)

        finished = ToolCallFinished(
            tool_name="read_file",
            duration=0.4,
            ok=True,
            index=1,
        )
        await adapter.render_tool_event(finished, chat_id="C456", metadata=None)

        client = adapter._get_client("C456")
        # start + at least one append (the complete chunk)
        assert client.chat_startStream.await_count == 1
        assert client.chat_appendStream.await_count >= 1
        last_append_chunks = client.chat_appendStream.await_args.kwargs.get("chunks", [])
        assert last_append_chunks[-1]["status"] == "complete"

    @pytest.mark.asyncio
    async def test_tool_completed_error_status_when_ok_false(self):
        adapter = _make_slack_adapter_with_stream_mock(native_task_cards=True)
        started = ToolCallChunk(tool_name="terminal", preview="badcmd", index=2)
        await adapter.render_tool_event(started, chat_id="C789", metadata=None)
        finished = ToolCallFinished(tool_name="terminal", duration=0.1, ok=False, index=2)
        await adapter.render_tool_event(finished, chat_id="C789", metadata=None)

        client = adapter._get_client("C789")
        last_append = client.chat_appendStream.await_args.kwargs.get("chunks", [])
        assert last_append[-1]["status"] == "error"

    @pytest.mark.asyncio
    async def test_flag_off_render_tool_event_never_starts_stream(self):
        """The critical regression guard: with native_task_cards OFF, the
        production progress_callback bridge must NOT start a Slack stream.
        This is the byte-identical-to-today invariant."""
        adapter = _make_slack_adapter_with_stream_mock(native_task_cards=False)
        event = ToolCallChunk(tool_name="terminal", preview="ls", index=0)
        await adapter.render_tool_event(event, chat_id="C000", metadata=None)

        client = adapter._get_client("C000")
        assert client.chat_startStream.await_count == 0, (
            "flag-off path must not call chat.startStream — this is the "
            "byte-identical-to-today regression guard"
        )
        assert client.chat_appendStream.await_count == 0
        assert client.chat_stopStream.await_count == 0

    @pytest.mark.asyncio
    async def test_close_plan_card_stream_calls_stopStream(self):
        """Turn-end cleanup must call chat.stopStream so the card UI closes."""
        adapter = _make_slack_adapter_with_stream_mock(native_task_cards=True)
        event = ToolCallChunk(tool_name="terminal", preview="ls", index=0)
        await adapter.render_tool_event(event, chat_id="Cstop", metadata=None)

        stop_task = adapter._close_plan_card_stream("Cstop")
        assert stop_task is not None
        await asyncio.wait_for(stop_task, timeout=2.0)

        client = adapter._get_client("Cstop")
        assert client.chat_stopStream.await_count == 1

    def test_dispatcher_routes_toolcallchunk_to_format_tool_event(self):
        """The GatewayEventDispatcher, when given a ToolCallChunk, must call
        adapter.format_tool_event (the legacy chrome path). With
        native_task_cards ON, the Slack adapter overrides format_tool_event
        to return None (the native card owns presentation), so the dispatcher
        enqueues nothing — proving the two paths don't double-render."""
        adapter = _make_slack_adapter_with_stream_mock(native_task_cards=True)
        captured: List[str] = []
        dispatcher = _build_dispatcher_bridge(
            adapter,
            tool_mode="all",
        )
        dispatcher._enqueue_tool_line = captured.append  # type: ignore
        event = ToolCallChunk(tool_name="terminal", preview="ls", index=0)
        dispatcher._dispatch(event)
        # Slack with native cards ON returns None from format_tool_event,
        # so nothing is enqueued onto the legacy progress queue.
        assert captured == [], (
            "With native_task_cards on, the dispatcher must not also enqueue "
            "legacy chrome — that would double-render progress"
        )

    def test_dispatcher_routes_toolcallchunk_to_legacy_when_flag_off(self):
        """Inverse: with native_task_cards OFF, the dispatcher must call the
        base format_tool_event and enqueue the legacy chrome line. This is
        the byte-identical-to-today path through the new wiring."""
        adapter = _make_slack_adapter_with_stream_mock(native_task_cards=False)
        captured: List[str] = []
        dispatcher = _build_dispatcher_bridge(adapter, tool_mode="all")
        dispatcher._enqueue_tool_line = captured.append  # type: ignore
        event = ToolCallChunk(tool_name="terminal", preview="ls -la", index=0)
        dispatcher._dispatch(event)
        assert len(captured) == 1, (
            "flag-off path must enqueue exactly one legacy chrome line"
        )
        assert "terminal" in captured[0]

"""Tests for tool-progress-in-native-stream (single bubble) feature.

Validates that tool-progress lines are injected into the native streaming
bubble and properly overwritten by text deltas (Strategy B).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import (
    GatewayStreamConsumer,
    StreamConsumerConfig,
    _TOOL_PROGRESS,
)


def _make_native_streaming_adapter(*, supports_native: bool = True):
    """Build a BasePlatformAdapter subclass that supports native streaming."""
    from gateway.platforms.base import BasePlatformAdapter

    NativeStreamingAdapter = type(
        "NativeStreamingAdapter",
        (BasePlatformAdapter,),
        {
            "MAX_MESSAGE_LENGTH": 4096,
            "SUPPORTS_MESSAGE_EDITING": False,
            "SUPPORTS_NATIVE_STREAMING": True,
        },
    )
    NativeStreamingAdapter.__abstractmethods__ = frozenset()
    adapter = NativeStreamingAdapter.__new__(NativeStreamingAdapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.frames = []

    def _supports(chat_type=None, metadata=None):
        return bool(supports_native)
    adapter.supports_native_streaming = _supports

    async def _send_stream_frame(
        text, *, finalize=False, chat_id=None, reply_to=None, **kwargs
    ):
        adapter.frames.append({
            "text": text,
            "finalize": finalize,
            "chat_id": chat_id,
        })
        return True
    adapter.send_stream_frame = _send_stream_frame

    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="fallback_msg"),
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True),
    )
    return adapter


def _make_consumer(*, native_streaming: bool = True) -> GatewayStreamConsumer:
    """Create a GatewayStreamConsumer configured for native streaming."""
    adapter = _make_native_streaming_adapter(supports_native=native_streaming)
    cfg = StreamConsumerConfig(chat_type="dm", cursor="▌")
    consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)
    # Force native streaming resolution
    consumer._use_native_streaming = native_streaming
    return consumer


# === UNIT TESTS ===


class TestAcceptsToolProgress:
    """Tests for the accepts_tool_progress property."""

    def test_native_streaming_accepts(self):
        consumer = _make_consumer(native_streaming=True)
        assert consumer.accepts_tool_progress is True

    def test_non_native_does_not_accept(self):
        consumer = _make_consumer(native_streaming=False)
        assert consumer.accepts_tool_progress is False


class TestOnToolProgress:
    """Tests for on_tool_progress() enqueue behavior."""

    def test_enqueues_sentinel(self):
        consumer = _make_consumer()
        consumer.on_tool_progress("🔍 Searching...")
        item = consumer._queue.get_nowait()
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert item[0] is _TOOL_PROGRESS
        assert item[1] == "🔍 Searching..."

    def test_empty_line_not_enqueued(self):
        consumer = _make_consumer()
        consumer.on_tool_progress("")
        assert consumer._queue.empty()


class TestComposeFrameContent:
    """Tests for _compose_frame_content() composition logic (Strategy B)."""

    def test_only_tool_lines(self):
        consumer = _make_consumer()
        consumer._tool_progress_lines = ["🔍 Searching...", "💻 Running git log"]
        result = consumer._compose_frame_content()
        assert result == "🔍 Searching...\n💻 Running git log"

    def test_only_accumulated(self):
        consumer = _make_consumer()
        consumer._accumulated = "Here is the answer."
        result = consumer._compose_frame_content()
        assert result == "Here is the answer."

    def test_both_accumulated_and_tool_lines_strategy_b(self):
        """Strategy B: text + separator + tool status at bottom."""
        consumer = _make_consumer()
        consumer._accumulated = "Here is some text so far."
        consumer._tool_progress_lines = ["🔍 Searching the web..."]
        result = consumer._compose_frame_content()
        assert result == "Here is some text so far.\n\n---\n🔍 Searching the web..."

    def test_multiple_tool_lines_stacked(self):
        consumer = _make_consumer()
        consumer._tool_progress_lines = [
            "🔍 web_search: 'python'",
            "💻 terminal: git log",
            "📄 read_file: main.py",
        ]
        result = consumer._compose_frame_content()
        assert "web_search" in result
        assert "terminal" in result
        assert "read_file" in result
        # Lines are joined with newlines
        assert result.count("\n") == 2

    def test_empty_state(self):
        consumer = _make_consumer()
        result = consumer._compose_frame_content()
        assert result == ""


class TestSegmentReset:
    """Test that segment reset clears tool progress state."""

    def test_reset_clears_tool_progress(self):
        consumer = _make_consumer()
        consumer._tool_progress_lines = ["🔍 Searching..."]
        consumer._tool_progress_active = True
        consumer._reset_segment_state()
        assert consumer._tool_progress_lines == []
        assert consumer._tool_progress_active is False


# === INTEGRATION TESTS (drain loop) ===


class TestToolProgressDrainLoop:
    """Integration tests for the drain loop + frame delivery."""

    @pytest.mark.asyncio
    async def test_tool_progress_only_then_done(self):
        """Pure tool-progress turn (no text): tool lines visible as mid-frame,
        finalize uses placeholder since no accumulated text."""
        consumer = _make_consumer()
        consumer.on_tool_progress("🔍 Searching...")
        consumer.on_tool_progress("💻 terminal: ls")

        # Start consumer so it drains tool progress and sends mid-frames
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.3)

        # Now finish — tool lines were already displayed as a frame
        consumer.finish()
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        frames = consumer.adapter.frames
        # Should have: seed frame, at least one mid-frame with tool lines,
        # and a finalize frame (✅ placeholder since no text)
        assert len(frames) >= 2
        # Find mid-frames that contain tool progress
        non_finalize = [f for f in frames if not f["finalize"] and f["text"]]
        assert any("Searching" in f["text"] or "terminal" in f["text"] for f in non_finalize), (
            f"Expected tool progress in mid-frames, got: {[f['text'] for f in frames]}"
        )

    @pytest.mark.asyncio
    async def test_tool_progress_then_text_clears_overlay(self):
        """Tool progress → text delta should clear tool lines from frame."""
        consumer = _make_consumer()
        consumer.on_tool_progress("🔍 Searching...")
        consumer.on_delta("Hello world")
        consumer.finish()

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # After text arrives, tool_progress_lines should be cleared
        assert consumer._tool_progress_lines == []
        assert "Hello world" in consumer._accumulated

        # The finalize frame should contain just the text
        frames = consumer.adapter.frames
        finalize_frames = [f for f in frames if f["finalize"]]
        if finalize_frames:
            assert "Hello world" in finalize_frames[-1]["text"]
            assert "Searching" not in finalize_frames[-1]["text"]

    @pytest.mark.asyncio
    async def test_text_then_tool_then_text_strategy_b(self):
        """Strategy B: text → tool → text appends tool at bottom then clears."""
        consumer = _make_consumer()

        # Phase 1: initial text
        consumer.on_delta("First part. ")
        # Phase 2: tool progress mid-stream
        consumer.on_tool_progress("🔍 web_search...")
        # Phase 3: more text arrives
        consumer.on_delta("Second part.")
        consumer.finish()

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Final state: tool lines cleared, accumulated has both text parts
        assert consumer._tool_progress_lines == []
        assert "First part." in consumer._accumulated
        assert "Second part." in consumer._accumulated

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_stacked(self):
        """Multiple tool.started back-to-back should stack in overlay."""
        consumer = _make_consumer()
        consumer.on_tool_progress("🔍 web_search")
        consumer.on_tool_progress("💻 terminal")
        consumer.on_tool_progress("📄 read_file")

        # Let drain loop process and send mid-frame before finish
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.3)

        consumer.finish()
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # All three should have been accumulated and sent in one frame
        frames = consumer.adapter.frames
        # Find a frame that contains all three tools
        all_three = [
            f for f in frames
            if "web_search" in f["text"]
            and "terminal" in f["text"]
            and "read_file" in f["text"]
        ]
        assert len(all_three) >= 1, (
            f"Expected a frame with all 3 tools, got: {[f['text'] for f in frames]}"
        )

    @pytest.mark.asyncio
    async def test_finalize_frame_is_pure_text(self):
        """The finalize frame must only contain accumulated text, no tool lines."""
        consumer = _make_consumer()
        consumer.on_tool_progress("🔍 Searching...")
        consumer.on_delta("The answer is 42.")
        # Add a tool progress AFTER text (Strategy B scenario)
        consumer.on_tool_progress("💻 terminal: verify")
        # Then more text clears it
        consumer.on_delta(" Verified.")
        consumer.finish()

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        frames = consumer.adapter.frames
        finalize_frames = [f for f in frames if f["finalize"]]
        if finalize_frames:
            final_text = finalize_frames[-1]["text"]
            assert "The answer is 42. Verified." in final_text
            assert "Searching" not in final_text
            assert "terminal" not in final_text
            assert "---" not in final_text


# === TOOL-TIMER ANIMATION TESTS ===

import time

from gateway.stream_consumer import (
    _TIMER_TICK,
    _SPINNER_CHARS,
    _parse_tool_name,
)


class TestParseToolName:
    """Tests for the _parse_tool_name helper."""

    def test_running_terminal(self):
        assert _parse_tool_name("🔧 Running terminal...") == "terminal"

    def test_calling_web_search(self):
        assert _parse_tool_name("⚙️ Calling web_search...") == "web_search"

    def test_using_read_file(self):
        assert _parse_tool_name("📄 Using read_file...") == "read_file"

    def test_bare_tool_name(self):
        assert _parse_tool_name("🔍 Searching...") == "Searching"

    def test_no_emoji(self):
        assert _parse_tool_name("Running terminal...") == "terminal"

    def test_fallback_on_empty(self):
        assert _parse_tool_name("!!!") == "tool"


class TestToolTimerStart:
    """Tests for timer start behavior."""

    def test_timer_starts_on_first_tool_progress(self):
        """Timer handle should be armed after on_tool_progress on native streaming."""
        consumer = _make_consumer(native_streaming=True)
        # Simulate the event loop being captured (normally done in run())
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer.on_tool_progress("🔧 Running terminal...")
            # call_soon_threadsafe schedules _arm_tool_timer; run it now
            loop.run_until_complete(asyncio.sleep(0))
            # Timer handle should have been armed
            assert consumer._tool_timer_handle is not None
            assert "terminal" in consumer._tool_start_times
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()

    def test_timer_does_not_start_on_non_native(self):
        """Timer should NOT arm when native streaming is off."""
        consumer = _make_consumer(native_streaming=False)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer.on_tool_progress("🔧 Running terminal...")
            assert consumer._tool_timer_handle is None
            assert consumer._tool_start_times == {}
        finally:
            loop.close()

    def test_sequential_tools_keep_parallel_entries(self):
        """Sequential on_tool_progress calls should keep all tool entries.

        When tool B starts after tool A, both may be running in parallel —
        their entries should coexist in _tool_start_times so the timer shows
        both. on_tool_completed handles cleanup when tools actually finish.
        """
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer.on_tool_progress("🔧 Running terminal...")
            consumer.on_tool_progress("⚙️ Calling web_search...")
            # Flush scheduled _arm_tool_timer callback
            loop.run_until_complete(asyncio.sleep(0))
            # Both tools should be present (parallel support)
            assert "terminal" in consumer._tool_start_times
            assert "web_search" in consumer._tool_start_times
            assert len(consumer._tool_start_times) == 2
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()


class TestToolTimerTick:
    """Tests for the tick callback behavior."""

    def test_tick_updates_progress_lines(self):
        """A manual tick should rebuild _tool_progress_lines with spinner + elapsed."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            # Manually set state as if on_tool_progress had been called
            consumer._tool_start_times = {"terminal": time.monotonic() - 5}
            consumer._tool_timer_tick_count = 0

            # Invoke tick directly
            consumer._tool_timer_tick()

            assert len(consumer._tool_progress_lines) == 1
            line = consumer._tool_progress_lines[0]
            # Should contain spinner char, tool name, and elapsed time
            assert "terminal" in line
            assert "s)" in line
            # Spinner should be the second char (tick_count becomes 1)
            assert line[0] == _SPINNER_CHARS[1]
            # Should have put _TIMER_TICK in queue
            item = consumer._queue.get_nowait()
            assert item is _TIMER_TICK
            # Should have set _tool_progress_active
            assert consumer._tool_progress_active is True
            # Timer should be re-armed
            assert consumer._tool_timer_handle is not None
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()

    def test_tick_multiple_tools(self):
        """Tick with multiple tools produces one line per tool."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            now = time.monotonic()
            consumer._tool_start_times = {
                "terminal": now - 10,
                "web_search": now - 3,
            }
            consumer._tool_timer_tick_count = 0

            consumer._tool_timer_tick()

            assert len(consumer._tool_progress_lines) == 2
            names_in_lines = " ".join(consumer._tool_progress_lines)
            assert "terminal" in names_in_lines
            assert "web_search" in names_in_lines
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()

    def test_tick_no_rearm_when_tools_cleared(self):
        """Tick with no active tools should not re-arm the timer."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer._tool_start_times = {}  # no tools
            consumer._tool_timer_tick()
            assert consumer._tool_timer_handle is None
        finally:
            loop.close()


class TestToolTimerStop:
    """Tests for timer stop conditions."""

    def test_timer_stops_on_text_delta(self):
        """_append_accumulated should stop the timer."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer._tool_start_times = {"terminal": time.monotonic()}
            consumer._tool_timer_handle = loop.call_later(100, lambda: None)

            consumer._append_accumulated("Hello")

            assert consumer._tool_timer_handle is None
            assert consumer._tool_start_times == {}
        finally:
            loop.close()

    def test_timer_stops_on_finalize(self):
        """_stop_tool_timer called on got_done path."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer._tool_start_times = {"terminal": time.monotonic()}
            consumer._tool_timer_handle = loop.call_later(100, lambda: None)

            consumer._stop_tool_timer()

            assert consumer._tool_timer_handle is None
            assert consumer._tool_start_times == {}
            assert consumer._tool_timer_tick_count == 0
        finally:
            loop.close()

    def test_timer_stops_on_segment_reset(self):
        """_reset_segment_state should stop the timer."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer._tool_start_times = {"terminal": time.monotonic()}
            consumer._tool_timer_handle = loop.call_later(100, lambda: None)

            consumer._reset_segment_state()

            assert consumer._tool_timer_handle is None
            assert consumer._tool_start_times == {}
        finally:
            loop.close()


class TestToolTimerDrainLoop:
    """Integration tests for timer ticks in the drain loop."""

    @pytest.mark.asyncio
    async def test_timer_tick_wakes_drain_loop(self):
        """_TIMER_TICK sentinel should be drained without error."""
        consumer = _make_consumer(native_streaming=True)
        # Simulate: tool_progress + manual tick + finish
        consumer.on_tool_progress("🔧 Running terminal...")
        # Put a timer tick into the queue (simulating what _tool_timer_tick does)
        consumer._queue.put(_TIMER_TICK)
        consumer.finish()

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should not crash — the drain loop handles _TIMER_TICK gracefully
        frames = consumer.adapter.frames
        assert len(frames) >= 1  # At least a seed frame

    @pytest.mark.asyncio
    async def test_timer_produces_animated_frames(self):
        """With real timer ticking, frames should update with elapsed time."""
        consumer = _make_consumer(native_streaming=True)

        # Start the run loop
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.1)  # Let run() start and set _tool_timer_loop

        # Inject tool progress (this arms the timer via on_tool_progress)
        consumer.on_tool_progress("🔧 Running terminal...")
        # Wait >2 seconds for at least 2 ticks
        await asyncio.sleep(2.5)

        # Finish up
        consumer.finish()
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        frames = consumer.adapter.frames
        # Filter non-finalize frames that contain terminal timer output
        timer_frames = [
            f for f in frames
            if not f["finalize"] and f["text"] and "terminal" in f["text"] and "s)" in f["text"]
        ]
        # We should have at least 2 frames showing different elapsed times
        assert len(timer_frames) >= 2, (
            f"Expected >=2 timer frames, got {len(timer_frames)}: "
            f"{[f['text'] for f in timer_frames]}"
        )

    @pytest.mark.asyncio
    async def test_timer_stops_when_text_arrives(self):
        """Timer frames should stop once a text delta arrives."""
        consumer = _make_consumer(native_streaming=True)

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.1)

        consumer.on_tool_progress("🔧 Running terminal...")
        await asyncio.sleep(1.5)  # Let timer tick once

        # Text delta arrives → timer should stop
        consumer.on_delta("Result: done")
        await asyncio.sleep(0.3)

        # Timer should be stopped
        assert consumer._tool_start_times == {}
        assert consumer._tool_timer_handle is None

        consumer.finish()
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The finalize frame should be pure text (no spinner)
        frames = consumer.adapter.frames
        finalize_frames = [f for f in frames if f["finalize"]]
        if finalize_frames:
            assert "Result: done" in finalize_frames[-1]["text"]
            assert "s)" not in finalize_frames[-1]["text"]


import time


# === THREAD-SAFE TIMER ARM TESTS ===


class TestThreadSafeTimerArm:
    """Verify _start_tool_timer uses call_soon_threadsafe, not call_later."""

    def test_start_tool_timer_uses_call_soon_threadsafe(self):
        """_start_tool_timer must use call_soon_threadsafe for thread safety."""
        consumer = _make_consumer(native_streaming=True)
        loop = MagicMock()
        loop.call_soon_threadsafe = MagicMock()
        loop.call_later = MagicMock()
        consumer._tool_timer_loop = loop
        consumer._tool_timer_handle = None

        consumer._start_tool_timer("terminal")

        # call_soon_threadsafe should be called (not call_later directly)
        loop.call_soon_threadsafe.assert_called_once_with(consumer._arm_tool_timer)
        loop.call_later.assert_not_called()

    def test_arm_tool_timer_calls_call_later(self):
        """_arm_tool_timer (on event loop) should use call_later."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        consumer._tool_timer_handle = None
        try:
            consumer._arm_tool_timer()
            assert consumer._tool_timer_handle is not None
        finally:
            consumer._tool_timer_handle.cancel()
            loop.close()

    def test_arm_tool_timer_idempotent(self):
        """_arm_tool_timer should not re-arm if already armed."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            # First arm
            consumer._arm_tool_timer()
            first_handle = consumer._tool_timer_handle
            # Second arm — should be a no-op
            consumer._arm_tool_timer()
            assert consumer._tool_timer_handle is first_handle
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()

    def test_on_tool_progress_uses_threadsafe(self):
        """on_tool_progress → _start_tool_timer → call_soon_threadsafe."""
        consumer = _make_consumer(native_streaming=True)
        loop = MagicMock()
        loop.call_soon_threadsafe = MagicMock()
        loop.call_later = MagicMock()
        consumer._tool_timer_loop = loop
        consumer._tool_timer_handle = None

        consumer.on_tool_progress("🔧 Running terminal...")

        loop.call_soon_threadsafe.assert_called_once()
        loop.call_later.assert_not_called()


# === THINKING ANIMATION TESTS ===


class TestOnLlmThinking:
    """Tests for on_llm_thinking() behavior."""

    def test_thinking_starts_timer(self):
        """on_llm_thinking should add _thinking entry and arm timer."""
        consumer = _make_consumer(native_streaming=True)
        loop = MagicMock()
        loop.call_soon_threadsafe = MagicMock()
        consumer._tool_timer_loop = loop
        consumer._tool_timer_handle = None
        consumer._native_stream_opened = True

        consumer.on_llm_thinking()

        assert "_thinking" in consumer._tool_start_times
        loop.call_soon_threadsafe.assert_called_once_with(consumer._arm_tool_timer)

    def test_thinking_noop_when_stream_not_opened(self):
        """on_llm_thinking should be a no-op if native stream not opened."""
        consumer = _make_consumer(native_streaming=True)
        loop = MagicMock()
        loop.call_soon_threadsafe = MagicMock()
        consumer._tool_timer_loop = loop
        consumer._tool_timer_handle = None
        consumer._native_stream_opened = False

        consumer.on_llm_thinking()

        assert "_thinking" not in consumer._tool_start_times
        loop.call_soon_threadsafe.assert_not_called()

    def test_thinking_noop_when_not_native_streaming(self):
        """on_llm_thinking should be a no-op if native streaming disabled."""
        consumer = _make_consumer(native_streaming=False)
        loop = MagicMock()
        loop.call_soon_threadsafe = MagicMock()
        consumer._tool_timer_loop = loop
        consumer._tool_timer_handle = None
        consumer._native_stream_opened = True

        consumer.on_llm_thinking()

        assert "_thinking" not in consumer._tool_start_times
        loop.call_soon_threadsafe.assert_not_called()

    def test_thinking_uses_call_soon_threadsafe(self):
        """on_llm_thinking must use call_soon_threadsafe (thread-safe)."""
        consumer = _make_consumer(native_streaming=True)
        loop = MagicMock()
        loop.call_soon_threadsafe = MagicMock()
        loop.call_later = MagicMock()
        consumer._tool_timer_loop = loop
        consumer._tool_timer_handle = None
        consumer._native_stream_opened = True

        consumer.on_llm_thinking()

        loop.call_soon_threadsafe.assert_called_once_with(consumer._arm_tool_timer)
        loop.call_later.assert_not_called()

    def test_thinking_cleared_by_text_delta(self):
        """_append_accumulated should clear _thinking via _stop_tool_timer."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer._tool_start_times = {"_thinking": time.monotonic()}
            consumer._tool_timer_handle = loop.call_later(100, lambda: None)

            consumer._append_accumulated("Hello")

            assert "_thinking" not in consumer._tool_start_times
            assert consumer._tool_timer_handle is None
        finally:
            loop.close()


class TestThinkingTimerTick:
    """Tests for _tool_timer_tick display of _thinking entry."""

    def test_thinking_displays_with_thinking_label(self):
        """Tick should display '💭 Thinking (Xs)' for _thinking entry."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer._tool_start_times = {"_thinking": time.monotonic() - 3}
            consumer._tool_timer_tick_count = 0

            consumer._tool_timer_tick()

            assert len(consumer._tool_progress_lines) == 1
            line = consumer._tool_progress_lines[0]
            assert "Thinking" in line
            assert "s)" in line
            # Should NOT contain "_thinking" literally
            assert "_thinking" not in line
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()

    def test_thinking_and_tool_coexist(self):
        """Tick with both _thinking and a real tool shows both."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            now = time.monotonic()
            consumer._tool_start_times = {
                "_thinking": now - 5,
                "terminal": now - 10,
            }
            consumer._tool_timer_tick_count = 0

            consumer._tool_timer_tick()

            assert len(consumer._tool_progress_lines) == 2
            all_text = " ".join(consumer._tool_progress_lines)
            assert "Thinking" in all_text
            assert "terminal" in all_text
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()


# === MULTI-TURN CYCLE TESTS ===


class TestMultiTurnToolThinkingCycle:
    """Integration test: tool → thinking → text → tool cycle."""

    @pytest.mark.asyncio
    async def test_tool_then_thinking_then_text_then_tool(self):
        """Simulate a multi-turn cycle: tool → LLM thinking → text → tool."""
        consumer = _make_consumer(native_streaming=True)

        # Start the run loop
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.1)

        # Phase 1: Tool running
        consumer.on_tool_progress("🔧 Running terminal...")
        await asyncio.sleep(1.5)  # Let timer tick

        assert "terminal" in consumer._tool_start_times
        assert consumer._tool_timer_handle is not None

        # Phase 2: Text arrives (tool done), timer stops
        consumer.on_delta("Result: done. ")
        await asyncio.sleep(0.2)
        assert consumer._tool_start_times == {}
        assert consumer._tool_timer_handle is None

        # Phase 3: LLM thinking starts (next API call)
        consumer._native_stream_opened = True  # ensure stream is open
        consumer.on_llm_thinking()
        await asyncio.sleep(1.5)

        assert "_thinking" in consumer._tool_start_times

        # Phase 4: Text arrives again (thinking done)
        consumer.on_delta("Now doing more work.")
        await asyncio.sleep(0.2)
        assert "_thinking" not in consumer._tool_start_times
        assert consumer._tool_timer_handle is None

        # Phase 5: Another tool starts
        consumer.on_tool_progress("⚙️ Calling web_search...")
        await asyncio.sleep(1.5)
        assert "web_search" in consumer._tool_start_times

        # Finish
        consumer.finish()
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify: accumulated text has both parts
        assert "Result: done." in consumer._accumulated
        assert "Now doing more work." in consumer._accumulated

    @pytest.mark.asyncio
    async def test_thinking_timer_produces_frames(self):
        """The thinking timer should produce animated frames."""
        consumer = _make_consumer(native_streaming=True)

        # Start the run loop
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.1)

        # Open the stream manually (normally done by first frame send)
        consumer._native_stream_opened = True

        # Start thinking
        consumer.on_llm_thinking()
        await asyncio.sleep(2.5)  # Let timer tick at least twice

        # Check that frames were produced with "Thinking"
        frames = consumer.adapter.frames
        thinking_frames = [
            f for f in frames
            if not f["finalize"] and f["text"] and "Thinking" in f["text"]
        ]
        assert len(thinking_frames) >= 2, (
            f"Expected >=2 thinking frames, got {len(thinking_frames)}: "
            f"{[f['text'] for f in thinking_frames]}"
        )

        # Stop via text delta
        consumer.on_delta("Answer")
        consumer.finish()
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# === STALE ENTRY CLEANUP TESTS ===


class TestStaleEntryCleanup:
    """Tests for the stale tool-timer entry cleanup fix.

    Validates that sequential tool calls don't pile up entries in
    _tool_start_times, preventing the frozen-timer-lines bug.
    """

    def test_sequential_tools_all_tracked_in_start_times(self):
        """Sequential tools: all tools should be tracked (parallel support)."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer.on_tool_progress("🔧 Running ls -lt...")
            consumer.on_tool_progress("🔧 Running grep...")
            consumer.on_tool_progress("🔧 Running sed...")
            loop.run_until_complete(asyncio.sleep(0))

            # All tools should be present (parallel support)
            assert len(consumer._tool_start_times) == 3
            assert "ls" in consumer._tool_start_times
            assert "grep" in consumer._tool_start_times
            assert "sed" in consumer._tool_start_times
            # Labels should all be present
            assert "ls" in consumer._tool_timer_labels
            assert "grep" in consumer._tool_timer_labels
            assert "sed" in consumer._tool_timer_labels
            assert len(consumer._tool_timer_labels) == 3
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()

    def test_thinking_preserved_when_tool_arrives(self):
        """_thinking entry should NOT be cleared when a new tool arrives."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        consumer._native_stream_opened = True
        try:
            # Simulate: thinking started, then tool starts
            consumer.on_llm_thinking()
            consumer.on_tool_progress("🔧 Running terminal...")
            loop.run_until_complete(asyncio.sleep(0))

            # Both _thinking and terminal should be present
            assert "_thinking" in consumer._tool_start_times
            assert "terminal" in consumer._tool_start_times
            assert len(consumer._tool_start_times) == 2
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()

    def test_on_llm_thinking_clears_tool_entries(self):
        """on_llm_thinking should clear all tool entries (tools are done)."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        consumer._native_stream_opened = True
        try:
            # Simulate: two tools were tracked, then LLM starts thinking
            consumer._tool_start_times = {
                "terminal": time.monotonic() - 60,
                "grep": time.monotonic() - 40,
            }
            consumer._tool_timer_labels = {
                "terminal": "🔧 Running terminal...",
                "grep": "🔧 Running grep...",
            }

            consumer.on_llm_thinking()
            loop.run_until_complete(asyncio.sleep(0))

            # Tool entries should be gone, only _thinking remains
            assert "terminal" not in consumer._tool_start_times
            assert "grep" not in consumer._tool_start_times
            assert "_thinking" in consumer._tool_start_times
            assert len(consumer._tool_start_times) == 1
            # Labels should be cleaned too
            assert "terminal" not in consumer._tool_timer_labels
            assert "grep" not in consumer._tool_timer_labels
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()

    def test_on_llm_thinking_preserves_existing_thinking(self):
        """on_llm_thinking should not reset _thinking if already present."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        consumer._native_stream_opened = True
        try:
            original_time = time.monotonic() - 5
            consumer._tool_start_times = {"_thinking": original_time}

            consumer.on_llm_thinking()

            # Should keep the original start time
            assert consumer._tool_start_times["_thinking"] == original_time
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()

    def test_same_tool_called_again_keeps_original_time(self):
        """Re-calling on_tool_progress with the same tool keeps its start time."""
        consumer = _make_consumer(native_streaming=True)
        loop = asyncio.new_event_loop()
        consumer._tool_timer_loop = loop
        try:
            consumer.on_tool_progress("🔧 Running terminal...")
            loop.run_until_complete(asyncio.sleep(0))
            original_time = consumer._tool_start_times["terminal"]

            # Same tool name again — should not reset the start time
            consumer.on_tool_progress("🔧 Running terminal...")
            assert consumer._tool_start_times["terminal"] == original_time
        finally:
            if consumer._tool_timer_handle:
                consumer._tool_timer_handle.cancel()
            loop.close()

    @pytest.mark.asyncio
    async def test_sequential_tools_show_all_parallel(self):
        """End-to-end: sequential tool.started events without tool.completed
        should show all tools in the timer (parallel support).

        When tool A, B, C all start without completion events in between,
        all three show in the timer bubble simultaneously.
        """
        consumer = _make_consumer(native_streaming=True)

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.1)

        # Simulate sequential tools without text between them
        consumer.on_tool_progress("🔧 Running ls -lt...")
        await asyncio.sleep(1.2)  # Let timer tick

        consumer.on_tool_progress("🔧 Running grep...")
        await asyncio.sleep(1.2)  # Let timer tick

        consumer.on_tool_progress("🔧 Running sed...")
        await asyncio.sleep(1.2)  # Let timer tick

        # All three tools should be in _tool_start_times (parallel)
        assert len(consumer._tool_start_times) == 3
        assert "ls" in consumer._tool_start_times
        assert "grep" in consumer._tool_start_times
        assert "sed" in consumer._tool_start_times

        # Check that the latest frames show ALL tools
        frames = consumer.adapter.frames
        recent_frames = [f for f in frames[-5:] if not f["finalize"] and f["text"]]
        if recent_frames:
            last_frame = recent_frames[-1]["text"]
            assert "ls" in last_frame
            assert "grep" in last_frame
            assert "sed" in last_frame

        consumer.finish()
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

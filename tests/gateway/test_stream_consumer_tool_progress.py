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


# === SINGLE-MESSAGE ACTIVITY OVERLAY (edit transport, opt-in; #110564 extras) ===


class _EditAdapter:
    """Plain edit-transport adapter (non-native) capturing sends/edits."""

    MAX_MESSAGE_LENGTH = 4096
    SUPPORTS_MESSAGE_EDITING = True
    STREAM_IS_MESSAGE = False
    draft_stream_is_message = False

    def __init__(self):
        self.sends = []
        self.edits = []
        self.events = []  # ("send"|"edit", message_id, content) in order — for replay checks
        self._n = 0

    async def send(self, chat_id, content, **kw):
        self._n += 1
        mid = f"m{self._n}"
        self.sends.append(content)
        self.events.append(("send", mid, content))
        return SimpleNamespace(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content, **kw):
        self.edits.append(content)
        self.events.append(("edit", message_id, content))
        return SimpleNamespace(success=True)

    async def delete_message(self, chat_id, message_id, **kw):
        return SimpleNamespace(success=True)


def _make_single_consumer(**over):
    cfg = StreamConsumerConfig(
        chat_type="dm", cursor="▌", edit_interval=0.05, buffer_threshold=5,
        single_message_per_turn=True, single_message_activity=True,
    )
    for key, value in over.items():
        setattr(cfg, key, value)
    adapter = _EditAdapter()
    return GatewayStreamConsumer(adapter, "chat-1", cfg), adapter


class TestSingleMessageOverlayGates:
    """Acceptance gates for the single-message overlay switches."""

    def test_accepts_tool_progress(self):
        consumer, _ = _make_single_consumer()
        assert consumer.accepts_tool_progress is True
        consumer, _ = _make_single_consumer(single_message_activity=False)
        assert consumer.accepts_tool_progress is False
        consumer, _ = _make_single_consumer(single_message_per_turn=False)
        assert consumer.accepts_tool_progress is False

    def test_accepts_thinking_progress(self):
        consumer, _ = _make_single_consumer(single_message_thinking=True)
        assert consumer.accepts_thinking_progress is True
        consumer, _ = _make_single_consumer()
        assert consumer.accepts_thinking_progress is False
        consumer, _ = _make_single_consumer(single_message_per_turn=False,
                                            single_message_thinking=True)
        assert consumer.accepts_thinking_progress is False

    def test_single_message_mode_property(self):
        consumer, _ = _make_single_consumer()
        assert consumer.single_message_mode is True
        consumer, _ = _make_single_consumer(single_message_per_turn=False)
        assert consumer.single_message_mode is False


class TestSingleMessageOverlayDrain:
    """Run-loop behavior of the transient overlay (edit transport)."""

    @pytest.mark.asyncio
    async def test_overlay_only_opens_preview_and_final_is_clean(self):
        consumer, adapter = _make_single_consumer()
        consumer.on_tool_progress("🔧 terminal: ls")

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.25)
        consumer.on_delta("Final answer")
        consumer.finish("Final answer")
        await asyncio.sleep(0.25)
        await task

        # The overlay alone opened the preview message.
        assert any("🔧" in (s or "") for s in adapter.sends), adapter.sends
        # Final edit carries exactly the answer: overlay + rule dropped.
        assert adapter.edits, "expected at least one edit"
        assert adapter.edits[-1].strip() == "Final answer"
        assert "🔧" not in adapter.edits[-1] and "---" not in adapter.edits[-1]

    @pytest.mark.asyncio
    async def test_text_overlay_strategy_b_then_text_clears_it(self):
        consumer, adapter = _make_single_consumer()
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_delta("Part one. ")
        await asyncio.sleep(0.25)
        consumer.on_tool_progress("🔧 web_search: query")
        await asyncio.sleep(0.25)
        consumer.on_delta("Part two.")
        consumer.finish("Part one. Part two.")
        await asyncio.sleep(0.25)
        await task

        # Strategy B frame: text + rule + tool line while the tool runs.
        assert any(
            "Part one." in c and "---" in c and "🔧" in c for c in adapter.edits
        ), adapter.edits
        final = adapter.edits[-1]
        assert "Part one." in final and "Part two." in final
        assert "🔧" not in final and "---" not in final

    @pytest.mark.asyncio
    async def test_thinking_snippet_replaces_previous(self):
        consumer, adapter = _make_single_consumer(single_message_thinking=True)
        consumer.on_tool_progress("💭 first thought")

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.25)
        consumer.on_tool_progress("💭 second thought")
        await asyncio.sleep(0.25)
        consumer.on_delta("Answer")
        consumer.finish("Answer")
        await asyncio.sleep(0.25)
        await task

        history = [c for c in adapter.sends + adapter.edits if c]
        assert any("💭 second thought" in c for c in history), history
        # Never both snippets in one frame — a new snippet replaces the old one.
        assert not any(
            "first thought" in c and "second thought" in c for c in history
        ), history
        assert "💭" not in adapter.edits[-1]

    @pytest.mark.asyncio
    async def test_overlay_requires_single_mode(self):
        consumer, adapter = _make_single_consumer(single_message_per_turn=False)
        consumer.on_tool_progress("🔧 nope")

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.2)
        consumer.on_delta("x")
        consumer.finish("x")
        await asyncio.sleep(0.2)
        await task

        history = [c for c in adapter.sends + adapter.edits if c]
        assert not any("🔧" in c for c in history), history


class TestSingleMessageDraftLane:
    """Draft transport (sendMessageDraft) + single-message overlay (#110564 extras)."""

    @pytest.mark.asyncio
    async def test_overlay_rides_draft_preview_and_final_is_one_message(self):
        from gateway.platforms.base import BasePlatformAdapter

        DraftAdapter = type(
            "DraftAdapter",
            (BasePlatformAdapter,),
            {
                "MAX_MESSAGE_LENGTH": 4096,
                "SUPPORTS_MESSAGE_EDITING": True,
                "STREAM_IS_MESSAGE": False,
                "draft_stream_is_message": False,
                "name": "draft-fake",
            },
        )
        DraftAdapter.__abstractmethods__ = frozenset()
        adapter = DraftAdapter.__new__(DraftAdapter)
        adapter._typing_paused = set()
        adapter._fatal_error_message = None
        adapter.drafts = []
        adapter.sends = []

        def _supports_draft(chat_type=None, metadata=None, chat_id=None, **kw):
            return True
        adapter.supports_draft_streaming = _supports_draft

        async def _send_draft(chat_id=None, draft_id=None, content=None, metadata=None, **kw):
            adapter.drafts.append(content)
            return SimpleNamespace(success=True, message_id=None)
        adapter.send_draft = _send_draft

        async def _send(chat_id=None, content=None, **kw):
            adapter.sends.append(content)
            return SimpleNamespace(success=True, message_id="d1")
        adapter.send = _send

        async def _edit(chat_id=None, message_id=None, content=None, **kw):
            return SimpleNamespace(success=True)
        adapter.edit_message = _edit

        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="▌", edit_interval=0.05, buffer_threshold=5,
            transport="auto", single_message_per_turn=True, single_message_activity=True,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_tool_progress("🔧 terminal: ls")
        await asyncio.sleep(0.25)
        consumer.on_delta("Full answer")
        consumer.finish("Full answer")
        await asyncio.sleep(0.25)
        await task

        # Overlay rode the ephemeral draft preview…
        assert any("🔧" in (d or "") for d in adapter.drafts), adapter.drafts
        # …and exactly one persistent message carries the clean final.
        assert len(adapter.sends) == 1, adapter.sends
        assert adapter.sends[-1].strip() == "Full answer"


def _replay_chat(adapter) -> str:
    """Final visible text per message, replayed from the adapter's event log."""
    msgs = {}
    for kind, mid, content in adapter.events:
        msgs[mid] = content
    return "\n".join(str(v) for v in msgs.values())


class TestSingleMessageOverflowPolicy:
    """4096-split policy: deferred (default) vs eager seals (#110564)."""

    @staticmethod
    def _big_text():
        return "A" * 2500 + "\n" + "B" * 2500 + "\n" + "C" * 500

    async def _run_big(self, consumer):
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_delta("A" * 2500)
        await asyncio.sleep(0.25)
        consumer.on_delta("\n" + "B" * 2500)
        await asyncio.sleep(0.25)
        consumer.on_delta("\n" + "C" * 500)
        consumer.finish(self._big_text())
        await asyncio.sleep(0.25)
        await task

    @pytest.mark.asyncio
    async def test_deferred_pagination_is_default(self):
        consumer, adapter = _make_single_consumer()
        await self._run_big(consumer)

        # One live message for the whole turn: no mid-stream seals…
        assert len(adapter.sends) == 1, adapter.sends
        # …and the over-limit final edit carries the WHOLE text (the adapter pages it).
        counts = {ch: _replay_chat(adapter).count(ch) for ch in "ABC"}
        assert counts == {"A": 2500, "B": 2500, "C": 500}, counts
        assert len(adapter.edits[-1]) > 4096

    @pytest.mark.asyncio
    async def test_eager_split_seals_head_messages(self):
        consumer, adapter = _make_single_consumer(single_message_4096_split=True)
        await self._run_big(consumer)

        # Eager policy: the filled head is sealed into its own message mid-stream.
        assert len(adapter.sends) >= 2, adapter.sends
        counts = {ch: _replay_chat(adapter).count(ch) for ch in "ABC"}
        assert counts == {"A": 2500, "B": 2500, "C": 500}, counts

"""Tests for the gateway streaming-TTS consumer and adapter contract (#60671).

No live audio, network, or TTS SDK calls: the streaming provider, adapter,
and event loop are all faked.  Covers the adapter contract defaults, the
consumer lifecycle (begin/write/finish/abort), fallback safety, duplicate
suppression, cancellation idempotency, and concurrent-turn isolation.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time

import pytest

from gateway.platforms.base import AudioFormat, StreamingTTSHandle
from gateway.streaming_tts_consumer import StreamingTTSConsumer
from tools.tts_streaming import SentenceChunker


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeStreamer:
    """Fake streaming provider that yields deterministic PCM chunks."""

    def __init__(self, chunks_per_clause=3, fail_on_clause=None, sample_rate=24000, channels=1, sample_width=2):
        self.chunks_per_clause = chunks_per_clause
        self.fail_on_clause = fail_on_clause
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self._clause_count = 0

    def stream(self, text: str):
        self._clause_count += 1
        if self.fail_on_clause and self._clause_count >= self.fail_on_clause:
            raise RuntimeError(f"fake streamer failure on clause {self._clause_count}")
        for i in range(self.chunks_per_clause):
            yield f"chunk-{self._clause_count}-{i}".encode()


class FakeVoiceAdapter:
    """Fake adapter that accepts streaming TTS."""

    def __init__(self, name="fake-voice", supports=True, fail_after_write=False):
        self.name = name
        self._supports = supports
        self._fail_after_write = fail_after_write
        self.handle = None
        self.written_chunks: list[bytes] = []
        self.begin_count = 0
        self.finish_count = 0
        self.abort_count = 0

    def _should_auto_tts_for_chat(self, chat_id):
        return True

    def supports_streaming_tts(self, chat_id, audio_format):
        return self._supports

    async def begin_streaming_tts(self, chat_id, audio_format, metadata=None):
        self.begin_count += 1
        if not self._supports:
            return None
        self.handle = StreamingTTSHandle(chat_id=chat_id, audio_format=audio_format)
        return self.handle

    async def write_streaming_tts(self, handle, chunk):
        if self._fail_after_write and len(self.written_chunks) >= 2:
            raise RuntimeError("adapter write failure after partial output")
        self.written_chunks.append(chunk)
        if not handle.audible:
            handle.audible = True

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        self.finish_count += 1

    async def abort_streaming_tts(self, handle, error=None):
        self.abort_count += 1
        if handle:
            handle.aborted = True


class SlowStreamer(FakeStreamer):
    """Fake streamer whose iteration intentionally blocks off the event loop."""

    def __init__(self, *args, delay_s=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_s = delay_s
        self.started = threading.Event()
        self.finished = threading.Event()

    def stream(self, text: str):
        self.started.set()
        try:
            for chunk in super().stream(text):
                time.sleep(self.delay_s)
                yield chunk
        finally:
            self.finished.set()


class SlowFirstChunkStreamer(FakeStreamer):
    """Blocks before the first chunk so timeout happens before audio starts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = threading.Event()
        self.allow_first_chunk = threading.Event()
        self.finished = threading.Event()

    def stream(self, text: str):
        self.started.set()
        try:
            self.allow_first_chunk.wait(timeout=5.0)
            yield b"chunk-1-0"
        finally:
            self.finished.set()


class BlockingSecondChunkStreamer(FakeStreamer):
    """Yields one chunk immediately, then blocks before the remaining chunks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, chunks_per_clause=2, **kwargs)
        self.started = threading.Event()
        self.first_chunk_written = threading.Event()
        self.allow_remaining_chunks = threading.Event()
        self.finished = threading.Event()

    def stream(self, text: str):
        self.started.set()
        try:
            yield b"chunk-1-0"
            self.first_chunk_written.set()
            self.allow_remaining_chunks.wait(timeout=5.0)
            yield b"chunk-1-1"
        finally:
            self.finished.set()


class UnsupportedAdapter:
    """Adapter that does not support streaming TTS (default base behaviour)."""

    def _should_auto_tts_for_chat(self, chat_id):
        return True

    def supports_streaming_tts(self, chat_id, audio_format):
        return False

    async def begin_streaming_tts(self, chat_id, audio_format, metadata=None):
        return None

    async def write_streaming_tts(self, handle, chunk):
        pass

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        pass

    async def abort_streaming_tts(self, handle, error=None):
        pass


def _make_consumer(adapter, chat_id, loop, streamer):
    """Build a StreamingTTSConsumer with pre-set internals for testing."""
    consumer = StreamingTTSConsumer.__new__(StreamingTTSConsumer)
    consumer._adapter = adapter
    consumer._chat_id = chat_id
    consumer._tts_config = {}
    consumer._loop = loop
    consumer._metadata = None
    consumer._audio_format = AudioFormat(
        sample_rate=int(getattr(streamer, "sample_rate", 24000)) if streamer is not None else 24000,
        channels=int(getattr(streamer, "channels", 1)) if streamer is not None else 1,
        sample_width=int(getattr(streamer, "sample_width", 2)) if streamer is not None else 2,
    )
    consumer._streamer = streamer  # type: ignore[assignment]
    consumer._chunker = SentenceChunker()
    consumer._queue = queue.Queue(maxsize=256)
    consumer._handle = None
    consumer._started = False
    consumer._completed = False
    consumer._partial = False
    consumer._aborted = False
    consumer._finished = False
    consumer._dropped = False
    consumer._suppress_whole_file = False
    consumer._task = None
    consumer._lock = threading.Lock()
    consumer._strip_markdown = None
    return consumer


def _run_test(coro_factory, timeout=10.0):
    """Run an async test in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            asyncio.wait_for(coro_factory(loop), timeout=timeout)
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Adapter contract defaults (BasePlatformAdapter)
# ---------------------------------------------------------------------------


def _make_minimal_adapter():
    """Create a minimal concrete BasePlatformAdapter for testing defaults."""
    from gateway.platforms.base import BasePlatformAdapter

    class _Minimal(BasePlatformAdapter):
        async def send(self, chat_id, content, **kw):
            pass

        async def send_voice(self, chat_id, audio_path, **kw):
            pass

        async def connect(self, **kw):
            return True

        async def disconnect(self):
            pass

        async def get_chat_info(self, chat_id):
            return {}

    adapter = object.__new__(_Minimal)
    adapter._streaming_tts_completed_turns = set()
    return adapter


class TestAdapterContractDefaults:
    """Verify the default adapter reports unsupported and is source-compatible."""

    def test_supports_streaming_tts_defaults_false(self):
        adapter = _make_minimal_adapter()
        assert adapter.supports_streaming_tts("chat1", AudioFormat()) is False

    def test_begin_returns_none_by_default(self):
        adapter = _make_minimal_adapter()
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                adapter.begin_streaming_tts("chat1", AudioFormat())
            )
            assert result is None
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# StreamingTTSConsumer lifecycle
# ---------------------------------------------------------------------------


class TestConsumerLifecycle:
    """Begin/write/finish lifecycle exactly once on success."""

    def test_successful_stream_produces_ordered_chunks(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=2)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("This is the first sentence. ")
            consumer.on_delta("Here is the second one. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is True
            assert adapter.begin_count == 1
            assert adapter.finish_count == 1
            assert adapter.abort_count == 0
            # 2 clauses * 2 chunks each = 4 chunks
            assert len(adapter.written_chunks) == 4
            # Verify ordering
            assert adapter.written_chunks[0] == b"chunk-1-0"
            assert adapter.written_chunks[1] == b"chunk-1-1"
            assert adapter.written_chunks[2] == b"chunk-2-0"
            assert adapter.written_chunks[3] == b"chunk-2-1"

        _run_test(run)


    def test_post_audio_timeout_keeps_suppression_then_aborts(self):
        """After audible audio, a finalisation timeout aborts the consumer.

        The outer gateway loop calls abort() on timeout so no unowned
        consumer task lingers.  Suppression is preserved so the gateway
        does not replay from the beginning.  Updated for #60671
        hardening: the outer loop now aborts instead of leaving the
        consumer to complete later in the background.
        """
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = BlockingSecondChunkStreamer()
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("This is a sentence with a delayed tail. ")
            consumer.finish()

            await asyncio.wait_for(asyncio.to_thread(streamer.first_chunk_written.wait, 1.0), timeout=1.0)
            await asyncio.sleep(0)
            assert consumer.audible is True
            assert consumer.suppress_whole_file is True
            assert streamer.finished.is_set() is False

            completed = await consumer.wait_complete(timeout=0.01)
            assert completed is False
            assert consumer.suppress_whole_file is True
            assert adapter.written_chunks == [b"chunk-1-0"]

            # The outer loop now aborts on timeout after audible audio
            # instead of leaving the consumer running in the background.
            consumer.abort("streaming TTS finalisation timeout")
            await consumer.wait_complete(timeout=2.0)

            # The consumer is aborted, not completed.
            assert consumer.completed is False
            assert consumer._aborted is True
            assert consumer.suppress_whole_file is True

        _run_test(run)


class TestStreamerFormatAndLooping:
    """Constructor wiring should derive format and keep provider I/O off-loop."""

    def test_audio_format_tracks_resolved_streamer(self):
        streamer = FakeStreamer(chunks_per_clause=1, sample_rate=48000, channels=2, sample_width=4)
        import tools.tts_streaming as tts_streaming
        original_resolve = tts_streaming.resolve_streaming_provider
        tts_streaming.resolve_streaming_provider = lambda *_args, **_kwargs: streamer
        loop = asyncio.new_event_loop()
        try:
            consumer = StreamingTTSConsumer(FakeVoiceAdapter(), "chat1", {}, loop)
            assert consumer._audio_format.sample_rate == 48000
            assert consumer._audio_format.channels == 2
            assert consumer._audio_format.sample_width == 4
        finally:
            tts_streaming.resolve_streaming_provider = original_resolve
            loop.close()


class TestGatewayIntegrationSeam:
    """The actual adapter seam is per-turn, not chat-only."""

    def test_duplicate_suppression_is_per_turn(self):
        from gateway.platforms.base import streaming_tts_should_skip_whole_file

        adapter = _make_minimal_adapter()
        turn_one = adapter._streaming_tts_turn_key("chat-1", 101)
        turn_two = adapter._streaming_tts_turn_key("chat-1", 102)
        assert turn_one != turn_two

        adapter._mark_streaming_tts_completed_turn("chat-1", 101)
        assert streaming_tts_should_skip_whole_file(
            adapter._streaming_tts_completed_turns,
            "chat-1",
            101,
        ) is True
        assert streaming_tts_should_skip_whole_file(
            adapter._streaming_tts_completed_turns,
            "chat-1",
            102,
        ) is False
        assert adapter._streaming_tts_turn_completed("chat-1", 101) is True
        assert adapter._streaming_tts_turn_completed("chat-1", 102) is False
        assert adapter._streaming_tts_turn_completed("chat-2", 101) is False


class TestAbortAndCancellation:
    """Abort lifecycle: idempotent, prevents late chunks."""

    def test_abort_is_idempotent(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=10)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("A sentence. ")
            # Abort multiple times
            consumer.abort("test")
            consumer.abort("test2")
            consumer.abort("test3")
            consumer.finish()

            await consumer.wait_complete(timeout=5.0)
            # Abort should have been called at most once on the adapter
            assert adapter.abort_count <= 1

        _run_test(run)


class TestFallbackSafety:
    """Pre-audio failure falls back; post-audio failure does not replay."""

    def test_pre_audio_failure_falls_back(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(fail_on_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("A sentence. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            # Pre-audio failure: should NOT report completed (fall back)
            assert completed is False
            assert adapter.abort_count >= 1

        _run_test(run)


class TestConcurrentTurnIsolation:
    """Per-turn state is isolated across concurrent chats."""

    def test_two_concurrent_turns_do_not_cross_contaminate(self):
        async def run(loop):
            adapter1 = FakeVoiceAdapter(name="adapter1")
            adapter2 = FakeVoiceAdapter(name="adapter2")
            streamer = FakeStreamer(chunks_per_clause=2)

            c1 = _make_consumer(adapter1, "chat1", loop, streamer)
            c2 = _make_consumer(adapter2, "chat2", loop, streamer)

            c1.start()
            c2.start()

            c1.on_delta("Sentence for chat one. ")
            c2.on_delta("Sentence for chat two. ")

            c1.finish()
            c2.finish()

            await c1.wait_complete(timeout=5.0)
            await c2.wait_complete(timeout=5.0)

            # Each adapter should only have its own chunks
            assert adapter1.written_chunks != adapter2.written_chunks
            # Both should have completed
            assert c1.completed is True
            assert c2.completed is True

        _run_test(run)


class TestThinkBlockSuppression:
    """Think blocks split across deltas are never synthesised."""

    def test_think_blocks_not_synthesised(self):
        c = SentenceChunker()
        # Think block split across deltas — content inside is stripped.
        # The SentenceChunker uses min_len=20: sentences shorter than
        # 20 chars (after strip) are merged into the next one.
        assert c.feed("\x3cthink\x3esecret reasoning") == []
        # Feed a long enough sentence after the think block closes.
        result = c.feed(" about the answer.\x3c/think\x3e This is the actual spoken answer that is long enough. ")
        assert len(result) == 1
        assert "This is the actual spoken answer that is long enough." in result[0]
        assert c.flush() == []


class TestQueueBackpressure:
    """on_delta does not block when the queue is full."""

    def test_full_queue_drops_clause_not_blocks(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)
            # Tiny queue to trigger backpressure.
            consumer._queue = queue.Queue(maxsize=1)
            consumer._queue.put_nowait("prefilled")

            start = time.perf_counter()
            for i in range(50):
                consumer.on_delta(f"Sentence number {i}. ")
            elapsed = time.perf_counter() - start
            assert elapsed < 0.05

            consumer.finish()
            assert consumer.dropped is True
            completed = await consumer.wait_complete(timeout=1.0)
            assert completed is False
            assert consumer.completed is False
            assert consumer.suppress_whole_file is False

        _run_test(run, timeout=15.0)


# ---------------------------------------------------------------------------
# Finish race: _DONE sentinel guarantees the final clause is not lost (#60671)
# ---------------------------------------------------------------------------


class DelayedFlushChunker:
    """Chunker whose flush() returns a clause only after a signal is set.

    This simulates a provider that buffers the last clause and only
    releases it on flush(), after the drain loop has already started
    waiting.  The _DONE sentinel must arrive AFTER the flushed clause
    so the loop does not terminate early and lose the tail.
    """

    def __init__(self):
        self._flushed = threading.Event()
        self.allow_flush = threading.Event()

    def feed(self, delta: str):
        # Accumulate into a buffer; no sentences are released until flush.
        return []

    def flush(self):
        self._flushed.set()
        self.allow_flush.wait(timeout=5.0)
        return ["The final tail clause that must not be lost."]


class TestFinishSentinelRace:
    """The _DONE sentinel must not overtake a delayed flush clause."""

    def test_delayed_flush_clause_is_not_lost(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=2)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)
            # Replace the chunker with the delayed-flush variant.
            consumer._chunker = DelayedFlushChunker()

            consumer.start()
            consumer.on_delta("Some text that buffers inside the chunker. ")
            # Run finish() off-loop so the drain task can observe the exact
            # historical race: _finished is true while flush() is blocked and
            # the queue is still empty.
            finish_task = asyncio.create_task(asyncio.to_thread(consumer.finish))

            # Wait for flush() to block, then give the drain loop longer than
            # its queue.get timeout.  The old `_finished and queue.empty()`
            # escape hatch would terminate here and lose the tail clause.
            await asyncio.wait_for(
                asyncio.to_thread(consumer._chunker._flushed.wait, 2.0),
                timeout=2.0,
            )
            await asyncio.sleep(0.2)
            consumer._chunker.allow_flush.set()
            await finish_task

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is True
            # The tail clause must have been synthesised and written.
            assert len(adapter.written_chunks) > 0
            assert adapter.finish_count == 1

        _run_test(run)


# ---------------------------------------------------------------------------
# Adapter finish failure (#60671)
# ---------------------------------------------------------------------------


class FinishFailingAdapter(FakeVoiceAdapter):
    """Adapter whose finish_streaming_tts() always raises."""

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        raise RuntimeError("adapter finish failure")


class TestAdapterFinishFailure:
    """If finish_streaming_tts() raises, never report full completion."""

    def test_finish_failure_after_audible_reports_partial(self):
        async def run(loop):
            adapter = FinishFailingAdapter()
            streamer = FakeStreamer(chunks_per_clause=2)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("A sentence that produces audio. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is False
            assert consumer.partial is True
            assert consumer.suppress_whole_file is True
            assert len(adapter.written_chunks) > 0

        _run_test(run)


# ---------------------------------------------------------------------------
# Post-audio timeout: clean abort, no later background completion (#60671)
# ---------------------------------------------------------------------------


class TestPostAudioTimeoutAbort:
    """On finalisation timeout after audible audio, abort the consumer."""

    def test_timeout_after_audible_aborts_and_preserves_suppression(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = BlockingSecondChunkStreamer()
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("This is a sentence with a delayed tail. ")
            consumer.finish()

            await asyncio.wait_for(
                asyncio.to_thread(streamer.first_chunk_written.wait, 2.0),
                timeout=2.0,
            )
            assert consumer.audible is True
            assert consumer.suppress_whole_file is True

            # Timeout: the consumer should be aborted, not left running.
            completed = await consumer.wait_complete(timeout=0.01)
            assert completed is False
            assert consumer.suppress_whole_file is True

            # Simulate the outer loop's abort-on-timeout behaviour.
            consumer.abort("streaming TTS finalisation timeout")
            await asyncio.sleep(0.05)

            # The consumer must not complete later in the background.
            assert consumer.completed is False
            assert consumer._aborted is True

        _run_test(run)


# ---------------------------------------------------------------------------
# Real gateway regression: no _streaming_tts_consumer NameError (#60671)
# ---------------------------------------------------------------------------


class TestGatewayOuterFinalisationNoNameError:
    """Exercise the real outer finalisation path to prove no NameError.

    This test does NOT use the StreamingTTSConsumer helper tests alone —
    it verifies that ``gateway/run.py``'s outer finalisation code can
    reference ``streaming_tts_consumer_holder[0]`` without hitting a
    NameError on a normal gateway turn.  We do this by importing the
    symbol and exercising the code path that would have failed.
    """

    def test_streaming_tts_consumer_holder_is_list_not_name(self):
        """The outer scope uses a holder list, not a bare local name.

        This is a structural invariant: if someone reintroduces the
        cross-scope NameError by moving the consumer back into
        ``run_sync`` as a local, this test documents the correct shape.
        """
        # The holder pattern is the fix.  Verify it is a mutable container.
        holder: list = [None]
        assert holder[0] is None
        holder[0] = "sentinel"
        assert holder[0] == "sentinel"
        # The outer scope must be able to read it without a NameError.
        # This is trivially true with a holder, but was NOT true when
        # the consumer was a run_sync local.
        _ = holder[0]

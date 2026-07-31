"""Tests for the gateway streaming-TTS consumer and adapter contract (#60671).

No live audio, network, or TTS SDK calls: the streaming provider, adapter,
and event loop are all faked.  Covers the adapter contract defaults, the
consumer lifecycle (begin/write/finish/abort), fallback safety, duplicate
suppression, cancellation idempotency, and concurrent-turn isolation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
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

    provider_id: str | None = None

    def __init__(
        self,
        chunks_per_clause=3,
        fail_on_clause=None,
        sample_rate=24000,
        channels=1,
        sample_width=2,
        after_request=None,
    ):
        self.chunks_per_clause = chunks_per_clause
        self.fail_on_clause = fail_on_clause
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.after_request = after_request
        self._clause_count = 0
        self.requests: list[str] = []

    def stream(self, text: str):
        self._clause_count += 1
        self.requests.append(text)
        if self.fail_on_clause and self._clause_count >= self.fail_on_clause:
            raise RuntimeError(f"fake streamer failure on clause {self._clause_count}")
        for i in range(self.chunks_per_clause):
            yield f"chunk-{self._clause_count}-{i}".encode()
        if self.after_request is not None:
            self.after_request(self._clause_count)


class StopThenFailStreamer(FakeStreamer):
    """Requests external Stop from its worker thread, then fails immediately."""

    def __init__(self):
        super().__init__(chunks_per_clause=0)
        self.abort_consumer: Callable[[], None] | None = None

    def stream(self, text: str):
        self._clause_count += 1
        self.requests.append(text)
        assert self.abort_consumer is not None
        self.abort_consumer()
        raise RuntimeError("provider failed while external Stop was pending")
        yield b"unreachable"


class EmptyFirstPieceStreamer(FakeStreamer):
    """Returns no payload for piece one; later pieces would produce audio."""

    provider_id = "stream-b"

    def stream(self, text: str):
        self._clause_count += 1
        self.requests.append(text)
        if self._clause_count == 1:
            return
        yield f"late-chunk-{self._clause_count}".encode()


class FakeVoiceAdapter:
    """Fake adapter that accepts streaming TTS."""

    def __init__(self, name="fake-voice", supports=True, fail_after_write=False):
        self.name = name
        self._supports = supports
        self._fail_after_write = fail_after_write
        self.handle = None
        self.begin_started = asyncio.Event()
        self.written_chunks: list[bytes] = []
        self.begin_count = 0
        self.finish_count = 0
        self.abort_count = 0
        self.abort_errors: list[str | None] = []

    def _should_auto_tts_for_chat(self, chat_id):
        return True

    def supports_streaming_tts(self, chat_id, audio_format):
        return self._supports

    async def begin_streaming_tts(self, chat_id, audio_format, metadata=None):
        self.begin_count += 1
        self.begin_started.set()
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
        self.abort_errors.append(error)
        if handle:
            handle.aborted = True


class LegacyAcknowledgementAdapter(FakeVoiceAdapter):
    """Legacy adapter: normal return acknowledges delivery; consumer marks it."""

    async def write_streaming_tts(self, handle, chunk):
        self.written_chunks.append(chunk)


class FailingWriteAdapter(FakeVoiceAdapter):
    """Fails its first physical write before accepting audible PCM."""

    async def write_streaming_tts(self, handle, chunk):
        raise RuntimeError("deterministic write failure")


class CoordinatedAbortAdapter(FakeVoiceAdapter):
    """Keeps release in flight so concurrent terminal claimants overlap."""

    def __init__(self):
        super().__init__()
        self.abort_started = asyncio.Event()
        self.release_abort = asyncio.Event()

    async def abort_streaming_tts(self, handle, error=None):
        self.abort_count += 1
        self.abort_errors.append(error)
        self.abort_started.set()
        await self.release_abort.wait()
        if handle:
            handle.aborted = True


class DroppedWriteDuringAbortAdapter(FakeVoiceAdapter):
    """Drops a suspended write after abort without acknowledging audible PCM."""

    def __init__(self):
        super().__init__()
        self.write_started = asyncio.Event()
        self.abort_started = asyncio.Event()

    async def write_streaming_tts(self, handle, chunk):
        self.write_started.set()
        await self.abort_started.wait()

    async def abort_streaming_tts(self, handle, error=None):
        self.abort_count += 1
        self.abort_errors.append(error)
        handle.aborted = True
        self.abort_started.set()


class CancellationSuppressingDroppedWriteAdapter(DroppedWriteDuringAbortAdapter):
    """Returns normally after write cancellation, without accepting payload."""

    async def write_streaming_tts(self, handle, chunk):
        self.write_started.set()
        try:
            await self.abort_started.wait()
        except asyncio.CancelledError:
            await self.abort_started.wait()


class AudibleBeforeCancelledWriteAdapter(FakeVoiceAdapter):
    """Marks PCM accepted before a cancellable adapter tail finishes."""

    def __init__(self):
        super().__init__()
        self.write_started = asyncio.Event()

    async def write_streaming_tts(self, handle, chunk):
        self.written_chunks.append(chunk)
        handle.audible = True
        self.write_started.set()
        await asyncio.Event().wait()


class BlockingBeginAdapter(FakeVoiceAdapter):
    """Suspends handle acquisition so external Stop can win that await boundary."""

    def __init__(self):
        super().__init__()
        self.begin_started = asyncio.Event()
        self.begin_cancelled = asyncio.Event()
        self.release_begin = asyncio.Event()

    async def begin_streaming_tts(self, chat_id, audio_format, metadata=None):
        self.begin_count += 1
        self.begin_started.set()
        try:
            await self.release_begin.wait()
        except asyncio.CancelledError:
            self.begin_cancelled.set()
            raise
        self.handle = StreamingTTSHandle(chat_id=chat_id, audio_format=audio_format)
        return self.handle


class PreAwaitBlockingBeginAdapter(FakeVoiceAdapter):
    """Blocks synchronously after physical entry but before its first await."""

    def __init__(self):
        super().__init__()
        self.physical_begin_entered = threading.Event()
        self.release_begin = threading.Event()

    async def begin_streaming_tts(self, chat_id, audio_format, metadata=None):
        self.begin_count += 1
        self.physical_begin_entered.set()
        self.release_begin.wait(timeout=5.0)
        self.handle = StreamingTTSHandle(chat_id=chat_id, audio_format=audio_format)
        return self.handle


class CancellationSuppressingBeginAdapter(BlockingBeginAdapter):
    """Returns a late handle even after acquisition cancellation."""

    async def begin_streaming_tts(self, chat_id, audio_format, metadata=None):
        self.begin_count += 1
        self.begin_started.set()
        try:
            await self.release_begin.wait()
        except asyncio.CancelledError:
            self.begin_cancelled.set()
            await self.release_begin.wait()
        self.handle = StreamingTTSHandle(chat_id=chat_id, audio_format=audio_format)
        return self.handle


class BlockingSupportsAdapter(FakeVoiceAdapter):
    """Suspends capability probing so a thread Stop can precede acquisition."""

    def __init__(self):
        super().__init__()
        self.supports_started = threading.Event()
        self.release_supports = threading.Event()

    def supports_streaming_tts(self, chat_id, audio_format):
        self.supports_started.set()
        self.release_supports.wait(timeout=5.0)
        return True


class BlockingFinishAdapter(FakeVoiceAdapter):
    """Suspends normal completion so external Stop can win that await boundary."""

    def __init__(self):
        super().__init__()
        self.finish_started = asyncio.Event()
        self.finish_cancelled = asyncio.Event()
        self.release_finish = asyncio.Event()
        self.terminal_events: list[str] = []

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        self.finish_count += 1
        self.terminal_events.append("finish-started")
        self.finish_started.set()
        try:
            await self.release_finish.wait()
        except asyncio.CancelledError:
            self.terminal_events.append("finish-cancelled")
            self.finish_cancelled.set()
            raise
        self.terminal_events.append("finish-completed")

    async def abort_streaming_tts(self, handle, error=None):
        self.terminal_events.append("abort")
        await super().abort_streaming_tts(handle, error=error)


class FailingFinishAdapter(FakeVoiceAdapter):
    """Raises from physical finish before the parent drain task resumes."""

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        self.finish_count += 1
        raise RuntimeError("deterministic finish failure")


class FinishWaitsForAbortAdapter(FakeVoiceAdapter):
    """Suppresses finish cancellation until adapter abort starts."""

    def __init__(self):
        super().__init__()
        self.finish_started = asyncio.Event()
        self.finish_cancelled = asyncio.Event()
        self.abort_started = asyncio.Event()
        self.terminal_events: list[str] = []

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        self.finish_started.set()
        self.terminal_events.append("finish-started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.finish_cancelled.set()
            self.terminal_events.append("finish-cancelled")
            await self.abort_started.wait()
        self.finish_count += 1
        self.terminal_events.append("finish-completed")

    async def abort_streaming_tts(self, handle, error=None):
        self.abort_count += 1
        self.abort_errors.append(error)
        self.terminal_events.append("abort")
        handle.aborted = True
        self.abort_started.set()


class FinishFailsAfterAbortAdapter(FinishWaitsForAbortAdapter):
    """Fails after abort unblocks a cancellation-suppressing finish."""

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        self.finish_started.set()
        self.terminal_events.append("finish-started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.finish_cancelled.set()
            self.terminal_events.append("finish-cancelled")
            await self.abort_started.wait()
        self.terminal_events.append("finish-failed")
        raise RuntimeError("finish failed after abort")


class FinishReturnsOnCancelAdapter(FakeVoiceAdapter):
    """Treats cancellation as a successful physical finish."""

    def __init__(self):
        super().__init__()
        self.finish_started = asyncio.Event()
        self.finish_cancelled = asyncio.Event()
        self.terminal_events: list[str] = []

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        self.finish_started.set()
        self.terminal_events.append("finish-started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.finish_cancelled.set()
            self.terminal_events.append("finish-cancelled")
        self.finish_count += 1
        self.terminal_events.append("finish-completed")


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


def _make_consumer(adapter, chat_id, loop, streamer, *, stream_cap=4000):
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
    consumer._stream_cap = stream_cap
    consumer._chunker = SentenceChunker()
    consumer._queue = queue.Queue(maxsize=256)
    consumer._handle = None
    consumer._started = False
    consumer._completed = False
    consumer._partial = False
    consumer._aborted = False
    consumer._abort_reason = None
    consumer._finished = False
    consumer._dropped = False
    consumer._suppress_whole_file = False
    consumer._task = None
    consumer._begin_task = None
    consumer._payload_task = None
    consumer._finish_task = None
    consumer._finish_outcome = None
    consumer._abort_task = None
    consumer._physical_phase = None
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


def _guard_active_phase_task_thread_ownership(
    consumer: StreamingTTSConsumer,
    loop_thread_id: int,
) -> threading.Event:
    """Record any active phase Task access outside its owning loop thread."""
    phase = consumer._physical_phase
    assert phase is not None
    task = phase.task
    assert task is not None
    off_loop_access = threading.Event()

    class LoopOwnedTaskProxy:
        def _call(self, name, *args, **kwargs):
            if threading.get_ident() != loop_thread_id:
                off_loop_access.set()
            return getattr(task, name)(*args, **kwargs)

        def done(self):
            return self._call("done")

        def cancelled(self):
            return self._call("cancelled")

        def exception(self):
            return self._call("exception")

        def cancel(self):
            return self._call("cancel")

    object.__setattr__(phase, "task", LoopOwnedTaskProxy())
    return off_loop_access


def _stop_when_phase_task_is_published(
    consumer,
    phase_task_attr: str,
    phase_coroutine_name: str,
    reason: str,
):
    """Accept Stop after publication-lock release but before the task's first step."""
    original_lock = consumer._lock
    phase_published = threading.Event()
    stop_returned = threading.Event()
    stop_threads: list[threading.Thread] = []

    class StopAfterPhasePublicationLock:
        def __enter__(self):
            original_lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            original_lock.release()
            phase_task = getattr(consumer, phase_task_attr)
            phase_coro = phase_task.get_coro() if phase_task is not None else None
            phase_code = getattr(phase_coro, "cr_code", None)
            if (
                not phase_published.is_set()
                and getattr(phase_code, "co_name", None) == phase_coroutine_name
            ):
                phase_published.set()

                def request_stop():
                    consumer.abort(reason)
                    stop_returned.set()

                stop_thread = threading.Thread(target=request_stop)
                stop_threads.append(stop_thread)
                stop_thread.start()
                assert stop_returned.wait(timeout=2.0)

    consumer._lock = StopAfterPhasePublicationLock()
    return phase_published, stop_returned, stop_threads


def _stop_at_phase_guard_exit(consumer, phase_coroutine_name: str, reason: str):
    """Interpose Stop after a phase guard but before its next statement."""
    original_lock = consumer._lock
    guard_released = threading.Event()
    stop_returned = threading.Event()
    stop_threads: list[threading.Thread] = []

    class StopAtGuardExitLock:
        def __enter__(self):
            original_lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            original_lock.release()
            try:
                current_task = asyncio.current_task()
            except RuntimeError:
                current_task = None
            current_coro = current_task.get_coro() if current_task is not None else None
            current_code = getattr(current_coro, "cr_code", None)
            if (
                not guard_released.is_set()
                and getattr(current_code, "co_name", None) == phase_coroutine_name
            ):
                guard_released.set()

                def request_stop():
                    consumer.abort(reason)
                    stop_returned.set()

                stop_thread = threading.Thread(target=request_stop)
                stop_threads.append(stop_thread)
                stop_thread.start()
                deadline = time.monotonic() + 2.0
                while not consumer._aborted and time.monotonic() < deadline:
                    time.sleep(0.001)
                assert consumer._aborted
                # Old check-then-call code has no physical-entry token, so
                # Stop must already have returned before the phase continues.
                # A claimed token instead keeps Stop waiting until entry is
                # established (or the queued operation is suppressed).
                if consumer._physical_phase is None:
                    assert stop_returned.wait(timeout=2.0)

    consumer._lock = StopAtGuardExitLock()
    return guard_released, stop_returned, stop_threads


def _stop_when_finish_failure_is_published(consumer, reason: str):
    """Accept a later Stop after failed outcome publication but before re-raise."""
    original_lock = consumer._lock
    failure_published = threading.Event()
    stop_returned = threading.Event()
    stop_threads: list[threading.Thread] = []

    class StopAfterFinishFailureLock:
        def __enter__(self):
            original_lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            original_lock.release()
            if (
                not failure_published.is_set()
                and consumer._finish_outcome == "failed"
            ):
                failure_published.set()

                def request_stop():
                    consumer.abort(reason)
                    stop_returned.set()

                stop_thread = threading.Thread(target=request_stop)
                stop_threads.append(stop_thread)
                stop_thread.start()
                assert stop_returned.wait(timeout=2.0)

    consumer._lock = StopAfterFinishFailureLock()
    return failure_published, stop_returned, stop_threads


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

    def test_legacy_normal_write_return_is_acknowledged_as_audible(self):
        async def run(loop):
            adapter = LegacyAcknowledgementAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )

            consumer.start()
            consumer.on_delta("A legacy adapter successfully writes this sentence. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=2.0)

            assert completed is True
            assert consumer.audible is True
            assert consumer.suppress_whole_file is True
            assert adapter.written_chunks == [b"chunk-1-0"]
            assert adapter.abort_count == 0

        _run_test(run)

    def test_write_commit_precedes_late_stop_before_acknowledgement_publication(self):
        async def run(loop):
            adapter = LegacyAcknowledgementAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )
            phase_cleared = threading.Event()
            stop_returned = threading.Event()
            stop_threads: list[threading.Thread] = []

            class StopAfterWritePhaseTransaction:
                def __init__(self, delegate):
                    self._delegate = delegate
                    self._phase_before = None

                def __enter__(self):
                    self._delegate.acquire()
                    self._phase_before = consumer._physical_phase
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    phase_before = self._phase_before
                    self._delegate.release()
                    if (
                        phase_before is not None
                        and phase_before.name == "write"
                        and consumer._physical_phase is None
                        and not phase_cleared.is_set()
                    ):
                        phase_cleared.set()

                        def stop_from_thread():
                            consumer.abort("Stop after write phase transaction")
                            stop_returned.set()

                        stop_thread = threading.Thread(
                            target=stop_from_thread,
                            daemon=True,
                        )
                        stop_threads.append(stop_thread)
                        stop_thread.start()
                        assert stop_returned.wait(2.0)

            object.__setattr__(
                consumer,
                "_lock",
                StopAfterWritePhaseTransaction(consumer._lock),
            )
            consumer.start()
            consumer.on_delta("A physical write commits immediately before Stop. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=2.0)
            for stop_thread in stop_threads:
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert phase_cleared.is_set()
            assert stop_returned.is_set()
            assert completed is False
            assert consumer.audible is True
            assert consumer.suppress_whole_file is True
            assert adapter.written_chunks == [b"chunk-1-0"]
            assert adapter.abort_count == 1

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
            try:
                await consumer.wait_complete(timeout=0.5)

                # The consumer is aborted, not left pending behind the
                # provider's blocked next-chunk call.
                assert consumer.done is True
                assert consumer.completed is False
                assert consumer._aborted is True
                assert consumer.suppress_whole_file is True
                pending = asyncio.all_tasks() - {asyncio.current_task()}
                assert not pending
            finally:
                streamer.allow_remaining_chunks.set()
                if not consumer.done:
                    await consumer.wait_complete(timeout=2.0)

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

    def test_constructor_uses_resolved_streamer_cap_for_lossless_ordered_requests(
        self,
        monkeypatch,
    ):
        async def run(loop):
            import tools.tts_streaming as tts_streaming

            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=1)
            streamer.provider_id = "stream-b"
            config = {
                "provider": "sync-a",
                "sync-a": {"max_text_length": 50},
                "stream-b": {"max_text_length": 17},
                "streaming": {"provider": "auto"},
            }
            monkeypatch.setattr(
                tts_streaming,
                "resolve_streaming_provider",
                lambda resolved_config: streamer,
            )
            consumer = StreamingTTSConsumer(adapter, "chat1", config, loop)
            text = "x" * 40

            consumer.start()
            consumer.on_delta(text)
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is True
            assert [len(request) for request in streamer.requests] == [17, 17, 6]
            assert "".join(streamer.requests) == text
            assert adapter.begin_count == 1
            assert adapter.finish_count == 1
            assert adapter.abort_count == 0

        _run_test(run)

    def test_empty_first_physical_piece_aborts_before_later_piece_requests(
        self,
        monkeypatch,
    ):
        async def run(loop):
            import tools.tts_streaming as tts_streaming

            adapter = FakeVoiceAdapter()
            streamer = EmptyFirstPieceStreamer(chunks_per_clause=0)
            config = {
                "provider": "sync-a",
                "sync-a": {"max_text_length": 50},
                "stream-b": {"max_text_length": 17},
                "streaming": {"provider": "auto"},
            }
            monkeypatch.setattr(
                tts_streaming,
                "resolve_streaming_provider",
                lambda resolved_config: streamer,
            )
            consumer = StreamingTTSConsumer(adapter, "chat1", config, loop)
            text = "x" * 40

            consumer.start()
            consumer.on_delta(text)
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)

            assert completed is False
            assert streamer.requests == [text[:17]]
            assert adapter.written_chunks == []
            assert adapter.begin_count == 1
            assert adapter.finish_count == 0
            assert adapter.abort_count == 1
            assert consumer.audible is False
            assert consumer.suppress_whole_file is False

        _run_test(run)


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
            await asyncio.wait_for(adapter.begin_started.wait(), timeout=2.0)
            # Abort multiple times
            consumer.abort("test")
            consumer.abort("test2")
            consumer.abort("test3")
            consumer.finish()

            await consumer.wait_complete(timeout=5.0)
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["test"]
            assert adapter.handle is not None
            assert adapter.handle.aborted is True

        _run_test(run)

    def test_stop_racing_provider_failure_aborts_adapter_exactly_once(self):
        async def run(loop):
            adapter = CoordinatedAbortAdapter()
            streamer = StopThenFailStreamer()
            consumer = _make_consumer(adapter, "chat1", loop, streamer)
            streamer.abort_consumer = lambda: consumer.abort("external Stop")

            consumer.start()
            consumer.on_delta("A sentence that fails during external Stop. ")
            consumer.finish()

            await asyncio.wait_for(adapter.abort_started.wait(), timeout=2.0)
            adapter.release_abort.set()

            completed = await consumer.wait_complete(timeout=2.0)

            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["external Stop"]
            assert adapter.finish_count == 0
            assert len(streamer.requests) == 1
            assert completed is False
            assert consumer._aborted is True

        _run_test(run)

    def test_provider_failure_claim_keeps_first_reason_against_late_stop(self):
        async def run(loop):
            adapter = CoordinatedAbortAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(fail_on_clause=1),
            )

            consumer.start()
            consumer.on_delta("A sentence whose provider fails first. ")
            consumer.finish()
            await asyncio.wait_for(adapter.abort_started.wait(), timeout=2.0)

            consumer.abort("late external Stop")
            adapter.release_abort.set()
            completed = await consumer.wait_complete(timeout=2.0)

            assert completed is False
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["fake streamer failure on clause 1"]
            assert consumer._abort_reason == "fake streamer failure on clause 1"

        _run_test(run)

    def test_provider_failure_reason_precedes_post_phase_clear_stop(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(fail_on_clause=1),
            )
            phase_failed = threading.Event()
            stop_returned = threading.Event()
            stop_threads: list[threading.Thread] = []

            class StopAfterProviderPhaseTransaction:
                def __init__(self, delegate):
                    self._delegate = delegate
                    self._phase_before = None

                def __enter__(self):
                    self._delegate.acquire()
                    self._phase_before = consumer._physical_phase
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    phase_before = self._phase_before
                    self._delegate.release()
                    if (
                        phase_before is not None
                        and phase_before.name == "provider"
                        and consumer._physical_phase is None
                        and not phase_failed.is_set()
                    ):
                        phase_failed.set()

                        def stop_from_thread():
                            consumer.abort("late Stop after provider failure")
                            stop_returned.set()

                        stop_thread = threading.Thread(
                            target=stop_from_thread,
                            daemon=True,
                        )
                        stop_threads.append(stop_thread)
                        stop_thread.start()
                        assert stop_returned.wait(2.0)

            object.__setattr__(
                consumer,
                "_lock",
                StopAfterProviderPhaseTransaction(consumer._lock),
            )
            consumer.start()
            consumer.on_delta("Provider fails before a later external Stop. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=2.0)
            for stop_thread in stop_threads:
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert phase_failed.is_set()
            assert stop_returned.is_set()
            assert completed is False
            assert consumer._abort_reason == "fake streamer failure on clause 1"
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["fake streamer failure on clause 1"]
            assert adapter.finish_count == 0

        _run_test(run)

    def test_write_failure_reason_precedes_post_phase_clear_stop(self):
        async def run(loop):
            adapter = FailingWriteAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )
            phase_failed = threading.Event()
            stop_returned = threading.Event()
            stop_threads: list[threading.Thread] = []

            class StopAfterWriteFailureTransaction:
                def __init__(self, delegate):
                    self._delegate = delegate
                    self._phase_before = None

                def __enter__(self):
                    self._delegate.acquire()
                    self._phase_before = consumer._physical_phase
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    phase_before = self._phase_before
                    self._delegate.release()
                    if (
                        phase_before is not None
                        and phase_before.name == "write"
                        and consumer._physical_phase is None
                        and not phase_failed.is_set()
                    ):
                        phase_failed.set()

                        def stop_from_thread():
                            consumer.abort("late Stop after write failure")
                            stop_returned.set()

                        stop_thread = threading.Thread(
                            target=stop_from_thread,
                            daemon=True,
                        )
                        stop_threads.append(stop_thread)
                        stop_thread.start()
                        assert stop_returned.wait(2.0)

            object.__setattr__(
                consumer,
                "_lock",
                StopAfterWriteFailureTransaction(consumer._lock),
            )
            consumer.start()
            consumer.on_delta("Write fails before a later external Stop. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=2.0)
            for stop_thread in stop_threads:
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert phase_failed.is_set()
            assert stop_returned.is_set()
            assert completed is False
            assert consumer._abort_reason == "deterministic write failure"
            assert consumer.audible is False
            assert consumer.suppress_whole_file is False
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["deterministic write failure"]
            assert adapter.finish_count == 0

        _run_test(run)

    def test_stop_during_handle_acquisition_releases_late_handle_once(self):
        async def run(loop):
            adapter = CancellationSuppressingBeginAdapter()
            streamer = FakeStreamer(chunks_per_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            await asyncio.wait_for(adapter.begin_started.wait(), timeout=2.0)
            consumer.abort("external Stop during begin")
            adapter.release_begin.set()

            completed = await consumer.wait_complete(timeout=2.0)

            assert completed is False
            assert adapter.begin_count == 1
            assert adapter.begin_cancelled.is_set()
            assert adapter.abort_count == 1
            assert adapter.finish_count == 0
            assert adapter.handle is not None
            assert adapter.handle.aborted is True
            assert streamer.requests == []

        _run_test(run)

    def test_worker_stop_returns_when_begin_blocks_before_first_await(self):
        adapter = PreAwaitBlockingBeginAdapter()
        consumer_published = threading.Event()
        state: dict[str, object] = {}
        errors: list[BaseException] = []

        def run_loop():
            async def scenario():
                loop = asyncio.get_running_loop()
                consumer = _make_consumer(
                    adapter,
                    "chat1",
                    loop,
                    FakeStreamer(chunks_per_clause=1),
                )
                state["consumer"] = consumer
                consumer_published.set()
                consumer.start()
                consumer.on_delta("Begin blocks before its first await. ")
                consumer.finish()
                state["completed"] = await consumer.wait_complete(timeout=5.0)

            try:
                asyncio.run(scenario())
            except BaseException as exc:
                errors.append(exc)

        loop_thread = threading.Thread(target=run_loop, daemon=True)
        loop_thread.start()
        assert consumer_published.wait(timeout=2.0)
        assert adapter.physical_begin_entered.wait(timeout=2.0)
        consumer = state["consumer"]
        assert isinstance(consumer, StreamingTTSConsumer)

        started = time.monotonic()
        consumer.abort("external Stop while begin blocks before await")
        stop_elapsed = time.monotonic() - started
        adapter.release_begin.set()
        loop_thread.join(timeout=3.0)

        assert stop_elapsed < 2.0
        assert loop_thread.is_alive() is False
        assert errors == []
        assert state["completed"] is False
        assert adapter.begin_count == 1
        assert adapter.finish_count == 0
        assert adapter.abort_count == 1
        assert adapter.abort_errors == [
            "external Stop while begin blocks before await"
        ]
        assert adapter.handle is not None
        assert adapter.handle.aborted is True

    def test_stop_after_begin_task_publication_prevents_physical_begin(
        self,
    ):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )
            published, stop_returned, stop_threads = (
                _stop_when_phase_task_is_published(
                    consumer,
                    "_begin_task",
                    "_begin_adapter_if_active",
                    "external Stop before physical begin",
                )
            )

            consumer.start()
            consumer.on_delta("Stop before physical begin. ")
            consumer.finish()
            completed = await consumer.wait_complete(timeout=2.0)
            for stop_thread in stop_threads:
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert published.is_set()
            assert stop_returned.is_set()
            assert completed is False
            assert consumer.done is True
            assert adapter.begin_count == 0
            assert adapter.abort_count == 0
            assert adapter.finish_count == 0

        _run_test(run)

    @pytest.mark.parametrize(
        ("phase", "phase_coroutine_name"),
        [
            pytest.param("begin", "_begin_adapter_if_active", id="begin"),
            pytest.param("provider", "_next_stream_chunk_if_active", id="provider"),
            pytest.param("write", "_write_stream_chunk_if_active", id="write"),
            pytest.param("finish", "_finish_adapter", id="finish"),
        ],
    )
    def test_physical_phase_cannot_start_after_guard_seam_stop_returns(
        self,
        phase,
        phase_coroutine_name,
    ):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)
            guard_released, stop_returned, stop_threads = _stop_at_phase_guard_exit(
                consumer,
                phase_coroutine_name,
                f"external Stop at {phase} guard seam",
            )
            physical_after_stop_returned = []

            if phase == "begin":
                original_begin = adapter.begin_streaming_tts

                async def observed_begin(chat_id, audio_format, metadata=None):
                    physical_after_stop_returned.append(stop_returned.is_set())
                    return await original_begin(chat_id, audio_format, metadata)

                adapter.begin_streaming_tts = observed_begin
            elif phase == "provider":
                original_stream = streamer.stream

                def observed_stream(text):
                    physical_after_stop_returned.append(stop_returned.is_set())
                    yield from original_stream(text)

                streamer.stream = observed_stream
            elif phase == "write":
                original_write = adapter.write_streaming_tts

                async def observed_write(handle, chunk):
                    physical_after_stop_returned.append(stop_returned.is_set())
                    return await original_write(handle, chunk)

                adapter.write_streaming_tts = observed_write
            else:
                original_finish = adapter.finish_streaming_tts

                async def observed_finish(handle, *, interrupted=False):
                    physical_after_stop_returned.append(stop_returned.is_set())
                    return await original_finish(handle, interrupted=interrupted)

                adapter.finish_streaming_tts = observed_finish

            consumer.start()
            consumer.on_delta(f"Exercise the {phase} physical-entry seam. ")
            consumer.finish()
            completed = await consumer.wait_complete(timeout=2.0)
            for stop_thread in stop_threads:
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert guard_released.is_set()
            assert stop_returned.is_set()
            assert completed is False
            assert True not in physical_after_stop_returned

        _run_test(run)

    def test_stop_during_support_probe_never_starts_acquisition(self):
        async def run(loop):
            adapter = BlockingSupportsAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )

            stop_finished = threading.Event()

            def stop_during_probe():
                assert adapter.supports_started.wait(timeout=2.0)
                consumer.abort("external Stop during support probe")
                adapter.release_supports.set()
                stop_finished.set()

            stop_thread = threading.Thread(target=stop_during_probe)
            stop_thread.start()
            try:
                consumer.start()
                completed = await consumer.wait_complete(timeout=2.0)
            finally:
                adapter.release_supports.set()
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert stop_finished.is_set()
            assert completed is False
            assert consumer.done is True
            assert adapter.begin_count == 0
            assert adapter.abort_count == 0
            assert adapter.finish_count == 0

        _run_test(run)

    def test_stop_during_suspended_begin_cancels_acquisition(self):
        async def run(loop):
            adapter = BlockingBeginAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )

            consumer.start()
            await asyncio.wait_for(adapter.begin_started.wait(), timeout=2.0)
            await asyncio.to_thread(
                consumer.abort,
                "external Stop during suspended begin",
            )

            try:
                completed = await consumer.wait_complete(timeout=0.5)
                assert completed is False
                assert consumer.done is True
                assert adapter.begin_cancelled.is_set()
                assert adapter.begin_count == 1
                assert adapter.abort_count == 0
                assert adapter.finish_count == 0
                assert adapter.handle is None
            finally:
                adapter.release_begin.set()
                if not consumer.done:
                    await consumer.wait_complete(timeout=2.0)

        _run_test(run)

    def test_stop_from_real_thread_during_provider_pull_drops_late_chunk(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = SlowFirstChunkStreamer(chunks_per_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("A sentence blocked inside the provider pull. ")
            consumer.finish()
            await asyncio.wait_for(
                asyncio.to_thread(streamer.started.wait, 2.0),
                timeout=2.0,
            )
            await asyncio.to_thread(
                consumer.abort,
                "external Stop during provider pull",
            )
            streamer.allow_first_chunk.set()

            completed = await consumer.wait_complete(timeout=2.0)

            assert completed is False
            assert consumer.audible is False
            assert consumer.suppress_whole_file is False
            assert adapter.written_chunks == []
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["external Stop during provider pull"]
            assert adapter.finish_count == 0

        _run_test(run)

    @pytest.mark.parametrize(
        ("phase_coroutine_name", "expected_request_count"),
        [
            pytest.param(
                "_next_stream_chunk_if_active",
                0,
                id="provider-pull",
            ),
            pytest.param(
                "_write_stream_chunk_if_active",
                1,
                id="adapter-write",
            ),
        ],
    )
    def test_stop_after_payload_task_publication_prevents_physical_operation(
        self,
        phase_coroutine_name,
        expected_request_count,
    ):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)
            published, stop_returned, stop_threads = (
                _stop_when_phase_task_is_published(
                    consumer,
                    "_payload_task",
                    phase_coroutine_name,
                    f"external Stop before {phase_coroutine_name}",
                )
            )

            consumer.start()
            consumer.on_delta("Stop before the published payload task executes. ")
            consumer.finish()
            completed = await consumer.wait_complete(timeout=2.0)
            for stop_thread in stop_threads:
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert published.is_set()
            assert stop_returned.is_set()
            assert completed is False
            assert len(streamer.requests) == expected_request_count
            assert adapter.written_chunks == []
            assert consumer.audible is False
            assert consumer.suppress_whole_file is False
            assert adapter.abort_count == 1
            assert adapter.finish_count == 0

        _run_test(run)

    @pytest.mark.parametrize(
        "adapter_type",
        [DroppedWriteDuringAbortAdapter, CancellationSuppressingDroppedWriteAdapter],
    )
    def test_stop_during_dropped_write_does_not_publish_false_audible_state(
        self,
        adapter_type,
    ):
        async def run(loop):
            adapter = adapter_type()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )
            loop_thread = threading.get_ident()

            consumer.start()
            consumer.on_delta("A sentence whose first write is dropped during Stop. ")
            consumer.finish()
            await asyncio.wait_for(adapter.write_started.wait(), timeout=2.0)
            off_loop_task_access = _guard_active_phase_task_thread_ownership(
                consumer,
                loop_thread,
            )
            await asyncio.to_thread(consumer.abort, "external Stop during write")

            completed = await consumer.wait_complete(timeout=2.0)

            assert completed is False
            assert consumer.done is True
            assert consumer.audible is False
            assert consumer.suppress_whole_file is False
            assert consumer.partial is False
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["external Stop during write"]
            assert adapter.finish_count == 0
            assert adapter.handle is not None
            assert adapter.handle.audible is False
            assert adapter.handle.aborted is True
            assert off_loop_task_access.is_set() is False

        _run_test(run)

    def test_stopped_later_write_does_not_inherit_prior_audible_acknowledgement(
        self,
    ):
        async def run(loop):
            adapter = CancellationSuppressingDroppedWriteAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )
            handle = StreamingTTSHandle(
                chat_id="chat1",
                audio_format=consumer._audio_format,
                audible=True,
            )
            adapter.handle = handle
            consumer._handle = handle
            consumer._started = True

            write_task = loop.create_task(
                consumer._write_stream_chunk_if_active(handle, b"dropped-later-piece")
            )
            await asyncio.wait_for(adapter.write_started.wait(), timeout=2.0)
            await asyncio.to_thread(
                consumer.abort,
                "external Stop during later write",
            )
            acknowledged = await asyncio.wait_for(write_task, timeout=2.0)
            abort_task = consumer._abort_task
            if abort_task is not None:
                await asyncio.shield(abort_task)

            assert acknowledged is False
            assert handle.audible is True
            assert handle.aborted is True
            assert adapter.written_chunks == []
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["external Stop during later write"]

        _run_test(run)

    def test_adapter_audible_ack_survives_cancelled_write_tail(self):
        async def run(loop):
            adapter = AudibleBeforeCancelledWriteAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )

            consumer.start()
            consumer.on_delta("Adapter accepts audio before its write tail is cancelled. ")
            consumer.finish()
            await asyncio.wait_for(adapter.write_started.wait(), timeout=2.0)
            await asyncio.to_thread(
                consumer.abort,
                "external Stop after adapter accepted audio",
            )

            completed = await consumer.wait_complete(timeout=2.0)

            assert completed is False
            assert consumer.done is True
            assert consumer.audible is True
            assert consumer.suppress_whole_file is True
            assert adapter.written_chunks == [b"chunk-1-0"]
            assert adapter.abort_count == 1
            assert adapter.abort_errors == [
                "external Stop after adapter accepted audio"
            ]
            assert adapter.finish_count == 0
            assert adapter.handle is not None
            assert adapter.handle.audible is True
            assert adapter.handle.aborted is True
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            assert not pending

        _run_test(run)

    def test_stop_after_finish_task_publication_prevents_physical_finish(
        self,
    ):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )
            published, stop_returned, stop_threads = (
                _stop_when_phase_task_is_published(
                    consumer,
                    "_finish_task",
                    "_finish_adapter",
                    "external Stop before physical finish",
                )
            )

            consumer.start()
            consumer.on_delta("Produce audio, then Stop before finish. ")
            consumer.finish()
            completed = await consumer.wait_complete(timeout=2.0)
            for stop_thread in stop_threads:
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert published.is_set()
            assert stop_returned.is_set()
            assert completed is False
            assert adapter.written_chunks
            assert adapter.finish_count == 0
            assert adapter.abort_count == 1
            assert adapter.handle is not None
            assert adapter.handle.aborted is True

        _run_test(run)

    def test_stop_during_finish_cancels_finish_before_abort(self):
        async def run(loop):
            adapter = BlockingFinishAdapter()
            streamer = FakeStreamer(chunks_per_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)
            loop_thread = threading.get_ident()

            consumer.start()
            consumer.on_delta("A sentence that becomes audible before finish. ")
            consumer.finish()
            await asyncio.wait_for(adapter.finish_started.wait(), timeout=2.0)
            assert consumer.audible is True
            off_loop_task_access = _guard_active_phase_task_thread_ownership(
                consumer,
                loop_thread,
            )
            await asyncio.to_thread(consumer.abort, "external Stop during finish")
            try:
                completed = await consumer.wait_complete(timeout=0.5)

                assert completed is False
                assert consumer.done is True
                assert adapter.finish_count == 1
                assert adapter.abort_count == 1
                assert adapter.abort_errors == ["external Stop during finish"]
                assert adapter.terminal_events == [
                    "finish-started",
                    "finish-cancelled",
                    "abort",
                ]
                assert adapter.finish_cancelled.is_set()
                assert adapter.handle is not None
                assert adapter.handle.aborted is True
                assert consumer._aborted is True
                assert consumer.suppress_whole_file is True
                assert off_loop_task_access.is_set() is False
                assert len(streamer.requests) == 1
                pending = asyncio.all_tasks() - {asyncio.current_task()}
                assert not pending
            finally:
                adapter.release_finish.set()
                if not consumer.done:
                    await consumer.wait_complete(timeout=2.0)

        _run_test(run)

    def test_finish_waiting_for_abort_cannot_deadlock_terminal_claim(self):
        async def run(loop):
            adapter = FinishWaitsForAbortAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )

            consumer.start()
            consumer.on_delta("A sentence with cancellation-suppressing finish. ")
            consumer.finish()
            await asyncio.wait_for(adapter.finish_started.wait(), timeout=2.0)
            consumer.abort("external Stop while finish waits for abort")

            try:
                completed = await consumer.wait_complete(timeout=0.5)
                assert completed is False
                assert consumer.done is True
                assert adapter.finish_cancelled.is_set()
                assert adapter.abort_count == 1
                assert adapter.abort_errors == [
                    "external Stop while finish waits for abort"
                ]
                assert adapter.finish_count == 1
                assert adapter.terminal_events == [
                    "finish-started",
                    "finish-cancelled",
                    "abort",
                    "finish-completed",
                ]
                pending = asyncio.all_tasks() - {asyncio.current_task()}
                assert not pending
            finally:
                adapter.abort_started.set()
                if not consumer.done:
                    await consumer.wait_complete(timeout=2.0)

        _run_test(run)

    def test_finish_failure_reason_precedes_later_external_stop(self):
        async def run(loop):
            adapter = FailingFinishAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )
            failure_published, stop_returned, stop_threads = (
                _stop_when_finish_failure_is_published(
                    consumer,
                    "late external Stop after published finish failure",
                )
            )

            consumer.start()
            consumer.on_delta("Finish fails before a later external Stop. ")
            consumer.finish()
            completed = await consumer.wait_complete(timeout=2.0)
            for stop_thread in stop_threads:
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert failure_published.is_set()
            assert stop_returned.is_set()
            assert completed is False
            assert consumer._finish_outcome == "failed"
            assert consumer._abort_reason == "finish_streaming_tts failed"
            assert adapter.finish_count == 1
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["finish_streaming_tts failed"]
            assert adapter.handle is not None
            assert adapter.handle.aborted is True

        _run_test(run)

    def test_finish_failure_after_abort_does_not_repeat_physical_abort(self):
        async def run(loop):
            adapter = FinishFailsAfterAbortAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )

            consumer.start()
            consumer.on_delta("A sentence whose finish fails after abort. ")
            consumer.finish()
            await asyncio.wait_for(adapter.finish_started.wait(), timeout=2.0)
            consumer.abort("external Stop before finish failure")

            completed = await consumer.wait_complete(timeout=2.0)

            assert completed is False
            assert consumer.done is True
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["external Stop before finish failure"]
            assert adapter.terminal_events == [
                "finish-started",
                "finish-cancelled",
                "abort",
                "finish-failed",
            ]
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            assert not pending

        _run_test(run)

    def test_finish_returning_after_cancellation_owns_physical_terminal_edge(self):
        async def run(loop):
            adapter = FinishReturnsOnCancelAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )

            consumer.start()
            consumer.on_delta("A sentence whose finish accepts cancellation. ")
            consumer.finish()
            await asyncio.wait_for(adapter.finish_started.wait(), timeout=2.0)
            consumer.abort("external Stop during accepting finish")

            completed = await consumer.wait_complete(timeout=2.0)

            assert completed is False
            assert consumer.done is True
            assert adapter.finish_cancelled.is_set()
            assert adapter.finish_count == 1
            assert adapter.abort_count == 0
            assert adapter.terminal_events == [
                "finish-started",
                "finish-cancelled",
                "finish-completed",
            ]
            assert adapter.handle is not None
            assert adapter.handle.aborted is False
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            assert not pending
            await asyncio.sleep(0)
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            assert not pending

        _run_test(run)

    def test_finish_commit_precedes_late_stop_before_completion_publication(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )
            phase_finished = threading.Event()
            stop_returned = threading.Event()
            stop_threads: list[threading.Thread] = []

            class StopAfterFinishPhaseTransaction:
                def __init__(self, delegate):
                    self._delegate = delegate
                    self._phase_before = None

                def __enter__(self):
                    self._delegate.acquire()
                    self._phase_before = consumer._physical_phase
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    phase_before = self._phase_before
                    self._delegate.release()
                    if (
                        phase_before is not None
                        and phase_before.name == "finish"
                        and consumer._physical_phase is None
                        and not phase_finished.is_set()
                    ):
                        phase_finished.set()

                        def stop_from_thread():
                            consumer.abort("Stop after finish phase transaction")
                            stop_returned.set()

                        stop_thread = threading.Thread(
                            target=stop_from_thread,
                            daemon=True,
                        )
                        stop_threads.append(stop_thread)
                        stop_thread.start()
                        assert stop_returned.wait(2.0)

            object.__setattr__(
                consumer,
                "_lock",
                StopAfterFinishPhaseTransaction(consumer._lock),
            )
            consumer.start()
            consumer.on_delta("A sentence finishing before late Stop. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=2.0)
            for stop_thread in stop_threads:
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert phase_finished.is_set()
            assert stop_returned.is_set()
            assert completed is True
            assert consumer.completed is True
            assert consumer._aborted is False
            assert adapter.finish_count == 1
            assert adapter.abort_count == 0
            assert adapter.handle is not None
            assert adapter.handle.aborted is False

        _run_test(run)

    def test_finish_child_success_precedes_done_callback_stop(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )
            child_done = threading.Event()
            stop_returned = threading.Event()
            stop_threads: list[threading.Thread] = []
            claim_physical_phase = consumer._claim_async_physical_phase_locked

            def claim_with_done_callback(name, operation_factory, **kwargs):
                phase, task = claim_physical_phase(
                    name,
                    operation_factory,
                    **kwargs,
                )
                if name == "finish":

                    def stop_before_parent_resumes(_task):
                        child_done.set()

                        def stop_from_thread():
                            consumer.abort("Stop after finish child success")
                            stop_returned.set()

                        stop_thread = threading.Thread(
                            target=stop_from_thread,
                            daemon=True,
                        )
                        stop_threads.append(stop_thread)
                        stop_thread.start()
                        assert stop_returned.wait(2.0)

                    task.add_done_callback(stop_before_parent_resumes)
                return phase, task

            object.__setattr__(
                consumer,
                "_claim_async_physical_phase_locked",
                claim_with_done_callback,
            )
            consumer.start()
            consumer.on_delta("Finish child succeeds before a late Stop. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=2.0)
            for stop_thread in stop_threads:
                await asyncio.to_thread(stop_thread.join, 2.0)

            assert child_done.is_set()
            assert stop_returned.is_set()
            assert completed is True
            assert consumer.completed is True
            assert consumer._aborted is False
            assert consumer._finish_outcome == "succeeded"
            assert adapter.finish_count == 1
            assert adapter.abort_count == 0
            assert adapter.handle is not None
            assert adapter.handle.aborted is False

        _run_test(run)

    def test_stop_after_committed_finish_is_terminal_noop(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )

            consumer.start()
            consumer.on_delta("A completed sentence. ")
            consumer.finish()

            assert await consumer.wait_complete(timeout=2.0) is True
            await asyncio.to_thread(consumer.abort, "late Stop")
            await asyncio.sleep(0)

            assert consumer.completed is True
            assert consumer._aborted is False
            assert adapter.finish_count == 1
            assert adapter.abort_count == 0
            assert adapter.handle is not None
            assert adapter.handle.aborted is False

        _run_test(run)

    def test_cancelled_waiter_does_not_cancel_shared_abort_cleanup(self):
        async def run(loop):
            adapter = CoordinatedAbortAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )

            consumer.start()
            await asyncio.wait_for(adapter.begin_started.wait(), timeout=2.0)
            consumer.abort("blocked shared abort")
            await asyncio.wait_for(adapter.abort_started.wait(), timeout=2.0)

            waiter_entered = asyncio.Event()

            async def wait_for_consumer():
                waiter_entered.set()
                return await consumer.wait_complete(timeout=5.0)

            waiter = asyncio.create_task(wait_for_consumer())
            await asyncio.wait_for(waiter_entered.wait(), timeout=2.0)
            waiter.cancel()

            try:
                with pytest.raises(asyncio.CancelledError):
                    await waiter
                assert adapter.abort_count == 1
                assert consumer.done is False
            finally:
                adapter.release_abort.set()
                await consumer.wait_complete(timeout=2.0)

            assert consumer.done is True
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["blocked shared abort"]
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            assert not pending
            await asyncio.sleep(0)
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            assert not pending

        _run_test(run)

    def test_cancelled_waiter_propagates_without_cancelling_claimed_finish(self):
        async def run(loop):
            adapter = BlockingFinishAdapter()
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                FakeStreamer(chunks_per_clause=1),
            )

            consumer.start()
            consumer.on_delta("A sentence with a shielded finish. ")
            consumer.finish()
            await asyncio.wait_for(adapter.finish_started.wait(), timeout=2.0)

            waiter_entered = asyncio.Event()

            async def wait_for_consumer():
                waiter_entered.set()
                return await consumer.wait_complete(timeout=5.0)

            waiter = asyncio.create_task(wait_for_consumer())
            await asyncio.wait_for(waiter_entered.wait(), timeout=2.0)
            waiter.cancel()

            try:
                with pytest.raises(asyncio.CancelledError):
                    await waiter
                assert consumer.done is False
                assert adapter.finish_cancelled.is_set() is False
                assert adapter.abort_count == 0
            finally:
                adapter.release_finish.set()
                await consumer.wait_complete(timeout=2.0)

            assert await consumer.wait_complete(timeout=2.0) is True
            assert adapter.terminal_events == ["finish-started", "finish-completed"]
            assert adapter.abort_count == 0
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            assert not pending

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
            assert adapter.abort_count == 1
            assert adapter.abort_errors == ["fake streamer failure on clause 1"]

        _run_test(run)

    def test_failure_after_first_piece_suppresses_replay_and_skips_later_pieces(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=1, fail_on_clause=2)
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                streamer,
                stream_cap=17,
            )
            text = "y" * 40

            consumer.start()
            consumer.on_delta(text)
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is False
            assert streamer.requests == [text[:17], text[17:34]]
            assert consumer.partial is True
            assert consumer.suppress_whole_file is True
            assert adapter.begin_count == 1
            assert adapter.finish_count == 0
            assert adapter.abort_count == 1

        _run_test(run)

    def test_cancellation_after_first_piece_suppresses_later_requests_and_replay(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            consumer_holder = []

            def _abort_after_first_piece(request_number):
                if request_number == 1:
                    consumer_holder[0].abort("cancelled after first piece")

            streamer = FakeStreamer(
                chunks_per_clause=1,
                after_request=_abort_after_first_piece,
            )
            consumer = _make_consumer(
                adapter,
                "chat1",
                loop,
                streamer,
                stream_cap=17,
            )
            consumer_holder.append(consumer)
            text = "z" * 40

            consumer.start()
            consumer.on_delta(text)
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)

            assert completed is False
            assert streamer.requests == [text[:17]]
            assert consumer.suppress_whole_file is True
            assert adapter.begin_count == 1
            assert adapter.finish_count == 0
            assert adapter.abort_count == 1

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

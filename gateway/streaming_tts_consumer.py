"""Gateway streaming-TTS consumer — LLM deltas to adapter PCM audio sink.

Bridges the synchronous agent ``stream_delta_callback`` (fired from the
worker thread) to a voice-capable platform adapter's streaming-audio
contract, so playback begins while the LLM is still generating.

Lifecycle::

    consumer = StreamingTTSConsumer(adapter, chat_id, tts_config, loop, metadata)
    agent.stream_delta_callback = consumer.on_delta   # sync, non-blocking
    ... agent runs in executor ...
    consumer.finish()            # signal end-of-text
    success = await consumer.wait_complete(timeout=10)
    if consumer.suppress_whole_file:
        # suppress whole-file auto-TTS for this turn
    consumer.abort("cancelled")  # idempotent cancellation

Design:
- ``on_delta`` is synchronous and never blocks the agent thread. It feeds
  deltas into a ``SentenceChunker`` and queues completed clauses onto a
  thread-safe ``queue.Queue``.
- An asyncio task (``run``) runs on the gateway event loop, draining the
  queue, synthesising each clause via a ``StreamingTTSProvider``, and
  writing PCM chunks to the adapter.
- Per-turn state is isolated: each consumer instance owns its own chunker,
  queue, handle, and flags. Concurrent chats cannot cross-contaminate.
- On successful completion (all clauses synthesised and written), the
  consumer reports ``completed=True`` so the gateway can suppress the
  duplicate whole-file auto-TTS.
- On failure before any audible output, the consumer reports
  ``completed=False`` and clears ``suppress_whole_file`` so the gateway can
  fall back to whole-file TTS.
- On failure after partial audible output, the consumer reports
  ``completed=False`` but keeps ``suppress_whole_file=True`` so the gateway
  does NOT replay the whole response from the beginning.
- Cancellation/abort is idempotent: late chunks are silently dropped.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Callable, Coroutine
from typing import Any, Dict, Optional

from gateway.platforms.base import AudioFormat, StreamingTTSHandle

logger = logging.getLogger("gateway.streaming_tts_consumer")

_ABORT = object()
_DONE = object()
_FINISH_FAILURE_REASON = "finish_streaming_tts failed"


class _PhysicalPhaseEntry:
    """One physical operation claimed against concurrent Stop."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.task: Optional[asyncio.Task[Any]] = None
        self.entered = False
        self.cancelled_before_entry = False
        self.outcome = "pending"
        self.write_accepted = False
        self.audible_before_write = False
        self.stop_preceded_write_commit = False
        self.resolved = threading.Event()


class StreamingTTSConsumer:
    """Consumes LLM text deltas and produces streaming PCM audio for an adapter."""

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        tts_config: Dict[str, Any],
        loop: asyncio.AbstractEventLoop,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        from tools.tts_streaming import (
            SentenceChunker,
            resolve_streaming_provider,
            resolve_streaming_text_limit,
        )

        self._adapter = adapter
        self._chat_id = chat_id
        self._tts_config = tts_config
        self._loop = loop
        self._metadata = metadata

        # Resolve the streaming provider once. If unavailable, the consumer is
        # inactive and the gateway falls back to whole-file TTS.
        self._streamer = resolve_streaming_provider(tts_config)
        self._stream_cap = 0
        if self._streamer is not None:
            try:
                self._stream_cap = resolve_streaming_text_limit(
                    self._streamer,
                    self._tts_config,
                )
            except Exception:
                from tools.tts_tool import FALLBACK_MAX_TEXT_LENGTH

                self._stream_cap = FALLBACK_MAX_TEXT_LENGTH
        self._chunker = SentenceChunker()

        if self._streamer is not None:
            self._audio_format = AudioFormat(
                sample_rate=int(getattr(self._streamer, "sample_rate", AudioFormat.sample_rate)),
                channels=int(getattr(self._streamer, "channels", AudioFormat.channels)),
                sample_width=int(getattr(self._streamer, "sample_width", AudioFormat.sample_width)),
            )
        else:
            self._audio_format = audio_format or AudioFormat()

        # Thread-safe queue: completed clauses and the occasional abort sentinel.
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=256)

        # Per-turn state.
        self._handle: Optional[StreamingTTSHandle] = None
        self._started = False
        self._completed = False
        self._partial = False
        self._aborted = False
        self._abort_reason: Optional[str] = None
        self._finished = False
        self._dropped = False
        self._suppress_whole_file = False
        self._task: Optional[asyncio.Task] = None
        self._begin_task: Optional[asyncio.Task[Optional[StreamingTTSHandle]]] = None
        self._payload_task: Optional[asyncio.Task[Any]] = None
        self._finish_task: Optional[asyncio.Task[None]] = None
        self._finish_outcome: Optional[str] = None
        self._abort_task: Optional[asyncio.Task[None]] = None
        self._physical_phase: Optional[_PhysicalPhaseEntry] = None
        self._lock = threading.Lock()

        # Pre-allocate the strip-markdown helper lazily to avoid import cycles.
        self._strip_markdown = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True when this consumer has a usable streaming provider."""
        return self._streamer is not None

    @property
    def completed(self) -> bool:
        """True when streaming audio was fully delivered."""
        return self._completed

    @property
    def partial(self) -> bool:
        """True when some audio was audible before a failure or drop."""
        return self._partial

    @property
    def started(self) -> bool:
        """True when the adapter accepted the streaming session."""
        return self._started

    @property
    def audible(self) -> bool:
        """True once the first PCM chunk has been written."""
        return bool(self._handle and self._handle.audible)

    @property
    def dropped(self) -> bool:
        """True when queue saturation dropped at least one clause."""
        return self._dropped

    @property
    def suppress_whole_file(self) -> bool:
        """True when the gateway should skip the legacy whole-file TTS fallback."""
        return self._suppress_whole_file

    @property
    def done(self) -> bool:
        """True once the async drain task has terminated."""
        return self._task is not None and self._task.done()

    # ------------------------------------------------------------------
    # Sync callback (agent worker thread)
    # ------------------------------------------------------------------

    def on_delta(self, text: str) -> None:
        """Receive a text delta from the agent. Non-blocking."""
        if self._aborted or not self.active or self._finished:
            return
        try:
            for clause in self._chunker.feed(text):
                self._queue.put_nowait(clause)
        except queue.Full:
            self._dropped = True
            logger.debug("streaming TTS queue full, dropping clause")
        except Exception:
            logger.debug("streaming TTS on_delta error", exc_info=True)

    def finish(self) -> None:
        """Signal end-of-text and flush the chunker tail.

        Enqueues a ``_DONE`` sentinel after all flushed clauses so the
        drain loop has a deterministic termination signal that cannot
        race with a late ``on_delta`` or be lost when the queue is full.
        """
        if self._finished:
            return
        self._finished = True
        if self._aborted or not self.active:
            return
        try:
            for clause in self._chunker.flush():
                self._queue.put_nowait(clause)
        except queue.Full:
            self._dropped = True
            logger.debug("streaming TTS queue full while flushing tail")
        except Exception:
            pass
        # Guarantee the _DONE sentinel reaches the queue.  If the bounded
        # queue is full, drain one item to make room — the sentinel is
        # load-bearing and must not be lost (#60671 hardening).
        self._enqueue_done()

    def _enqueue_done(self) -> None:
        """Enqueue the _DONE sentinel, evicting a queued clause if necessary."""
        while True:
            try:
                self._queue.put_nowait(_DONE)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._dropped = True
                except queue.Empty:
                    continue

    # ------------------------------------------------------------------
    # Async lifecycle (gateway event loop)
    # ------------------------------------------------------------------

    def start(self) -> asyncio.Task:
        """Create and return the async drain task on the gateway loop."""
        if self._task is not None:
            return self._task
        self._task = self._loop.create_task(self._run())
        return self._task

    async def _run(self) -> None:
        """Drain clauses from the queue, synthesise, and write to the adapter."""
        if not self.active or self._aborted:
            return

        if not self._adapter.supports_streaming_tts(self._chat_id, self._audio_format):
            logger.debug("adapter %s does not support streaming TTS", getattr(self._adapter, "name", "?"))
            return
        begin_task = self._publish_begin_task()
        if begin_task is None:
            return
        try:
            handle = await asyncio.shield(begin_task)
        except asyncio.CancelledError:
            if self._aborted:
                return
            raise
        except Exception as exc:
            logger.debug("begin_streaming_tts failed: %s", exc)
            return
        finally:
            if self._begin_task is begin_task:
                self._begin_task = None

        if handle is None:
            return
        self._handle = handle

        self._started = True
        self._suppress_whole_file = False
        if self._aborted:
            await self._safe_abort("cancelled")
            return

        try:
            while True:
                if self._aborted:
                    break
                try:
                    item = await asyncio.to_thread(self._queue.get, True, 0.1)
                except queue.Empty:
                    continue

                if item is _ABORT:
                    break
                if item is _DONE:
                    break
                if not isinstance(item, str):
                    continue
                if self._aborted:
                    break

                try:
                    await self._synthesise_and_write(item)
                except Exception as exc:
                    logger.warning("streaming TTS clause failed: %s", exc)
                    if self._handle and self._handle.audible:
                        self._partial = True
                        self._suppress_whole_file = True
                    else:
                        self._suppress_whole_file = False
                    self._completed = False
                    await self._safe_abort(str(exc))
                    return

            if self._aborted:
                await self._safe_abort("cancelled")
                return
            if self._handle is not None:
                with self._lock:
                    if self._aborted:
                        finish_task = None
                    else:
                        self._finish_outcome = "pending"
                        finish_task = self._loop.create_task(
                            self._finish_adapter(self._handle)
                        )
                        self._finish_task = finish_task
                if finish_task is None:
                    await self._safe_abort("cancelled")
                    return

                finish_failed = False
                try:
                    await asyncio.shield(finish_task)
                except asyncio.CancelledError:
                    finish_failed = True
                    with self._lock:
                        if not self._aborted:
                            self._aborted = True
                            self._abort_reason = "finish_streaming_tts cancelled"
                except Exception as exc:
                    logger.debug("finish_streaming_tts error: %s", exc)
                    finish_failed = True

                with self._lock:
                    aborted = self._aborted
                    if not aborted:
                        if finish_failed:
                            # finish_streaming_tts() failed — never report full
                            # completion.  Preserve suppression only after audio
                            # became audible, so pre-audio failure can fall back.
                            if self._handle.audible:
                                self._partial = True
                                self._completed = False
                                self._suppress_whole_file = True
                            else:
                                self._completed = False
                                self._suppress_whole_file = False
                        elif self._handle.audible and not self._dropped:
                            self._completed = True
                            self._suppress_whole_file = True
                        elif self._handle.audible and self._dropped:
                            self._partial = True
                            self._completed = False
                            self._suppress_whole_file = True
                        else:
                            self._completed = False
                            self._suppress_whole_file = False

                if aborted:
                    await self._safe_abort("cancelled")
                    return
                if finish_failed:
                    await self._safe_abort(_FINISH_FAILURE_REASON)
        except Exception as exc:
            logger.warning("streaming TTS consumer error: %s", exc)
            await self._safe_abort(str(exc))
        finally:
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except Exception:
                pass

    def _publish_begin_task(
        self,
    ) -> Optional[asyncio.Task[Optional[StreamingTTSHandle]]]:
        """Atomically publish handle acquisition against cross-thread Stop."""
        with self._lock:
            if self._aborted:
                return None
            task = self._loop.create_task(self._begin_adapter_if_active())
            self._begin_task = task
            return task

    def _claim_async_physical_phase_locked(
        self,
        name: str,
        operation_factory: Callable[[], Coroutine[Any, Any, Any]],
        *,
        handle: Optional[StreamingTTSHandle] = None,
    ) -> tuple[_PhysicalPhaseEntry, asyncio.Task[Any]]:
        """Queue a physical coroutine while Stop is excluded by ``_lock``."""
        phase = _PhysicalPhaseEntry(name)
        if name == "write" and handle is not None:
            phase.audible_before_write = handle.audible
        task = self._loop.create_task(
            self._run_async_physical_phase(phase, operation_factory, handle=handle)
        )
        phase.task = task
        self._physical_phase = phase
        return phase, task

    async def _run_async_physical_phase(
        self,
        phase: _PhysicalPhaseEntry,
        operation_factory: Callable[[], Coroutine[Any, Any, Any]],
        *,
        handle: Optional[StreamingTTSHandle],
    ) -> Any:
        """Publish entry and outcome before the physical child becomes done."""
        with self._lock:
            if self._aborted or (handle is not None and handle.aborted):
                phase.cancelled_before_entry = True
                phase.outcome = "cancelled"
                suppressed = True
            else:
                phase.entered = True
                suppressed = False
        phase.resolved.set()
        if suppressed:
            raise asyncio.CancelledError

        try:
            result = await operation_factory()
        except asyncio.CancelledError:
            with self._lock:
                phase.outcome = "cancelled"
                if (
                    phase.name == "write"
                    and handle is not None
                    and handle.audible
                    and not phase.audible_before_write
                ):
                    phase.write_accepted = True
                    self._suppress_whole_file = True
                if phase.name == "finish" and self._finish_outcome is None:
                    self._finish_outcome = "cancelled"
            raise
        except Exception as exc:
            with self._lock:
                phase.outcome = "failed"
                if (
                    phase.name == "write"
                    and handle is not None
                    and handle.audible
                    and not phase.audible_before_write
                ):
                    phase.write_accepted = True
                    self._suppress_whole_file = True
                if phase.name == "finish":
                    self._finish_outcome = "failed"
                    failure_reason = _FINISH_FAILURE_REASON
                else:
                    failure_reason = str(exc)
                if self._abort_reason is None:
                    self._abort_reason = failure_reason
            raise
        else:
            with self._lock:
                phase.outcome = "succeeded"
                if phase.name == "begin" and isinstance(result, StreamingTTSHandle):
                    self._handle = result
                elif phase.name == "write" and handle is not None:
                    phase.write_accepted = (
                        (
                            handle.audible
                            and not phase.audible_before_write
                        )
                        or not phase.stop_preceded_write_commit
                    )
                    if phase.write_accepted:
                        handle.audible = True
                        self._suppress_whole_file = True
                elif phase.name == "finish":
                    self._finish_outcome = "succeeded"
            return result

    def _clear_physical_phase(self, phase: _PhysicalPhaseEntry) -> None:
        with self._lock:
            if self._physical_phase is phase:
                self._physical_phase = None
        # A completed or cancelled operation is also a resolved entry claim.
        phase.resolved.set()

    def _next_stream_chunk_with_entry(
        self,
        phase: _PhysicalPhaseEntry,
        iterator: Any,
    ) -> tuple[bool, Optional[bytes]]:
        """Resolve provider entry from its executor thread before pulling."""
        with self._lock:
            handle = self._handle
            if self._aborted or (handle is not None and handle.aborted):
                phase.cancelled_before_entry = True
                phase.outcome = "cancelled"
                suppressed = True
            else:
                phase.entered = True
                suppressed = False
        phase.resolved.set()
        if suppressed:
            return False, None
        try:
            result = self._next_stream_chunk(iterator)
        except Exception as exc:
            with self._lock:
                phase.outcome = "failed"
                if self._abort_reason is None:
                    self._abort_reason = str(exc)
            raise
        else:
            with self._lock:
                phase.outcome = "succeeded"
            return result

    async def _begin_adapter_if_active(self) -> Optional[StreamingTTSHandle]:
        """Atomically claim physical acquisition against concurrent Stop."""
        with self._lock:
            if self._aborted:
                return None
            phase, operation_task = self._claim_async_physical_phase_locked(
                "begin",
                lambda: self._adapter.begin_streaming_tts(
                    self._chat_id,
                    self._audio_format,
                    metadata=self._metadata,
                ),
            )
        try:
            return await operation_task
        finally:
            self._clear_physical_phase(phase)

    def _publish_payload_task(
        self,
        factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> Optional[asyncio.Task[Any]]:
        """Atomically publish provider/write work against cross-thread Stop."""
        with self._lock:
            handle = self._handle
            if self._aborted or (handle is not None and handle.aborted):
                return None
            task = self._loop.create_task(factory())
            self._payload_task = task
            return task

    async def _next_stream_chunk_if_active(
        self,
        iterator: Any,
    ) -> tuple[bool, Optional[bytes]]:
        """Atomically claim one provider pull against concurrent Stop."""
        with self._lock:
            handle = self._handle
            if self._aborted or (handle is not None and handle.aborted):
                return False, None
            phase = _PhysicalPhaseEntry("provider")
            operation_task = self._loop.create_task(
                asyncio.to_thread(self._next_stream_chunk_with_entry, phase, iterator)
            )
            phase.task = operation_task
            self._physical_phase = phase
        try:
            return await operation_task
        finally:
            self._clear_physical_phase(phase)

    async def _write_stream_chunk_if_active(
        self,
        handle: StreamingTTSHandle,
        chunk: bytes,
    ) -> bool:
        """Claim one write and acknowledge it only on a live return."""
        with self._lock:
            if self._aborted or handle.aborted:
                return False
            phase, operation_task = self._claim_async_physical_phase_locked(
                "write",
                lambda: self._adapter.write_streaming_tts(handle, chunk),
                handle=handle,
            )
        try:
            await operation_task
        finally:
            self._clear_physical_phase(phase)
        return phase.write_accepted

    async def _synthesise_and_write(self, clause: str) -> None:
        """Synthesise one clause via the streamer and write PCM chunks."""
        handle = self._handle
        if self._aborted or handle is None or handle.aborted:
            return

        cleaned = self._strip_markdown_for_tts(clause)
        if not cleaned or not cleaned.strip():
            return

        if self._streamer is None:
            return

        from tools.tts_streaming import split_text_for_tts

        for piece in split_text_for_tts(cleaned, self._stream_cap):
            if self._aborted or handle.aborted:
                return
            piece_acknowledged = False
            async for chunk in self._iter_stream_chunks(piece):
                if self._aborted or handle.aborted:
                    return
                if not chunk:
                    continue
                payload_task = self._publish_payload_task(
                    lambda: self._write_stream_chunk_if_active(handle, chunk)
                )
                if payload_task is None:
                    return
                try:
                    acknowledged = await asyncio.shield(payload_task)
                except asyncio.CancelledError:
                    if self._aborted:
                        return
                    raise
                finally:
                    if self._payload_task is payload_task:
                        self._payload_task = None
                if not acknowledged:
                    return
                piece_acknowledged = True
                self._suppress_whole_file = True

            if (
                not piece_acknowledged
                and not self._aborted
                and not handle.aborted
            ):
                raise RuntimeError("streaming TTS provider produced no audio for text piece")

    async def _iter_stream_chunks(self, text: str):
        """Yield provider PCM chunks one at a time without blocking the loop."""
        if self._streamer is None:
            return
        iterator = iter(self._streamer.stream(text))
        while True:
            if self._aborted or (self._handle is not None and self._handle.aborted):
                break
            payload_task = self._publish_payload_task(
                lambda: self._next_stream_chunk_if_active(iterator)
            )
            if payload_task is None:
                break
            try:
                has_chunk, chunk = await asyncio.shield(payload_task)
            except asyncio.CancelledError:
                if self._aborted:
                    break
                raise
            finally:
                if self._payload_task is payload_task:
                    self._payload_task = None
            if not has_chunk:
                break
            yield chunk

    @staticmethod
    def _next_stream_chunk(iterator: Any) -> tuple[bool, Optional[bytes]]:
        try:
            return True, next(iterator)
        except StopIteration:
            return False, None

    def _strip_markdown_for_tts(self, text: str) -> str:
        """Lazy-import and apply the TTS markdown stripper."""
        if self._strip_markdown is None:
            try:
                from tools.tts_tool import _strip_markdown_for_tts as _strip
                self._strip_markdown = _strip
            except ImportError:
                self._strip_markdown = lambda t: t  # noqa: E731
        return self._strip_markdown(text).strip()

    async def _finish_adapter(self, handle: StreamingTTSHandle) -> None:
        """Publish adapter-finish outcome before the task becomes observable as done."""
        with self._lock:
            if self._aborted or handle.aborted:
                self._finish_outcome = "cancelled"
                raise asyncio.CancelledError
            phase, operation_task = self._claim_async_physical_phase_locked(
                "finish",
                lambda: self._adapter.finish_streaming_tts(
                    handle,
                    interrupted=False,
                ),
                handle=handle,
            )
        try:
            await operation_task
        finally:
            self._clear_physical_phase(phase)

    def _finish_succeeded_locked(self) -> bool:
        """Return whether adapter finish committed before Stop could win."""
        return self._finish_outcome == "succeeded"

    def _claim_abort_task(self, fallback_reason: str) -> Optional[asyncio.Task[None]]:
        """Atomically claim one serialized adapter-abort task on the loop."""
        with self._lock:
            abort_task = self._abort_task
            if abort_task is not None:
                return abort_task
            if (
                self._handle is None
                or self._completed
                or self._finish_succeeded_locked()
            ):
                return None
            if self._abort_reason is None:
                self._abort_reason = fallback_reason
            self._aborted = True
            reason = self._abort_reason
            abort_task = self._loop.create_task(
                self._cancel_finish_then_abort(self._finish_task, reason)
            )
            self._abort_task = abort_task
            return abort_task

    async def _safe_abort(self, fallback_reason: str) -> None:
        """Claim and await the operation's single serialized abort task."""
        abort_task = self._claim_abort_task(fallback_reason)
        if abort_task is not None:
            await asyncio.shield(abort_task)

    def _handle_abort_requested(self, fallback_reason: str) -> None:
        """Cancel loop-owned acquisition and claim adapter cleanup on the loop."""
        with self._lock:
            physical_phase = self._physical_phase
            if physical_phase is not None and not physical_phase.resolved.is_set():
                physical_task = physical_phase.task
                if physical_task is not None and physical_task.cancel():
                    physical_phase.cancelled_before_entry = True
                    physical_phase.outcome = "cancelled"
                    physical_phase.resolved.set()
        begin_task = self._begin_task
        begin_child_committed = (
            physical_phase is not None
            and physical_phase.name == "begin"
            and physical_phase.outcome == "succeeded"
        )
        if begin_task is not None and not begin_task.done() and not begin_child_committed:
            begin_task.cancel()
        payload_task = self._payload_task
        if payload_task is not None and not payload_task.done():
            payload_task.cancel()
        self._claim_abort_task(fallback_reason)

    async def _cancel_finish_then_abort(
        self,
        finish_task: Optional[asyncio.Task[None]],
        reason: str,
    ) -> None:
        """Cancel finish, then abort unless finish commits on that checkpoint."""
        abort_issued = False
        if finish_task is not None:
            with self._lock:
                physical_phase = self._physical_phase
                finish_child_committed = (
                    physical_phase is not None
                    and physical_phase.name == "finish"
                    and physical_phase.outcome == "succeeded"
                )
            if not finish_task.done():
                if not finish_child_committed:
                    finish_task.cancel()
                    await asyncio.sleep(0)
                with self._lock:
                    finish_child_committed = self._finish_succeeded_locked()
                if finish_child_committed:
                    try:
                        await asyncio.shield(finish_task)
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                    return
            with self._lock:
                finish_succeeded = self._finish_succeeded_locked()
            if finish_succeeded:
                return
            if not finish_task.done():
                await self._abort_adapter_once(reason)
                abort_issued = True
            try:
                await asyncio.shield(finish_task)
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            else:
                return
        if not abort_issued:
            await self._abort_adapter_once(reason)

    async def _abort_adapter_once(self, reason: str) -> None:
        """Abort the claimed adapter stream once, swallowing adapter errors."""
        handle = self._handle
        if handle is None:
            return
        try:
            await self._adapter.abort_streaming_tts(handle, error=reason)
        except Exception:
            pass
        finally:
            handle.aborted = True

    # ------------------------------------------------------------------
    # Cancellation and completion
    # ------------------------------------------------------------------

    def abort(self, reason: str = "cancelled") -> None:
        """Idempotent cancellation from any thread."""
        try:
            on_loop_thread = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop_thread = False

        claimed_reason: str
        physical_phase: Optional[_PhysicalPhaseEntry]
        schedule_abort = False
        with self._lock:
            active_phase = self._physical_phase
            physical_write_committed = (
                active_phase is not None
                and active_phase.name == "write"
                and not active_phase.stop_preceded_write_commit
                and active_phase.outcome == "succeeded"
                and active_phase.write_accepted
            )
            if physical_write_committed and self._handle is not None:
                self._handle.audible = True
                self._suppress_whole_file = True
            elif active_phase is not None and active_phase.name == "write":
                active_phase.stop_preceded_write_commit = True
            physical_finish_committed = (
                active_phase is not None
                and active_phase.name == "finish"
                and active_phase.outcome == "succeeded"
            )
            if physical_finish_committed:
                self._finish_outcome = "succeeded"
            if (
                self._completed
                or self._finish_succeeded_locked()
                or physical_finish_committed
            ):
                return
            if not self._aborted:
                self._aborted = True
                if self._abort_reason is None:
                    self._abort_reason = reason
                schedule_abort = True
            if self._abort_reason is None:
                self._abort_reason = reason
            claimed_reason = self._abort_reason
            physical_phase = self._physical_phase
            if schedule_abort and not on_loop_thread:
                try:
                    self._loop.call_soon_threadsafe(
                        self._handle_abort_requested,
                        claimed_reason,
                    )
                except Exception:
                    if physical_phase is not None:
                        physical_phase.cancelled_before_entry = True
                        physical_phase.resolved.set()
        if on_loop_thread:
            self._handle_abort_requested(claimed_reason)
        elif physical_phase is not None:
            physical_phase.resolved.wait()
        # Guarantee the _ABORT sentinel reaches the queue.  If the bounded
        # queue is full, drain one item to make room — the sentinel must
        # not be lost (#60671 hardening).
        for _attempt in range(3):
            try:
                self._queue.put_nowait(_ABORT)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        else:
            logger.debug("streaming TTS _ABORT sentinel could not be enqueued")


    async def wait_complete(self, timeout: float = 10.0) -> bool:
        """Wait for the drain task to finish. Returns True only on full success."""
        if self._task is None:
            return self._completed
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass
        return self._completed

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
from typing import Any, Dict, Optional

from gateway.platforms.base import AudioFormat, StreamingTTSHandle

logger = logging.getLogger("gateway.streaming_tts_consumer")

_ABORT = object()
_DONE = object()


class _PcmResampler:
    """Stateful linear-interpolation int16 resampler for streaming PCM.

    An OpenAI-compatible TTS endpoint may return audio at a sample rate
    different from the adapter-declared ``AudioFormat`` (issue #76466). The
    adapter contract requires every delivered chunk to conform to the
    declared format, so the consumer resamples on the fly. State is carried
    across chunk boundaries so interpolation stays click-free, and the
    source tail is buffered until enough input accumulates.
    """

    def __init__(self, src_rate: int, dst_rate: int):
        self._ratio = src_rate / dst_rate  # source samples per output sample
        self._pos = 0.0  # source index of the next output sample
        self._buf = None  # lazily-imported numpy arrays; pending source samples
        self._pending = None  # stray odd byte waiting for its pair

    def _numpy(self):
        import numpy as np

        return np

    def process(self, chunk: bytes) -> bytes:
        """Resample one PCM chunk; may return fewer bytes than input."""
        np = self._numpy()
        if self._pending is not None:
            chunk = self._pending + chunk
            self._pending = None
        if len(chunk) % 2:  # int16 samples must stay aligned
            self._pending = chunk[-1:]
            chunk = chunk[:-1]
        src = np.frombuffer(chunk, dtype=np.int16).astype(np.float64)
        if self._buf is None:
            buf = src
        else:
            buf = np.concatenate([self._buf, src])
        n_out = int((len(buf) - self._pos) // self._ratio)
        if n_out <= 0:
            self._buf = buf
            return b""
        idx = self._pos + np.arange(n_out) * self._ratio
        i0 = idx.astype(np.int64)
        i1 = np.minimum(i0 + 1, len(buf) - 1)
        frac = idx - i0
        out = buf[i0] * (1.0 - frac) + buf[i1] * frac
        consumed = int(self._pos + n_out * self._ratio)
        self._buf = buf[consumed:]
        self._pos = (self._pos + n_out * self._ratio) - consumed
        return out.astype(np.int16).tobytes()

    def flush(self) -> bytes:
        """Emit the final partial output sample (duration-preserving)."""
        if self._buf is None or not len(self._buf):
            return b""
        np = self._numpy()
        i0 = int(self._pos)
        if i0 >= len(self._buf):
            self._buf = None
            return b""
        i1 = min(i0 + 1, len(self._buf) - 1)
        frac = self._pos - i0
        out = self._buf[i0] * (1.0 - frac) + self._buf[i1] * frac
        self._buf = None
        return np.array([out], dtype=np.int16).tobytes()


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
        from tools.tts_streaming import SentenceChunker, resolve_streaming_provider

        self._adapter = adapter
        self._chat_id = chat_id
        self._tts_config = tts_config
        self._loop = loop
        self._metadata = metadata

        # Resolve the streaming provider once. If unavailable, the consumer is
        # inactive and the gateway falls back to whole-file TTS.
        self._streamer = resolve_streaming_provider(tts_config)
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
        self._finished = False
        self._dropped = False
        self._suppress_whole_file = False
        self._task: Optional[asyncio.Task] = None
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
        if not self.active:
            return

        if not self._adapter.supports_streaming_tts(self._chat_id, self._audio_format):
            logger.debug("adapter %s does not support streaming TTS", getattr(self._adapter, "name", "?"))
            return

        try:
            self._handle = await self._adapter.begin_streaming_tts(
                self._chat_id,
                self._audio_format,
                metadata=self._metadata,
            )
        except Exception as exc:
            logger.debug("begin_streaming_tts failed: %s", exc)
            self._handle = None
            return

        if self._handle is None:
            return

        self._started = True
        self._suppress_whole_file = False

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

            if not self._aborted and self._handle is not None:
                _finish_failed = False
                try:
                    await self._adapter.finish_streaming_tts(self._handle, interrupted=self._aborted)
                except Exception as exc:
                    logger.debug("finish_streaming_tts error: %s", exc)
                    _finish_failed = True

                if _finish_failed:
                    # finish_streaming_tts() raised — never report full
                    # completion.  If audio was already audible, report
                    # partial and preserve suppression so the gateway
                    # does not replay from the beginning.  If no audio
                    # was audible, permit whole-file fallback.
                    if self._handle.audible:
                        self._partial = True
                        self._completed = False
                        self._suppress_whole_file = True
                    else:
                        self._completed = False
                        self._suppress_whole_file = False
                    await self._safe_abort("finish_streaming_tts failed")
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
        except Exception as exc:
            logger.warning("streaming TTS consumer error: %s", exc)
            await self._safe_abort(str(exc))
        finally:
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except Exception:
                pass

    async def _synthesise_and_write(self, clause: str) -> None:
        """Synthesise one clause via the streamer and write PCM chunks."""
        if self._handle is None or self._handle.aborted:
            return

        cleaned = self._strip_markdown_for_tts(clause)
        if not cleaned or not cleaned.strip():
            return

        if self._streamer is None:
            return

        async for chunk in self._iter_stream_chunks(cleaned):
            if self._aborted or self._handle.aborted:
                return
            if not chunk:
                continue
            was_audible = self._handle.audible
            await self._adapter.write_streaming_tts(self._handle, chunk)
            if not was_audible:
                self._handle.audible = True
                self._suppress_whole_file = True

    async def _iter_stream_chunks(self, text: str):
        """Yield provider PCM chunks one at a time without blocking the loop.

        When the streamer's endpoint returns a sample rate different from the
        adapter-declared ``AudioFormat`` (issue #76466), chunks are resampled
        on the fly so the adapter contract — every chunk conforms to the
        declared format — still holds.
        """
        if self._streamer is None:
            return
        iterator = iter(self._streamer.stream(text))
        resampler = None
        while True:
            has_chunk, chunk = await asyncio.to_thread(self._next_stream_chunk, iterator)
            if not has_chunk:
                break
            if chunk:
                if resampler is None and self._streamer.sample_rate != self._audio_format.sample_rate:
                    resampler = _PcmResampler(
                        int(self._streamer.sample_rate),
                        int(self._audio_format.sample_rate),
                    )
                    logger.info(
                        "TTS endpoint sample rate %d Hz differs from adapter "
                        "format %d Hz; resampling stream",
                        self._streamer.sample_rate,
                        self._audio_format.sample_rate,
                    )
                if resampler is not None:
                    chunk = resampler.process(chunk)
            if chunk:
                yield chunk
        if resampler is not None:
            tail = resampler.flush()
            if tail:
                yield tail

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

    async def _safe_abort(self, reason: str) -> None:
        """Abort the adapter stream, swallowing errors (idempotent)."""
        if self._handle is None:
            return
        try:
            await self._adapter.abort_streaming_tts(self._handle, error=reason)
        except Exception:
            pass
        finally:
            if self._handle:
                self._handle.aborted = True

    # ------------------------------------------------------------------
    # Cancellation and completion
    # ------------------------------------------------------------------

    def abort(self, reason: str = "cancelled") -> None:
        """Idempotent cancellation from any thread."""
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
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
        if self._handle is not None and not self._handle.aborted:
            try:
                self._loop.call_soon_threadsafe(
                    asyncio.create_task,
                    self._safe_abort(reason),
                )
            except Exception:
                pass

    async def wait_complete(self, timeout: float = 10.0) -> bool:
        """Wait for the drain task to finish. Returns True only on full success."""
        if self._task is None:
            return self._completed
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass
        return self._completed

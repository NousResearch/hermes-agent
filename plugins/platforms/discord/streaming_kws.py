"""Playback-scoped streaming keyword spotting for Discord voice.

The SocketReader thread must never run inference. It offers decoded Discord PCM
into a bounded queue; one worker owns all local ASR streams and emits
privacy-safe detection metadata. No PCM, transcript, or hypothesis is persisted.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DISCORD_SAMPLE_RATE = 48000
_DISCORD_CHANNELS = 2
_BYTES_PER_SAMPLE = 2


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _clamped_int(
    data: Dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(data.get(key, default))
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


def _string_tuple(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    result = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    # Whisper may expand this common Korean contraction. This is a semantic
    # equivalent, not fuzzy matching, and does not weaken the wake name.
    text = text.replace("멈추어", "멈춰")
    return re.sub(r"[^0-9a-z가-힣]+", "", text)


def _matches_phrase_prefix(text: str, phrase: str) -> bool:
    """Match a phrase prefix only at a following lexical boundary."""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    phrase = unicodedata.normalize("NFKC", str(phrase or "")).lower()
    text = text.replace("멈추어", "멈춰")
    phrase = phrase.replace("멈추어", "멈춰")
    parts = re.findall(r"[0-9a-z가-힣]+", phrase)
    if not parts:
        return False
    separators = r"[^0-9a-z가-힣]*"
    pattern = (
        r"^[^0-9a-z가-힣]*"
        + separators.join(re.escape(part) for part in parts)
        + r"(?![0-9a-z가-힣])"
    )
    return re.match(pattern, text) is not None


@dataclass(frozen=True)
class StreamingKwsConfig:
    enabled: bool = False
    shadow_only: bool = True
    provider: str = "faster_whisper"
    model_dir: str = ""
    model: str = "base"
    compute_type: str = "int8"
    window_ms: int = 1600
    stride_ms: int = 320
    min_audio_ms: int = 640
    hotword_bias: bool = False
    contrast_wake_names: Tuple[str, ...] = ()
    num_threads: int = 4
    queue_frames: int = 256

    @classmethod
    def from_mapping(cls, raw: Any) -> "StreamingKwsConfig":
        data = raw if isinstance(raw, dict) else {}
        try:
            threads = int(data.get("num_threads", 4))
        except (TypeError, ValueError):
            threads = 4
        try:
            frames = int(data.get("queue_frames", 256))
        except (TypeError, ValueError):
            frames = 256
        window_ms = _clamped_int(data, "window_ms", 1600, 800, 4000)
        stride_ms = min(window_ms, _clamped_int(data, "stride_ms", 320, 160, 1000))
        min_audio_ms = min(
            window_ms,
            _clamped_int(data, "min_audio_ms", 640, 400, 2000),
        )
        return cls(
            enabled=_as_bool(data.get("enabled"), False),
            shadow_only=_as_bool(data.get("shadow_only"), True),
            provider=str(data.get("provider") or "faster_whisper").strip().lower(),
            model_dir=str(data.get("model_dir") or "").strip(),
            model=str(data.get("model") or "base").strip(),
            compute_type=str(data.get("compute_type") or "int8").strip().lower(),
            window_ms=window_ms,
            stride_ms=stride_ms,
            min_audio_ms=min_audio_ms,
            hotword_bias=_as_bool(data.get("hotword_bias"), False),
            contrast_wake_names=_string_tuple(data.get("contrast_wake_names")),
            num_threads=min(4, max(1, threads)),
            queue_frames=min(1024, max(32, frames)),
        )


class FasterWhisperRollingEngine:
    """Rolling in-memory Korean phrase detector backed by faster-whisper."""

    def __init__(self, config: StreamingKwsConfig, phrases: Tuple[str, ...]):
        from tools.transcription_tools import ensure_faster_whisper_dependency

        ensure_faster_whisper_dependency()
        import numpy as np
        from faster_whisper import WhisperModel

        self._np = np
        self._phrases = tuple(str(p).strip() for p in phrases if str(p).strip())
        if not self._phrases:
            raise RuntimeError("Discord streaming KWS requires at least one phrase")
        source = str(Path(config.model_dir).expanduser()) if config.model_dir else config.model
        self._window_samples = round(config.window_ms * 16000 / 1000)
        self._stride_samples = round(config.stride_ms * 16000 / 1000)
        self._min_samples = round(config.min_audio_ms * 16000 / 1000)
        hotwords = list(self._phrases)
        tails = []
        for phrase in self._phrases:
            parts = phrase.split(maxsplit=1)
            if len(parts) == 2 and parts[1] not in tails:
                tails.append(parts[1])
        for wake_name in config.contrast_wake_names:
            for tail in tails:
                hotwords.append(f"{wake_name} {tail}")
        self._hotwords = " ".join(hotwords) if config.hotword_bias else None
        self._model = WhisperModel(
            source,
            device="cpu",
            compute_type=config.compute_type,
            cpu_threads=max(1, config.num_threads),
        )

    def create_stream(self):
        return {
            "samples": self._np.zeros(0, dtype=self._np.float32),
            "new_samples": 0,
        }

    def _append_pcm(self, stream, pcm: bytes) -> None:
        usable = len(pcm) - (len(pcm) % (_DISCORD_CHANNELS * _BYTES_PER_SAMPLE))
        if usable <= 0:
            return
        stereo = self._np.frombuffer(pcm[:usable], dtype=self._np.int16).reshape(-1, 2)
        mono_48k = stereo.astype(self._np.float32).mean(axis=1)
        # Discord is exactly 48 kHz. A three-sample boxcar is a small bounded
        # anti-alias filter and exact 3:1 decimator to Whisper-native 16 kHz.
        usable_mono = len(mono_48k) - (len(mono_48k) % 3)
        if usable_mono <= 0:
            return
        mono_16k = mono_48k[:usable_mono].reshape(-1, 3).mean(axis=1) / 32768.0
        stream["samples"] = self._np.concatenate((stream["samples"], mono_16k))[
            -self._window_samples :
        ]
        stream["new_samples"] += len(mono_16k)

    def _detect(self, stream, *, force: bool) -> Optional[int]:
        samples = stream["samples"]
        if len(samples) < self._min_samples:
            return None
        if not force and stream["new_samples"] < self._stride_samples:
            return None
        if force and stream["new_samples"] <= 0:
            return None
        stream["new_samples"] = 0
        segments, _info = self._model.transcribe(
            samples,
            language="ko",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            initial_prompt=None,
            hotwords=self._hotwords,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        for index, phrase in enumerate(self._phrases):
            if _matches_phrase_prefix(text, phrase):
                return index
        return None

    def process(self, stream, pcm: bytes) -> Optional[int]:
        self._append_pcm(stream, pcm)
        return self._detect(stream, force=False)

    def flush(self, stream) -> Optional[int]:
        return self._detect(stream, force=True)

    def close(self) -> None:
        self._model = None


def _build_engine(config: StreamingKwsConfig, phrases: Tuple[str, ...]):
    if config.provider not in {"faster_whisper", "faster-whisper", "whisper"}:
        raise ValueError(f"Unsupported Discord streaming KWS provider: {config.provider}")
    return FasterWhisperRollingEngine(config, phrases)


@dataclass(frozen=True)
class _QueueItem:
    kind: str
    guild_id: int
    token: int
    user_id: int = 0
    pcm: bytes = b""
    received_at: float = 0.0


class DiscordStreamingKwsState(Enum):
    NEW = auto()
    STARTING = auto()
    RUNNING = auto()
    FAILED = auto()
    CLOSING = auto()
    CLOSED = auto()


class DiscordStreamingKwsManager:
    """One bounded worker for all playback-scoped Discord KWS streams."""

    def __init__(
        self,
        config: StreamingKwsConfig,
        phrases: Tuple[str, ...],
        on_detection: Callable[[Dict[str, Any]], None],
        *,
        engine_factory: Callable[[StreamingKwsConfig, Tuple[str, ...]], Any] = _build_engine,
        start_timeout: float = 0.0,
    ):
        self.config = config
        self.phrases = tuple(phrases)
        self._on_detection = on_detection
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=config.queue_frames)
        self._lifecycle_lock = threading.RLock()
        self._state = DiscordStreamingKwsState.NEW
        self._stats_lock = threading.Lock()
        self._forced_end_lock = threading.Lock()
        self._forced_ends: set[Tuple[int, int]] = set()
        self._stats: Dict[str, int] = {
            "offered_frames": 0,
            "processed_frames": 0,
            "queue_drops": 0,
            "detections": 0,
            "worker_errors": 0,
        }
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._worker_waiting = False
        self._startup_error: Optional[BaseException] = None
        self._engine_factory = engine_factory
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="discord-streaming-kws",
        )
        with self._lifecycle_lock:
            self._state = DiscordStreamingKwsState.STARTING
        self._thread.start()
        # Production callers use the non-blocking default so a first model
        # download/load can never stall the Discord event loop. Tests and
        # diagnostics may opt into a bounded synchronous startup check.
        if start_timeout > 0:
            if not self._ready.wait(start_timeout):
                self.close()
                raise TimeoutError("Timed out while starting Discord streaming KWS")
            if self._startup_error is not None:
                self.close()
                raise RuntimeError("Discord streaming KWS failed to start") from self._startup_error

    def _bump(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + amount

    def snapshot_stats(self) -> Dict[str, Any]:
        # Global lock order is lifecycle -> stats -> forced-end. Admission,
        # close, and the worker all follow this order; diagnostics must not
        # invert it or they can deadlock against an in-flight `_bump()`.
        with self._lifecycle_lock:
            with self._stats_lock:
                stats: Dict[str, Any] = dict(self._stats)
            stats["queue_depth"] = self._queue.qsize()
            with self._forced_end_lock:
                stats["forced_end_depth"] = len(self._forced_ends)
            stats["state"] = self._state.name
        stats["ready"] = int(self._ready.is_set())
        stats["startup_failed"] = int(self._startup_error is not None)
        return stats

    def _put_control(self, item: _QueueItem) -> bool:
        with self._lifecycle_lock:
            if self._state is not DiscordStreamingKwsState.RUNNING:
                return False
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                self._bump("queue_drops")
                return False

    def begin_playback(self, guild_id: int, token: int) -> bool:
        return self._put_control(_QueueItem("begin", int(guild_id), int(token)))

    def offer_pcm(
        self,
        guild_id: int,
        token: int,
        user_id: int,
        pcm: bytes,
        *,
        received_at: Optional[float] = None,
    ) -> bool:
        if not pcm or not user_id:
            return False
        item = _QueueItem(
            "pcm",
            int(guild_id),
            int(token),
            int(user_id),
            bytes(pcm),
            time.monotonic() if received_at is None else float(received_at),
        )
        with self._lifecycle_lock:
            if self._state is not DiscordStreamingKwsState.RUNNING:
                return False
            self._bump("offered_frames")
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                self._bump("queue_drops")
                return False

    def end_playback(self, guild_id: int, token: int) -> bool:
        guild_id = int(guild_id)
        token = int(token)
        with self._lifecycle_lock:
            if self._state is not DiscordStreamingKwsState.RUNNING:
                return False
            try:
                self._queue.put_nowait(_QueueItem("end", guild_id, token))
                return True
            except queue.Full:
                self._bump("queue_drops")
                # Never block the Discord SocketReader. If the bounded PCM queue
                # is saturated, the worker consumes this side-channel before its
                # next queued frame and drops all remaining audio for the token.
                with self._forced_end_lock:
                    self._forced_ends.add((guild_id, token))
                return False

    def close(self) -> None:
        with self._lifecycle_lock:
            if not self._closed.is_set():
                self._state = DiscordStreamingKwsState.CLOSING
                self._closed.set()
                self._ready.set()
                # Shutdown is terminal: discard every queued PCM/control item
                # before waking the worker. In-flight inference may finish, but
                # it cannot resume stale queued audio after this boundary.
                self._discard_queue()
                with self._forced_end_lock:
                    self._forced_ends.clear()
                if self._worker_waiting:
                    self._queue.put_nowait(_QueueItem("stop", 0, 0))
        if self._thread is not threading.current_thread():
            # Native inference cannot be force-cancelled safely. Keep close
            # prompt and bounded; the worker observes _closed as soon as the
            # in-flight call returns and exits without consuming more PCM.
            self._thread.join(timeout=0.01)
        if not self._thread.is_alive():
            with self._lifecycle_lock:
                if self._state is DiscordStreamingKwsState.CLOSING:
                    self._state = DiscordStreamingKwsState.CLOSED

    def _discard_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        engine = None
        active: set[Tuple[int, int]] = set()
        fired: set[Tuple[int, int]] = set()
        streams: Dict[Tuple[int, int, int], Any] = {}
        first_received: Dict[Tuple[int, int, int], float] = {}
        last_received: Dict[Tuple[int, int, int], float] = {}
        audio_bytes: Dict[Tuple[int, int, int], int] = {}
        flushed_bytes: Dict[Tuple[int, int, int], int] = {}

        def _cleanup(token_key: Tuple[int, int]) -> None:
            for key in [key for key in streams if key[:2] == token_key]:
                streams.pop(key, None)
                first_received.pop(key, None)
                last_received.pop(key, None)
                audio_bytes.pop(key, None)
                flushed_bytes.pop(key, None)

        def _drain_forced_ends() -> None:
            with self._forced_end_lock:
                pending = tuple(self._forced_ends)
                self._forced_ends.clear()
            for token_key in pending:
                active.discard(token_key)
                fired.discard(token_key)
                _cleanup(token_key)

        def _emit(
            stream_key: Tuple[int, int, int],
            keyword_index: int,
            observed_at: float,
        ) -> None:
            # Reserve callback delivery under the lifecycle gate, then invoke
            # user code outside every manager lock. The reservation is the
            # linearization point: close either wins first (no callback) or the
            # callback is admitted first, but a blocked callback can never hold
            # close hostage.
            with self._lifecycle_lock:
                token_key = stream_key[:2]
                if (
                    self._state is not DiscordStreamingKwsState.RUNNING
                    or token_key in fired
                    or token_key not in active
                ):
                    return
                fired.add(token_key)
                self._bump("detections")
                first = first_received.get(stream_key, observed_at)
                bytes_seen = audio_bytes.get(stream_key, 0)
                event = {
                    "guild_id": stream_key[0],
                    "token": stream_key[1],
                    "user_id": stream_key[2],
                    "keyword_index": int(keyword_index),
                    "latency_ms": round((time.monotonic() - first) * 1000),
                    "audio_ms": round(
                        bytes_seen
                        / (
                            _DISCORD_SAMPLE_RATE
                            * _DISCORD_CHANNELS
                            * _BYTES_PER_SAMPLE
                        )
                        * 1000
                    ),
                    "queue_delay_ms": round(
                        (time.monotonic() - observed_at) * 1000
                    ),
                }
            try:
                self._on_detection(event)
            except Exception as exc:
                self._bump("worker_errors")
                logger.info(
                    "Discord streaming KWS callback failed type=%s",
                    type(exc).__name__,
                )

        try:
            engine = self._engine_factory(self.config, self.phrases)
        except BaseException as exc:
            self._bump("worker_errors")
            logger.warning(
                "Discord streaming KWS startup failed type=%s",
                type(exc).__name__,
            )
            with self._lifecycle_lock:
                self._startup_error = exc
                if self._state is DiscordStreamingKwsState.STARTING:
                    self._state = DiscordStreamingKwsState.FAILED
                    self._closed.set()
                elif self._state is DiscordStreamingKwsState.CLOSING:
                    # close() won while the native factory was blocked. The
                    # worker is now exiting, so publish the terminal state even
                    # though no engine object was created.
                    self._state = DiscordStreamingKwsState.CLOSED
                self._discard_queue()
                with self._forced_end_lock:
                    self._forced_ends.clear()
                self._ready.set()
            return
        exit_before_run = False
        with self._lifecycle_lock:
            if self._state is not DiscordStreamingKwsState.STARTING:
                exit_before_run = True
            else:
                self._state = DiscordStreamingKwsState.RUNNING
                self._ready.set()
        if exit_before_run:
            # close() may win while native engine construction is still
            # blocked. The constructor result must still be closed and the
            # terminal state published once construction returns.
            try:
                engine.close()
            except Exception:
                pass
            with self._lifecycle_lock:
                if self._state is DiscordStreamingKwsState.CLOSING:
                    self._state = DiscordStreamingKwsState.CLOSED
            return
        try:
            while not self._closed.is_set():
                _drain_forced_ends()
                with self._lifecycle_lock:
                    if self._state is not DiscordStreamingKwsState.RUNNING:
                        return
                    self._worker_waiting = True
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    with self._lifecycle_lock:
                        self._worker_waiting = False
                    flush = getattr(engine, "flush", None)
                    if not callable(flush):
                        continue
                    now = time.monotonic()
                    for stream_key, stream in list(streams.items()):
                        token_key = stream_key[:2]
                        if token_key not in active or token_key in fired:
                            continue
                        if now - last_received.get(stream_key, now) < 0.2:
                            continue
                        bytes_seen = audio_bytes.get(stream_key, 0)
                        if flushed_bytes.get(stream_key) == bytes_seen:
                            continue
                        flushed_bytes[stream_key] = bytes_seen
                        try:
                            keyword_index = flush(stream)
                        except Exception as exc:
                            self._bump("worker_errors")
                            logger.info(
                                "Discord streaming KWS idle flush failed type=%s",
                                type(exc).__name__,
                            )
                            continue
                        if keyword_index is not None:
                            _emit(
                                stream_key,
                                keyword_index,
                                last_received.get(stream_key, now),
                            )
                    continue
                with self._lifecycle_lock:
                    self._worker_waiting = False

                token_key = (item.guild_id, item.token)
                if item.kind == "stop":
                    return
                if item.kind == "begin":
                    active.add(token_key)
                    fired.discard(token_key)
                    _cleanup(token_key)
                    continue
                if item.kind == "end":
                    active.discard(token_key)
                    fired.discard(token_key)
                    _cleanup(token_key)
                    continue
                if item.kind != "pcm" or token_key not in active or token_key in fired:
                    continue

                stream_key = (item.guild_id, item.token, item.user_id)
                stream = streams.get(stream_key)
                if stream is None:
                    stream = engine.create_stream()
                    streams[stream_key] = stream
                    first_received[stream_key] = item.received_at
                    audio_bytes[stream_key] = 0
                last_received[stream_key] = item.received_at
                audio_bytes[stream_key] += len(item.pcm)
                try:
                    keyword_index = engine.process(stream, item.pcm)
                    self._bump("processed_frames")
                except Exception as exc:
                    self._bump("worker_errors")
                    logger.info(
                        "Discord streaming KWS frame failed type=%s",
                        type(exc).__name__,
                    )
                    continue
                if self._closed.is_set():
                    return
                if keyword_index is not None:
                    _emit(stream_key, keyword_index, item.received_at)
        finally:
            with self._lifecycle_lock:
                self._worker_waiting = False
            self._discard_queue()
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass
            with self._lifecycle_lock:
                if self._state is DiscordStreamingKwsState.CLOSING:
                    self._state = DiscordStreamingKwsState.CLOSED

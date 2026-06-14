"""Daily WebRTC transport via daily-python virtual audio devices.

Audio geometry (chosen to avoid resampling entirely — daily-python gives
every virtual device its own sample_rate, verified against the installed
SDK 0.29.1: Daily.create_microphone_device / create_speaker_device both
take ``sample_rate``):
  inbound  (caller -> STT): virtual SPEAKER device @ 24 kHz mono. Read 1920
           frames (80 ms) per blocking read_frames() call; the device paces
           reads at real time. Deepgram Flux resamples 24k->16k internally.
  outbound (TTS -> caller): virtual MICROPHONE device @ 48 kHz mono — matches
           Cartesia TTS output. Blocking write_frames() paces playback;
           barge-in drains the local queue and stops writing.

daily-python allows ONE active virtual speaker per process (and the
upstream demos treat the mic the same way), so devices are process-level
singletons and the adapter enforces a single active call.

API facts verified against the installed daily-python 0.29.1 and the
upstream demos (demos/audio/wav_audio_send.py, wav_audio_receive.py):
  - Daily.init(worker_threads=2, log_level=...) — no args required.
  - create_microphone_device(device_name, sample_rate=16000, channels=1,
    non_blocking=False); same signature for create_speaker_device.
  - CallClient(event_handler=None);
    join(meeting_url, meeting_token=None, client_settings=None,
         completion=None) — completion(JoinData, CallClientError);
    leave(completion=None) — completion(CallClientError);
    update_subscription_profiles(profile_settings, completion=None);
    participants() — mapping keyed by participant id (the local
    participant under the "local" key), each value carrying
    {"id": ..., "info": {"isLocal": bool, ...}, ...}.
  - daily.EventHandler — subclass and pass to CallClient(event_handler=);
    callbacks are invoked on daily-python's internal event thread:
      on_participant_joined(participant)        (remote participants)
      on_participant_left(participant, reason)
      on_call_state_updated(state)              ("joining"/"joined"/
                                                 "leaving"/"left")
      on_error(message)
  - client_settings mic selection shape (wav_audio_send.py):
    {"inputs": {"camera": False, "microphone":
        {"isEnabled": True, "settings": {"deviceId": <name>}}}}
  - VirtualMicrophoneDevice.write_frames(frames, completion=None) -> int;
    VirtualSpeakerDevice.read_frames(num_frames, completion=None) ->
    bytestring (empty when no frames were read).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

MIC_DEVICE = "hermes-voice-mic"
SPEAKER_DEVICE = "hermes-voice-speaker"
MIC_RATE = 48000                # agent TTS output rate (Cartesia 48k)
OUT_CHUNK_BYTES = 7680          # 80ms @ 48kHz s16le mono — barge-in flush granularity
SPEAKER_RATE = 24000            # inbound caller rate (Flux STT resamples 24k->16k)
IN_CHUNK_FRAMES = 1920          # 80 ms @ 24 kHz — one ASR chunk per read

# Subscribe to participant microphones only (wav_audio_receive.py pattern);
# video is never wanted on a voice call.
_SUBSCRIPTION_PROFILES = {
    "base": {"camera": "unsubscribed", "microphone": "subscribed"}
}

_init_lock = threading.Lock()
_initialized = False
_mic = None
_speaker = None


def _is_local_participant(key: str, participant: dict) -> bool:
    """True for the agent's own entry in participants()/events. The local
    participant appears under the "local" key in participants() and carries
    info.isLocal in event payloads."""
    if key == "local":
        return True
    info = participant.get("info") or {}
    return bool(info.get("isLocal"))


def _make_event_handler(transport: "DailyTransport"):
    """EventHandler subclass wired to *transport* (deferred daily import so
    the module stays importable without the SDK). Callbacks arrive on
    daily-python's internal event thread — handlers must be thread-safe."""
    from daily import EventHandler

    class _TransportEvents(EventHandler):
        def on_participant_joined(self, participant) -> None:
            transport._on_participant_joined(participant)

        def on_participant_left(self, participant, reason) -> None:
            transport._on_participant_left(participant, reason)

        def on_call_state_updated(self, state) -> None:
            transport._on_call_state_updated(state)

        def on_error(self, message) -> None:
            transport._on_client_error(message)

    return _TransportEvents()


def _ensure_daily() -> None:
    """Daily.init() + virtual device creation, once per process."""
    global _initialized, _mic, _speaker
    with _init_lock:
        if _initialized:
            return
        from daily import Daily
        Daily.init()
        _mic = Daily.create_microphone_device(
            MIC_DEVICE, sample_rate=MIC_RATE, channels=1)
        _speaker = Daily.create_speaker_device(
            SPEAKER_DEVICE, sample_rate=SPEAKER_RATE, channels=1)
        Daily.select_speaker_device(SPEAKER_DEVICE)
        _initialized = True


class DailyTransport:
    """One Daily call. on_audio_in(pcm) is scheduled onto *loop* for every
    80 ms chunk of caller audio (s16le mono 24 kHz)."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_audio_in: Callable[[bytes], Awaitable[None]],
    ):
        self._loop = loop
        self._on_audio_in = on_audio_in
        self._client = None
        self._running = False
        self._reader: Optional[threading.Thread] = None
        self._writer: Optional[threading.Thread] = None
        self._keepalive: Optional[threading.Thread] = None
        self._out_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._joined = threading.Event()
        self._join_error: Optional[str] = None
        self._last_audio_in_t = 0.0
        # Presence + lifecycle state for the adapter's call watchdog
        # (ENG-555 cost leak: abandoned calls kept the billable ASR stream
        # alive forever). Mutated from daily-python's event thread.
        self._presence_lock = threading.Lock()
        self._remote_ids: set = set()
        self._leaving = False
        self._abnormal_end: Optional[str] = None
        # Teardown gate: set the moment teardown starts, BEFORE the call is
        # left, so the DTX keep-alive stops feeding billable silence into
        # the ASR stream immediately (the keep-alive thread must never
        # outlive the call).
        self._teardown = threading.Event()
        # Telemetry write mark (notes §16 first_frame_written): the turn
        # loop arms it at the start of a turn cycle; the writer thread
        # stamps the first real frame write after arming.
        self._first_write_t: Optional[float] = None
        self._write_mark_armed = False

    async def join(self, room_url: str, token: str, timeout: float = 15.0) -> None:
        _ensure_daily()
        from daily import CallClient
        self._client = CallClient(event_handler=_make_event_handler(self))
        self._client.update_subscription_profiles(_SUBSCRIPTION_PROFILES)

        def _on_join(data, error):
            self._join_error = str(error) if error else None
            self._joined.set()

        self._client.join(
            room_url,
            meeting_token=token,
            client_settings={
                "inputs": {
                    "camera": False,
                    "microphone": {
                        "isEnabled": True,
                        "settings": {"deviceId": MIC_DEVICE},
                    },
                }
            },
            completion=_on_join,
        )
        await self._loop.run_in_executor(None, self._joined.wait, timeout)
        if not self._joined.is_set():
            self._client.release()
            self._client = None
            raise RuntimeError(f"Daily join timed out after {timeout}s")
        if self._join_error:
            self._client.release()
            self._client = None
            raise RuntimeError(f"Daily join failed: {self._join_error}")
        # Seed presence from the post-join roster: a human may already be
        # waiting in the room when the agent joins (their joined event fired
        # before our handler existed).
        participants = self._client.participants()
        with self._presence_lock:
            for key, part in participants.items():
                if not _is_local_participant(key, part):
                    self._remote_ids.add(part.get("id") or key)
        self._running = True
        self._last_audio_in_t = time.monotonic()
        self._reader = threading.Thread(
            target=self._read_loop, name="voice-daily-reader", daemon=True)
        self._writer = threading.Thread(
            target=self._write_loop, name="voice-daily-writer", daemon=True)
        self._keepalive = threading.Thread(
            target=self._keepalive_loop, name="voice-daily-keepalive",
            daemon=True)
        self._reader.start()
        self._writer.start()
        self._keepalive.start()
        logger.info("voice/daily: joined %s", room_url)

    # -- presence + call-state events (daily-python event thread) ----------

    def _on_participant_joined(self, participant) -> None:
        if _is_local_participant("", participant or {}):
            return
        pid = (participant or {}).get("id")
        if pid is None:
            return
        with self._presence_lock:
            self._remote_ids.add(pid)
            count = len(self._remote_ids)
        logger.info("voice/daily: participant joined id=%s remote_count=%d",
                    pid, count)

    def _on_participant_left(self, participant, reason) -> None:
        pid = (participant or {}).get("id")
        with self._presence_lock:
            self._remote_ids.discard(pid)
            count = len(self._remote_ids)
        logger.warning(
            "voice/daily: participant left id=%s reason=%s remote_count=%d",
            pid, reason, count)

    def _on_call_state_updated(self, state) -> None:
        logger.info("voice/daily: call state -> %s", state)
        if state == "left" and not self._leaving:
            # We did not initiate this: room expired / agent ejected /
            # connection lost. The adapter watchdog tears the call down.
            self._abnormal_end = "left"
            logger.warning(
                "voice/daily: call ended remotely (ejected/expired) — "
                "flagging for teardown")

    def _on_client_error(self, message) -> None:
        self._abnormal_end = f"error: {message}"
        logger.error("voice/daily: fatal client error: %s", message)

    @property
    def remote_participant_count(self) -> int:
        """Remote (non-agent) participants currently in the room."""
        with self._presence_lock:
            return len(self._remote_ids)

    @property
    def abnormal_end(self) -> Optional[str]:
        """Set when the call ended without us leaving: "left" (ejection /
        room expiry) or "error: ..." (fatal client error). None while the
        call is healthy."""
        return self._abnormal_end

    def _read_loop(self) -> None:
        while self._running:
            frames = _speaker.read_frames(IN_CHUNK_FRAMES)   # blocking 80 ms
            if not frames:
                # Empty reads happen at teardown / before audio flows; avoid
                # a hot spin since only non-empty reads pace real time.
                time.sleep(0.01)
                continue
            self._last_audio_in_t = time.monotonic()
            asyncio.run_coroutine_threadsafe(self._on_audio_in(frames), self._loop)

    def _keepalive_loop(self) -> None:
        # WebRTC DTX: a silent caller stops sending packets entirely and
        # read_frames() BLOCKS (it cannot be relied on to return empties at
        # cadence). Streaming STT turn detection assumes a continuous
        # timeline — Flux needs to SEE the post-speech silence to fire its
        # end-of-turn, and a starved socket cannot deliver it. This thread
        # feeds synthesized 80ms silence whenever real audio stops flowing.
        # COST GUARD: this synthesized silence is BILLABLE STT input. The
        # _teardown gate stops the feed the moment teardown starts — without
        # it an abandoned call would keep the STT stream alive indefinitely.
        silence = b"\x00" * (IN_CHUNK_FRAMES * 2)
        while self._running and not self._teardown.is_set():
            self._teardown.wait(0.08)
            if not self._running or self._teardown.is_set():
                return
            if time.monotonic() - self._last_audio_in_t >= 0.16:
                asyncio.run_coroutine_threadsafe(
                    self._on_audio_in(silence), self._loop)

    def _write_loop(self) -> None:
        # WALL-CLOCK paced: write_frames sometimes returns faster than real
        # time (Daily buffers internally — measured live 2026-06-12: the
        # writer ran ~1.6s ahead on long replies, and barge-in's
        # clear_output cannot claw audio back out of Daily's buffer, so the
        # caller kept hearing the dead reply for that long). Never hand
        # Daily chunk N+1 before chunk N's real-time due point: buffered
        # audio is then capped at ~one 80ms chunk and barge-in cutoff stays
        # bounded by the trigger latency, not the backlog.
        burst_t0 = 0.0
        burst_audio_s = 0.0
        next_due = 0.0
        while self._running:
            try:
                chunk = self._out_q.get(timeout=0.5)
            except queue.Empty:
                if burst_audio_s > 0.0:
                    logger.info(
                        "voice/daily: write burst ended audio_s=%.2f wall_s=%.2f",
                        burst_audio_s, time.monotonic() - burst_t0)
                    burst_audio_s = 0.0
                continue
            if chunk is None:
                continue
            now = time.monotonic()
            if burst_audio_s == 0.0:
                burst_t0 = now
                next_due = now
                logger.info("voice/daily: write burst started qsize=%d",
                            self._out_q.qsize())
            delay = next_due - now
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                next_due = time.monotonic()   # writer fell behind; resync
            t0 = time.monotonic()
            if self._write_mark_armed:
                self._write_mark_armed = False
                self._first_write_t = t0
            _mic.write_frames(chunk)
            chunk_s = len(chunk) / 2.0 / MIC_RATE
            next_due += chunk_s
            burst_audio_s += chunk_s

    async def send_audio(self, pcm: bytes) -> None:
        """Queue agent speech (s16le mono 48 kHz) for the caller.

        Split into <=80ms sub-chunks so the wall-clock writer caps Daily's
        internal buffer tightly: barge-in's clear_output then leaves only
        ~one 80ms chunk of unstoppable audio regardless of the TTS provider's
        native chunk size (Cartesia emits ~190ms chunks)."""
        if len(pcm) <= OUT_CHUNK_BYTES:
            self._out_q.put(pcm)
            return
        for i in range(0, len(pcm), OUT_CHUNK_BYTES):
            self._out_q.put(pcm[i:i + OUT_CHUNK_BYTES])

    def reset_write_mark(self) -> None:
        """Arm the first_frame_written telemetry mark for a new turn."""
        self._first_write_t = None
        self._write_mark_armed = True

    @property
    def first_write_t(self) -> Optional[float]:
        """Monotonic time of the first frame written after the last
        reset_write_mark(), or None if nothing was written yet."""
        return self._first_write_t

    def is_playing(self) -> bool:
        """True while agent audio is still queued to play out to the caller.

        The turn loop flips its state back to LISTENING as soon as the TTS has
        FINISHED SENDING audio (tts.end() returns when the last chunk is
        queued), but the wall-clock writer then plays that audio out over the
        following seconds. Barge-in must fire during this tail too — otherwise
        the caller hears the agent talk over them with no way to cut it."""
        return not self._out_q.empty()

    def clear_output(self) -> None:
        """Barge-in: drop all queued (unplayed) agent audio."""
        dropped = 0
        try:
            while True:
                self._out_q.get_nowait()
                dropped += 1
        except queue.Empty:
            pass
        logger.info("voice/daily: cleared %d queued chunks", dropped)

    def begin_teardown(self) -> None:
        """Stop feeding billable keep-alive silence IMMEDIATELY. Called by
        the adapter as the very first teardown step — before the turn loop,
        STT, and transport are wound down (each of which can await) — so no
        more ASR audio is paid for past this point."""
        if not self._teardown.is_set():
            self._teardown.set()
            logger.info("voice/daily: teardown begun — keep-alive stopped")

    async def leave(self) -> None:
        self._teardown.set()
        self._leaving = True
        self._running = False
        self.clear_output()
        if self._client is not None:
            done = threading.Event()
            self._client.leave(completion=lambda _error: done.set())
            await self._loop.run_in_executor(None, done.wait, 10.0)
            self._client.release()
            self._client = None
        for worker in (self._reader, self._writer, self._keepalive):
            if worker is not None and worker.is_alive():
                await self._loop.run_in_executor(None, worker.join, 2.0)
        self._reader = None
        self._writer = None
        self._keepalive = None
        logger.info("voice/daily: left room")

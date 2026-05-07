from __future__ import annotations

"""Discord voice-channel bridge for OpenAI Realtime.

The bridge is intentionally small and self-contained:
- Discord inbound voice packets arrive as 48 kHz stereo PCM16.
- OpenAI Realtime expects 24 kHz mono PCM16 input and emits 24 kHz mono PCM16.
- discord.py voice playback expects an AudioSource that returns 20 ms chunks of
  48 kHz stereo PCM16 (3840 bytes) unless the source is Opus.

This module does not import discord at module import time so the Discord adapter
remains importable without optional voice dependencies.
"""

import audioop
import base64
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime"
DISCORD_RATE = 48_000
DISCORD_CHANNELS = 2
REALTIME_RATE = 24_000
REALTIME_CHANNELS = 1
SAMPLE_WIDTH = 2
DISCORD_FRAME_BYTES = int(DISCORD_RATE * 0.02) * DISCORD_CHANNELS * SAMPLE_WIDTH


def _require_ws_connect():
    try:
        from websockets.sync.client import connect as _connect  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "websockets package is required for OpenAI Realtime; install with: pip install websockets"
        ) from exc
    return _connect


def discord_pcm_to_realtime(pcm: bytes) -> bytes:
    """Convert Discord 48 kHz stereo PCM16 to OpenAI 24 kHz mono PCM16."""
    if not pcm:
        return b""
    mono = audioop.tomono(pcm, SAMPLE_WIDTH, 0.5, 0.5)
    converted, _ = audioop.ratecv(
        mono, SAMPLE_WIDTH, REALTIME_CHANNELS, DISCORD_RATE, REALTIME_RATE, None
    )
    return converted


def realtime_pcm_to_discord(pcm: bytes) -> bytes:
    """Convert OpenAI 24 kHz mono PCM16 to Discord 48 kHz stereo PCM16."""
    if not pcm:
        return b""
    upsampled, _ = audioop.ratecv(
        pcm, SAMPLE_WIDTH, REALTIME_CHANNELS, REALTIME_RATE, DISCORD_RATE, None
    )
    return audioop.tostereo(upsampled, SAMPLE_WIDTH, 1.0, 1.0)


class QueuePCMAudioSource:
    """PCM frame reader backed by a byte queue.

    The bare class is intentionally free of a module-level discord.py import so
    this module remains importable without optional Discord voice dependencies.
    When handing it to discord.py, wrap it with `_make_discord_audio_source()` so
    `VoiceClient.play()` sees an actual `discord.AudioSource` instance.
    """

    def __init__(self, pcm_queue: "queue.Queue[Optional[bytes]]") -> None:
        self._queue = pcm_queue
        self._buffer = bytearray()
        self._closed = False

    def is_opus(self) -> bool:
        return False

    def read(self) -> bytes:
        if self._closed:
            return b""
        deadline = time.monotonic() + 0.25
        while len(self._buffer) < DISCORD_FRAME_BYTES:
            timeout = max(0.0, deadline - time.monotonic())
            if timeout <= 0:
                break
            try:
                chunk = self._queue.get(timeout=timeout)
            except queue.Empty:
                break
            if chunk is None:
                self._closed = True
                break
            self._buffer.extend(chunk)
        if not self._buffer and self._closed:
            return b""
        if len(self._buffer) < DISCORD_FRAME_BYTES:
            self._buffer.extend(b"\x00" * (DISCORD_FRAME_BYTES - len(self._buffer)))
        out = bytes(self._buffer[:DISCORD_FRAME_BYTES])
        del self._buffer[:DISCORD_FRAME_BYTES]
        return out

    def cleanup(self) -> None:
        self._closed = True


def _make_discord_audio_source(pcm_queue: "queue.Queue[Optional[bytes]]") -> Any:
    """Create a queue-backed source that passes discord.py's AudioSource check."""
    try:
        import discord  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("discord.py is required for Discord voice playback") from exc

    audio_source_base = getattr(discord, "AudioSource", None)
    if not isinstance(audio_source_base, type):  # pragma: no cover - test suites may install a MagicMock discord module
        return QueuePCMAudioSource(pcm_queue)

    class DiscordQueuePCMAudioSource(audio_source_base):  # type: ignore[misc, valid-type]
        def __init__(self, queue_: "queue.Queue[Optional[bytes]]") -> None:
            self._inner = QueuePCMAudioSource(queue_)

        def is_opus(self) -> bool:
            return self._inner.is_opus()

        def read(self) -> bytes:
            return self._inner.read()

        def cleanup(self) -> None:
            self._inner.cleanup()
            try:
                super().cleanup()
            except AttributeError:
                pass

    return DiscordQueuePCMAudioSource(pcm_queue)


@dataclass
class RealtimeBridgeStatus:
    connected: bool
    input_bytes: int
    output_bytes: int
    last_input_at: Optional[float]
    last_output_at: Optional[float]
    last_error: Optional[str]


class OpenAIRealtimeDiscordBridge:
    """Duplex Discord voice <-> OpenAI Realtime WebSocket bridge."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_client: Any,
        model: str = "gpt-realtime-2",
        voice: str = "alloy",
        instructions: str = "You are Hermes in a Discord voice channel. Keep replies brief and conversational.",
    ) -> None:
        self.api_key = api_key
        self.voice_client = voice_client
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self._input_q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._output_q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=400)
        self._stop = threading.Event()
        self._send_lock = threading.Lock()
        self._ws: Any = None
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None
        self.input_bytes = 0
        self.output_bytes = 0
        self.last_input_at: Optional[float] = None
        self.last_output_at: Optional[float] = None
        self.last_error: Optional[str] = None

    def start(self) -> None:
        connect = _require_ws_connect()
        url = f"{REALTIME_URL}?model={self.model}"
        headers = [("Authorization", f"Bearer {self.api_key}")]
        if not self._uses_ga_api():
            headers.append(("OpenAI-Beta", "realtime=v1"))
        try:
            self._ws = connect(url, additional_headers=headers)
        except TypeError:  # pragma: no cover - older websockets
            self._ws = connect(url, extra_headers=headers)
        self._send_json(self._session_update_payload())
        self._rx_thread = threading.Thread(target=self._recv_loop, name="discord-openai-realtime-rx", daemon=True)
        self._tx_thread = threading.Thread(target=self._send_loop, name="discord-openai-realtime-tx", daemon=True)
        self._rx_thread.start()
        self._tx_thread.start()
        self._start_discord_playback()

    def _uses_ga_api(self) -> bool:
        # New GA-only models such as gpt-realtime-2 reject the legacy
        # OpenAI-Beta: realtime=v1 header and expect the GA session shape.
        return self.model == "gpt-realtime-2" or self.model.startswith("gpt-realtime-2-")

    def _session_update_payload(self) -> dict:
        if self._uses_ga_api():
            return {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.model,
                    "instructions": self.instructions,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": 500,
                                "create_response": True,
                            },
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                            "voice": self.voice,
                        },
                    },
                },
            }
        return {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": self.instructions,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "create_response": True,
                },
            },
        }

    def stop(self) -> None:
        self._stop.set()
        try:
            self._output_q.put_nowait(None)
        except queue.Full:
            pass
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        try:
            if self.voice_client and self.voice_client.is_playing():
                self.voice_client.stop()
        except Exception:
            pass

    def accept_discord_pcm(self, pcm: bytes) -> None:
        if self._stop.is_set() or not pcm:
            return
        converted = discord_pcm_to_realtime(pcm)
        if not converted:
            return
        try:
            self._input_q.put_nowait(converted)
            self.input_bytes += len(converted)
            self.last_input_at = time.time()
        except queue.Full:
            # Prefer dropping old audio over building latency.
            try:
                self._input_q.get_nowait()
                self._input_q.put_nowait(converted)
            except queue.Empty:
                pass

    def status(self) -> RealtimeBridgeStatus:
        return RealtimeBridgeStatus(
            connected=self._ws is not None and not self._stop.is_set(),
            input_bytes=self.input_bytes,
            output_bytes=self.output_bytes,
            last_input_at=self.last_input_at,
            last_output_at=self.last_output_at,
            last_error=self.last_error,
        )

    def _start_discord_playback(self) -> None:
        if not self.voice_client or not self.voice_client.is_connected():
            raise RuntimeError("Discord voice client is not connected")
        source = _make_discord_audio_source(self._output_q)

        def _after(error):
            if error:
                logger.warning("Realtime Discord playback ended with error: %s", error)

        if self.voice_client.is_playing():
            self.voice_client.stop()
        self.voice_client.play(source, after=_after)

    def _send_loop(self) -> None:
        while not self._stop.is_set():
            try:
                pcm = self._input_q.get(timeout=0.2)
            except queue.Empty:
                continue
            b64 = base64.b64encode(pcm).decode("ascii")
            try:
                self._send_json({"type": "input_audio_buffer.append", "audio": b64})
            except Exception as exc:
                self.last_error = str(exc)
                logger.warning("OpenAI Realtime input send failed: %s", exc)
                self.stop()
                return

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except TypeError:  # pragma: no cover - older websockets
                raw = self._ws.recv()
            except Exception as exc:
                if not self._stop.is_set():
                    self.last_error = str(exc)
                    logger.warning("OpenAI Realtime receive failed: %s", exc)
                self.stop()
                return
            if raw is None:
                self.stop()
                return
            try:
                frame = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
            except (TypeError, ValueError):
                continue
            if not isinstance(frame, dict):
                continue
            ftype = frame.get("type")
            if ftype == "response.audio.delta":
                b64 = frame.get("delta") or frame.get("audio") or ""
                try:
                    pcm24 = base64.b64decode(b64)
                except (TypeError, ValueError):
                    continue
                pcm48 = realtime_pcm_to_discord(pcm24)
                if pcm48:
                    self._enqueue_output(pcm48)
                    self.output_bytes += len(pcm48)
                    self.last_output_at = time.time()
            elif ftype == "input_audio_buffer.speech_started":
                # Barge-in: stop any queued/spoken model audio as soon as the user speaks.
                self._clear_output_queue()
            elif ftype == "error":
                self.last_error = str(frame.get("error") or frame)
                logger.warning("OpenAI Realtime error: %s", self.last_error)

    def _enqueue_output(self, pcm: bytes) -> None:
        try:
            self._output_q.put_nowait(pcm)
        except queue.Full:
            self._clear_output_queue()
            try:
                self._output_q.put_nowait(pcm)
            except queue.Full:
                pass

    def _clear_output_queue(self) -> None:
        while True:
            try:
                self._output_q.get_nowait()
            except queue.Empty:
                break

    def _send_json(self, payload: dict) -> None:
        if self._ws is None:
            raise RuntimeError("OpenAI Realtime WebSocket is not connected")
        with self._send_lock:
            self._ws.send(json.dumps(payload))

"""OpenAI Realtime API WebSocket client + file-queue speaker.

Text is sent to OpenAI Realtime and returned PCM is appended to a file that the
Google Meet audio bridge continuously forwards to Chrome's virtual microphone.
"""

from __future__ import annotations

import base64
import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..queue_io import append_jsonl, read_jsonl, remove_jsonl_entry


REALTIME_URL = "wss://api.openai.com/v1/realtime"
_TERMINAL_FRAMES = {"response.done", "response.completed", "response.cancelled"}


def _decode_audio(encoded: str) -> bytes:
    try:
        return base64.b64decode(encoded) if encoded else b""
    except (TypeError, ValueError):
        return b""


class RealtimeSession:
    """Synchronous OpenAI Realtime connection with serialized WebSocket writes."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-realtime",
        voice: str = "alloy",
        instructions: str = "",
        audio_sink_path: Optional[Path] = None,
        sample_rate: int = 24000,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.audio_sink_path = Path(audio_sink_path) if audio_sink_path else None
        self.sample_rate = sample_rate
        self._ws: Any = None
        self._send_lock = threading.Lock()
        self.audio_bytes_out = 0
        self.last_audio_out_at: Optional[float] = None

    def connect(self) -> None:
        """Open the connection and configure the server-side realtime session."""
        try:
            from websockets.sync.client import connect  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency is optional.
            raise RuntimeError(
                "websockets package is required for OpenAI Realtime; install with: pip install websockets"
            ) from exc
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        url = f"{REALTIME_URL}?model={self.model}"
        try:
            self._ws = connect(url, additional_headers=headers)
        except TypeError:
            self._ws = connect(url, extra_headers=headers)
        self._send_json(
            {
                "type": "session.update",
                "session": {
                    "voice": self.voice,
                    "instructions": self.instructions,
                    "modalities": ["audio", "text"],
                    "output_audio_format": "pcm16",
                    "input_audio_format": "pcm16",
                },
            }
        )

    def close(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.close()
            self._ws = None

    def speak(self, text: str, timeout: float = 30.0) -> dict:
        """Request one audio response and append its PCM deltas to the sink file."""
        if self._ws is None:
            raise RuntimeError("RealtimeSession.connect() must be called first")
        started = time.monotonic()
        self._send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        self._send_json(
            {"type": "response.create", "response": {"modalities": ["audio"]}}
        )
        bytes_written = 0
        with contextlib.ExitStack() as stack:
            sink = None
            if self.audio_sink_path is not None:
                self.audio_sink_path.parent.mkdir(parents=True, exist_ok=True)
                sink = stack.enter_context(self.audio_sink_path.open("ab"))
            while True:
                frame = self._recv_frame(started + timeout, timeout)
                if frame is None or frame.get("type") in _TERMINAL_FRAMES:
                    break
                frame_type = frame.get("type")
                if frame_type == "error":
                    raise RuntimeError(f"realtime error: {frame.get('error') or frame}")
                if frame_type != "response.audio.delta" or sink is None:
                    continue
                chunk = _decode_audio(frame.get("delta") or frame.get("audio") or "")
                if not chunk:
                    continue
                sink.write(chunk)
                sink.flush()
                bytes_written += len(chunk)
                self.audio_bytes_out += len(chunk)
                self.last_audio_out_at = time.time()
        return {
            "ok": True,
            "bytes_written": bytes_written,
            "duration_ms": (time.monotonic() - started) * 1000.0,
        }

    def cancel_response(self) -> bool:
        """Cancel the in-flight response for a human barge-in when connected."""
        if self._ws is None:
            return False
        try:
            self._send_json({"type": "response.cancel"})
        except Exception:
            return False
        return True

    def _send_json(self, payload: dict) -> None:
        assert self._ws is not None
        with self._send_lock:
            self._ws.send(json.dumps(payload))

    def _recv_frame(self, deadline: float, timeout: float) -> Optional[dict]:
        """Return the next JSON object before *deadline*, ignoring malformed frames."""
        assert self._ws is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"realtime response did not complete within {timeout}s"
                )
            try:
                raw = self._ws.recv(timeout=remaining)
            except TypeError:
                raw = self._ws.recv()
            if raw is None:
                return None
            with contextlib.suppress(TypeError, ValueError):
                frame = (
                    json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
                )
                if isinstance(frame, dict):
                    return frame


class RealtimeSpeaker:
    """Consume the addressable JSONL speech queue one entry at a time."""

    def __init__(self, session: RealtimeSession, queue_path: Path, processed_path: Optional[Path] = None) -> None:
        self.session = session
        self.queue_path = Path(queue_path)
        self.processed_path = Path(processed_path) if processed_path else None

    def _read_queue(self) -> list[dict]:
        return read_jsonl(self.queue_path)

    def _remove_processed(self, entry: dict) -> None:
        remove_jsonl_entry(self.queue_path, str(entry.get("id") or ""))

    def _append_processed(self, entry: dict, result: dict) -> None:
        if self.processed_path is None:
            return
        append_jsonl(
            self.processed_path,
            {"id": entry.get("id"), "text": entry.get("text", ""), "result": result},
        )

    def run_until_stopped(
        self, stop_fn: Callable[[], bool], poll_interval: float = 0.5
    ) -> None:
        while not stop_fn():
            entries = self._read_queue()
            if not entries:
                time.sleep(poll_interval)
                continue
            entry = entries[0]
            text = str(entry.get("text") or "").strip()
            result = {"ok": True, "bytes_written": 0, "duration_ms": 0.0}
            if text:
                try:
                    result = self.session.speak(text)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
            self._append_processed(entry, result)
            self._remove_processed(entry)

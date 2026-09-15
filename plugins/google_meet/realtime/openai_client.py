"""OpenAI Realtime API WebSocket client + file-queue speaker.

text → OpenAI Realtime → audio deltas appended as PCM to a file the audio bridge streams into
Chrome's fake mic. One sync WebSocket per session; ``websockets`` is imported lazily.
"""

from __future__ import annotations

import base64
import contextlib
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional


REALTIME_URL = "wss://api.openai.com/v1/realtime"

_TERMINAL_FRAMES = {"response.done", "response.completed", "response.cancelled"}


def _decode_audio(b64: str) -> bytes:
    try:
        return base64.b64decode(b64) if b64 else b""
    except (ValueError, TypeError):
        return b""


class RealtimeSession:
    """Minimal sync client for the OpenAI Realtime WebSocket API; ``speak`` and ``cancel_response``
    may run on different threads — a lock serializes WebSocket writes."""

    def __init__(self, api_key: str, model: str = "gpt-realtime", voice: str = "alloy",
                 instructions: str = "", audio_sink_path: Optional[Path] = None, sample_rate: int = 24000) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.audio_sink_path = Path(audio_sink_path) if audio_sink_path else None
        self.sample_rate = sample_rate
        self._ws: Any = None
        self._send_lock = threading.Lock()
        self.audio_bytes_out: int = 0  # public counters for status reporting
        self.last_audio_out_at: Optional[float] = None

    def connect(self) -> None:
        """Open the WS and send ``session.update`` with voice + instructions."""
        try:
            from websockets.sync.client import connect  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised via test
            raise RuntimeError("websockets package is required for OpenAI Realtime; "
                               "install with: pip install websockets") from exc
        url = f"{REALTIME_URL}?model={self.model}"
        headers = [("Authorization", f"Bearer {self.api_key}"), ("OpenAI-Beta", "realtime=v1")]
        # Newer websockets takes additional_headers=, older extra_headers=.
        try:
            self._ws = connect(url, additional_headers=headers)
        except TypeError:
            self._ws = connect(url, extra_headers=headers)
        self._send_json({"type": "session.update", "session": {
            "voice": self.voice, "instructions": self.instructions, "modalities": ["audio", "text"],
            "output_audio_format": "pcm16", "input_audio_format": "pcm16"}})

    def close(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.close()
            self._ws = None

    def speak(self, text: str, timeout: float = 30.0) -> dict:
        """Send ``text`` and append the audio response to ``audio_sink_path`` (opened 'ab' per call
        so a streaming reader can consume it). Frames other than audio deltas/terminal/error are ignored."""
        if self._ws is None:
            raise RuntimeError("RealtimeSession.connect() must be called first")
        start = time.monotonic()
        self._send_json({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
        self._send_json({"type": "response.create", "response": {"modalities": ["audio"]}})
        bytes_written = 0
        with contextlib.ExitStack() as stack:
            sink_fp = None
            if self.audio_sink_path is not None:
                self.audio_sink_path.parent.mkdir(parents=True, exist_ok=True)
                sink_fp = stack.enter_context(open(self.audio_sink_path, "ab"))
            while True:
                frame = self._recv_frame(start + timeout, timeout)
                if frame is None or frame.get("type") in _TERMINAL_FRAMES:  # peer closed / response done
                    break
                ftype = frame.get("type")
                if ftype == "error":
                    raise RuntimeError(f"realtime error: {frame.get('error') or frame}")
                chunk = _decode_audio(frame.get("delta") or frame.get("audio") or "") if (
                    ftype == "response.audio.delta" and sink_fp is not None) else b""
                if chunk:
                    sink_fp.write(chunk)
                    sink_fp.flush()
                    bytes_written += len(chunk)
                    self.audio_bytes_out += len(chunk)
                    self.last_audio_out_at = time.time()
        return {"ok": True, "bytes_written": bytes_written, "duration_ms": (time.monotonic() - start) * 1000.0}

    def cancel_response(self) -> bool:
        """Barge-in: send ``response.cancel``. True if sent, False if nothing to cancel / socket closed."""
        if self._ws is None:
            return False
        try:
            self._send_json({"type": "response.cancel"})
            return True
        except Exception:
            return False

    def _send_json(self, payload: dict) -> None:
        assert self._ws is not None
        with self._send_lock:
            self._ws.send(json.dumps(payload))

    def _recv_frame(self, deadline: float, timeout: float) -> Optional[dict]:
        """Next dict frame before *deadline* (monotonic), ``None`` once the peer closes.
        Non-dict / unparseable frames are skipped; TimeoutError past the deadline."""
        assert self._ws is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"realtime response did not complete within {timeout}s")
            try:
                raw = self._ws.recv(timeout=remaining)
            except TypeError:  # older websockets: no timeout kwarg
                raw = self._ws.recv()
            if raw is None:
                return None
            with contextlib.suppress(TypeError, ValueError):
                frame = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
                if isinstance(frame, dict):
                    return frame


class RealtimeSpeaker:
    """JSONL queue (``{"id", "text"}`` per line) wrapper around :class:`RealtimeSession`; processed
    lines are appended to ``processed_path`` (if set) and removed from the queue."""

    def __init__(self, session: RealtimeSession, queue_path: Path, processed_path: Optional[Path] = None) -> None:
        self.session = session
        self.queue_path = Path(queue_path)
        self.processed_path = Path(processed_path) if processed_path else None

    def _read_queue(self) -> list[dict]:
        """Parse the JSONL queue, skipping blank/malformed lines; entries lacking an ``id`` get one."""
        if not self.queue_path.exists():
            return []
        out: list[dict] = []
        for line in self.queue_path.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(ValueError):
                entry = json.loads(line) if line.strip() else None
                if isinstance(entry, dict):
                    entry.setdefault("id", str(uuid.uuid4()))
                    out.append(entry)
        return out

    def _rewrite_queue(self, remaining: list[dict]) -> None:
        # Always keep the file (empty when drained): consumers may watch its mtime.
        body = "".join(json.dumps(e) + "\n" for e in remaining)
        self.queue_path.write_text(body, encoding="utf-8")

    def _append_processed(self, entry: dict, result: dict) -> None:
        if self.processed_path is None:
            return
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"id": entry.get("id"), "text": entry.get("text", ""), "result": result}
        with open(self.processed_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")

    def run_until_stopped(self, stop_fn: Callable[[], bool], poll_interval: float = 0.5) -> None:
        while not stop_fn():
            entries = self._read_queue()
            if not entries:
                time.sleep(poll_interval)
                continue
            # One entry per iteration: the queue may grow while we speak.
            head = entries[0]
            text = (head.get("text") or "").strip()
            result = {"ok": True, "bytes_written": 0, "duration_ms": 0.0}
            if text:
                try:
                    result = self.session.speak(text)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
            self._append_processed(head, result)
            # Re-read (new entries may have arrived), then drop the head by position or id.
            latest = self._read_queue()
            self._rewrite_queue(latest[1:] if latest and latest[0].get("id") == head.get("id")
                                else [e for e in latest if e.get("id") != head.get("id")])

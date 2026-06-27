"""OpenAI Realtime API WebSocket client + file-queue speaker.

This module is the "output" side of the v2 voice bridge: it takes text,
sends it to the OpenAI Realtime API, receives audio deltas back, and
appends the PCM bytes to a file. A separate consumer (the audio
bridge) streams that file into Chrome's fake microphone.

Designed for simplicity: a single synchronous WebSocket connection per
speaker, per session. The ``websockets`` package is imported lazily so
that importing this module never fails just because the optional dep
is missing.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional


REALTIME_URL = "wss://api.openai.com/v1/realtime"


def _require_websockets():
    """Import ``websockets.sync.client.connect`` or raise with hint."""
    try:
        from websockets.sync.client import connect as _connect  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via test
        raise RuntimeError(
            "websockets package is required for OpenAI Realtime; "
            "install with: pip install websockets"
        ) from exc
    return _connect


class RealtimeSession:
    """Minimal sync client for the OpenAI Realtime WebSocket API.

    Usage:
        sess = RealtimeSession(api_key=..., audio_sink_path=Path("out.pcm"))
        sess.connect()
        sess.speak("Hello team.")
        sess.close()

    Thread safety: ``speak`` and ``cancel_response`` may be called from
    different threads; a lock serializes WebSocket writes.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-realtime",
        voice: str = "alloy",
        instructions: str = "",
        audio_sink_path: Optional[Path] = None,
        sample_rate: int = 24000,
    ) -> None:
        import threading as _threading
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.audio_sink_path = Path(audio_sink_path) if audio_sink_path else None
        self.sample_rate = sample_rate
        self._ws: Any = None
        self._send_lock = _threading.Lock()
        self._last_response_id: Optional[str] = None
        # Public counters for status reporting.
        self.audio_bytes_out: int = 0
        self.last_audio_out_at: Optional[float] = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open WS and send session.update with voice+instructions."""
        connect = _require_websockets()
        url = f"{REALTIME_URL}?model={self.model}"
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        # websockets.sync.client.connect accepts either additional_headers=
        # (newer) or extra_headers= depending on version; try the newer
        # name first and fall back.
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
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ── speaking ──────────────────────────────────────────────────────────

    def speak(self, text: str, timeout: float = 30.0) -> dict:
        """Send ``text`` and accumulate the audio response.

        Audio deltas are base64-decoded and appended to
        ``audio_sink_path`` (opened 'ab' and closed per call, so a
        separate streaming reader can consume whatever is there).
        """
        if self._ws is None:
            raise RuntimeError("RealtimeSession.connect() must be called first")

        start = time.monotonic()

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
            {
                "type": "response.create",
                "response": {"modalities": ["audio"]},
            }
        )

        bytes_written = 0
        sink_fp = None
        if self.audio_sink_path is not None:
            self.audio_sink_path.parent.mkdir(parents=True, exist_ok=True)
            sink_fp = open(self.audio_sink_path, "ab")

        try:
            while True:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError(
                        f"realtime response did not complete within {timeout}s"
                    )
                raw = self._recv(timeout=remaining)
                if raw is None:
                    # Connection closed by peer.
                    break
                try:
                    frame = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
                except (TypeError, ValueError):
                    continue
                if not isinstance(frame, dict):
                    continue
                ftype = frame.get("type")
                if ftype == "response.audio.delta":
                    b64 = frame.get("delta") or frame.get("audio") or ""
                    if b64 and sink_fp is not None:
                        try:
                            chunk = base64.b64decode(b64)
                        except (ValueError, TypeError):
                            chunk = b""
                        if chunk:
                            sink_fp.write(chunk)
                            sink_fp.flush()
                            bytes_written += len(chunk)
                            self.audio_bytes_out += len(chunk)
                            self.last_audio_out_at = time.time()
                elif ftype == "response.created":
                    rid = (frame.get("response") or {}).get("id")
                    if rid:
                        self._last_response_id = rid
                elif ftype in ("response.done", "response.completed", "response.cancelled"):
                    break
                elif ftype == "error":
                    err = frame.get("error") or frame
                    raise RuntimeError(f"realtime error: {err}")
                # All other frames (response.created, response.output_item.*,
                # response.audio_transcript.delta, rate_limits.updated, ...)
                # are ignored for v2.
        finally:
            if sink_fp is not None:
                sink_fp.close()

        duration_ms = (time.monotonic() - start) * 1000.0
        return {
            "ok": True,
            "bytes_written": bytes_written,
            "duration_ms": duration_ms,
        }

    # ── ws plumbing ───────────────────────────────────────────────────────

    def cancel_response(self) -> bool:
        """Interrupt the in-flight response (barge-in).

        Sends ``response.cancel`` on the current WebSocket so the model
        stops generating audio immediately. Safe to call at any time;
        returns True if a cancel was actually sent, False when there's
        nothing to cancel or the socket isn't open.
        """
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

    def _recv(self, timeout: Optional[float] = None):
        assert self._ws is not None
        try:
            if timeout is None:
                return self._ws.recv()
            return self._ws.recv(timeout=timeout)
        except TypeError:
            # Older websockets may not accept timeout kwarg.
            return self._ws.recv()


class RealtimeSpeaker:
    """File-based JSONL queue wrapper around :class:`RealtimeSession`.

    Each line in ``queue_path`` is a JSON object of the form
    ``{"id": "<uuid>", "text": "..."}``. Processed lines are appended
    to ``processed_path`` (if set) and then removed from the queue;
    if ``processed_path`` is ``None``, processed lines are simply
    dropped.
    """

    def __init__(
        self,
        session: RealtimeSession,
        queue_path: Path,
        processed_path: Optional[Path] = None,
    ) -> None:
        self.session = session
        self.queue_path = Path(queue_path)
        self.processed_path = Path(processed_path) if processed_path else None

    # ── helpers ──────────────────────────────────────────────────────────

    def _read_queue(self) -> list[dict]:
        if not self.queue_path.exists():
            return []
        out: list[dict] = []
        for line in self.queue_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            if "id" not in entry:
                entry["id"] = str(uuid.uuid4())
            out.append(entry)
        return out

    def _rewrite_queue(self, remaining: list[dict]) -> None:
        if not remaining:
            # Keep the file but empty — consumers may be watching for
            # new writes via mtime, and delete-then-recreate is a race.
            self.queue_path.write_text("")
            return
        self.queue_path.write_text(
            "\n".join(json.dumps(e) for e in remaining) + "\n"
        )

    def _append_processed(self, entry: dict, result: dict) -> None:
        if self.processed_path is None:
            return
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"id": entry.get("id"), "text": entry.get("text", ""), "result": result}
        with open(self.processed_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")

    # ── main loop ────────────────────────────────────────────────────────

    def run_until_stopped(
        self,
        stop_fn: Callable[[], bool],
        poll_interval: float = 0.5,
    ) -> None:
        while not stop_fn():
            entries = self._read_queue()
            if not entries:
                time.sleep(poll_interval)
                continue
            # Process one at a time; re-check the queue file after each
            # speak() call because new entries may have arrived.
            head = entries[0]
            text = (head.get("text") or "").strip()
            if text:
                try:
                    result = self.session.speak(text)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
            else:
                result = {"ok": True, "bytes_written": 0, "duration_ms": 0.0}
            self._append_processed(head, result)

            # Re-read the queue from disk in case it was appended to
            # while we were speaking, then drop the head.
            latest = self._read_queue()
            if latest and latest[0].get("id") == head.get("id"):
                self._rewrite_queue(latest[1:])
            else:
                # Fallback: drop-by-id anywhere in the queue.
                self._rewrite_queue(
                    [e for e in latest if e.get("id") != head.get("id")]
                )

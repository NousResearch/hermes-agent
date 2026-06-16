"""Cartesia Ink-2 streaming STT — conversational ASR with model-integrated
turn detection (the Realtime "Auto" turns websocket).

Like Deepgram Flux, Ink-2 is a single model that does transcription AND
turn-taking: Cartesia owns turn detection server-side and emits
conversation-native turn events. This adapter normalizes those onto the same
EV_* contract Flux uses, so the turn loop drives it identically — the seam
exists so we can A/B Ink-2 against Flux on turn-taking + latency. Accuracy is
out of scope here.

Input: raw s16le mono PCM at the call's native rate. Unlike Flux (which we
resample 24k->16k), Ink-2 accepts the declared sample_rate directly, so we send
the inbound mic audio AS-IS and just declare its rate in the URL — no resample.

Wire schema VERIFIED live 2026-06-16 (probe against the real socket):
  {"type":"connected","request_id":...}                       -> ignored ack
  {"type":"turn.start","turn_id":"1","request_id":...}
  {"type":"turn.update","transcript":"Hi, I'm","turn_id":"1",...}   (cumulative)
  {"type":"turn.eager_end","transcript":"...","turn_id":"1",...}
  {"type":"turn.resume","turn_id":"1",...}                     (no transcript)
  {"type":"turn.end","transcript":"...","turn_id":"1",...}     (final)
Transcript lives in ``transcript``; the id field is ``turn_id`` (string);
there is no ``confidence`` or per-word list; eager-end fires by default (no
opt-in param). Connection params + ``Authorization: Bearer`` + ``Cartesia-Version``
header are confirmed working.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)

# Canonical normalized event names live in stt.py (the port).
try:
    from stt import EV_EAGER_EOT, EV_END, EV_RESUMED, EV_START, EV_UPDATE
except ImportError:
    from .stt import EV_EAGER_EOT, EV_END, EV_RESUMED, EV_START, EV_UPDATE

WS_BASE = "wss://api.cartesia.ai/stt/turns/websocket"
INK_MODEL = "ink-2"
CARTESIA_VERSION = "2026-03-01"

# Verified live 2026-06-16: top-level message "type" -> normalized EV_*.
_INK_EVENT_MAP = {
    "turn.start": EV_START,
    "turn.update": EV_UPDATE,
    "turn.eager_end": EV_EAGER_EOT,
    "turn.resume": EV_RESUMED,
    "turn.end": EV_END,
}


def _normalize_ink_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a raw Ink-2 server message to the normalized turn event the loop
    consumes, or None for messages we ignore. ALL wire-schema assumptions are
    isolated here so a first live call corrects them in one place.

    Yields the same shape Flux yields:
        {"event", "transcript", "confidence", "words", "turn_index"}
    """
    ev = _INK_EVENT_MAP.get(msg.get("type"))
    if ev is None:
        # Non-turn frames ("connected" ack, metadata, errors) are not turn
        # events; the loop ignores them.
        return None
    # Verified: transcript in "transcript"; id in "turn_id" (string). Ink-2
    # sends no confidence or per-word list, so those stay None/[] (the turn
    # loop does not require them) — kept for shape-parity with the Flux adapter.
    return {
        "event": ev,
        "transcript": msg.get("transcript") or "",
        "confidence": None,
        "words": [],
        "turn_index": msg.get("turn_id"),
    }


class CartesiaInkSTT:
    """One Cartesia Ink-2 websocket for a call. ``events()`` yields normalized
    turn events; ``send_audio`` streams the mic audio at its native rate (no
    resample)."""

    provider = "cartesia_ink"   # telemetry tag (the STT port surface)

    def __init__(self, api_key: str, *, input_rate: int = 24000):
        if not api_key:
            raise ValueError("CARTESIA_API_KEY is required for Ink-2 STT")
        self._api_key = api_key
        self._input_rate = input_rate
        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None
        self._queue: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()
        self._closed = False
        self._asr_seconds = 0.0   # cost telemetry: audio seconds streamed

    @property
    def asr_seconds_est(self) -> float:
        """Estimated seconds of audio sent to ASR (for the per-call cost
        record)."""
        return self._asr_seconds

    def _url(self) -> str:
        # Realtime "Auto" endpoint (Cartesia owns turn detection). Verified
        # live: these params connect and eager-end fires by default.
        return (
            f"{WS_BASE}?model={INK_MODEL}"
            f"&encoding=pcm_s16le&sample_rate={self._input_rate}"
            f"&language=en"
        )

    async def start(self) -> None:
        import websockets
        self._ws = await websockets.connect(
            self._url(),
            additional_headers={
                "Authorization": f"Bearer {self._api_key}",
                "Cartesia-Version": CARTESIA_VERSION,
            },
            max_size=None, ping_interval=5, ping_timeout=10,
        )
        self._recv_task = asyncio.create_task(self._recv_loop(self._ws))
        logger.info("voice/stt cartesia-ink: connected model=%s rate=%d",
                    INK_MODEL, self._input_rate)

    async def _recv_loop(self, ws) -> None:
        try:
            async for raw in ws:
                msg = json.loads(raw)
                norm = _normalize_ink_message(msg)
                if norm is not None:
                    await self._queue.put(norm)
                # VERIFY-AGAINST-LIVE-API: error frame "type" string (assumed
                # "error"). Log it so live debugging surfaces wire mismatches.
                elif msg.get("type") in ("error", "Error"):
                    logger.warning("voice/stt cartesia-ink: %s: %s",
                                   msg.get("type"), msg)
                # Acks / metadata / unknown turn-detection frames: informational.
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closed:
                logger.warning("voice/stt cartesia-ink: socket closed: %s", e)
        finally:
            await self._queue.put(None)   # unblock events()

    async def send_audio(self, pcm: bytes) -> None:
        if self._closed or self._ws is None:
            return
        self._asr_seconds += len(pcm) / 2 / self._input_rate   # s16le mono
        try:
            # Ink-2 accepts the declared sample_rate directly: send raw s16le
            # binary frames at the native input rate (no resample).
            await self._ws.send(pcm)
        except Exception as e:
            if not self._closed:
                logger.warning("voice/stt cartesia-ink: send failed: %s", e)

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield normalized turn events until the stream ends."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def configure(self, **kw: Any) -> None:
        """No-op: Ink-2 turn detection is model-controlled (no eot/eager
        thresholds yet), so there is nothing to tune mid-stream. Logged once at
        INFO so a tuning attempt isn't silently swallowed."""
        if not getattr(self, "_configure_warned", False):
            self._configure_warned = True
            logger.info(
                "voice/stt cartesia-ink: configure() is a no-op — Ink-2 turn "
                "detection is model-controlled (no tunable thresholds)")

    async def stop(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                # VERIFY-AGAINST-LIVE-API: close/finalize control message shape
                # (assumed {"type":"close"}; Flux uses {"type":"CloseStream"}).
                await self._ws.send(json.dumps({"type": "close"}))
            except Exception:
                pass
        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

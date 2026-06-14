"""Cartesia streaming TTS over a PERSISTENT websocket.

Why persistent (not one socket per turn): a fresh Cartesia websocket costs
~600-1400ms to connect (TLS + handshake), measured live
2026-06-13. Paying that per turn would dwarf the ~200-310ms synthesis
latency. Cartesia's own guidance is to amortize connect across turns and use
one ``context_id`` per utterance ("Live session (WebSocket)" is the
recommended endpoint for streaming LLM tokens). So a single
``CartesiaTTSClient`` holds the socket for the whole call and hands out a
lightweight ``CartesiaTTSTurn`` per agent turn.

Model: ``sonic-3.5`` (latest — "fastest, most natural", sub-90ms model
latency). Output: 48 kHz s16le mono PCM to match the Daily transport exactly
(no resampling). Managed buffering: sentences are sent with ``continue=true``
ending in punctuation, so the model has enough context to start promptly; the
turn closes with ``continue=false``. The turn loop's ``<flush>`` sentinel maps
to Cartesia's native manual flush (``flush=true`` + empty transcript), never
spoken as text.

Barge-in: server-side cancel is unreliable (verified live — audio chunks keep
arriving after a ``cancel``), so barge-in is CLIENT-SIDE: ``mute()`` drops
audio immediately and ``context_id`` routing guarantees a finished/aborted
turn's audio never reaches the transport.

Wire facts verified live (2026-06-13): audio arrives as
``{"type":"chunk","context_id":...,"data":"<base64 pcm>"}`` (the ``data``
field, not the doc summary's ``audio``); turn ends with ``{"type":"done"}``;
errors as ``{"type":"error","error":...}``. We read ``data`` first and fall
back to ``audio`` defensively across API versions.

Resilience: the websockets library answers ping/pong automatically; we set an
explicit ping_interval so the socket survives silent gaps BETWEEN turns
(Cartesia closes idle sockets). If the socket drops, the next send lazily
reconnects.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_VERSION = "2024-11-13"
OUTPUT_SAMPLE_RATE = 48000          # matches the Daily transport (no resample)
DEFAULT_MODEL = "sonic-3.5"
FLUSH_SENTINEL = "<flush>"          # force-synth token the turn loop emits
PING_INTERVAL_S = 15                # keep the socket alive across idle turn gaps
PING_TIMEOUT_S = 15


class CartesiaTTSClient:
    """Persistent Cartesia websocket for one call.

    One turn is active at a time (the turn loop guarantees this). A single
    recv loop routes audio chunks to the active turn by ``context_id``;
    chunks for a finished/barged-in context are dropped.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        *,
        model: str = DEFAULT_MODEL,
        speed: Optional[float] = None,
        volume: Optional[float] = None,
        emotion: Optional[str] = None,
        max_buffer_delay_ms: Optional[int] = None,
    ):
        if not api_key:
            raise ValueError("CARTESIA_API_KEY is required for cartesia TTS")
        if not voice_id:
            raise ValueError("a Cartesia voice_id is required for cartesia TTS")
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model or DEFAULT_MODEL
        self._gen_cfg: Dict[str, Any] = {}
        if speed is not None:
            self._gen_cfg["speed"] = speed
        if volume is not None:
            self._gen_cfg["volume"] = volume
        if emotion:
            self._gen_cfg["emotion"] = emotion
        self._max_buffer_delay_ms = max_buffer_delay_ms
        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None
        self._active: Optional["CartesiaTTSTurn"] = None
        self._turn_seq = 0
        self._reconnects = 0          # observability: socket drops this call

    async def connect(self) -> None:
        """Pre-pay the socket connect during call setup so turn 1 is fast."""
        t0 = time.monotonic()
        await self._ensure_connected()
        logger.info("voice/tts cartesia: connected model=%s voice=%s in %dms",
                    self._model, self._voice_id,
                    int((time.monotonic() - t0) * 1000))

    async def _ensure_connected(self) -> None:
        if self._ws is not None and not getattr(self._ws, "closed", False):
            return
        import websockets
        if self._ws is not None:
            self._reconnects += 1
            logger.warning("voice/tts cartesia: socket was closed; "
                           "reconnecting (#%d this call)", self._reconnects)
        url = f"{WS_URL}?api_key={self._api_key}&cartesia_version={CARTESIA_VERSION}"
        self._ws = await websockets.connect(
            url, max_size=None,
            ping_interval=PING_INTERVAL_S, ping_timeout=PING_TIMEOUT_S,
            close_timeout=2)
        self._recv_task = asyncio.create_task(self._recv_loop(self._ws))

    async def _recv_loop(self, ws) -> None:
        try:
            async for raw in ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                active = self._active
                if active is None or msg.get("context_id") != active.context_id:
                    continue  # stale (previous/barged-in turn) — never forward
                if mtype == "chunk":
                    if not active._muted:
                        data = msg.get("data") or msg.get("audio")
                        if data:
                            pcm = base64.b64decode(data)
                            if active._first_audio_t is None:
                                active._first_audio_t = time.monotonic()
                            active._chunks += 1
                            active._bytes += len(pcm)
                            await active._on_audio(pcm)
                elif mtype == "done":
                    active._done.set()
                elif mtype == "error":
                    active._error = msg.get("error") or "error"
                    logger.warning(
                        "voice/tts cartesia: server error ctx=%s req=%s: %s",
                        msg.get("context_id"), msg.get("request_id"),
                        active._error)
                    active._done.set()
                elif mtype == "flush_done":
                    pass  # context boundary marker; single-turn doesn't need it
                # timestamps / word messages: not used here
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("voice/tts cartesia: recv loop ended: %s", e)
        finally:
            if self._active is not None:
                # never let end() hang if the socket dropped mid-turn
                self._active._done.set()
            if self._ws is ws:
                self._ws = None

    def new_turn(
        self, on_audio: Callable[[bytes], Awaitable[None]]
    ) -> "CartesiaTTSTurn":
        self._turn_seq += 1
        return CartesiaTTSTurn(self, f"turn-{self._turn_seq}", on_audio)

    def _request(self, context_id: str, transcript: str, cont: bool,
                 flush: bool = False) -> Dict[str, Any]:
        req: Dict[str, Any] = {
            "model_id": self._model,
            "transcript": transcript,
            "voice": {"mode": "id", "id": self._voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": OUTPUT_SAMPLE_RATE,
            },
            "language": "en",
            "context_id": context_id,
            "continue": cont,
        }
        if flush:
            req["flush"] = True
        if self._gen_cfg:
            req["generation_config"] = dict(self._gen_cfg)
        if self._max_buffer_delay_ms is not None:
            req["max_buffer_delay_ms"] = self._max_buffer_delay_ms
        return req

    async def _send(self, payload: Dict[str, Any]) -> None:
        await self._ensure_connected()
        await self._ws.send(json.dumps(payload))

    async def close(self) -> None:
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
        self._active = None


class CartesiaTTSTurn:
    """One agent turn on the client's shared socket. Interface the turn loop
    expects: open / send_text / send_filler / end / mute / abort."""

    def __init__(self, client: CartesiaTTSClient, context_id: str,
                 on_audio: Callable[[bytes], Awaitable[None]]):
        self._client = client
        self.context_id = context_id
        self._on_audio = on_audio
        self._done = asyncio.Event()
        self._muted = False
        self._aborted = False
        self.chars_sent = 0
        # per-turn observability
        self._first_send_t: Optional[float] = None
        self._first_audio_t: Optional[float] = None
        self._chunks = 0
        self._bytes = 0
        self._error: Optional[str] = None

    async def open(self) -> None:
        # Become the active turn; the socket is already connected (or lazily
        # connects). Audio for this context_id now routes to our on_audio.
        self._client._active = self
        await self._client._ensure_connected()

    async def send_text(self, text: str) -> None:
        if self._aborted or not text:
            return
        if text.strip() == FLUSH_SENTINEL:
            # <flush> sentinel -> Cartesia native manual flush. Never spoken.
            # continue=true keeps the context open for more text.
            try:
                await self._client._send(
                    self._client._request(self.context_id, "", True, flush=True))
            except Exception:
                pass
            return
        text = text.replace(FLUSH_SENTINEL, "").strip()
        if not text:
            return
        if self._first_send_t is None:
            self._first_send_t = time.monotonic()
        self.chars_sent += len(text)
        await self._client._send(
            self._client._request(self.context_id, text, True))

    async def send_filler(self, text: str) -> None:
        """Filler utterance (tool-latency masking) + an immediate flush so it
        is heard right away."""
        if self._aborted or not text:
            return
        await self.send_text(text)
        try:
            await self._client._send(
                self._client._request(self.context_id, "", True, flush=True))
        except Exception:
            pass

    async def end(self) -> None:
        """Close the context (continue=false flushes remaining audio) and wait
        for the final ``done``. Callers wrap this in asyncio.wait_for."""
        if self._aborted:
            return
        try:
            await self._client._send(
                self._client._request(self.context_id, "", False))
        except Exception:
            pass
        try:
            await self._done.wait()
        finally:
            self._log_turn()
            if self._client._active is self:
                self._client._active = None

    def _log_turn(self) -> None:
        ttfb = None
        if self._first_send_t is not None and self._first_audio_t is not None:
            ttfb = int((self._first_audio_t - self._first_send_t) * 1000)
        audio_s = self._bytes / 2 / OUTPUT_SAMPLE_RATE
        if self._error:
            logger.warning(
                "voice/tts cartesia: turn ctx=%s ERROR=%s chars=%d chunks=%d",
                self.context_id, self._error, self.chars_sent, self._chunks)
        else:
            logger.info(
                "voice/tts cartesia: turn ctx=%s ttfb=%sms chars=%d chunks=%d "
                "audio=%.1fs", self.context_id,
                ttfb if ttfb is not None else "-", self.chars_sent,
                self._chunks, audio_s)

    def mute(self) -> None:
        """Synchronous barge-in step: stop forwarding audio IMMEDIATELY.
        Callers clear the transport queue right after."""
        self._muted = True

    async def abort(self) -> None:
        """Barge-in: drop pending audio (client-side) and best-effort tell the
        server to stop (frees credits; not relied on — verified unreliable)."""
        self._aborted = True
        self._muted = True
        try:
            await self._client._send(
                {"context_id": self.context_id, "cancel": True})
        except Exception:
            pass
        self._done.set()
        if self._client._active is self:
            self._client._active = None

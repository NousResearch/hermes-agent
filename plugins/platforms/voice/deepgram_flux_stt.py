"""Deepgram Flux streaming STT — conversational ASR with model-integrated
turn detection (the /v2/listen websocket API).

Flux is a single fused model that does transcription AND turn-taking. Instead
of a separate ASR + VAD + end-of-turn-detection stack, Flux emits
conversation-native events that the turn loop reacts to directly:

  - StartOfTurn     -> the user began speaking (barge-in trigger)
  - Update          -> incremental partial transcript (~every 250ms)
  - EagerEndOfTurn  -> medium-confidence end (~150-250ms early) -> start the
                       LLM speculatively
  - TurnResumed     -> the user kept talking past the eager signal -> cancel
                       the speculative response
  - EndOfTurn       -> high-confidence end -> commit the turn (synthesize)

Verified live 2026-06-14: every server message is ``{"type":"TurnInfo",
"event":"StartOfTurn"|...,"transcript":...,"end_of_turn_confidence":...,
"words":[...]}``; StartOfTurn carries a non-empty transcript; EagerEndOfTurn
lands ~185ms before EndOfTurn.

Input: linear16 @ 16 kHz mono. Our inbound mic is 24 kHz (the Daily ASR rate),
so audio is resampled 24k->16k here (linear interpolation; the 24->16 ratio of
speech-band audio tolerates it — confirmed accurate transcription live).

Opinionated: Deepgram-only, no provider seam — Flux's model-integrated turn
detection IS the architecture, not a swappable component.
"""

from __future__ import annotations

import array
import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)

WS_BASE = "wss://api.deepgram.com/v2/listen"
FLUX_MODEL = "flux-general-en"
FLUX_RATE = 16000               # Flux input rate (linear16 mono)
DEFAULT_EOT_THRESHOLD = 0.7
DEFAULT_EAGER_EOT_THRESHOLD = 0.5
DEFAULT_EOT_TIMEOUT_MS = 5000

# Normalized event names the turn loop consumes (provider-agnostic). Their
# canonical home is stt.py (the port); re-exported here so existing
# `from deepgram_flux_stt import EV_*` imports keep working.
try:
    from stt import EV_EAGER_EOT, EV_END, EV_RESUMED, EV_START, EV_UPDATE
except ImportError:
    from .stt import EV_EAGER_EOT, EV_END, EV_RESUMED, EV_START, EV_UPDATE

_FLUX_EVENT_MAP = {
    "StartOfTurn": EV_START,
    "Update": EV_UPDATE,
    "EagerEndOfTurn": EV_EAGER_EOT,
    "TurnResumed": EV_RESUMED,
    "EndOfTurn": EV_END,
}


def _normalize_flux_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a raw Flux server message to the normalized turn event the loop
    consumes, or None for non-TurnInfo / unknown-event messages (which the loop
    ignores). Keeps the wire-format knowledge in one testable place."""
    if msg.get("type") != "TurnInfo":
        return None
    ev = _FLUX_EVENT_MAP.get(msg.get("event"))
    if ev is None:
        return None
    return {
        "event": ev,
        "transcript": msg.get("transcript") or "",
        "confidence": msg.get("end_of_turn_confidence"),
        "words": msg.get("words") or [],
        "turn_index": msg.get("turn_index"),
    }


def _resample_to_16k(pcm: bytes, in_rate: int) -> bytes:
    """Linear-interpolate s16le mono PCM from ``in_rate`` to 16 kHz.

    Pure stdlib (no numpy): the inbound chunk is ~1920 samples per 80 ms, so a
    plain interpolation loop is negligible and keeps the voice-platform extra to
    just daily-python + websockets. Per-chunk (stateless): the sub-sample
    boundary error is inaudible to an ASR model for 24k->16k speech-band audio.
    Assumes a little-endian host (``array('h')``), which every Hermes target is.
    """
    if in_rate == FLUX_RATE:
        return pcm
    src = array.array("h")
    src.frombytes(pcm)
    n_in = len(src)
    if n_in == 0:
        return b""
    n_out = max(1, int(round(n_in * FLUX_RATE / in_rate)))
    out = array.array("h", bytes(2 * n_out))
    if n_in == 1 or n_out == 1:
        for i in range(n_out):
            out[i] = src[0]
        return out.tobytes()
    # Map output index i -> source position, endpoints inclusive (i=0 -> 0,
    # i=n_out-1 -> n_in-1), matching a linspace+interp resample.
    step = (n_in - 1) / (n_out - 1)
    for i in range(n_out):
        pos = i * step
        j = int(pos)
        if j + 1 < n_in:
            frac = pos - j
            out[i] = int(src[j] + (src[j + 1] - src[j]) * frac)
        else:
            out[i] = src[j]
    return out.tobytes()


class DeepgramFluxSTT:
    """One Flux websocket for a call. ``events()`` yields normalized turn
    events; ``send_audio`` streams the (resampled) mic audio."""

    provider = "deepgram_flux"   # telemetry tag (the STT port surface)

    def __init__(
        self,
        api_key: str,
        *,
        input_rate: int = 24000,
        eot_threshold: float = DEFAULT_EOT_THRESHOLD,
        eager_eot_threshold: float = DEFAULT_EAGER_EOT_THRESHOLD,
        eot_timeout_ms: int = DEFAULT_EOT_TIMEOUT_MS,
    ):
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is required for Flux STT")
        if eager_eot_threshold > eot_threshold:
            # Flux rejects eager > eot; clamp so a misconfig can't 400 the call.
            eager_eot_threshold = eot_threshold
        self._api_key = api_key
        self._input_rate = input_rate
        self._eot_threshold = eot_threshold
        self._eager_eot_threshold = eager_eot_threshold
        self._eot_timeout_ms = eot_timeout_ms
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
        return (
            f"{WS_BASE}?model={FLUX_MODEL}"
            f"&encoding=linear16&sample_rate={FLUX_RATE}"
            f"&eot_threshold={self._eot_threshold}"
            f"&eager_eot_threshold={self._eager_eot_threshold}"
            f"&eot_timeout_ms={self._eot_timeout_ms}"
        )

    async def start(self) -> None:
        import websockets
        self._ws = await websockets.connect(
            self._url(),
            additional_headers={"Authorization": f"Token {self._api_key}"},
            max_size=None, ping_interval=5, ping_timeout=10,
        )
        self._recv_task = asyncio.create_task(self._recv_loop(self._ws))
        logger.info("voice/stt flux: connected model=%s eot=%.2f eager=%.2f",
                    FLUX_MODEL, self._eot_threshold, self._eager_eot_threshold)

    async def _recv_loop(self, ws) -> None:
        try:
            async for raw in ws:
                msg = json.loads(raw)
                norm = _normalize_flux_message(msg)
                if norm is not None:
                    await self._queue.put(norm)
                elif msg.get("type") in ("Error", "ConfigureFailure"):
                    logger.warning("voice/stt flux: %s: %s",
                                   msg.get("type"), msg)
                # Connected / ConfigureSuccess / Metadata: informational
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closed:
                logger.warning("voice/stt flux: socket closed: %s", e)
        finally:
            await self._queue.put(None)   # unblock events()

    async def send_audio(self, pcm: bytes) -> None:
        if self._closed or self._ws is None:
            return
        self._asr_seconds += len(pcm) / 2 / self._input_rate   # s16le mono
        try:
            await self._ws.send(_resample_to_16k(pcm, self._input_rate))
        except Exception as e:
            if not self._closed:
                logger.warning("voice/stt flux: send failed: %s", e)

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield normalized turn events until the stream ends."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def configure(
        self,
        *,
        eot_threshold: Optional[float] = None,
        eager_eot_threshold: Optional[float] = None,
        eot_timeout_ms: Optional[int] = None,
    ) -> None:
        """Update turn-detection thresholds mid-stream (no reconnect).
        Omitted values keep their current setting."""
        if self._ws is None:
            return
        th: Dict[str, Any] = {}
        if eot_threshold is not None:
            th["eot_threshold"] = eot_threshold
            self._eot_threshold = eot_threshold
        if eager_eot_threshold is not None:
            th["eager_eot_threshold"] = eager_eot_threshold
            self._eager_eot_threshold = eager_eot_threshold
        if eot_timeout_ms is not None:
            th["eot_timeout_ms"] = eot_timeout_ms
            self._eot_timeout_ms = eot_timeout_ms
        if not th:
            return
        try:
            await self._ws.send(json.dumps({"type": "Configure", "thresholds": th}))
        except Exception:
            pass

    async def stop(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
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

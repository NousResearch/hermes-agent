"""The realtime bridge: carrier media stream ⇄ realtime voice model.

One :class:`RealtimeCallBridge` per live realtime call:

- the carrier opens a WebSocket to ``/voice/stream/{token}`` (the one-shot
  token was minted when the call was dialed/answered and embedded in the
  ``stream_url`` the carrier received)
- inbound pump: carrier frame → µ-law decode → resample → model
- outbound pump: model audio → resample to 8 kHz → µ-law → 20 ms pacer →
  carrier; barge-in (caller starts speaking) clears the carrier's queued
  audio and cancels the in-flight model response
- ``agent_consult`` tool calls run a host-owned completion through the
  plugin LLM facade (``ctx.llm``) so the realtime model can answer with
  the user's data without leaving the audio loop
- transcripts mirror into the CallRecord for history/persistence
"""

import asyncio
import logging
import secrets
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from .. import audio
from ..events import CallRecord
from .base import RealtimeVoiceSession
from .frames import StreamFrameAdapter, adapter_for_provider
from .gemini_live import create_realtime_session

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import VoiceCallRuntime

logger = logging.getLogger(__name__)

TOKEN_TTL_S = 300.0
CONSULT_TIMEOUT_S = 45.0
CONSULT_MAX_CHARS = 2000

# Set by the plugin's register() so the bridge can reach ctx.llm without
# holding a PluginContext reference through every layer.
_plugin_llm_factory: Optional[Callable[[], Any]] = None


def set_plugin_llm_factory(factory: Callable[[], Any]) -> None:
    global _plugin_llm_factory
    _plugin_llm_factory = factory


class AudioPacer:
    """Sends one 160-byte µ-law frame every 20 ms on an absolute schedule
    (drift-corrected — no cumulative delay), like OpenClaw's audio pacer."""

    def __init__(self, send_frame: Callable[[bytes], Any]):
        self._send = send_frame
        self._frames: deque = deque()
        self._running = False

    def push(self, ulaw: bytes) -> None:
        self._frames.extend(audio.chunk_frames(ulaw))

    def clear(self) -> int:
        dropped = len(self._frames)
        self._frames.clear()
        return dropped

    @property
    def pending(self) -> int:
        return len(self._frames)

    async def run(self) -> None:
        self._running = True
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        try:
            while self._running:
                if self._frames:
                    frame = self._frames.popleft()
                    await self._send(frame)
                    next_at += audio.FRAME_SECONDS
                    delay = next_at - loop.time()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    elif delay < -1.0:
                        next_at = loop.time()  # fell badly behind; resync
                else:
                    next_at = loop.time()
                    await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._running = False


class RealtimeBridgeManager:
    """Owned by the runtime when ``realtime.enabled``: mints one-shot stream
    tokens at dial/answer time and turns carrier WS upgrades into bridges."""

    def __init__(self, runtime: "VoiceCallRuntime"):
        self.runtime = runtime
        self._tokens: Dict[str, Tuple[str, float]] = {}  # token → (call_id, ts)
        self.active_bridges: Dict[str, "RealtimeCallBridge"] = {}

    # -- token lifecycle ------------------------------------------------------

    def mint_token(self, call_id: str) -> str:
        now = time.time()
        for token in [t for t, (_, ts) in self._tokens.items()
                      if ts < now - TOKEN_TTL_S]:
            self._tokens.pop(token, None)
        token = secrets.token_urlsafe(24)
        self._tokens[token] = (call_id, now)
        return token

    def consume_token(self, token: str) -> Optional[str]:
        """One-shot: a second use of the same token fails."""
        entry = self._tokens.pop(token, None)
        if entry is None:
            return None
        call_id, minted_at = entry
        if minted_at < time.time() - TOKEN_TTL_S:
            return None
        return call_id

    # -- dial/answer hook -------------------------------------------------------

    def prepare_call(self, record: CallRecord) -> None:
        """Attach stream metadata before the provider dials/answers."""
        if record.mode != "conversation":
            return  # notify calls keep plain carrier TTS
        config = self.runtime.config
        public = (self.runtime.public_url or "").rstrip("/")
        if not public:
            logger.warning(
                "voice_call realtime: no public URL — cannot attach stream to %s",
                record.call_id,
            )
            return
        wss_base = public.replace("https://", "wss://").replace("http://", "ws://")
        token = self.mint_token(record.call_id)
        stream_url = f"{wss_base}{config.serve.stream_path}/{token}"
        record.metadata["stream_url"] = stream_url
        record.metadata["stream_auth_token"] = token
        record.metadata["realtime"] = True
        # Twilio attaches streams via TwiML, not dial fields.
        provider = self.runtime.provider
        if provider is not None and hasattr(provider, "register_pending_stream"):
            provider.register_pending_stream(record, stream_url)

    # -- WS upgrade (installed as webhook_server.stream_handler) -----------------

    async def handle_stream_request(self, request):
        from aiohttp import web

        token = request.match_info.get("token", "")
        call_id = self.consume_token(token)
        if call_id is None:
            logger.warning("voice_call realtime: rejected stream with bad token")
            return web.json_response({"error": "invalid token"}, status=403)
        manager = self.runtime.manager
        record = manager.get_call(call_id) if manager else None
        if record is None:
            return web.json_response({"error": "no such call"}, status=404)

        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)

        try:
            session = create_realtime_session(self.runtime.config.realtime)
        except Exception as e:  # noqa: BLE001 — config/key errors
            logger.error("voice_call realtime: session create failed: %s", e)
            await ws.close()
            return ws

        bridge = RealtimeCallBridge(
            runtime=self.runtime,
            record=record,
            frame_adapter=adapter_for_provider(record.provider),
            session=session,
        )
        self.active_bridges[call_id] = bridge
        try:
            await bridge.run(ws)
        finally:
            self.active_bridges.pop(call_id, None)
        return ws


class RealtimeCallBridge:
    def __init__(
        self,
        runtime: "VoiceCallRuntime",
        record: CallRecord,
        frame_adapter: StreamFrameAdapter,
        session: RealtimeVoiceSession,
    ):
        self.runtime = runtime
        self.record = record
        self.adapter = frame_adapter
        self.session = session
        self._ws = None
        self._greeting_until = 0.0

    async def run(self, ws) -> None:
        self._ws = ws
        try:
            await self.session.connect()
        except Exception as e:  # noqa: BLE001
            logger.error("voice_call realtime: model connect failed: %s", e)
            await ws.close()
            return

        if self.record.direction == "inbound" and self.runtime.config.inbound_greeting:
            await self.session.inject_text(self.runtime.config.inbound_greeting)
            # Suppress barge-in while the greeting plays (OpenClaw behavior).
            self._greeting_until = time.time() + 3.0
        elif self.record.direction == "outbound":
            initial = self.record.metadata.pop("initial_message", None)
            if initial:
                await self.session.inject_text(str(initial))
                self._greeting_until = time.time() + 3.0

        pacer = AudioPacer(self._send_media_frame)
        tasks = [
            asyncio.create_task(self._inbound_pump(ws), name="vc-rt-inbound"),
            asyncio.create_task(self._outbound_pump(pacer), name="vc-rt-outbound"),
            asyncio.create_task(pacer.run(), name="vc-rt-pacer"),
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                exc = task.exception()
                if exc is not None:
                    logger.warning("voice_call realtime: %s failed: %s",
                                   task.get_name(), exc)
        finally:
            pacer.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.session.close()
            if not ws.closed:
                await ws.close()
        logger.info("voice_call realtime: bridge for %s ended", self.record.call_id)

    async def _send_media_frame(self, frame: bytes) -> None:
        if self._ws is not None and not self._ws.closed:
            await self._ws.send_str(self.adapter.serialize_media(frame))

    # -- carrier → model ---------------------------------------------------------

    async def _inbound_pump(self, ws) -> None:
        from aiohttp import WSMsgType

        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    return
                continue
            frame = self.adapter.parse(msg.data)
            if frame.type == "start":
                if frame.stream_id:
                    self.adapter.set_stream_id(frame.stream_id)
            elif frame.type == "media":
                if frame.track and "outbound" in frame.track:
                    continue  # never echo our own audio into the model
                pcm = audio.ulaw_to_pcm16(frame.payload)
                pcm = audio.resample_pcm16(pcm, 8000, self.session.input_sample_rate)
                await self.session.send_audio(pcm)
            elif frame.type == "stop":
                return
            elif frame.type == "error":
                logger.warning(
                    "voice_call realtime: carrier stream error: %s", frame.error
                )

    # -- model → carrier ----------------------------------------------------------

    async def _outbound_pump(self, pacer: AudioPacer) -> None:
        async for event in self.session.events():
            if event.type == "audio":
                pcm = audio.resample_pcm16(
                    event.audio, self.session.output_sample_rate, 8000
                )
                pacer.push(audio.pcm16_to_ulaw(pcm))
            elif event.type == "speech_started":
                await self._handle_barge_in(pacer)
            elif event.type == "transcript":
                self._record_transcript(event.role, event.text)
            elif event.type == "tool_call":
                asyncio.get_running_loop().create_task(
                    self._handle_tool_call(event)
                )
            elif event.type == "error":
                logger.warning("voice_call realtime: model error: %s", event.text)
            elif event.type == "closed":
                return

    async def _handle_barge_in(self, pacer: AudioPacer) -> None:
        if time.time() < self._greeting_until:
            return  # don't let line noise cut off the greeting
        dropped = pacer.clear()
        if self._ws is not None and not self._ws.closed:
            await self._ws.send_str(self.adapter.serialize_clear())
        try:
            await self.session.cancel_response()
        except Exception:  # noqa: BLE001
            logger.debug("voice_call realtime: cancel_response failed", exc_info=True)
        if dropped:
            logger.debug(
                "voice_call realtime: barge-in dropped %d queued frames", dropped
            )

    def _record_transcript(self, role: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        manager = self.runtime.manager
        if manager is not None:
            manager.append_transcript(
                self.record.call_id,
                "bot" if role == "assistant" else "user",
                text,
            )

    # -- agent_consult ---------------------------------------------------------------

    async def _handle_tool_call(self, event) -> None:
        question = str(event.tool_args.get("question", "")).strip()
        try:
            answer = await asyncio.wait_for(
                self._consult_agent(question), timeout=CONSULT_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            answer = "Sorry, looking that up took too long."
        except Exception as e:  # noqa: BLE001
            logger.exception("voice_call realtime: agent_consult failed")
            answer = f"Sorry, I could not check that: {e}"
        try:
            await self.session.send_tool_result(event.tool_call_id, answer)
        except Exception:  # noqa: BLE001
            logger.warning("voice_call realtime: tool result delivery failed",
                           exc_info=True)

    async def _consult_agent(self, question: str) -> str:
        if not question:
            return "No question was provided."
        llm = _plugin_llm_factory() if _plugin_llm_factory is not None else None
        if llm is None:
            return "The agent is not available right now."
        peer = self.record.peer_number or "unknown"

        def _run() -> str:
            result = llm.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the Hermes agent assisting a live phone "
                            f"call with {peer}. Answer in 1-3 short spoken-"
                            "style sentences of plain text. No markdown, "
                            "URLs, or lists. Never reveal secrets."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                purpose="voice_call.agent_consult",
                timeout=CONSULT_TIMEOUT_S - 5,
            )
            return (result.text or "").strip()

        answer = await asyncio.get_running_loop().run_in_executor(None, _run)
        return answer[:CONSULT_MAX_CHARS] or "I could not find an answer."

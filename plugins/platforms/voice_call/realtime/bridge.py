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
        # token → (call_id, ts)
        self._tokens: Dict[str, Tuple[str, float]] = {}
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
        wss_base = public.replace(
            "https://", "wss://").replace("http://", "ws://")
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
            logger.warning(
                "voice_call realtime: rejected stream with bad token")
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
        self._pacer: Optional[AudioPacer] = None
        # Tool results that arrived while the model was mid-utterance (e.g.
        # speaking the "let me check" filler) — sending response.create then
        # would collide with the active response, so they flush on
        # response_done instead.
        self._deferred_tool_results: list = []
        # While an agent_consult routes through the gateway, the agent's
        # reply arrives via adapter.send() → runtime.speak_for_chat() →
        # deliver_agent_text(); this future hands it back to the consult.
        self._consult_future: Optional[asyncio.Future] = None
        # The "let me check" filler can't play at tool_call time (the
        # function-call response is still active) — it plays on the next
        # response_done while the consult is still pending.
        self._consult_filler_pending = False

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
        self._pacer = pacer
        tasks = [
            asyncio.create_task(self._inbound_pump(ws), name="vc-rt-inbound"),
            asyncio.create_task(self._outbound_pump(pacer),
                                name="vc-rt-outbound"),
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
        logger.info("voice_call realtime: bridge for %s ended",
                    self.record.call_id)

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
                if self.session.audio_wire_format == "ulaw":
                    # Model speaks the phone line's codec — pass through.
                    await self.session.send_audio(frame.payload)
                else:
                    pcm = audio.ulaw_to_pcm16(frame.payload)
                    pcm = audio.resample_pcm16(
                        pcm, 8000, self.session.input_sample_rate
                    )
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
                if self.session.audio_wire_format == "ulaw":
                    pacer.push(event.audio)  # already µ-law @ 8 kHz
                else:
                    pcm = audio.resample_pcm16(
                        event.audio, self.session.output_sample_rate, 8000
                    )
                    pacer.push(audio.pcm16_to_ulaw(pcm))
            elif event.type == "speech_started":
                await self._handle_barge_in(pacer)
            elif event.type == "transcript":
                self._record_transcript(event.role, event.text)
            elif event.type == "tool_call":
                logger.info(
                    "voice_call realtime: tool call %s id=%s call=%s args=%.200s",
                    event.tool_name, event.tool_call_id,
                    self.record.call_id, event.tool_args,
                )
                asyncio.get_running_loop().create_task(
                    self._handle_tool_call(event)
                )
            elif event.type == "response_done":
                await self._flush_deferred_tool_results()
                await self._maybe_speak_filler()
            elif event.type == "error":
                # A cancel racing a just-finished response is benign noise.
                if "no active response" in (event.text or "").lower():
                    logger.debug("voice_call realtime: %s", event.text)
                else:
                    logger.warning(
                        "voice_call realtime: model error: %s", event.text)
            elif event.type == "closed":
                return

    async def _handle_barge_in(self, pacer: AudioPacer) -> None:
        if time.time() < self._greeting_until:
            return  # don't let line noise cut off the greeting
        # Caller speech is only an interruption when the bot is actually
        # talking (response in flight or audio still queued). Otherwise it's
        # a normal conversational turn — clearing the pacer here would dump
        # answers the model already finished generating (it produces audio
        # faster than realtime), and cancelling would just spam
        # "no active response found" errors.
        if not (self.session.response_active or pacer.pending):
            return
        dropped = pacer.clear()
        if self._ws is not None and not self._ws.closed:
            await self._ws.send_str(self.adapter.serialize_clear())
        if self.session.response_active:
            try:
                await self.session.cancel_response()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "voice_call realtime: cancel_response failed", exc_info=True
                )
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

    async def _maybe_speak_filler(self) -> None:
        """Speak the "let me check" filler once a consult is running and the
        model's function-call response has finished (it can't play earlier —
        the response is still active at tool_call time; the flag is cleared
        when the consult completes)."""
        if not self._consult_filler_pending or self.session.response_active:
            return
        self._consult_filler_pending = False
        phrase = self.runtime.config.responder.thinking_phrase
        if not phrase:
            return
        try:
            await self.session.inject_text(phrase)
            logger.info(
                "voice_call realtime: spoke consult filler call=%s",
                self.record.call_id,
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "voice_call realtime: filler inject failed", exc_info=True)

    async def _handle_tool_call(self, event) -> None:
        if event.tool_name == "end_call":
            await self._handle_end_call(event)
            return
        if event.tool_name != "agent_consult":
            logger.warning(
                "voice_call realtime: unknown tool %r requested", event.tool_name
            )
            try:
                await self.session.send_tool_result(
                    event.tool_call_id, f"Unknown tool {event.tool_name!r}."
                )
            except Exception:  # noqa: BLE001
                logger.debug("voice_call realtime: tool error reply failed",
                             exc_info=True)
            return
        question = str(event.tool_args.get("question", "")).strip()
        started_at = time.time()
        timeout = max(15.0, float(
            self.runtime.config.responder.response_timeout_s))

        # Empty-args consults happen when the model flails (e.g. retrying
        # after a timeout). OpenClaw's fallback: join an in-flight consult
        # if one exists, else use the caller's last utterance as the question.
        if not question:
            pending = self._consult_future
            if pending is not None and not pending.done():
                logger.info(
                    "voice_call realtime: empty consult joins the in-flight "
                    "one call=%s", self.record.call_id,
                )
                try:
                    answer = await asyncio.wait_for(
                        asyncio.shield(pending), timeout=timeout
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    answer = (
                        "Still looking that up — the answer will arrive "
                        "shortly; do not call this tool again for the same "
                        "question."
                    )
                await self._deliver_tool_result(event.tool_call_id, answer)
                return
            question = next(
                (t.text for t in reversed(self.record.transcript)
                 if t.speaker == "user"),
                "",
            )

        # Queue the spoken "let me check" filler — usually the function-call
        # response is still active here, so it plays on the next
        # response_done; if the session is already idle it plays now.
        if self.runtime.config.responder.thinking_phrase:
            self._consult_filler_pending = True
            await self._maybe_speak_filler()
        try:
            answer = await asyncio.wait_for(
                self._consult_agent(question), timeout=timeout
            )
            status = "ok"
        except asyncio.TimeoutError:
            # The gateway turn is usually still running; its answer will be
            # spoken when it lands (deliver_agent_text → inject). Keep the
            # model from re-asking in the meantime.
            answer = (
                "This is taking longer than expected. Tell the caller you are "
                "still checking; the answer will be spoken as soon as it is "
                "ready. Do not call this tool again for the same question."
            )
            status = "timeout"
        except Exception as e:  # noqa: BLE001
            logger.exception("voice_call realtime: agent_consult failed")
            answer = f"Sorry, I could not check that: {e}"
            status = "error"
        finally:
            self._consult_filler_pending = False
        logger.info(
            "voice_call realtime: consult completed call=%s status=%s "
            "elapsed=%.1fs answer_chars=%d",
            self.record.call_id, status, time.time() - started_at, len(answer),
        )
        await self._deliver_tool_result(event.tool_call_id, answer)

    async def _handle_end_call(self, event) -> None:
        """The model asked to hang up (caller said goodbye). Let any
        in-flight goodbye finish playing, then end the call at the carrier —
        works for every provider via manager.end_call → provider.hangup_call."""
        reason = str(event.tool_args.get("reason", "") or "model requested hangup")
        logger.info(
            "voice_call realtime: model requested hangup call=%s reason=%s",
            self.record.call_id, reason,
        )
        # Grace: drain the goodbye (active response + queued frames), max 8s.
        deadline = time.time() + 8.0
        while time.time() < deadline and (
            self.session.response_active
            or (self._pacer is not None and self._pacer.pending)
        ):
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)  # let the carrier flush the last frames
        manager = self.runtime.manager
        if manager is None or manager.get_call(self.record.call_id) is None:
            return  # already ended (caller hung up first)
        try:
            await manager.end_call(self.record.call_id, reason="hangup-bot")
        except Exception:  # noqa: BLE001
            logger.exception("voice_call realtime: model-requested hangup failed")

    async def deliver_agent_text(self, text: str, final: bool = True) -> bool:
        """Agent output routed to a realtime call (via adapter.send()).

        Only the turn's FINAL response counts: a pending consult consumes
        it as the tool result; otherwise the realtime model is asked to
        speak it (e.g. cron/agent-initiated messages while a realtime call
        is live — carrier TTS would talk over the media stream). Interim
        sends (tool-progress chrome, status notices) are acknowledged but
        dropped — they must not resolve a consult or be read aloud."""
        if not final:
            logger.debug(
                "voice_call realtime: suppressed interim agent send "
                "call=%s text=%.120s", self.record.call_id, text,
            )
            return True
        fut = self._consult_future
        if fut is not None and not fut.done():
            logger.info(
                "voice_call realtime: consult answer received call=%s chars=%d",
                self.record.call_id, len(text),
            )
            fut.set_result(text)
            return True
        logger.info(
            "voice_call realtime: speaking agent message call=%s chars=%d",
            self.record.call_id, len(text),
        )
        try:
            await self.session.inject_text(text)
            return True
        except Exception:  # noqa: BLE001
            logger.warning("voice_call realtime: inject of agent text failed",
                           exc_info=True)
            return False

    async def _deliver_tool_result(self, tool_call_id: str, answer: str) -> None:
        """Send now, or defer until the in-flight utterance (filler) ends —
        response.create during an active response is rejected by the API."""
        if self.session.response_active:
            self._deferred_tool_results.append((tool_call_id, answer))
            # Re-check: the response may have ended between the check and
            # the append (its response_done already processed).
            if not self.session.response_active:
                await self._flush_deferred_tool_results()
            return
        try:
            await self.session.send_tool_result(tool_call_id, answer)
        except Exception:  # noqa: BLE001
            logger.warning("voice_call realtime: tool result delivery failed",
                           exc_info=True)

    async def _flush_deferred_tool_results(self) -> None:
        while self._deferred_tool_results and not self.session.response_active:
            tool_call_id, answer = self._deferred_tool_results.pop(0)
            try:
                await self.session.send_tool_result(tool_call_id, answer)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "voice_call realtime: deferred tool result delivery failed",
                    exc_info=True,
                )

    async def _consult_agent(self, question: str) -> str:
        if not question:
            return "No question was provided."
        # Preferred: route through the gateway as a normal message so the
        # full Hermes agent answers — with its real tools (web search,
        # weather, memory, ...) and the same per-phone session as
        # turn-based calls. This mirrors OpenClaw, whose consult runs the
        # embedded agent with the tool catalog rather than a bare LLM.
        if self.runtime.adapter is not None:
            logger.info(
                "voice_call realtime: consulting gateway agent call=%s "
                "question=%.200s", self.record.call_id, question,
            )
            return await self._consult_via_gateway(question)
        # Headless fallback: tool-less host completion (can only answer
        # from model knowledge).
        logger.info(
            "voice_call realtime: consulting plugin LLM (no gateway adapter) "
            "call=%s question=%.200s", self.record.call_id, question,
        )
        return await self._consult_via_completion(question)

    async def _consult_via_gateway(self, question: str) -> str:
        from ..responder import dispatch_transcript

        # A newer question supersedes an in-flight consult: resolve the old
        # future benignly (its tool result completes immediately) — the new
        # dispatch interrupts the old gateway turn, which is the same
        # semantics as a user changing their mind mid-message.
        previous = self._consult_future
        if previous is not None and not previous.done():
            logger.info(
                "voice_call realtime: consult superseded by newer question "
                "call=%s", self.record.call_id,
            )
            previous.set_result(
                "Superseded by a newer question from the caller; answer that "
                "instead."
            )
        # The speed contract rides WITH the question — a distant system-
        # prompt hint loses to session history full of thorough-research
        # precedent; an instruction adjacent to the request does not.
        framed = (
            "[Voice consult — I am on a live phone call and waiting. Answer "
            "in 1-3 short spoken sentences as fast as possible: answer "
            "directly from what you know or at most ONE quick web_search. "
            "Do not use web_extract, browser, or multi-step research. If a "
            "thorough answer would take longer, give your best short answer "
            "now and note what you'd verify later.] "
            f"{question}"
        )
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._consult_future = fut
        try:
            await dispatch_transcript(self.runtime, self.record, framed)
            answer = (await fut).strip()
            return answer[:CONSULT_MAX_CHARS] or "I could not find an answer."
        finally:
            if self._consult_future is fut:
                self._consult_future = None

    async def _consult_via_completion(self, question: str) -> str:
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

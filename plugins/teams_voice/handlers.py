"""Call-session handlers — the dialogue brains the bridge dispatches into.

* :class:`EchoCallSessionHandler` — dependency-light smoke test: smiles on connect
  and echoes the caller's audio so the worker's RMS lip-sync animates the avatar.

* :class:`RealtimeCallSessionHandler` — the full speech-to-speech brain:
  recording gate, **echo guard** (self-answer fix), bidirectional resampled audio,
  expression cues + **realtime visemes**, **barge-in**, and the realtime tool set:
  **agent delegation** (`hermes_agent_consult` → `run_agent`), **vision**
  (`look_at_screen`), **show_to_caller** (image → tile), and **outbound call-back**
  (`call_me_back`, delivered on the worker's outbound leg).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path

from . import audio, expression, protocol, realtime_tools, viseme_estimate
from .agent_consult import AgentConsult
from .bridge_server import CallSession, CallSessionHandler
from .config import BYTES_PER_FRAME, FRAME_DURATION_MS, PCM_SAMPLE_RATE_HZ, TeamsVoiceConfig
from .echo_guard import EchoGuard
from .outbound import OutboundError, place_call
from .realtime.openai_client import REALTIME_SAMPLE_RATE_HZ, RealtimeConfig, RealtimeSession
from .vision_store import StoredFrame, VisionStore

logger = logging.getLogger(__name__)

# Shared across connections: the inbound call that requests a callback and the
# outbound leg that delivers it are *different* WebSocket connections, so the
# pending spoken result is keyed by the worker's callId here.
_PENDING_OUTBOUND: dict[str, str] = {}


class EchoCallSessionHandler(CallSessionHandler):
    """Smoke-test handler — visible proof the driver path works end to end."""

    def __init__(self) -> None:
        self._seq = 0
        self._ts = 0

    async def on_session_start(self, session: CallSession, msg: protocol.SessionStart) -> None:
        await super().on_session_start(session, msg)
        try:
            await session.send_expression(expression.HAPPY)
        except Exception:  # noqa: BLE001 — cosmetic; never fail the call
            logger.debug("[teams_voice] echo: expression send failed", exc_info=True)

    async def on_audio_frame(self, session: CallSession, msg: protocol.AudioFrame) -> None:
        if not session.recording_active:
            return
        try:
            await session.send_audio_frame(self._seq, self._ts, msg.payload_base64)
        except Exception:  # noqa: BLE001
            return
        self._seq += 1
        self._ts += FRAME_DURATION_MS


class RealtimeCallSessionHandler(CallSessionHandler):
    """Bridges a Teams call to an OpenAI/Azure realtime speech-to-speech model."""

    def __init__(self, config: RealtimeConfig, bridge_config: TeamsVoiceConfig | None = None) -> None:
        self._cfg = config
        self._bridge = bridge_config
        self._rt: RealtimeSession | None = None
        self._session: CallSession | None = None
        self._caller: protocol.CallerInfo | None = None
        self._outbound = False
        self._pending_greeting: str | None = None
        # Outbound (model -> worker) framing state.
        self._out_seq = 0
        self._out_ts = 0
        self._out_residual = b""
        # Dialogue state.
        self._turn_id = 0
        self._transcript = ""
        self._last_emotion: str | None = None
        self._echo = EchoGuard()
        self._vision = VisionStore()
        self._consult = AgentConsult()

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def on_session_start(self, session: CallSession, msg: protocol.SessionStart) -> None:
        await super().on_session_start(session, msg)
        self._session = session
        self._caller = msg.caller
        self._outbound = (msg.direction or "").lower() == "outbound"
        # If this is the delivery leg of a call-back, fetch the pending result.
        if self._outbound:
            self._pending_greeting = _PENDING_OUTBOUND.pop(msg.call_id, None)

        rt = RealtimeSession(self._cfg)
        rt.tools = realtime_tools.default_tools()
        rt.on_audio_delta = self._on_model_audio
        rt.on_transcript_delta = self._on_transcript
        rt.on_speech_started = self._on_barge_in
        rt.on_response_done = self._on_response_done
        rt.on_function_call = self._on_function_call
        self._rt = rt
        try:
            await rt.connect()
        except Exception:  # noqa: BLE001 — keep socket; worker shows neutral avatar
            logger.error("[teams_voice] realtime connect failed for %s", session.call_id, exc_info=True)
            return
        # Inbound greeting fires from the model; outbound waits for recording-active.
        if not self._outbound:
            await self._safe_expression(expression.NEUTRAL)

    async def on_recording_status(self, session: CallSession, msg: protocol.RecordingStatus) -> None:
        await super().on_recording_status(session, msg)
        # Outbound delivery: speak the result only once the callee has answered
        # (recording active), not while the phone is still ringing (greet-on-answer).
        if self._outbound and session.recording_active and self._pending_greeting and self._rt:
            greeting, self._pending_greeting = self._pending_greeting, None
            await self._rt.request_say(
                f"The caller just answered. Deliver this result clearly and concisely, "
                f"then say goodbye: {greeting}"
            )

    async def on_audio_frame(self, session: CallSession, msg: protocol.AudioFrame) -> None:
        if not session.recording_active or self._rt is None:
            return
        pcm16 = base64.b64decode(msg.payload_base64)
        if not self._echo.allow_input(audio.pcm16_rms(pcm16)):  # echo guard
            return
        pcm24 = audio.resample_pcm16(pcm16, PCM_SAMPLE_RATE_HZ, REALTIME_SAMPLE_RATE_HZ)
        await self._rt.push_audio(pcm24)

    async def on_video_frame(self, session: CallSession, msg: protocol.VideoFrame) -> None:
        if not session.recording_active:
            return
        self._vision.store(
            StoredFrame(
                source=msg.source,
                data_base64=msg.data_base64,
                mime=msg.mime or "image/jpeg",
                ts=msg.ts,
                participant_name=msg.participant_name,
            )
        )

    async def on_session_end(self, session: CallSession, msg: protocol.SessionEnd) -> None:
        await super().on_session_end(session, msg)
        self._vision.clear()
        if self._rt is not None:
            await self._rt.close()
            self._rt = None

    # ── model -> worker callbacks ────────────────────────────────────────────

    async def _on_model_audio(self, pcm24: bytes) -> None:
        session = self._session
        if session is None:
            return
        pcm16 = audio.resample_pcm16(pcm24, REALTIME_SAMPLE_RATE_HZ, PCM_SAMPLE_RATE_HZ)
        frames, self._out_residual = audio.frame_pcm16(self._out_residual + pcm16, BYTES_PER_FRAME)
        for frame in frames:
            try:
                await session.send_audio_frame(
                    self._out_seq, self._out_ts, base64.b64encode(frame).decode("ascii")
                )
            except Exception:  # noqa: BLE001
                return
            self._echo.note_output(FRAME_DURATION_MS)  # advance the playout clock
            self._out_seq += 1
            self._out_ts += FRAME_DURATION_MS

    async def _on_transcript(self, text: str) -> None:
        session = self._session
        if session is None:
            return
        self._transcript += text
        emotion = expression.infer_emotion(self._transcript)
        if emotion != self._last_emotion:
            self._last_emotion = emotion
            await self._safe_expression(emotion)
        # Approximate realtime visemes: estimate over this delta, anchored at the
        # current playout position. The worker blends them over RMS openness.
        marks = viseme_estimate.estimate_visemes(text, max(len(text) * 60, 60))
        if marks:
            try:
                await session.send_speech_marks(viseme_estimate.marks_to_payload(marks), ts=self._out_ts)
            except Exception:  # noqa: BLE001
                pass

    async def _on_barge_in(self) -> None:
        self._turn_id += 1
        self._echo.collapse()
        self._echo.mark_caller_turn()
        self._out_residual = b""
        if self._session is not None:
            try:
                await self._session.send_assistant_cancel(self._turn_id)
            except Exception:  # noqa: BLE001
                pass
        if self._rt is not None:
            await self._rt.cancel_response()

    async def _on_response_done(self) -> None:
        session = self._session
        if session is not None and self._out_residual:
            pad = self._out_residual + b"\x00" * (BYTES_PER_FRAME - len(self._out_residual))
            try:
                await session.send_audio_frame(
                    self._out_seq, self._out_ts, base64.b64encode(pad).decode("ascii")
                )
                self._out_seq += 1
                self._out_ts += FRAME_DURATION_MS
            except Exception:  # noqa: BLE001
                pass
        self._out_residual = b""
        self._transcript = ""
        self._last_emotion = None

    # ── tool dispatch ────────────────────────────────────────────────────────

    async def _on_function_call(self, name: str, call_id: str, args_json: str) -> None:
        try:
            args = json.loads(args_json or "{}")
        except (TypeError, ValueError):
            args = {}
        result = await self._run_tool(name, args if isinstance(args, dict) else {})
        if self._rt is not None:
            await self._rt.send_function_result(call_id, result or "Done.")

    async def _run_tool(self, name: str, args: dict) -> str:
        try:
            if name == "hermes_agent_consult":
                return await self._consult.ask(str(args.get("query", "")))
            if name == "look_at_screen":
                return await self._look_at_screen(str(args.get("question", "")), args.get("source"))
            if name == "show_to_caller":
                return await self._show_to_caller(str(args.get("prompt", "")))
            if name == "call_me_back":
                return await self._call_me_back(str(args.get("message", "")))
        except Exception:  # noqa: BLE001 — a tool fault must not break the call
            logger.error("[teams_voice] tool %s failed", name, exc_info=True)
            return "Sorry, that didn't work."
        return f"Unknown tool: {name}."

    async def _look_at_screen(self, question: str, source: str | None) -> str:
        want = "camera" if str(source or "").lower() == "camera" else "screenshare"
        frame = self._vision.latest(want) or self._vision.latest()
        if frame is None:
            return "I can't see a shared screen or camera right now."
        prompt = question.strip() or "Describe what you see."
        try:
            from agent.auxiliary_client import async_call_llm

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": frame.data_url()}},
                    ],
                }
            ]
            resp = await async_call_llm(task="vision", messages=messages, max_tokens=400)
            text = resp.choices[0].message.content if resp and resp.choices else ""
            return (text or "").strip() or "I couldn't quite make that out."
        except Exception:  # noqa: BLE001
            logger.error("[teams_voice] look_at_screen vision call failed", exc_info=True)
            return "I had trouble looking at that."

    async def _show_to_caller(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            return "What would you like me to show?"
        try:
            from tools.image_generation_tool import image_generate_tool

            raw = await asyncio.to_thread(lambda: image_generate_tool(prompt=prompt, aspect_ratio="landscape"))
            data = json.loads(raw)
            if not data.get("success") or not data.get("image"):
                return "I couldn't create that image."
            img_bytes = Path(data["image"]).read_bytes()
            mime = "image/png" if str(data["image"]).lower().endswith(".png") else "image/jpeg"
            if self._session is not None:
                await self._session.send_display_image(
                    base64.b64encode(img_bytes).decode("ascii"),
                    mime,
                    duration_ms=6000,
                    mode="overlay",
                    caption=prompt[:80],
                )
            return "I'm showing it on screen now."
        except Exception:  # noqa: BLE001
            logger.error("[teams_voice] show_to_caller failed", exc_info=True)
            return "I made the image but couldn't display it."

    async def _call_me_back(self, message: str) -> str:
        message = message.strip()
        caller = self._caller
        if self._bridge is None or caller is None or not caller.aad_id:
            return "I can't call you back — I don't have a number to reach you."
        tenant = caller.tenant_id or self._bridge.tenant_id
        if not tenant:
            return "I can't call you back — missing your tenant."
        try:
            result = await place_call(
                user_object_id=caller.aad_id,
                tenant_id=tenant,
                shared_secret=self._bridge.shared_secret,
                worker_base_url=self._bridge.worker_base_url,
            )
        except OutboundError as exc:
            logger.warning("[teams_voice] call_me_back failed: %s", exc)
            return "I couldn't place the call-back just now."
        call_id = result.get("callId")
        if call_id:
            _PENDING_OUTBOUND[call_id] = message or "Here's what you asked for."
        return "Okay — I'll call you right back with that."

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _safe_expression(self, emotion: str) -> None:
        if self._session is None:
            return
        try:
            await self._session.send_expression(emotion)
        except Exception:  # noqa: BLE001 — cosmetic
            pass

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
import time
from dataclasses import replace
from pathlib import Path

from . import audio, expression, group_call_gate, protocol, realtime_tools, verbal_interrupts, viseme_estimate
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
        self._greeted = False
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
        # Group-call gate state.
        self._gate_cfg = group_call_gate.GroupCallGateConfig(wake_phrases=("assistant", "hermes"))
        self._last_addressed_ms: float | None = None
        self._drop_response = False  # deterministic egress drop for gated turns
        # Ambient continuous vision (push the latest changed frame every ~6s).
        self._ambient_task: asyncio.Task | None = None
        self._ambient_interval_s = 6.0

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def on_session_start(self, session: CallSession, msg: protocol.SessionStart) -> None:
        await super().on_session_start(session, msg)
        self._session = session
        self._caller = msg.caller
        self._outbound = (msg.direction or "").lower() == "outbound"
        # If this is the delivery leg of a call-back, fetch the pending result.
        if self._outbound:
            self._pending_greeting = _PENDING_OUTBOUND.pop(msg.call_id, None)

        rt = RealtimeSession(replace(self._cfg, instructions=self._build_instructions()))
        rt.tools = realtime_tools.default_tools()
        rt.on_audio_delta = self._on_model_audio
        rt.on_transcript_delta = self._on_transcript
        rt.on_input_transcript = self._on_input_transcript
        rt.on_speech_started = self._on_barge_in
        rt.on_response_done = self._on_response_done
        rt.on_function_call = self._on_function_call
        self._rt = rt
        try:
            await rt.connect()
        except Exception:  # noqa: BLE001 — keep socket; worker shows neutral avatar
            logger.error("[teams_voice] realtime connect failed for %s", session.call_id, exc_info=True)
            return
        # Greeting fires on recording-active (greet-on-answer); show a neutral face now.
        await self._safe_expression(expression.NEUTRAL)
        self._ambient_task = asyncio.create_task(self._ambient_vision_loop())

    def _first_name(self) -> str:
        name = (self._caller.display_name if self._caller else "") or ""
        return name.strip().split(" ")[0] if name.strip() else ""

    def _build_instructions(self) -> str:
        """Augment base instructions with roster name + group-gate etiquette."""
        parts = [self._cfg.instructions]
        name = self._first_name()
        if name:
            parts.append(f"The caller's first name is {name}; address them by name naturally.")
        phrases = ", ".join(f'"{p}"' for p in self._gate_cfg.wake_phrases)
        parts.append(
            "If more than one person is on the call, stay silent unless someone "
            f"addresses you by name ({phrases}); in a one-on-one call respond normally."
        )
        if getattr(self._cfg, "bilingual", False):
            parts.append(
                "You are bilingual in Arabic and English: detect the caller's language, "
                "reply in that language, switch when they switch, and translate on request."
            )
        return " ".join(parts)

    async def on_recording_status(self, session: CallSession, msg: protocol.RecordingStatus) -> None:
        await super().on_recording_status(session, msg)
        # Outbound delivery: speak the result only once the callee has answered
        # (recording active), not while the phone is still ringing (greet-on-answer).
        if not session.recording_active or self._rt is None:
            return
        if self._outbound and self._pending_greeting:
            greeting, self._pending_greeting = self._pending_greeting, None
            await self._rt.request_say(
                f"The caller just answered. Deliver this result clearly and concisely, "
                f"then say goodbye: {greeting}"
            )
        elif not self._outbound and not self._greeted:
            # Roster greeting by name, on answer (not while ringing).
            self._greeted = True
            name = self._first_name()
            who = f" the caller, {name}," if name else " the caller"
            await self._rt.request_say(
                f"Greet{who} warmly and briefly, then ask how you can help."
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

    async def on_dtmf(self, session: CallSession, msg: protocol.Dtmf) -> None:
        # Surface keypad input to the realtime model (recording-gated) so it can
        # run "press 1 to…" flows.
        if not session.recording_active or self._rt is None:
            return
        await self._rt.send_user_text(f"The caller pressed the {msg.digit} key on the keypad.")

    async def on_session_end(self, session: CallSession, msg: protocol.SessionEnd) -> None:
        await super().on_session_end(session, msg)
        if self._ambient_task is not None:
            self._ambient_task.cancel()
            self._ambient_task = None
        self._vision.clear()
        if self._rt is not None:
            await self._rt.close()
            self._rt = None

    async def _ambient_vision_loop(self) -> None:
        """Every ~6s, push the latest *changed* frame to the model (no forced
        response), so it stays visually aware between explicit look_at_screen calls."""
        last_ts: int | None = None
        try:
            while True:
                await asyncio.sleep(self._ambient_interval_s)
                session = self._session
                if self._rt is None or session is None or not session.recording_active:
                    continue
                frame = self._vision.latest()
                if frame is None or frame.ts == last_ts:
                    continue  # nothing new since last push (dedup)
                last_ts = frame.ts
                try:
                    await self._rt.send_image(frame.data_url())
                except Exception:  # noqa: BLE001 — ambient, best-effort
                    pass
        except asyncio.CancelledError:
            raise

    # ── model -> worker callbacks ────────────────────────────────────────────

    async def _on_model_audio(self, pcm24: bytes) -> None:
        session = self._session
        if session is None:
            return
        if self._drop_response:  # group gate dropped this (unaddressed) turn
            self._out_residual = b""
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

    async def _cut_playback(self) -> None:
        """Stop playback immediately: flush the worker queue and cancel the model."""
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

    async def _on_barge_in(self) -> None:
        await self._cut_playback()

    async def _on_input_transcript(self, text: str) -> None:
        """Caller's finished turn — drive verbal interrupts and the group gate."""
        self._echo.mark_caller_turn()
        # 1) Deterministic verbal interrupt ("stop" / "توقف" / "⟨name⟩, stop").
        if verbal_interrupts.is_verbal_interrupt(text, self._gate_cfg.wake_phrases):
            self._drop_response = True  # suppress any reply to the interrupt itself
            await self._cut_playback()
            return
        # 2) Group-call gate: stay silent unless addressed (2+ humans).
        is_group = (self._session.human_count if self._session else 0) >= 2
        now = time.monotonic() * 1000.0
        decision = group_call_gate.should_respond_to_group_turn(
            transcript=text,
            is_group=is_group,
            config=self._gate_cfg,
            last_addressed_at_ms=self._last_addressed_ms,
            now_ms=now,
        )
        if decision.respond:
            if decision.addressed:
                self._last_addressed_ms = now
        else:
            # Unaddressed meeting turn: drop the auto-created reply at the egress.
            self._drop_response = True
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
        self._drop_response = False  # next turn starts fresh

    # ── tool dispatch ────────────────────────────────────────────────────────

    async def _on_function_call(self, name: str, call_id: str, args_json: str) -> None:
        try:
            args = json.loads(args_json or "{}")
        except (TypeError, ValueError):
            args = {}
        # Show a "thinking" face while the tool runs; the reply re-cues the emotion.
        await self._safe_expression(expression.THINKING)
        result = await self._run_tool(name, args if isinstance(args, dict) else {})
        if self._rt is not None:
            await self._rt.send_function_result(call_id, result or "Done.")

    async def _run_tool(self, name: str, args: dict) -> str:
        try:
            if name == "hermes_agent_consult":
                return await self._consult.ask(str(args.get("query", "")))
            if name == "hermes_agent_task":
                return await self._agent_task(str(args.get("query", "")))
            if name == "look_at_screen":
                return await self._look_at_screen(
                    str(args.get("question", "")), args.get("source"), str(args.get("scope") or "live")
                )
            if name == "show_to_caller":
                return await self._show_to_caller(str(args.get("prompt", "")))
            if name == "call_me_back":
                return await self._call_me_back(str(args.get("message", "")))
        except Exception:  # noqa: BLE001 — a tool fault must not break the call
            logger.error("[teams_voice] tool %s failed", name, exc_info=True)
            return "Sorry, that didn't work."
        return f"Unknown tool: {name}."

    async def _look_at_screen(self, question: str, source: str | None, scope: str = "live") -> str:
        prompt = question.strip() or "Describe what you see."
        if scope == "history":
            frames = self._vision.history(limit=6)
            if not frames:
                return "I don't have any earlier frames to look back on."
            content: list[dict] = [{"type": "text", "text": prompt}]
            for f in frames:  # timestamped, attributed keyframes
                content.append({"type": "text", "text": f"(earlier, from {f.describe()})"})
                content.append({"type": "image_url", "image_url": {"url": f.data_url()}})
        else:
            want = "camera" if str(source or "").lower() == "camera" else "screenshare"
            frame = self._vision.latest(want) or self._vision.latest()
            if frame is None:
                return "I can't see a shared screen or camera right now."
            content = [
                {"type": "text", "text": f"{prompt} (looking at the {frame.describe()})"},
                {"type": "image_url", "image_url": {"url": frame.data_url()}},
            ]
        return await self._vision_consult(content)

    async def _vision_consult(self, content: list[dict]) -> str:
        try:
            from agent.auxiliary_client import async_call_llm

            resp = await async_call_llm(
                task="vision", messages=[{"role": "user", "content": content}], max_tokens=400
            )
            text = resp.choices[0].message.content if resp and resp.choices else ""
            return (text or "").strip() or "I couldn't quite make that out."
        except Exception:  # noqa: BLE001
            logger.error("[teams_voice] vision consult failed", exc_info=True)
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

    async def _agent_task(self, query: str) -> str:
        """Run a long job in the background; deliver the result via a call-back."""
        query = query.strip()
        caller = self._caller
        if not query:
            return "What would you like me to work on?"
        if self._bridge is None or caller is None or not caller.aad_id:
            # No way to call back — fall back to an inline consult.
            return await self._consult.ask(query)
        asyncio.create_task(self._run_background_task(query, caller))
        return "Got it — I'll work on that in the background and call you back with the result."

    async def _run_background_task(self, query: str, caller: protocol.CallerInfo) -> None:
        try:
            result = await self._consult.ask(query, timeout_s=300.0)
        except Exception:  # noqa: BLE001
            logger.error("[teams_voice] background task failed", exc_info=True)
            result = "I couldn't complete that task."
        if self._bridge is None or not caller.aad_id:
            return
        tenant = caller.tenant_id or self._bridge.tenant_id
        if not tenant:
            return
        try:
            res = await place_call(
                user_object_id=caller.aad_id,
                tenant_id=tenant,
                shared_secret=self._bridge.shared_secret,
                worker_base_url=self._bridge.worker_base_url,
            )
        except OutboundError as exc:
            logger.warning("[teams_voice] background callback failed: %s", exc)
            return
        cid = res.get("callId")
        if cid:
            _PENDING_OUTBOUND[cid] = result

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _safe_expression(self, emotion: str) -> None:
        if self._session is None:
            return
        try:
            await self._session.send_expression(emotion)
        except Exception:  # noqa: BLE001 — cosmetic
            pass

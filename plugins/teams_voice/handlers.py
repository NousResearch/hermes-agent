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
import uuid
from dataclasses import replace
from pathlib import Path

from . import audio, expression, group_call_gate, meeting, protocol, realtime_tools, verbal_interrupts, viseme_estimate
from .agent_consult import AgentConsult
from .meeting import MeetingTranscript
from .bridge_server import CallSession, CallSessionHandler
from .config import BYTES_PER_FRAME, FRAME_DURATION_MS, PCM_SAMPLE_RATE_HZ, TeamsVoiceConfig, caller_allowed
from .echo_guard import EchoGuard
from .outbound import OutboundError, place_call
from .realtime.openai_client import REALTIME_SAMPLE_RATE_HZ, RealtimeConfig, RealtimeSession
from .vision_budget import VisionBudget
from .vision_store import StoredFrame, VisionStore

PCM_SAMPLE_RATE_HZ_MS = PCM_SAMPLE_RATE_HZ // 1000  # samples per ms (16) — duration math

logger = logging.getLogger(__name__)

# Shared across connections: the inbound call that requests a callback and the
# outbound leg that delivers it are *different* WebSocket connections, so the
# pending spoken result is keyed by the worker's callId here. Entries carry a TTL
# so a never-answered call-back can't leak its result string indefinitely.
_PENDING_OUTBOUND: dict[str, tuple[str, float]] = {}
_PENDING_TTL_S = 600.0


def _pending_prune() -> None:
    now = time.monotonic()
    for k in [k for k, (_t, exp) in _PENDING_OUTBOUND.items() if exp <= now]:
        _PENDING_OUTBOUND.pop(k, None)


def _pending_set(call_id: str, text: str) -> None:
    _pending_prune()
    _PENDING_OUTBOUND[call_id] = (text, time.monotonic() + _PENDING_TTL_S)


def _pending_pop(call_id: str) -> str | None:
    _pending_prune()
    entry = _PENDING_OUTBOUND.pop(call_id, None)
    return entry[0] if entry else None


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
        self._require_recording = bridge_config.require_recording_status if bridge_config else True
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
        wake = tuple(bridge_config.wake_phrases) if (bridge_config and bridge_config.wake_phrases) else ("assistant", "hermes")
        self._gate_cfg = group_call_gate.GroupCallGateConfig(wake_phrases=wake)
        self._last_addressed_ms: float | None = None
        self._drop_response = False  # deterministic egress drop for gated turns
        # Ambient continuous vision (push the latest changed frame per source ~6s).
        self._ambient_task: asyncio.Task | None = None
        self._ambient_interval_s = 6.0
        self._ambient_last_ts: dict[str, int] = {}
        self._vision_budget = VisionBudget(bridge_config.max_vision_per_minute if bridge_config else 30)
        self._meeting = MeetingTranscript()
        self._thread_id = ""
        self._last_speaker = ""  # from unmixed-audio speaker_name, for attribution

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def on_session_start(self, session: CallSession, msg: protocol.SessionStart) -> None:
        await super().on_session_start(session, msg)
        self._session = session
        self._caller = msg.caller
        self._thread_id = msg.thread_id
        self._outbound = (msg.direction or "").lower() == "outbound"
        # If this is the delivery leg of a call-back, fetch the pending result.
        if self._outbound:
            self._pending_greeting = _pending_pop(msg.call_id)

        # Caller allowlist (enforced only when configured): reject unmatched inbound.
        if not self._outbound and self._bridge and not caller_allowed(
            self._bridge, msg.caller.aad_id, msg.caller.display_name
        ):
            logger.info("[teams_voice] caller not allowlisted; rejecting %s", session.call_id)
            await session._ws.close()
            return

        # Agent session continuity scope.
        scope = self._bridge.session_scope if self._bridge else "per-call"
        if scope == "per-thread":
            skey = msg.thread_id or msg.call_id
        elif scope == "per-aad":
            skey = msg.caller.aad_id or msg.call_id
        else:
            skey = msg.call_id
        self._consult = AgentConsult(session_id=f"teams:{skey}")

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

    async def on_participants(self, session: CallSession, msg: protocol.Participants) -> None:
        await super().on_participants(session, msg)  # sets session.human_count
        # Race-free group gate: in a meeting (2+ humans) turn OFF server-VAD
        # auto-response, so the model only speaks when we explicitly create a
        # response for an addressed turn (no audio can leak before a cancel).
        if self._rt is not None:
            await self._rt.set_auto_response(session.human_count < 2)

    async def on_audio_frame(self, session: CallSession, msg: protocol.AudioFrame) -> None:
        if self._require_recording and not session.recording_active:
            return
        if self._rt is None:
            return
        if msg.speaker_name:  # unmixed-audio attribution for the meeting transcript
            self._last_speaker = msg.speaker_name
        pcm16 = base64.b64decode(msg.payload_base64)
        if not self._echo.allow_input(audio.pcm16_rms(pcm16)):  # echo guard
            return
        pcm24 = audio.resample_pcm16(pcm16, PCM_SAMPLE_RATE_HZ, REALTIME_SAMPLE_RATE_HZ)
        await self._rt.push_audio(pcm24)

    async def on_video_frame(self, session: CallSession, msg: protocol.VideoFrame) -> None:
        if self._require_recording and not session.recording_active:
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
        if (self._require_recording and not session.recording_active) or self._rt is None:
            return
        await self._rt.send_user_text(f"The caller pressed the {msg.digit} key on the keypad.")

    async def on_session_end(self, session: CallSession, msg: protocol.SessionEnd) -> None:
        await super().on_session_end(session, msg)
        if self._ambient_task is not None:
            self._ambient_task.cancel()
            self._ambient_task = None
        # End-of-meeting recap (opt-in) — run detached so teardown isn't blocked.
        if self._bridge and self._bridge.meeting_recap and not self._meeting.is_empty():
            asyncio.create_task(
                meeting.post_minutes(self._consult, self._meeting, self._thread_id)
            )
        self._vision.clear()
        if self._rt is not None:
            await self._rt.close()
            self._rt = None

    async def _ambient_vision_loop(self) -> None:
        """Every ~6s, push the latest *changed* frame to the model (no forced
        response), so it stays visually aware between explicit look_at_screen calls."""
        try:
            while True:
                await asyncio.sleep(self._ambient_interval_s)
                session = self._session
                if self._rt is None or session is None or not session.recording_active:
                    continue
                # Push each source (screen + camera) that changed since last time.
                # The worker only emits scene-change frames, so a new ts == a new scene.
                for src in ("screenshare", "camera"):
                    frame = self._vision.latest(src)
                    if frame is None or frame.ts == self._ambient_last_ts.get(src):
                        continue
                    if not self._vision_budget.try_consume():
                        break  # over the per-minute vision cap
                    self._ambient_last_ts[src] = frame.ts
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
        self._meeting.add(self._last_speaker or self._first_name() or "Caller", text)
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
            # In group mode auto-response is OFF, so trigger the reply ourselves.
            if is_group and self._rt is not None:
                await self._rt.create_response()
        else:
            # Unaddressed meeting turn: egress-drop backstop + cancel any response.
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
        if self._transcript.strip():
            self._meeting.add("Assistant", self._transcript)
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
                return await self._show_to_caller(str(args.get("prompt", "")), args.get("count", 1))
            if name == "call_me_back":
                return await self._call_me_back(str(args.get("message", "")))
            if name == "post_meeting_minutes":
                return await meeting.post_minutes(self._consult, self._meeting, self._thread_id)
        except Exception:  # noqa: BLE001 — a tool fault must not break the call
            logger.error("[teams_voice] tool %s failed", name, exc_info=True)
            return "Sorry, that didn't work."
        return f"Unknown tool: {name}."

    async def _look_at_screen(self, question: str, source: str | None, scope: str = "live") -> str:
        if not self._vision_budget.try_consume():
            return "I've looked at a lot just now — give me a moment before the next one."
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
            self._vision_budget.refund()  # consult failed before the model — give it back
            logger.error("[teams_voice] vision consult failed", exc_info=True)
            return "I had trouble looking at that."

    async def _show_to_caller(self, prompt: str, count: object = 1) -> str:
        prompt = prompt.strip()
        if not prompt:
            return "What would you like me to show?"
        try:
            n = max(1, min(int(count), 3))
        except (TypeError, ValueError):
            n = 1
        try:
            from tools.image_generation_tool import image_generate_tool

            paths: list[str] = []
            for _ in range(n):
                raw = await asyncio.to_thread(
                    lambda: image_generate_tool(prompt=prompt, aspect_ratio="landscape")
                )
                data = json.loads(raw)
                if data.get("success") and data.get("image"):
                    paths.append(data["image"])
            if not paths:
                return "I couldn't create that image."
            # Paced slideshow: 4.5s hold for non-final, 5s for the final image.
            for idx, path in enumerate(paths):
                final = idx == len(paths) - 1
                img_bytes = Path(path).read_bytes()
                mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
                if self._session is not None:
                    await self._session.send_display_image(
                        base64.b64encode(img_bytes).decode("ascii"),
                        mime,
                        duration_ms=5000 if final else 4500,
                        mode="overlay",
                        caption=prompt[:80],
                    )
                if not final:
                    await asyncio.sleep(4.0)
            return "I'm showing it on screen now." if len(paths) == 1 else f"Showing you {len(paths)} images."
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
                allow_remote=self._bridge.allow_remote_worker,
            )
        except OutboundError as exc:
            logger.warning("[teams_voice] call_me_back failed: %s", exc)
            return "I couldn't place the call-back just now."
        call_id = result.get("callId")
        if call_id:
            _pending_set(call_id, message or "Here's what you asked for.")
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
                allow_remote=self._bridge.allow_remote_worker,
            )
        except OutboundError as exc:
            logger.warning("[teams_voice] background callback failed: %s", exc)
            return
        cid = res.get("callId")
        if cid:
            _pending_set(cid, result)

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _safe_expression(self, emotion: str) -> None:
        if self._session is None:
            return
        try:
            await self._session.send_expression(emotion)
        except Exception:  # noqa: BLE001 — cosmetic
            pass


class StreamingCallSessionHandler(CallSessionHandler):
    """Streaming voice path: STT → agent → TTS (half-duplex, turn-based).

    Segments caller audio into utterances (VAD), transcribes them, applies the
    verbal-interrupt + group gate on the transcript, runs the Hermes agent, then
    speaks the reply via TTS with expression + estimated visemes. Simpler than the
    realtime path but works with any STT/TTS provider and no realtime model.
    """

    def __init__(self, bridge_config: TeamsVoiceConfig | None = None) -> None:
        from .streaming_audio import UtteranceBuffer

        self._bridge = bridge_config
        self._require_recording = bridge_config.require_recording_status if bridge_config else True
        self._utterance_task: asyncio.Task | None = None
        self._session: CallSession | None = None
        self._caller: protocol.CallerInfo | None = None
        self._buf = UtteranceBuffer()
        self._consult = AgentConsult()
        wake = tuple(bridge_config.wake_phrases) if (bridge_config and bridge_config.wake_phrases) else ("assistant", "hermes")
        self._gate_cfg = group_call_gate.GroupCallGateConfig(wake_phrases=wake)
        self._last_addressed_ms: float | None = None
        self._out_seq = 0
        self._out_ts = 0
        self._processing = False  # half-duplex: one utterance at a time
        self._meeting = MeetingTranscript()
        self._thread_id = ""

    async def on_session_start(self, session: CallSession, msg: protocol.SessionStart) -> None:
        await super().on_session_start(session, msg)
        self._session = session
        self._caller = msg.caller
        self._thread_id = msg.thread_id
        # Caller allowlist parity with the realtime path.
        if (msg.direction or "").lower() != "outbound" and self._bridge and not caller_allowed(
            self._bridge, msg.caller.aad_id, msg.caller.display_name
        ):
            logger.info("[teams_voice] caller not allowlisted; rejecting %s", session.call_id)
            await session._ws.close()
            return
        scope = self._bridge.session_scope if self._bridge else "per-call"
        key = msg.thread_id if scope == "per-thread" else (msg.caller.aad_id if scope == "per-aad" else msg.call_id)
        self._consult = AgentConsult(session_id=f"teams:{key or msg.call_id}")
        await self._safe_expression(expression.NEUTRAL)

    async def on_session_end(self, session: CallSession, msg: protocol.SessionEnd) -> None:
        await super().on_session_end(session, msg)
        # Cancel an in-flight utterance job so we don't speak after hangup.
        if self._utterance_task is not None:
            self._utterance_task.cancel()
            self._utterance_task = None
        if self._bridge and self._bridge.meeting_recap and not self._meeting.is_empty():
            asyncio.create_task(
                meeting.post_minutes(self._consult, self._meeting, self._thread_id)
            )

    async def on_audio_frame(self, session: CallSession, msg: protocol.AudioFrame) -> None:
        if (self._require_recording and not session.recording_active) or self._processing:
            return
        pcm = base64.b64decode(msg.payload_base64)
        utterance = self._buf.push(pcm, audio.pcm16_rms(pcm))
        if utterance is not None:
            self._processing = True
            self._utterance_task = asyncio.create_task(self._handle_utterance(utterance))

    async def _handle_utterance(self, pcm: bytes) -> None:
        try:
            transcript = await self._transcribe(pcm)
            if not transcript:
                return
            if verbal_interrupts.is_verbal_interrupt(transcript, self._gate_cfg.wake_phrases):
                return  # nothing playing in half-duplex; just don't reply
            is_group = (self._session.human_count if self._session else 0) >= 2
            now = time.monotonic() * 1000.0
            decision = group_call_gate.should_respond_to_group_turn(
                transcript=transcript, is_group=is_group, config=self._gate_cfg,
                last_addressed_at_ms=self._last_addressed_ms, now_ms=now,
            )
            if not decision.respond:
                return
            if decision.addressed:
                self._last_addressed_ms = now
            caller_name = (self._caller.display_name.split(" ")[0] if self._caller and self._caller.display_name else "Caller")
            self._meeting.add(caller_name, transcript)
            # On-demand "summarize the meeting" → post minutes instead of a normal reply.
            if meeting.is_summary_request(transcript):
                await self._speak(await meeting.post_minutes(self._consult, self._meeting, self._thread_id))
                return
            await self._safe_expression(expression.THINKING)
            reply = await self._consult.ask(transcript)
            self._meeting.add("Assistant", reply)
            await self._speak(reply)
        except Exception:  # noqa: BLE001 — never let a turn crash the call
            logger.error("[teams_voice] streaming turn failed", exc_info=True)
        finally:
            self._buf.reset()
            self._processing = False

    async def _transcribe(self, pcm: bytes) -> str:
        from hermes_constants import get_hermes_home

        from .streaming_audio import write_wav_pcm16

        d = Path(get_hermes_home()) / "cache" / "teams_voice"
        d.mkdir(parents=True, exist_ok=True)
        wav = d / f"utt_{uuid.uuid4().hex}.wav"
        try:
            await asyncio.to_thread(write_wav_pcm16, pcm, str(wav), PCM_SAMPLE_RATE_HZ)
            from tools.transcription_tools import transcribe_audio

            res = await asyncio.to_thread(transcribe_audio, str(wav))
            return (res.get("transcript") or "").strip() if res.get("success") else ""
        finally:
            try:
                wav.unlink(missing_ok=True)
            except OSError:
                pass

    async def _speak(self, text: str) -> None:
        text = (text or "").strip()
        session = self._session
        if not text or session is None:
            return
        await self._safe_expression(expression.infer_emotion(text))
        from hermes_constants import get_hermes_home

        from .streaming_audio import decode_to_pcm16k

        d = Path(get_hermes_home()) / "cache" / "teams_voice"
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"tts_{uuid.uuid4().hex}.mp3"
        try:
            from tools.tts_tool import text_to_speech_tool

            raw = await asyncio.to_thread(lambda: text_to_speech_tool(text, output_path=str(out)))
            path = str(out)
            try:
                fp = json.loads(raw).get("file_path")
                if fp:
                    path = fp
            except (TypeError, ValueError):
                pass
            pcm16k = await asyncio.to_thread(decode_to_pcm16k, path)
            if not pcm16k:
                return
            # TODO(viseme): swap the estimator for real ElevenLabs /with-timestamps
            # alignment when the TTS path exposes per-character timing.
            dur_ms = (len(pcm16k) // 2) // PCM_SAMPLE_RATE_HZ_MS
            marks = viseme_estimate.estimate_visemes(text, dur_ms)
            if marks:
                try:
                    await session.send_speech_marks(viseme_estimate.marks_to_payload(marks), ts=self._out_ts)
                except Exception:  # noqa: BLE001
                    pass
            frames, _ = audio.frame_pcm16(pcm16k, BYTES_PER_FRAME)
            for frame in frames:
                try:
                    await session.send_audio_frame(
                        self._out_seq, self._out_ts, base64.b64encode(frame).decode("ascii")
                    )
                except Exception:  # noqa: BLE001
                    return
                self._out_seq += 1
                self._out_ts += FRAME_DURATION_MS
        finally:
            try:
                Path(out).unlink(missing_ok=True)
            except OSError:
                pass

    async def _safe_expression(self, emotion: str) -> None:
        if self._session is None:
            return
        try:
            await self._session.send_expression(emotion)
        except Exception:  # noqa: BLE001
            pass

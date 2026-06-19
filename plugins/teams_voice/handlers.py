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
from .call_session_base import (
    _PENDING_OUTBOUND,
    BaseTeamsCallHandler,
    _pending_pop,
    _pending_set,
)
from .call_tools import CallToolRunner
from .config import BYTES_PER_FRAME, FRAME_DURATION_MS, PCM_SAMPLE_RATE_HZ, TeamsVoiceConfig
from .echo_guard import EchoGuard
from .outbound import OutboundError, place_call
from .realtime.openai_client import REALTIME_SAMPLE_RATE_HZ, RealtimeConfig, RealtimeSession
from .vision_budget import VisionBudget
from .vision_store import StoredFrame, VisionStore

PCM_SAMPLE_RATE_HZ_MS = PCM_SAMPLE_RATE_HZ // 1000  # samples per ms (16) — duration math

logger = logging.getLogger(__name__)


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


class RealtimeCallSessionHandler(BaseTeamsCallHandler):
    """Bridges a Teams call to an OpenAI/Azure realtime speech-to-speech model."""

    def __init__(self, config: RealtimeConfig, bridge_config: TeamsVoiceConfig | None = None) -> None:
        super().__init__(bridge_config)  # shared session policy / state
        self._cfg = config
        self._rt: RealtimeSession | None = None
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
        self._drop_response = False  # deterministic egress drop for gated turns
        # Ambient continuous vision (push the latest changed frame per source ~6s).
        self._ambient_task: asyncio.Task | None = None
        self._ambient_interval_s = 6.0
        self._ambient_last_ts: dict[str, int] = {}
        self._vision_budget = VisionBudget(bridge_config.max_vision_per_minute if bridge_config else 30)
        self._last_speaker = ""  # from unmixed-audio speaker_name, for attribution
        self._tools = CallToolRunner(self)

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def on_session_start(self, session: CallSession, msg: protocol.SessionStart) -> None:
        if not await self._begin_session(session, msg):  # state + allowlist + scope
            return

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
        # Capture all speech for the minutes (full meeting, not just addressed turns).
        self._meeting.add(self._last_speaker or self._first_name() or "Caller", text)
        # 1) Deterministic verbal interrupt ("stop" / "توقف" / "⟨name⟩, stop").
        if verbal_interrupts.is_verbal_interrupt(text, self._gate_cfg.wake_phrases):
            self._drop_response = True  # suppress any reply to the interrupt itself
            await self._cut_playback()
            return
        # 2) Group-call gate: stay silent unless addressed (2+ humans).
        now = time.monotonic() * 1000.0
        is_group, decision = self._group_decision(text, now)
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
        result = await self._tools.run_tool(name, args if isinstance(args, dict) else {})
        if self._rt is not None:
            await self._rt.send_function_result(call_id, result or "Done.")


class StreamingCallSessionHandler(BaseTeamsCallHandler):
    """Streaming voice path: STT → agent → TTS (half-duplex, turn-based).

    Segments caller audio into utterances (VAD), transcribes them, applies the
    verbal-interrupt + group gate on the transcript, runs the Hermes agent, then
    speaks the reply via TTS with expression + estimated visemes. Simpler than the
    realtime path but works with any STT/TTS provider and no realtime model.
    """

    def __init__(self, bridge_config: TeamsVoiceConfig | None = None) -> None:
        super().__init__(bridge_config)  # shared session policy / state
        from .streaming_audio import UtteranceBuffer

        self._utterance_task: asyncio.Task | None = None
        self._buf = UtteranceBuffer()
        self._out_seq = 0
        self._out_ts = 0
        self._processing = False  # half-duplex: one utterance at a time

    async def on_session_start(self, session: CallSession, msg: protocol.SessionStart) -> None:
        if not await self._begin_session(session, msg):  # state + allowlist + scope
            return
        await self._safe_expression(expression.NEUTRAL)

    async def on_recording_status(self, session: CallSession, msg: protocol.RecordingStatus) -> None:
        await super().on_recording_status(session, msg)
        if not session.recording_active or self._greeted:
            return
        self._greeted = True
        # Greet-on-answer: outbound speaks the pending result; inbound greets by name.
        if self._outbound and self._pending_greeting:
            text, self._pending_greeting = self._pending_greeting, None
        elif not self._outbound:
            name = self._first_name()
            text = f"Hello{(' ' + name) if name else ''}, how can I help you?"
        else:
            return
        self._processing = True  # half-duplex: hold the turn while we greet
        self._utterance_task = asyncio.create_task(self._speak_turn(text))

    async def _speak_turn(self, text: str) -> None:
        try:
            await self._speak(text)
        finally:
            self._processing = False

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
            # Capture ALL caller speech for the minutes — including unaddressed
            # meeting discussion — before the respond gate.
            self._meeting.add(self._first_name() or "Caller", transcript)
            # On-demand "summarize the meeting" → post minutes instead of a normal reply.
            if meeting.is_summary_request(transcript):
                await self._speak(await meeting.post_minutes(self._consult, self._meeting, self._thread_id))
                return
            now = time.monotonic() * 1000.0
            _is_group, decision = self._group_decision(transcript, now)
            if not decision.respond:
                return
            if decision.addressed:
                self._last_addressed_ms = now
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

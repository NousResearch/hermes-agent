"""Realtime (xAI S2S) voice-mode handlers for the interactive CLI.

Lifted out of ``cli.py`` so the god-file does not grow another 400 lines of
session glue. ``HermesCLI`` inherits ``CLIVoiceRealtimeMixin``; methods
resolve unchanged via the MRO.

Import discipline mirrors ``hermes_cli.cli_billing_mixin``:
  * this module never imports ``cli`` at load time
  * ``_cprint`` / theme constants are imported lazily inside methods
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _ui():
    from cli import _ACCENT, _DIM, _RST, _cprint
    return _cprint, _DIM, _ACCENT, _RST


class _CLIVoiceTurnRunner:
    """TurnRunner for the voice supervisor: turns ride the CLI input queue."""

    def __init__(self, cli):
        self._cli = cli

    def submit(self, task: str) -> bool:
        from cli import _SeededQueryMessage  # lazy: cli imports this mixin

        # The task is the voice model's restatement, not keyboard input: it
        # must be submitted literally as a plain turn (full toolset, no voice
        # prefix) — a bare str would go through `!` shell dispatch, slash
        # routing and file-drop detection in process_loop.
        self._cli._pending_input.put(_SeededQueryMessage(task))
        app = getattr(self._cli, "_app", None)
        if app:
            app.invalidate()
        return True

    def interrupt(self) -> None:
        if self._cli.agent is not None:
            self._cli.agent.interrupt()

    def is_busy(self) -> bool:
        return bool(getattr(self._cli, "_agent_running", False))

    def is_queue_empty(self) -> bool:
        return self._cli._pending_input.empty()


class CLIVoiceRealtimeMixin:
    """Mixin holding interactive-CLI realtime voice session glue."""

    def _init_voice_realtime_state(self) -> None:
        """Realtime (xAI S2S) input backend state — see _voice_realtime_start."""
        self._voice_rt_session = None
        self._voice_rt_lock = threading.Lock()
        self._voice_rt_failed = False
        self._voice_rt_playback_t0 = None
        self._voice_rt_reconnect_notified = False
        self._voice_rt_barge_enabled = True
        self._voice_rt_barge_grace_s = 0.5
        self._voice_rt_supervisor = False
        self._voice_rt_ctrl = None
        self._voice_rt_narrate = False
        self._voice_rt_full_duplex = False
        self._voice_rt_output_ended_at = None

    # ── Realtime voice input backend (xAI Grok S2S) ──────────────────────
    # tools/voice_realtime.py owns the session. While it is alive it owns the
    # mic — classic recorder and RMS listener must not run.

    def _voice_realtime_config_enabled(self) -> bool:
        """True when ``voice.realtime.enabled`` is on in config (shape-safe)."""
        try:
            from hermes_cli.config import load_config
            from tools.voice_realtime_config import realtime_voice_enabled
            _vc = load_config().get("voice")
            return realtime_voice_enabled(_vc if isinstance(_vc, dict) else {})
        except Exception:
            return False

    def _voice_realtime_configured_supervisor(self) -> bool:
        """The brain ``/voice on`` WOULD start (config, not a live session)."""
        from hermes_cli.config import load_config
        from tools.voice_realtime_config import load_realtime_config
        _vc = load_config().get("voice")
        return load_realtime_config(_vc if isinstance(_vc, dict) else {}).supervisor

    # The predicates below are hooks on shared CLI paths (chat turn, stream, TUI keys); like the
    # rest of cli_voice_mixin they tolerate a HermesCLI built without __init__.
    def _voice_realtime_session_alive(self) -> bool:
        sess = getattr(self, "_voice_rt_session", None)
        return sess is not None and sess.alive

    def _voice_realtime_supervisor_active(self) -> bool:
        """True while a supervisor-brain (grok chats, Hermes consults) session lives."""
        return getattr(self, "_voice_rt_supervisor", False) and self._voice_realtime_session_alive()

    def _voice_realtime_start(self) -> bool:
        """Ensure the realtime session exists and is armed (idempotent).
        False → backend unavailable; caller decides the fallback."""
        _cprint, _DIM, _ACCENT, _RST = _ui()
        with self._voice_rt_lock:
            sess = self._voice_rt_session
            if sess is not None and sess.alive:
                sess.set_armed(True)
                with self._voice_lock:
                    self._voice_recording = True
                if hasattr(self, '_app') and self._app:
                    self._app.invalidate()
                return True
            if self._voice_rt_failed:
                return False
            try:
                from hermes_cli.config import load_config
                from tools.voice_realtime import RealtimeVoiceSession
                from tools.voice_realtime_config import load_realtime_config
                _vc = load_config().get("voice")
                voice_cfg = _vc if isinstance(_vc, dict) else {}
                self._voice_rt_barge_enabled = bool(voice_cfg.get("barge_in", True))
                try:
                    self._voice_rt_barge_grace_s = max(
                        0.0, float(voice_cfg.get("barge_in_grace_seconds", 0.5))
                    )
                except (TypeError, ValueError):
                    self._voice_rt_barge_grace_s = 0.5
                rt_cfg = load_realtime_config(voice_cfg)
                self._voice_rt_supervisor = rt_cfg.supervisor
                self._voice_rt_full_duplex = rt_cfg.full_duplex
                self._voice_rt_output_ended_at = None
                self._voice_rt_narrate = rt_cfg.narrate_progress
                sess = RealtimeVoiceSession(
                    rt_cfg,
                    on_transcript=self._voice_realtime_on_transcript,
                    on_speech_started=self._voice_realtime_on_speech_started,
                    on_speech_stopped=self._voice_realtime_on_speech_stopped,
                    on_state=self._voice_realtime_on_state,
                    on_idle_pause=self._voice_realtime_on_idle_pause,
                    input_gate=self._voice_realtime_gate,
                    activity_hold=self._voice_realtime_activity_hold,
                    on_function_call=self._voice_realtime_on_function_call,
                )
                if rt_cfg.supervisor:
                    from agent.voice_supervisor import VoiceSupervisorController
                    self._voice_rt_ctrl = VoiceSupervisorController(
                        sess,
                        _CLIVoiceTurnRunner(self),
                        narrate=self._voice_rt_narrate,
                        on_event=self._voice_realtime_on_supervisor_event,
                    )
                else:
                    self._voice_rt_ctrl = None
                sess.start()
            except Exception as e:
                self._voice_rt_failed = True
                _cprint(f"\n{_DIM}Realtime voice failed to start: {e}{_RST}")
                return False
            self._voice_rt_session = sess
            sess.set_armed(True)
            with self._voice_lock:
                self._voice_recording = True
        if self._voice_beeps_enabled():
            try:
                from tools.voice_mode import play_beep
                play_beep(frequency=880, count=1)
            except Exception:
                pass
        _cprint(f"\n{_DIM}Connecting realtime voice (grok)...{_RST}")
        if hasattr(self, '_app') and self._app:
            self._app.invalidate()
        return True

    def _voice_realtime_start_or_fallback(self) -> None:
        """Thread target for /voice on and key-resume: realtime, else classic."""
        if not self._voice_realtime_start():
            self._voice_realtime_fallback_to_classic()

    def _voice_realtime_fallback_to_classic(self) -> None:
        """Hand the mic to the classic recorder. A realtime-only setup (no
        STT) can't fall back — end voice mode instead of a deaf live chat."""
        _cprint, _DIM, _ACCENT, _RST = _ui()
        if not (self._voice_mode and self._voice_continuous):
            return
        try:
            self._voice_start_recording()
        except Exception as e:
            _cprint(f"\n{_DIM}Voice input unavailable: {e}{_RST}")
            self._disable_voice_mode()

    def _voice_realtime_pause(self, announce: bool = True) -> None:
        """Pause realtime listening (session stays connected)."""
        _cprint, _DIM, _ACCENT, _RST = _ui()
        sess = self._voice_rt_session
        with self._voice_lock:
            self._voice_continuous = False
            self._voice_recording = False
        if sess is not None:
            try:
                sess.set_armed(False)
            except Exception:
                pass
        if announce:
            _label = self._voice_record_key_label()
            _cprint(f"\n{_DIM}Realtime listening paused — {_label} to resume.{_RST}")
        if hasattr(self, '_app') and self._app:
            self._app.invalidate()

    def _voice_realtime_stop(self) -> None:
        """Tear down the realtime session (voice off / CLI exit)."""
        with self._voice_rt_lock:
            sess, self._voice_rt_session = self._voice_rt_session, None
            self._voice_rt_failed = False  # fresh chance on the next /voice on
            ctrl, self._voice_rt_ctrl = self._voice_rt_ctrl, None
            if ctrl is not None:
                ctrl.reset()
            self._voice_rt_supervisor = False
        with self._voice_lock:
            if self._voice_recorder is None:
                # No classic recorder to stop — "listening" flag clears here.
                self._voice_recording = False
        if sess is not None:
            threading.Thread(target=sess.stop, daemon=True).start()

    def _voice_realtime_gate(self) -> bool:
        """Input gate polled by the session; closed → frames/events dropped.

        Supervisor brain: half-duplex by default — the mic is muted while
        speech plays (+ a short tail hangover) so open speakers can't feed
        the assistant its own voice; ``full_duplex: true`` keeps it hot
        (headphones). Hermes working in the background never mutes the mic.
        Ears brain keeps classic rules: half-duplex when barge-in is off, a
        grace window after playback onset. No capture during modal prompts.
        """
        if not (self._voice_mode and self._voice_continuous):
            return False
        if (
            getattr(self, "_clarify_state", None)
            or getattr(self, "_sudo_state", None)
            or getattr(self, "_approval_state", None)
            or getattr(self, "_slash_confirm_state", None)
            or getattr(self, "_secret_state", None)
        ):
            return False
        try:
            from tools.voice_mode import is_audio_output_active
            output_active = (
                not self._voice_tts_done.is_set() or is_audio_output_active()
            )
        except Exception:
            output_active = not self._voice_tts_done.is_set()
        now = time.monotonic()

        if self._voice_rt_supervisor:
            _sess = self._voice_rt_session
            if _sess is not None and _sess.barge_active:
                return True  # loud-barge window: user talked over the speech
            if output_active:
                self._voice_rt_output_ended_at = None
                if not self._voice_rt_full_duplex:
                    return False
                t0 = self._voice_rt_playback_t0
                if t0 is None:
                    self._voice_rt_playback_t0 = t0 = now
                return now - t0 >= self._voice_rt_barge_grace_s
            self._voice_rt_playback_t0 = None
            if not self._voice_rt_full_duplex:
                # Speaker tail can outlive the playback refcount — hold the
                # gate briefly so the echo can't commit an utterance.
                ended = self._voice_rt_output_ended_at
                if ended is None:
                    self._voice_rt_output_ended_at = ended = now
                if now - ended < 0.35:
                    return False
            return True

        barge = self._voice_rt_barge_enabled
        if output_active:
            if not barge:
                return False
            t0 = self._voice_rt_playback_t0
            if t0 is None:
                self._voice_rt_playback_t0 = t0 = now
            if now - t0 < self._voice_rt_barge_grace_s:
                return False
        else:
            self._voice_rt_playback_t0 = None
            if getattr(self, "_agent_running", False) and not barge:
                return False
        return True

    def _voice_realtime_activity_hold(self) -> bool:
        """True while the user is correctly silent (agent busy / TTS live) —
        those periods never count toward the idle pause."""
        if getattr(self, "_agent_running", False):
            return True
        if not self._voice_tts_done.is_set():
            return True
        try:
            from tools.voice_mode import is_audio_output_active
            return is_audio_output_active()
        except Exception:
            return False

    def _voice_realtime_on_transcript(self, transcript: str) -> None:
        """Finished utterance: ears brain submits it as a Hermes turn;
        supervisor brain only prints it (grok answers) + stop-phrase check."""
        _cprint, _DIM, _ACCENT, _RST = _ui()
        from cli import _VoiceInputMessage
        transcript = (transcript or "").strip()
        if not transcript:
            return
        try:
            from tools.voice_mode import is_voice_stop_phrase
            if is_voice_stop_phrase(transcript):
                _cprint(f"\n{_DIM}Stop phrase detected — ending voice chat.{_RST}")
                self._disable_voice_mode()
                return
        except Exception:
            pass
        self._no_speech_count = 0
        if self._voice_realtime_supervisor_active():
            # grok heard the audio itself; the sidecar ASR only serves
            # the stop-phrase check above.
            return
        self._attached_images.clear()
        self._pending_input.put(_VoiceInputMessage(transcript))
        if hasattr(self, '_app') and self._app:
            self._app.invalidate()

    def _voice_realtime_on_speech_started(self) -> None:
        """Barge-in trigger (same semantics as the full-duplex listener):
        cut TTS during playback, interrupt the turn during generation."""
        _cprint, _DIM, _ACCENT, _RST = _ui()
        if self._voice_realtime_supervisor_active():
            # The session already dropped queued grok speech, and the server
            # interrupts its own response. Hermes keeps working — chatting
            # over a background consult must never cancel it.
            return
        try:
            from tools.voice_mode import is_audio_output_active, stop_playback
            output_active = (
                not self._voice_tts_done.is_set() or is_audio_output_active()
            )
            if output_active:
                logger.debug("TTS CUT: realtime speech_started during playback")
                from tools.tts_streaming import mark_speech_interrupted
                mark_speech_interrupted()
                _pipe_stop = getattr(self, "_voice_tts_stop", None)
                if _pipe_stop is not None:
                    _pipe_stop.set()
                stop_playback()
            elif getattr(self, "_agent_running", False):
                logger.debug(
                    "realtime speech_started during generation — interrupting turn"
                )
                _pipe_stop = getattr(self, "_voice_tts_stop", None)
                if _pipe_stop is not None:
                    _pipe_stop.set()  # never let the stale reply speak
                if self.agent is not None:
                    _cprint(f"\n{_DIM}🎤 Voice interjection — interrupting…{_RST}")
                    self.agent.interrupt()
        except Exception as e:
            logger.debug("realtime speech_started handling failed: %s", e)

    def _voice_realtime_on_speech_stopped(self) -> None:
        """Utterance endpointed — audible capture cue while idle-listening."""
        if getattr(self, "_agent_running", False) or not self._voice_tts_done.is_set():
            return
        if self._voice_beeps_enabled():
            try:
                from tools.voice_mode import play_beep
                play_beep(frequency=660, count=2)
            except Exception:
                pass

    def _voice_realtime_on_idle_pause(self) -> None:
        """Session auto-disarmed after prolonged silence (billing guard)."""
        _cprint, _DIM, _ACCENT, _RST = _ui()
        with self._voice_lock:
            # Match manual pause: continuous must clear too, or the next
            # completed turn's auto-restart re-arms the mic behind the user.
            self._voice_continuous = False
            self._voice_recording = False
        _label = self._voice_record_key_label()
        _cprint(
            f"\n{_DIM}No speech for a while — realtime listening paused "
            f"({_label} to resume).{_RST}"
        )
        if hasattr(self, '_app') and self._app:
            self._app.invalidate()

    def _voice_realtime_on_state(self, state: str, detail: str) -> None:
        """Connection lifecycle notifications from the session threads."""
        _cprint, _DIM, _ACCENT, _RST = _ui()
        if state == "connected":
            self._voice_rt_reconnect_notified = False
            _label = self._voice_record_key_label()
            _cprint(
                f"\n{_ACCENT}🎙 Realtime voice ready — just talk.{_RST} "
                f"{_DIM}({_label} pauses; say \"stop\" to end){_RST}"
            )
            if hasattr(self, '_app') and self._app:
                self._app.invalidate()
        elif state == "reconnecting":
            if not self._voice_rt_reconnect_notified:
                self._voice_rt_reconnect_notified = True
                _cprint(f"\n{_DIM}Realtime voice reconnecting…{_RST}")
        elif state == "dead":
            _cprint(f"\n{_DIM}Realtime voice unavailable ({detail}).{_RST}")
            with self._voice_rt_lock:
                sess, self._voice_rt_session = self._voice_rt_session, None
                self._voice_rt_failed = True
            with self._voice_lock:
                self._voice_recording = False
            if sess is not None:
                try:
                    sess.stop()  # safe from session threads (skips self-join)
                except Exception:
                    pass
            if self._voice_mode and self._voice_continuous:
                threading.Thread(
                    target=self._voice_realtime_fallback_to_classic, daemon=True
                ).start()

    def _voice_realtime_handle_ctrl_c(self, app) -> bool:
        """Ctrl+C during a live realtime voice chat ends the chat, not Hermes.

        The classic recording branch can't catch this — realtime has no
        AudioRecorder, so ``_voice_recorder`` is None and the press would
        fall through to the interrupt/exit path. The caller runs this only
        after prompt/overlay cancels, so a press aimed at an approval panel
        doesn't tear the call down. True → the press was consumed.
        """
        if not (self._voice_mode and self._voice_realtime_session_alive()):
            return False
        # A running agent turn owns Ctrl+C (interrupt the turn, as the
        # composer hint says); only an idle voice chat is ended by it.
        if getattr(self, "_agent_running", False) and getattr(self, "agent", None):
            return False
        threading.Thread(target=self._disable_voice_mode, daemon=True).start()
        app.invalidate()
        return True

    def _voice_realtime_handle_record_key(self, app) -> bool:
        """Record key while a realtime session lives: pause/resume the
        always-on listener. True → the press was consumed."""
        if not self._voice_realtime_session_alive():
            return False
        sess = self._voice_rt_session
        if (
            self._voice_rt_supervisor
            and self._voice_recording
            and sess is not None
            and sess.speaking
        ):
            # Mid-speech: cut the audio, keep listening. A second press
            # (once quiet) pauses as usual.
            sess.clear_playout()
            app.invalidate()
            return True
        if self._voice_recording:
            threading.Thread(target=self._voice_realtime_pause, daemon=True).start()
        else:
            if self._clarify_state or self._sudo_state or self._approval_state or self._slash_confirm_state:
                return True
            if not self._voice_tts_done.is_set():
                try:
                    logger.info("TTS CUT: record key handler cutting TTS (realtime)")
                    from tools.tts_streaming import mark_speech_interrupted
                    mark_speech_interrupted()
                    if self._voice_tts_stop is not None:
                        self._voice_tts_stop.set()
                    from tools.voice_mode import stop_playback
                    stop_playback()
                    self._voice_tts_done.set()
                except Exception:
                    pass
            with self._voice_lock:
                self._voice_continuous = True
            threading.Thread(
                target=self._voice_realtime_start_or_fallback, daemon=True
            ).start()
        app.invalidate()
        return True

    def _voice_realtime_ready(self, reqs: dict) -> bool:
        """True when `/voice on` can run on the realtime backend.

        Realtime transcribes server-side — only local audio I/O is required
        (``reqs`` is the classic ``check_voice_requirements()`` dict). Prints
        the unavailability detail so a misconfigured backend is not silent.
        """
        if not self._voice_realtime_config_enabled():
            return False
        _cprint, _DIM, _ACCENT, _RST = _ui()
        try:
            from tools.voice_realtime_config import check_realtime_requirements
            ok, detail = check_realtime_requirements()
        except Exception as exc:
            ok, detail = False, str(exc)
        if not ok:
            _cprint(f"\n{_DIM}Realtime voice configured but unavailable: {detail}{_RST}")
            return False
        return bool(reqs.get("audio_available"))

    def _voice_realtime_status_line(self) -> Optional[str]:
        """The ``/voice status`` realtime line, or None when not configured."""
        if not self._voice_realtime_config_enabled():
            return None
        sess = self._voice_rt_session
        if sess is not None and sess.alive:
            state = "connected" if sess.connected else "connecting"
            state += ", listening" if sess.armed else ", paused"
            supervisor = self._voice_rt_supervisor
        else:
            state = "failed (classic fallback active)" if self._voice_rt_failed else "enabled (not active)"
            # No live session: report the brain config would start, not the last session's.
            supervisor = self._voice_realtime_configured_supervisor()
        brain = (
            "supervisor (grok chats, Hermes consults)" if supervisor
            else "ears (every utterance → Hermes)"
        )
        return f"  Realtime:  {state} — xAI grok S2S, brain: {brain}"

    # -- supervisor brain: grok-voice chats, Hermes consults in background --

    def _voice_realtime_on_function_call(self, name: str, call_id: str, args_json: str) -> None:
        """grok-voice tool call → the shared supervisor controller."""
        ctrl = self._voice_rt_ctrl
        if ctrl is not None:
            ctrl.on_function_call(name, call_id, args_json)

    def _voice_realtime_on_supervisor_event(self, kind: str, text: str) -> None:
        """Render controller events (consult/steer accepted) in the terminal."""
        _cprint, _DIM, _ACCENT, _RST = _ui()
        _cprint(f"\n{_DIM}🎙→⚕ {kind}: {text}{_RST}")
        if hasattr(self, '_app') and self._app:
            self._app.invalidate()

    def _voice_realtime_consult_complete(
        self, message, response: str, *, interrupted: bool = False
    ) -> bool:
        """Report a finished consult turn to the voice session.
        True → the caller must skip local TTS (grok speaks the summary).
        An interrupted turn (steer) never completes the consult."""
        ctrl = getattr(self, "_voice_rt_ctrl", None)
        if ctrl is None:
            return False
        return ctrl.on_turn_complete(message, response, interrupted=interrupted)

    def _voice_realtime_narrate_tool(self, function_name: str) -> None:
        """Throttled spoken progress while a consult runs (never raises)."""
        ctrl = getattr(self, "_voice_rt_ctrl", None)
        if ctrl is not None:
            ctrl.narrate_tool(function_name)

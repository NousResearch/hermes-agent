"""Voice mode (recording, STT, TTS, full-duplex barge-in) and wake-word listener handlers for the interactive CLI

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time

from hermes_constants import is_termux as _is_termux_environment
from typing import Optional


def _config_section(name: str) -> dict:
    """``load_config()[name]`` coerced to a dict.

    Shape-safe: a hand-edited ``voice: true`` / ``voice: cmd+b`` leaves the section as a
    non-dict; return {} so callers fall back to documented defaults instead of crashing on
    ``.get()``. Config load failures also yield {}.
    """
    try:
        from hermes_cli.config import load_config
        section = load_config().get(name)
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _numeric_or(value, default):
    """``value`` if it is a real int/float, else ``default``.

    ``bool`` is excluded — it subclasses int, so a hand-edited ``silence_threshold: true``
    would otherwise be forwarded as ``1`` instead of falling back.
    """
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _unlink_quietly(path) -> None:
    try:
        if path and os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


class CLIVoiceMixin:
    """Voice mode (recording, STT, TTS, full-duplex barge-in) and wake-word listener handlers for the interactive CLI"""

    def _voice_invalidate(self) -> None:
        """Repaint the TUI (audio level indicator / status) when the app is live."""
        if hasattr(self, '_app') and self._app:
            self._app.invalidate()

    def _voice_start_recording(self):
        """Start capturing audio from the microphone."""
        from cli import _ACCENT, _DIM, _RST, _cprint
        if getattr(self, '_should_exit', False):
            return
        from tools.voice_mode import create_audio_recorder, check_voice_requirements

        reqs = check_voice_requirements()
        if not reqs["audio_available"]:
            if _is_termux_environment():
                if "Termux:API Android app is not installed" in reqs.get("details", ""):
                    raise RuntimeError(
                        "Termux:API command package detected, but the Android app is missing.\n"
                        "Install/update the Termux:API Android app, then retry /voice on.\n"
                        "Fallback: pkg install python-numpy portaudio && python -m pip install sounddevice"
                    )
                raise RuntimeError(
                    "Voice mode requires either Termux:API microphone access or Python audio libraries.\n"
                    "Option 1: pkg install termux-api and install the Termux:API Android app\n"
                    "Option 2: pkg install python-numpy portaudio && python -m pip install sounddevice"
                )
            raise RuntimeError(
                "Voice mode requires sounddevice and numpy.\n"
                f"Install with: {sys.executable} -m pip install sounddevice numpy")
        if not reqs.get("stt_available", reqs.get("stt_key_set")):
            raise RuntimeError(
                "Voice mode requires an STT provider for transcription.\n"
                "Option 1: uv pip install faster-whisper  "
                "(free, local; `pip install faster-whisper` also works if pip is on PATH)\n"
                "Option 2: Set GROQ_API_KEY (free tier)\n"
                "Option 3: Set VOICE_TOOLS_OPENAI_KEY (paid)")

        # Prevent double-start from concurrent threads (atomic check-and-set)
        with self._voice_lock:
            if self._voice_recording:
                return
            self._voice_recording = True

        voice_cfg = _config_section("voice")

        # Recorder creation can fail (no input device, PortAudio init). Reset the flag on
        # failure or every future voice start is silently skipped by the guard above.
        if self._voice_recorder is None:
            try:
                self._voice_recorder = create_audio_recorder()
            except Exception:
                with self._voice_lock:
                    self._voice_recording = False
                raise

        # Config-driven silence params, numeric-guarded against YAML scalar corruption.
        rec = self._voice_recorder
        rec._silence_threshold = _numeric_or(voice_cfg.get("silence_threshold"), 200)
        rec._silence_duration = _numeric_or(voice_cfg.get("silence_duration"), 3.0)
        # voice.max_recording_seconds — hard cap on one recording; explicit <= 0 disables it.
        _max_rec = _numeric_or(voice_cfg.get("max_recording_seconds"), None)
        rec._max_recording_seconds = (_max_rec if _max_rec > 0 else 0.0) if _max_rec is not None else 120.0

        def _on_silence():
            """Called by AudioRecorder when silence is detected after speech."""
            with self._voice_lock:
                if not self._voice_recording:
                    return
            _cprint(f"\n{_DIM}Silence detected, auto-stopping...{_RST}")
            self._voice_invalidate()
            self._voice_stop_and_transcribe()

        # Audio cue: single beep BEFORE starting stream (avoid CoreAudio conflict)
        self._voice_beep(frequency=880, count=1)

        try:
            self._voice_recorder.start(on_silence_stop=_on_silence)
        except Exception:
            with self._voice_lock:
                self._voice_recording = False
            raise
        _label = self._voice_record_key_label()
        if getattr(self._voice_recorder, "supports_silence_autostop", True):
            _recording_hint = f"auto-stops on silence | {_label} to stop & exit continuous"
        elif _is_termux_environment():
            _recording_hint = f"Termux:API capture | {_label} to stop"
        else:
            _recording_hint = f"{_label} to stop"
        _cprint(f"\n{_ACCENT}● Recording...{_RST} {_DIM}({_recording_hint}){_RST}")

        # Periodically refresh prompt to update audio level indicator
        def _refresh_level():
            while True:
                with self._voice_lock:
                    still_recording = self._voice_recording
                if not still_recording:
                    break
                self._voice_invalidate()
                time.sleep(0.15)
        threading.Thread(target=_refresh_level, daemon=True).start()

    def _voice_beep(self, *, frequency: int, count: int) -> None:
        """Play a record start/stop beep when enabled; never raises."""
        if self._voice_beeps_enabled():
            try:
                from tools.voice_mode import play_beep
                play_beep(frequency=frequency, count=count)
            except Exception:
                pass

    def _voice_stt_model(self) -> Optional[str]:
        """STT model override from config, or None for the provider default.

        For the local provider, prefer stt.local.model (default ``base``) so the CLI passes
        a real model name into the local STT backend.
        """
        stt_config = _config_section("stt")
        if str(stt_config.get("provider") or "").strip().lower() == "local":
            local_config = stt_config.get("local") or {}
            if not isinstance(local_config, dict):
                local_config = {}
            return local_config.get("model") or "base"
        return stt_config.get("model")

    def _voice_stt_provider(self) -> str:
        """Configured STT provider name (lowercased), or empty string."""
        return str(_config_section("stt").get("provider") or "").strip().lower()

    def _voice_restart_recording_async(self) -> None:
        """Restart continuous-mode recording off-thread (start() can block)."""
        from cli import _DIM, _RST, _cprint
        def _restart_recording():
            try:
                self._voice_start_recording()
                self._voice_invalidate()
            except Exception as e:
                _cprint(f"{_DIM}Voice auto-restart failed: {e}{_RST}")
        threading.Thread(target=_restart_recording, daemon=True).start()

    def _voice_stop_and_transcribe(self):
        """Stop recording, transcribe via STT, and queue the transcript as input."""
        from cli import _DIM, _RST, _VoiceInputMessage, _cprint
        # Atomic guard; _voice_processing is set immediately so concurrent Ctrl+B presses
        # don't race into the START path while recorder.stop() holds its lock.
        with self._voice_lock:
            if not self._voice_recording:
                return
            self._voice_recording = False
            self._voice_processing = True

        submitted = False
        transcription_failed = False
        wav_path = None
        try:
            if self._voice_recorder is None:
                return
            wav_path = self._voice_recorder.stop()
            # Audio cue: double beep after stream stopped (no CoreAudio conflict)
            self._voice_beep(frequency=660, count=2)
            if wav_path is None:
                _cprint(f"{_DIM}No speech detected.{_RST}")
                return
            self._voice_invalidate()
            stt_model = self._voice_stt_model()
            if self._voice_stt_provider() == "local":
                _cprint(
                    f"{_DIM}Preparing local STT model '{stt_model}' "
                    f"(first use may download it from Hugging Face)...{_RST}")
            else:
                _cprint(f"{_DIM}Transcribing...{_RST}")
            from tools.voice_mode import is_voice_stop_phrase, transcribe_recording
            result = transcribe_recording(wav_path, model=stt_model)
            if result.get("success") and result.get("transcript", "").strip():
                transcript = result["transcript"].strip()
                if is_voice_stop_phrase(transcript):
                    # Bare "stop" (or configured phrase) ends the voice chat, not a turn.
                    _cprint(f"{_DIM}Stop phrase detected — ending voice chat.{_RST}")
                    self._disable_voice_mode()
                    return
                self._attached_images.clear()
                self._voice_invalidate()
                self._pending_input.put(_VoiceInputMessage(transcript))
                submitted = True
            elif result.get("success"):
                _cprint(f"{_DIM}No speech detected.{_RST}")
            else:
                _cprint(f"\n{_DIM}Transcription failed: {result.get('error', 'Unknown error')}{_RST}")
                transcription_failed = True
        except Exception as e:
            _cprint(f"\n{_DIM}Voice processing error: {e}{_RST}")
            transcription_failed = wav_path is not None
        finally:
            with self._voice_lock:
                self._voice_processing = False
            self._voice_invalidate()
            # On failure keep the source recording so long dictation is not lost.
            try:
                if wav_path and os.path.isfile(wav_path):
                    if transcription_failed:
                        _cprint(f"{_DIM}Recording preserved at: {wav_path}{_RST}")
                    else:
                        os.unlink(wav_path)
            except Exception:
                pass

            # Three consecutive no-speech cycles end continuous mode (no infinite restart
            # loop). While the agent is mid-turn or TTS is speaking the user is CORRECTLY
            # silent — those cycles must not count, or a multi-minute tool run ends the voice
            # chat under the user (stop phrase and barge-in still work during the hold).
            stop_continuous_restart = False
            _tts_done = getattr(self, "_voice_tts_done", None)
            _activity_hold = bool(
                getattr(self, "_agent_running", False)
                or (_tts_done is not None and not _tts_done.is_set()))
            if submitted:
                self._no_speech_count = 0
            elif not _activity_hold:
                self._no_speech_count = getattr(self, '_no_speech_count', 0) + 1
                if self._no_speech_count >= 3:
                    self._voice_continuous = False
                    self._no_speech_count = 0
                    _cprint(f"{_DIM}No speech detected 3 times, continuous mode stopped.{_RST}")
                    stop_continuous_restart = True
            # No transcript but continuous mode active: restart so the user can keep talking
            # (when a transcript IS submitted, process_loop restarts after chat()).
            if (
                self._voice_continuous
                and not submitted
                and not self._voice_recording
                and not stop_continuous_restart):
                self._voice_restart_recording_async()

    def _voice_speak_response_async(self, text: str) -> None:
        """Schedule TTS and mark it pending before continuous recording can restart."""
        if not self._voice_tts or not text:
            return
        self._voice_tts_done.clear()
        threading.Thread(target=self._voice_speak_response, args=(text,), daemon=True).start()
        # Barge-in safety net for speak calls outside a chat turn (the agent-turn listener
        # armed in chat() normally covers playback); idempotent via _voice_fd_active.
        if self._voice_continuous:
            threading.Thread(target=self._voice_full_duplex_listener, daemon=True).start()

    def _voice_speak_response(self, text: str):
        """Speak the agent's response aloud using TTS (runs in background thread)."""
        from cli import _DIM, _RST, _cprint, logger
        if not self._voice_tts:
            return
        self._voice_tts_done.clear()
        try:
            from tools.tts_tool import text_to_speech_tool
            from tools.voice_mode import play_audio_file
            # Shared cleaner strips markdown/emoji/⋗ blocks/verifier footer; the TTS tool owns
            # provider request limits and long-form chunking.
            try:
                from tools.tts_text_normalize import prepare_spoken_text
                tts_text = prepare_spoken_text(text, max_chars=None)
            except Exception:
                # Legacy fallback pipeline — keep voice replies best-effort.
                tts_text = re.sub(r'```[\s\S]*?```', ' ', text)   # fenced code blocks
                tts_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', tts_text)  # [text](url) -> text
                tts_text = re.sub(r'https?://\S+', '', tts_text)      # URLs
                tts_text = re.sub(r'\*\*(.+?)\*\*', r'\1', tts_text)  # bold
                tts_text = re.sub(r'\*(.+?)\*', r'\1', tts_text)      # italic
                tts_text = re.sub(r'`(.+?)`', r'\1', tts_text)        # inline code
                tts_text = re.sub(r'^#+\s*', '', tts_text, flags=re.MULTILINE)  # headers
                tts_text = re.sub(r'^\s*[-*]\s+', '', tts_text, flags=re.MULTILINE)  # list items
                tts_text = re.sub(r'---+', '', tts_text)              # horizontal rules
                tts_text = re.sub(r'\n{3,}', '\n\n', tts_text)        # excessive newlines
                tts_text = tts_text.strip()
            if not tts_text:
                return
            self._voice_last_tts_text = tts_text
            # MP3 for CLI playback (afplay doesn't handle OGG well); the TTS tool may
            # auto-convert MP3->OGG but the original MP3 remains.
            out_dir = os.path.join(tempfile.gettempdir(), "hermes_voice")
            os.makedirs(out_dir, exist_ok=True)
            mp3_path = os.path.join(out_dir, f"tts_{time.strftime('%Y%m%d_%H%M%S')}.mp3")

            raw_result = text_to_speech_tool(text=tts_text, output_path=mp3_path)
            try:
                tts_result = json.loads(raw_result) if isinstance(raw_result, str) else {}
            except Exception:
                tts_result = {}
            # The tool result is authoritative — chunked long-form output returns several files.
            play_paths = tts_result.get("file_paths") or [tts_result.get("file_path") or mp3_path]
            for play_path in play_paths if tts_result.get("success") else []:
                if os.path.isfile(play_path) and os.path.getsize(play_path) > 0:
                    play_audio_file(play_path)
            # Clean up all generated files (play_paths + mp3_path + ogg variant)
            for path in set(play_paths + [mp3_path, mp3_path.rsplit(".", 1)[0] + ".ogg"]):
                _unlink_quietly(path)
        except Exception as e:
            logger.warning("Voice TTS playback failed: %s", e)
            _cprint(f"{_DIM}TTS playback failed: {e}{_RST}")
        finally:
            self._voice_tts_done.set()

    def _voice_full_duplex_listener(self) -> None:
        """Full-duplex agent-turn listener: mic live for the WHOLE turn.

        Armed at utterance-submit (chat() start in continuous voice mode), disarmed when agent
        finished + TTS played, so the user can interject during generation too. Generation
        phase: speech interrupts the turn via ``self.agent.interrupt()`` (same seam as
        Ctrl+C); playback phase: speech cuts TTS. Either way the captured utterance is
        submitted as the next message; the stop phrase ends the voice chat in BOTH phases.
        """
        from cli import _DIM, _RST, _cprint, logger
        fd_active = getattr(self, "_voice_fd_active", None)
        if fd_active is None:
            fd_active = threading.Event()
            self._voice_fd_active = fd_active
        if fd_active.is_set():
            return  # one listener owns the mic for this turn
        fd_active.set()
        try:
            from hermes_cli.config import load_config
            voice_cfg = load_config().get("voice") or {}
            if not (isinstance(voice_cfg, dict) and voice_cfg.get("barge_in", True)):
                return
            from tools.voice_mode import full_duplex_listen, is_audio_output_active, stop_playback

            try:
                _mult = float(voice_cfg.get("barge_in_threshold_multiplier", 0) or 0)
            except (TypeError, ValueError):
                _mult = 0.0
            try:
                _grace_ms = int(float(voice_cfg.get("barge_in_grace_seconds", 0.5)) * 1000)
            except (TypeError, ValueError):
                _grace_ms = 500

            tts_done = getattr(self, "_voice_tts_done", None)

            def _should_stop() -> bool:
                if not (getattr(self, "_voice_mode", False) and getattr(self, "_voice_continuous", False)):
                    return True
                if getattr(self, "_agent_running", False):
                    return False
                # Agent finished — keep listening until TTS fully played.
                if tts_done is not None and not tts_done.is_set():
                    return False
                return not is_audio_output_active()

            def _on_trigger(phase: str) -> None:
                # Latch BEFORE cutting anything: suppresses process_loop's auto-restart until
                # the capture is submitted.
                self._voice_barge_capture.set()
                self._voice_barge_phase = phase
                _pipe_stop = getattr(self, "_voice_tts_stop", None)
                if phase == "playback":
                    logger.debug("TTS CUT: full-duplex listener tripped during playback")
                    from tools.tts_streaming import mark_speech_interrupted
                    mark_speech_interrupted()
                    if _pipe_stop is not None:
                        _pipe_stop.set()
                    stop_playback()
                else:
                    # Generation phase: no audio to cut — interrupt the in-flight agent turn.
                    logger.debug(
                        "full-duplex listener tripped during generation — "
                        "interrupting agent turn")
                    if _pipe_stop is not None:
                        _pipe_stop.set()  # never let the stale reply speak
                    try:
                        if self.agent is not None and getattr(self, "_agent_running", False):
                            _cprint(f"\n{_DIM}🎤 Voice interjection — interrupting…{_RST}")
                            self.agent.interrupt()
                    except Exception as e:
                        logger.debug("voice interjection interrupt failed: %s", e)

            wav_path = full_duplex_listen(
                _should_stop, is_playing=is_audio_output_active, on_trigger=_on_trigger,
                multiplier=_mult or None, grace_ms=max(0, _grace_ms))
            if wav_path and self._voice_barge_capture.is_set():
                self._voice_submit_barge_utterance(wav_path)
            else:
                self._voice_barge_capture.clear()
        except Exception as e:
            self._voice_barge_capture.clear()
            logger.debug("Voice full-duplex listener failed: %s", e)
        finally:
            fd_active.clear()

    def _voice_submit_barge_utterance(self, wav_path: str) -> None:
        """Transcribe a barge-captured interruption and queue it as the next turn."""
        from cli import _DIM, _RST, _VoiceInputMessage, _cprint, logger
        submitted = False
        try:
            from tools.voice_mode import transcribe_recording
            result = transcribe_recording(wav_path, model=self._voice_stt_model())
            transcript = (result.get("transcript") or "").strip() if result.get("success") else ""
            if transcript:
                from tools.voice_mode import is_voice_stop_phrase
                if is_voice_stop_phrase(transcript):
                    _cprint(f"\n{_DIM}Stop phrase detected — ending voice chat.{_RST}")
                    self._disable_voice_mode()
                    return
                # Fail-closed echo guard: playback-phase capture has no echo cancellation, so
                # a close match for what Hermes just spoke is speaker bleed, not a user turn.
                if getattr(self, "_voice_barge_phase", None) == "playback":
                    from tools.voice_mode_transcript import is_tts_echo
                    if is_tts_echo(transcript, getattr(self, "_voice_last_tts_text", "")):
                        logger.debug(
                            "Dropping playback-phase barge transcript as TTS echo: %r", transcript)
                        _cprint(f"\n{_DIM}Ignored likely TTS echo (not queued).{_RST}")
                        return
                self._pending_input.put(_VoiceInputMessage(transcript))
                submitted = True
            elif not result.get("success"):
                _cprint(f"\n{_DIM}Transcription failed: {result.get('error', 'Unknown error')}{_RST}")
        except Exception as e:
            _cprint(f"\n{_DIM}Voice processing error: {e}{_RST}")
        finally:
            _unlink_quietly(wav_path)
            self._voice_barge_capture.clear()
            self._voice_barge_phase = None
            # No usable transcript: hand the mic back to the normal loop.
            if not submitted and self._voice_mode and self._voice_continuous and not self._voice_recording:
                self._voice_restart_recording_async()

    def _voice_beeps_enabled(self) -> bool:
        """Return whether CLI voice mode should play record start/stop beeps."""
        try:
            from utils import is_truthy_value  # handles quoted YAML "false" (bool() would not)
            return is_truthy_value(_config_section("voice").get("beep_enabled", True), default=True)
        except Exception:
            return True

    def _enable_voice_mode(self):
        """Enable voice mode after checking requirements."""
        from cli import _ACCENT, _BOLD, _DIM, _RST, _cprint
        if self._voice_mode:
            _cprint(f"{_DIM}Voice mode is already enabled.{_RST}")
            return

        from tools.voice_mode import check_voice_requirements, detect_audio_environment
        env_check = detect_audio_environment()
        if not env_check["available"]:
            _cprint(f"\n{_ACCENT}Voice mode unavailable in this environment:{_RST}")
            for warning in env_check["warnings"]:
                _cprint(f"  {_DIM}{warning}{_RST}")
            return

        reqs = check_voice_requirements()
        if not reqs["available"]:
            _cprint(f"\n{_ACCENT}Voice mode requirements not met:{_RST}")
            for line in reqs["details"].split("\n"):
                _cprint(f"  {_DIM}{line}{_RST}")
            if reqs["missing_packages"]:
                if _is_termux_environment():
                    _cprint(f"\n  {_BOLD}Option 1: pkg install termux-api{_RST}")
                    _cprint(f"  {_DIM}Then install/update the Termux:API Android app for microphone capture{_RST}")
                    _cprint(f"  {_BOLD}Option 2: pkg install python-numpy portaudio && python -m pip install sounddevice{_RST}")
                else:
                    _cprint(f"\n  {_BOLD}Install: {sys.executable} -m pip install {' '.join(reqs['missing_packages'])}{_RST}")
            return

        with self._voice_lock:
            self._voice_mode = True
        if _config_section("voice").get("auto_tts", False):
            with self._voice_lock:
                self._voice_tts = True

        # The voice-mode instruction is injected as a user message prefix (not a system
        # prompt change) to avoid invalidating the prompt cache — see _voice_message_prefix.
        tts_status = " (TTS enabled)" if self._voice_tts else ""
        if self._voice_tts:
            self._tts_lease_async(True)  # warm the engine so the first reply isn't dead air
        # Startup-pinned label so the advertised shortcut always matches the live
        # prompt_toolkit binding (live config would drift after a mid-session edit).
        # See #19835.
        _cprint(f"\n{_ACCENT}Voice mode enabled{tts_status}{_RST}")
        _cprint(f"  {_DIM}{self._voice_record_key_label()} to start/stop recording{_RST}")
        # Spoken-stop hint from voice.stop_phrases (first entry); "" when disabled.
        try:
            from tools.voice_mode_transcript import voice_stop_hint
            _stop_hint = voice_stop_hint()
        except Exception:
            _stop_hint = ""
        if _stop_hint:
            _cprint(f"  {_DIM}{_stop_hint}{_RST}")
        _cprint(f"  {_DIM}/voice tts  to toggle speech output{_RST}")
        _cprint(f"  {_DIM}/voice off  to disable voice mode{_RST}")

    def _typed_voice_stop(self, user_input) -> bool:
        """Typed bare stop phrase during an active voice chat ends the chat (mirrors the spoken
        one; outside voice mode "stop" passes through to the agent). Exact-match via
        ``is_voice_stop_phrase``, so longer messages containing "stop" are never swallowed.

        Saying "stop" ends the voice chat (PR #73106); TYPING the same bare stop phrase while voice mode is
        on must behave identically instead of sending "stop" to the agent as a turn.
        """
        from cli import _DIM, _RST, _cprint
        if not isinstance(user_input, str):
            return False
        with self._voice_lock:
            voice_on = self._voice_mode or self._voice_continuous
        if not voice_on:
            return False
        try:
            from tools.voice_mode import is_voice_stop_phrase
            if not is_voice_stop_phrase(user_input):
                return False
        except Exception:
            return False
        _cprint(f"\n{_DIM}Stop phrase typed — ending voice chat.{_RST}")
        self._disable_voice_mode()
        return True

    def _disable_voice_mode(self):
        """Disable voice mode, cancel any active recording, and stop TTS."""
        from cli import _DIM, _RST, _cprint, logger
        with self._voice_lock:
            if self._voice_recording and self._voice_recorder:
                self._voice_recorder.cancel()
                self._voice_recording = False
            recorder = self._voice_recorder
            self._voice_mode = False
            self._voice_tts = False
            self._voice_continuous = False

        # Release the TTS lease so a resident local model (piper/kittentts) can be freed.
        self._tts_lease_async(False)
        # Shut down the persistent audio stream in background
        if recorder is not None:
            def _bg_shutdown(rec=recorder):
                try:
                    rec.shutdown()
                except Exception:
                    pass
            threading.Thread(target=_bg_shutdown, daemon=True).start()
            self._voice_recorder = None
        # Stop any active TTS playback (file player + streaming pipeline)
        try:
            if self._voice_tts_stop is not None:
                logger.info("TTS CUT: _disable_voice_mode setting stop event")
                self._voice_tts_stop.set()
            from tools.voice_mode import stop_playback
            stop_playback()
        except Exception:
            pass
        self._voice_tts_done.set()
        _cprint(f"\n{_DIM}Voice mode disabled.{_RST}")

    def _maybe_start_wake_word(self):
        """Start the wake-word listener at CLI startup if this surface is eligible."""
        try:
            from tools.wake_word import wake_surface_enabled
            if not wake_surface_enabled("cli"):
                return
        except Exception:
            return
        self._start_wake_word_listener(announce=True)

    def _start_wake_word_listener(self, announce: bool = False) -> bool:
        """Build + start the hotword detector. Returns True on success."""
        from cli import _ACCENT, _DIM, _RST, _cprint
        say = _cprint if announce else (lambda *_a: None)
        try:
            from tools.wake_word import (
                check_wake_word_requirements, load_wake_word_config, owns_listener, start_listening)
        except Exception as e:
            say(f"{_DIM}Wake word unavailable: {e}{_RST}")
            return False

        if getattr(self, "_wake_word_active", False) and owns_listener(self):
            say(f"{_DIM}Wake word is already listening.{_RST}")
            return True
        self._wake_word_active = False

        cfg = load_wake_word_config()
        reqs = check_wake_word_requirements(cfg)
        if not reqs["available"]:
            say(f"\n{_ACCENT}Wake word requirements not met:{_RST}")
            if reqs.get("hint"):
                say(f"  {_DIM}{reqs['hint']}{_RST}")
            return False
        if not reqs.get("deps_available", True):
            # Fresh install: the engine constructor lazy-installs its deps (onnxruntime is
            # a large wheel) — tell the user why this is slow.
            say(f"{_DIM}Installing wake word engine (first use — this may take a minute)...{_RST}")

        self._wake_start_new_session = bool(cfg.get("start_new_session", True))
        try:
            start_listening(self._on_wake_word, owner=self, config=cfg)
        except Exception as e:
            say(f"\n{_DIM}Failed to start wake word: {e}{_RST}")
            return False

        self._wake_word_active = True
        self._wake_suspended = False
        import cli as _cli
        _cli._cli_wake_owner = self
        self._start_wake_watchdog()
        say(f"\n{_ACCENT}Wake word listening{_RST} "
            f"{_DIM}(say \"{reqs['phrase']}\" — /wake off to stop){_RST}")
        return True

    def _stop_wake_word_listener(self, announce: bool = False):
        """Stop and tear down the hotword detector."""
        from cli import _DIM, _RST, _cprint
        import cli as _cli
        was_active = getattr(self, "_wake_word_active", False)
        self._wake_word_active = False
        self._wake_suspended = False
        try:
            from tools.wake_word import stop_listening
            stop_listening(owner=self)
        except Exception:
            pass
        if _cli._cli_wake_owner is self:
            _cli._cli_wake_owner = None
        if announce:
            _cprint(f"{_DIM}Wake word {'stopped' if was_active else 'is not running'}.{_RST}")

    def _on_wake_word(self):
        """Fired after the detector hears the wake phrase."""
        from cli import _ACCENT, _DIM, _RST, _cprint, logger
        if getattr(self, "_should_exit", False):
            return
        # Ignore wake while a turn is in flight or the mic is already in use.
        if self._agent_running or self._voice_recording or getattr(self, "_voice_processing", False):
            return

        # Release the mic so STT can capture the command utterance.
        try:
            from tools.wake_word import pause_listening
            if not pause_listening(owner=self):
                self._wake_word_active = False
                return
        except Exception as e:
            logger.debug("wake word pause failed: %s", e)
            return
        self._wake_suspended = True

        # The CLI is single-profile: a phrase enrolled by ANOTHER profile can't be routed
        # here — print the switch command and re-arm rather than answer as the wrong profile.
        try:
            from tools.wake_word import get_last_match
            _match = get_last_match()
        except Exception:
            _match = None
        if _match and _match[1]:
            from tools.wake_word import _active_profile_name
            if _match[1] != _active_profile_name():
                _cprint(f"\n{_DIM}Wake phrase for profile '{_match[1]}' — "
                        f"run: hermes -p {_match[1]}{_RST}")
                self._wake_suspended = True  # watchdog resumes the listener
                return

        _cprint(f"\n{_ACCENT}✦ Wake word detected — listening...{_RST}")
        if getattr(self, "_app", None):
            try:
                self._app.invalidate()
            except Exception:
                pass

        if getattr(self, "_wake_start_new_session", True):
            try:
                self.new_session(silent=True)
            except Exception as e:
                logger.debug("wake word new_session failed: %s", e)

        # Single-utterance capture; VAD auto-stop transcribes and queues for process_loop.
        with self._voice_lock:
            self._voice_mode = True
        self._voice_continuous = False
        try:
            self._voice_start_recording()
        except Exception as e:
            _cprint(f"{_DIM}Wake capture failed: {e}{_RST}")

    def _start_wake_watchdog(self):
        """Resume the paused detector when the CLI returns to a stable idle."""
        from cli import logger
        if getattr(self, "_wake_watchdog_started", False):
            return
        self._wake_watchdog_started = True

        def _loop():
            idle_polls = 0
            try:
                while getattr(self, "_wake_word_active", False) and not getattr(self, "_should_exit", False):
                    time.sleep(0.25)
                    if not getattr(self, "_wake_suspended", False):
                        idle_polls = 0
                        continue
                    busy = (
                        self._agent_running
                        or self._voice_recording
                        or getattr(self, "_voice_processing", False)
                        or not self._pending_input.empty())
                    if busy:
                        idle_polls = 0
                        continue
                    # Require a few consecutive idle polls (~0.75s) so we don't resume in
                    # the gap between VAD stop and the agent starting.
                    idle_polls += 1
                    if idle_polls >= 3:
                        idle_polls = 0
                        try:
                            from tools.wake_word import resume_listening
                            if resume_listening(owner=self):
                                self._wake_suspended = False
                            else:
                                self._wake_word_active = False
                        except Exception as e:
                            logger.debug("wake word resume failed: %s", e)
            finally:
                self._wake_watchdog_started = False

        threading.Thread(target=_loop, daemon=True, name="wake-watchdog").start()

    def _show_wake_word_status(self):
        """Show current wake-word listener status."""
        from cli import _ACCENT, _BOLD, _DIM, _RST, _cprint
        from tools.wake_word import (
            audio_is_silent, check_wake_word_requirements, is_listening, load_wake_word_config,
            owns_listener)

        cfg = load_wake_word_config()
        reqs = check_wake_word_requirements(cfg)
        owned = owns_listener(self)
        state = "LISTENING" if owned and is_listening() else "PAUSED" if owned else "OFF"
        _cprint(f"\n{_BOLD}Wake Word Status{_RST}")
        _cprint(f"  State:       {state}")
        _cprint(f"  Phrase:      \"{reqs['phrase']}\"")
        _cprint(f"  Provider:    {reqs['provider']}")
        _cprint(f"  Surface:     {cfg.get('surface', 'auto')}")
        _cprint(f"  New session: {'yes' if cfg.get('start_new_session', True) else 'no'}")
        if state == "LISTENING" and audio_is_silent():
            _cprint(f"  {_ACCENT}⚠ Microphone delivers only silence — the listener can't hear anything.{_RST}")
            _cprint(f"  {_DIM}On macOS: System Settings > Privacy & Security > Microphone — allow your"
                    f" terminal/Hermes, then /wake off + /wake on.{_RST}")
        if not reqs["available"] and reqs.get("hint"):
            _cprint(f"  {_DIM}{reqs['hint']}{_RST}")
        if not owned:
            _cprint(f"  {_DIM}Enable with /wake on{_RST}")

    def _tts_lease_async(self, active: bool) -> None:
        """Acquire/release this CLI's TTS engine lease in the background.

        Acquiring pre-loads the configured provider so the first reply starts hot; releasing
        lets the last-holder path unload resident local models. Never blocks or fails the toggle.
        """
        from cli import logger

        def _run():
            try:
                from tools.tts_tool_lifecycle import acquire_tts_lease, release_tts_lease
                if active:
                    acquire_tts_lease("cli:voice-tts")
                else:
                    release_tts_lease("cli:voice-tts")
            except Exception as e:
                logger.debug("voice: tts lease active=%s failed: %s", active, e)

        threading.Thread(target=_run, name="tts-lease-cli", daemon=True).start()

    def _toggle_voice_tts(self):
        """Toggle TTS output for voice mode."""
        from cli import _ACCENT, _DIM, _RST, _cprint
        if not self._voice_mode:
            _cprint(f"{_DIM}Enable voice mode first: /voice on{_RST}")
            return

        with self._voice_lock:
            self._voice_tts = not self._voice_tts
        status = "enabled" if self._voice_tts else "disabled"
        if self._voice_tts:
            from tools.tts_tool import check_tts_requirements
            if not check_tts_requirements():
                _cprint(f"{_DIM}Warning: No TTS provider available. Install edge-tts or set API keys.{_RST}")
        self._tts_lease_async(self._voice_tts)  # warm-up / release signal for the TTS engine
        _cprint(f"{_ACCENT}Voice TTS {status}.{_RST}")

    def _show_voice_status(self):
        """Show current voice mode status."""
        from cli import _BOLD, _RST, _cprint
        from tools.voice_mode import check_voice_requirements

        reqs = check_voice_requirements()
        _cprint(f"\n{_BOLD}Voice Mode Status{_RST}")
        _cprint(f"  Mode:      {'ON' if self._voice_mode else 'OFF'}")
        _cprint(f"  TTS:       {'ON' if self._voice_tts else 'OFF'}")
        _cprint(f"  Recording: {'YES' if self._voice_recording else 'no'}")
        # Startup-pinned label so /voice status always matches the live prompt_toolkit
        # binding (live config would drift after a mid-session config edit).
        # See #19835.
        _cprint(f"  Record key: {self._voice_record_key_label()}")
        _cprint(f"\n  {_BOLD}Requirements:{_RST}")
        for line in reqs["details"].split("\n"):
            _cprint(f"    {line}")

"""Process-wide voice recording + TTS API for the TUI gateway."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Optional

# Modifier aliases mirrored from the TUI parser (``ui-tui/src/lib/platform.ts`` ``_MOD_ALIASES``)
# so one config value binds the same shortcut in both runtimes. ``super``/``win``/``windows`` are
# deliberately absent: prompt_toolkit has no super/meta modifier, so those spellings are TUI-only
# and normalize to the default (a silent fallback beats a hard startup crash; cli.py warns).
_VOICE_MOD_ALIASES = {"ctrl": "c-", "control": "c-", "alt": "a-", "option": "a-", "opt": "a-"}

# Named keys prompt_toolkit accepts as ``c-<name>`` / ``a-<name>``; aliases collapse to canonical.
# Aliases collapse to prompt_toolkit's canonical spelling so the same config value binds identically in both
# runtimes (Copilot round-10 on #19835).
_VOICE_NAMED_KEYS = {
    "space": "space", "spc": "space", "enter": "enter", "return": "enter", "ret": "enter", "tab": "tab",
    "escape": "escape", "esc": "escape", "backspace": "backspace", "bs": "backspace", "delete": "delete", "del": "delete",
}

# ``useInputHandlers()`` intercepts ctrl+c/d/l (interrupt/quit/clear) before the voice check, so
# such a binding would be advertised but never fire (same blocklist as the TUI parser). On macOS
# the CLI's copy/exit/clear bindings also claim ``a-c``/``a-d``/``a-l`` (hermes-ink reports Alt as
# ``key.meta``), mirroring the TUI's darwin-only reservation — alt is reserved on darwin only.
_VOICE_RESERVED_CHARS = frozenset({"c", "d", "l"})

# On macOS the classic CLI's prompt_toolkit bindings for copy / exit / clear also claim ``a-c`` / ``a-d`` /
# ``a-l`` via the action-modifier lookup, and hermes-ink reports Alt as ``key.meta`` on many terminals.
# Mirror the TUI parser's darwin-only reservation so ``option+c`` etc. don't bind Alt+C in the CLI while the
# TUI silently falls back to Ctrl+B (Copilot round-14 on #19835).
_DEFAULT_PT_KEY = "c-b"


def voice_record_key_from_config(cfg: Any) -> Any:
    """Shape-safe ``cfg.voice.record_key``: a hand-edited ``voice: true`` / ``voice: cmd+b`` leaves
    ``cfg["voice"]`` as a bool/str and the naive ``.get`` chain would raise before voice starts.

    ``load_config()`` deep-merges raw YAML and preserves scalar overrides, so a hand-edited ``voice: true``
    / ``voice: cmd+b`` leaves ``cfg["voice"]`` as a bool/str instead of a dict, and the naive
    ``.get("voice", {}).get("record_key")`` chain raises AttributeError before voice can even start (Copilot
    round-11 on 19835). Return ``None`` for malformed shapes so call sites can feed the result straight into
    the normalizer/formatter and get the documented default. See #19835.
    """
    voice = cfg.get("voice") if isinstance(cfg, dict) else None
    return voice.get("record_key") if isinstance(voice, dict) else None


def normalize_voice_record_key_for_prompt_toolkit(raw: Any) -> str:
    """Coerce ``voice.record_key`` into prompt_toolkit's ``c-x`` / ``a-x`` format.

    Mirrors the TUI parser contract (``ui-tui/src/lib/platform.ts``): non-string / empty / typo'd /
    bare-char / multi-modifier / reserved ``ctrl+c|d|l`` / ``super``-family → the documented default
    ``c-b``; named keys collapse to canonical spelling (``ctrl+return`` → ``c-enter``). Exactly one
    modifier: multi-modifier chords bind different shortcuts in prompt_toolkit (a-c-r) and
    hermes-ink rejects them; a bare key is refused by the TUI parser.

    * ``super`` / ``win`` / ``windows`` → ``c-b`` (TUI-only modifiers — prompt_toolkit has no super mod; the
    CLI binding site is expected to warn when this fallback fires so users see the cross-runtime split,
    Copilot round-11 on #19835)
    """
    if not isinstance(raw, str):
        return _DEFAULT_PT_KEY
    parts = [p.strip() for p in raw.strip().lower().split("+") if p.strip()]
    if len(parts) != 2:
        return _DEFAULT_PT_KEY
    modifier_token, key_token = parts
    # ``super`` / ``win`` / ``windows`` are TUI-only (prompt_toolkit has no super modifier, so
    # ``@kb.add(super+b)`` crashes the CLI at startup). Fall back to the documented default here; the CLI
    # binding site is expected to log a warning when the configured value is one of these spellings so users
    # know the TUI+CLI runtimes diverge on that shortcut (Copilot round-11 on #19835).
    normalized_mod = _VOICE_MOD_ALIASES.get(modifier_token)
    if not normalized_mod:
        return _DEFAULT_PT_KEY
    if len(key_token) == 1:
        reserved = (normalized_mod == "c-" or sys.platform == "darwin") and key_token in _VOICE_RESERVED_CHARS
        return _DEFAULT_PT_KEY if reserved else f"{normalized_mod}{key_token}"
    # Multi-char token must be a known named key; ``ctrl+spcae`` must not pass through as
    # ``c-spcae`` (prompt_toolkit would reject it).
    named = _VOICE_NAMED_KEYS.get(key_token)
    return f"{normalized_mod}{named}" if named else _DEFAULT_PT_KEY


def pt_key_to_sequence(pt_key: str) -> tuple[str, ...]:
    """Convert a prompt_toolkit key specifier (e.g. 'c-b' or 'a-v') to a sequence tuple."""
    if isinstance(pt_key, str) and pt_key.startswith("a-"):
        return ("escape", pt_key[2:])
    return (pt_key,)


def format_voice_record_key_for_status(raw: Any) -> str:
    """Render ``voice.record_key`` for ``/voice status`` as ``Ctrl+B`` / ``Alt+Space``; malformed
    configs surface as the default so status never advertises a shortcut that won't bind.

    Mirrors the TUI's ``formatVoiceRecordKey``: returns ``Ctrl+B`` / ``Alt+Space`` / ``Ctrl+Enter``. See
    #19835.
    """
    normalized = normalize_voice_record_key_for_prompt_toolkit(raw)
    prefix = "Alt+" if normalized.startswith("a-") else "Ctrl+"
    key = normalized[2:]
    return prefix + key[0].upper() + key[1:]


from tools.voice_mode import (
    create_audio_recorder,
    is_voice_stop_phrase,
    is_whisper_hallucination,
    play_audio_file,
    transcribe_recording,
)

logger = logging.getLogger(__name__)


def _debug(msg: str) -> None:
    """HERMES_VOICE_DEBUG=1 breadcrumb on stderr (the TUI gateway shows it as a gateway.stderr
    Activity line). Broken pipes are swallowed: this fires from background threads where a dead
    stderr must not kill the gateway — the stdin/stdout command pipe is what matters."""
    if os.environ.get("HERMES_VOICE_DEBUG", "").strip() == "1":
        with contextlib.suppress(BrokenPipeError, OSError):
            print(f"[voice] {msg}", file=sys.stderr, flush=True)


def _beeps_enabled() -> bool:
    """CLI parity: voice.beep_enabled in config.yaml (default True)."""
    try:
        from hermes_cli.config import load_config
        from utils import is_truthy_value

        voice_cfg = load_config().get("voice", {})
        if isinstance(voice_cfg, dict):
            # is_truthy_value handles quoted YAML strings like "false" that bool() misreads.
            # See #49883.
            # See #49883.
            return is_truthy_value(voice_cfg.get("beep_enabled", True), default=True)
    except Exception:
        pass
    return True


def _play_beep(frequency: int, count: int = 1) -> None:
    """Audible cue matching cli.py's beeps (880 Hz once on start, 660 Hz twice on stop).
    Best-effort — a missing speaker must never break the voice loop."""
    if not _beeps_enabled():
        return
    try:
        from tools.voice_mode import play_beep

        play_beep(frequency=frequency, count=count)
    except Exception as e:
        _debug(f"beep {frequency}Hz failed: {e}")


def _safe_call(cb: Optional[Callable], *args: Any, warn: Optional[str] = None) -> None:
    """Invoke an optional callback, swallowing its exceptions. ``warn`` is a ``logger.warning``
    format with one ``%s`` slot; without it failures are silent (status callbacks are fire-and-forget)."""
    if not cb:
        return
    try:
        cb(*args)
    except Exception as e:
        if warn:
            logger.warning(warn, e)


def _transcribe_wav(wav_path: str, fail_msg: str, debug_prefix: Optional[str] = None) -> Optional[str]:
    """Transcribe ``wav_path``, unlink it, and return the cleaned transcript (or None).

    transcribe_recording returns {"success", "transcript", "error"?} — NOT {"text"}; the wrong key
    silently masqueraded as "not hearing the user". Empty text and Whisper hallucinations are
    dropped; failures are logged with ``fail_msg``.
    """
    try:
        result = transcribe_recording(wav_path)
        success = bool(result.get("success"))
        text = (result.get("transcript") or "").strip()
        if debug_prefix:
            _debug(f"{debug_prefix}: transcribe -> success={success} text={text!r} err={result.get('error')!r}")
        if success and text and not is_whisper_hallucination(text):
            return text
    except Exception as e:
        logger.warning(fail_msg, e)
        if debug_prefix:
            _debug(f"{debug_prefix}: transcribe raised {type(e).__name__}: {e}")
    finally:
        with contextlib.suppress(Exception):
            if os.path.isfile(wav_path):
                os.unlink(wav_path)
    return None


def _deactivate(on_status: Optional[Callable[[str], None]] = None) -> None:
    """Mark the continuous loop inactive and (optionally) report ``"idle"``."""
    global _continuous_active
    with _continuous_lock:
        _continuous_active = False
    _safe_call(on_status, "idle")


# Push-to-talk state.
_recorder = None
_recorder_lock = threading.Lock()

# Continuous (VAD) state.
_continuous_lock = threading.Lock()
_continuous_active = False
_continuous_stopping = False
_continuous_auto_restart: bool = True
_continuous_recorder: Any = None

# TTS-vs-STT feedback guard: TTS over the speakers lands in the live mic and gets transcribed as
# user input — an infinite loop the agent happily joins. Mirrors cli.py:_voice_tts_done: cleared
# while speak_text plays, set while silent. The silence callback waits on it before re-arming;
# speak_text cancels live capture before playback so the previous utterance's tail doesn't leak.
_tts_playing = threading.Event()
_tts_playing.set()  # initially "not playing"

# Silence-count hold: while the agent is mid-turn (possibly minutes) or TTS plays, the user is
# CORRECTLY silent — those cycles must not count toward the no-speech limit or a long tool run
# ends the voice chat under the user. The host surface registers a probe reporting "agent busy".
_voice_busy_probe: Optional[Callable[[], bool]] = None


def set_voice_busy_probe(probe: Optional[Callable[[], bool]]) -> None:
    """Register a callable returning True while the agent is mid-turn; ``None`` clears it.
    Must be cheap and thread-safe — it runs on the silence-callback thread."""
    global _voice_busy_probe
    _voice_busy_probe = probe


def _voice_activity_held() -> bool:
    """True while silent cycles must NOT count toward the no-speech limit (TTS playing or agent
    mid-turn). Fail-open to "not held" so a broken probe can never make the voice chat immortal."""
    if not _tts_playing.is_set():
        return True
    try:
        return _voice_busy_probe is not None and bool(_voice_busy_probe())
    except Exception:
        return False


# (on_transcript, on_status, on_silent_limit, on_stop_phrase) of the active loop; guarded by
# ``_continuous_lock``. on_stop_phrase is the explicit user-intent stop (the user SAYS a bare stop
# phrase), distinct from on_silent_limit (a timeout) so consumers end the conversation like a
# manual stop instead of reporting "no speech detected"; unset → on_silent_limit fires.
_NO_CALLBACKS: tuple = (None, None, None, None)
_continuous_callbacks: tuple = _NO_CALLBACKS
_continuous_no_speech_count = 0
_CONTINUOUS_NO_SPEECH_LIMIT = 3


def _turn_transcript(
    wav_path: Optional[str], fail_msg: str, where: str, tail: str, trace: bool = False
) -> tuple[Optional[str], bool, str]:
    """Transcribe a finished capture → (deliverable text, is_stop_phrase, stop_text).

    A bare stop phrase ("stop") is explicit user intent to end the voice chat: it is never sent
    to the agent, so the deliverable text becomes None. ``trace`` adds transcription breadcrumbs.
    """
    transcript = _transcribe_wav(wav_path, fail_msg, where if trace else None) if wav_path else None
    if not (transcript and is_voice_stop_phrase(transcript)):
        return transcript, False, ""
    _debug(f"{where}: stop phrase {transcript!r} — {tail}")
    return None, True, transcript


def _signal_halt(stop_phrase: bool, stop_text: str, on_stop_phrase, on_silent_limit) -> None:
    """Dedicated stop-phrase signal when wired, else the legacy on_silent_limit fallback."""
    if stop_phrase and on_stop_phrase is not None:
        _safe_call(on_stop_phrase, stop_text)
    else:
        _safe_call(on_silent_limit)


def _tally_silence(spoke: bool, held: bool, where: str) -> tuple[bool, int]:
    """Update the no-speech counter (caller holds ``_continuous_lock``): speech resets it, a held
    cycle (agent busy / TTS playing) is ignored, otherwise it bumps. Returns (limit_hit,
    count_after_bump); the counter is reset when the limit is hit."""
    global _continuous_no_speech_count
    if spoke:
        _continuous_no_speech_count = 0
        return False, 0
    if held:
        _debug(f"{where}: silent cycle ignored (agent busy or TTS playing)")
        return False, _continuous_no_speech_count
    _continuous_no_speech_count += 1
    count = _continuous_no_speech_count
    if count >= _CONTINUOUS_NO_SPEECH_LIMIT:
        _continuous_no_speech_count = 0
        return True, count
    return False, count


# ── Push-to-talk API ─────────────────────────────────────────────────


def start_recording() -> None:
    """Begin capturing from the default input device (push-to-talk)."""
    global _recorder
    with _recorder_lock:
        if _recorder is not None and getattr(_recorder, "is_recording", False):
            return
        rec = create_audio_recorder()
        rec.start()  # only publish a recorder that actually started
        _recorder = rec


def stop_and_transcribe() -> Optional[str]:
    """Stop the active push-to-talk recording, transcribe, return text."""
    global _recorder
    with _recorder_lock:
        rec = _recorder
        _recorder = None
    if rec is None:
        return None
    wav_path = rec.stop()
    return _transcribe_wav(wav_path, "voice transcription failed: %s") if wav_path else None


# ── Continuous (VAD) API ─────────────────────────────────────────────


def start_continuous(
    on_transcript: Callable[[str], None],
    on_status: Optional[Callable[[str], None]] = None,
    on_silent_limit: Optional[Callable[[], None]] = None,
    silence_threshold: int = 200,
    silence_duration: float = 3.0,
    auto_restart: bool = True,
    max_recording_seconds: float = 0.0,
    on_stop_phrase: Optional[Callable[[str], None]] = None,
) -> bool:
    """Start a VAD-driven continuous recording loop.

    ``max_recording_seconds`` caps a single recording (``voice.max_recording_seconds``);
    non-positive / non-numeric disables the cap. ``on_stop_phrase`` receives the stripped
    transcript when the user utters a bare stop phrase (``voice.stop_phrases``); the loop halts
    first, so the consumer only reflects "voice off" — like the manual stop control.
    """
    global _continuous_active, _continuous_recorder, _continuous_auto_restart, _continuous_no_speech_count
    global _continuous_callbacks

    with _continuous_lock:
        if _continuous_active:
            _debug("start_continuous: already active — no-op")
            return True
        if _continuous_stopping:
            _debug("start_continuous: stop/transcribe in progress — busy")
            return False
        _continuous_active = True
        _continuous_auto_restart = auto_restart
        _continuous_callbacks = (on_transcript, on_status, on_silent_limit, on_stop_phrase)
        if auto_restart:
            _continuous_no_speech_count = 0

        if _continuous_recorder is None:
            _continuous_recorder = create_audio_recorder()
        rec = _continuous_recorder
        rec._silence_threshold = silence_threshold
        rec._silence_duration = silence_duration
        # Same numeric-with-bool-excluded guard as cli.py:_voice_start_recording.
        cap_ok = isinstance(max_recording_seconds, (int, float)) and not isinstance(max_recording_seconds, bool)
        rec._max_recording_seconds = max_recording_seconds if cap_ok and max_recording_seconds > 0 else 0.0

    _debug(f"start_continuous: begin (threshold={silence_threshold}, duration={silence_duration}s)")
    # CLI parity: beep *before* opening the stream — after stream.start() it triggers a CoreAudio
    # conflict on macOS.
    _play_beep(frequency=880, count=1)
    try:
        rec.start(on_silence_stop=_continuous_on_silence)
    except Exception as e:
        logger.error("failed to start continuous recording: %s", e)
        _debug(f"start_continuous: rec.start raised {type(e).__name__}: {e}")
        _deactivate()
        raise
    _safe_call(on_status, "listening")
    return True


def stop_continuous(force_transcribe: bool = False) -> None:
    """Stop the active continuous loop and release the microphone.

    Idempotent. With ``force_transcribe`` the recorder stops synchronously, then
    transcription/cleanup runs on a background thread before reporting ``"idle"``; otherwise
    the buffer is discarded.
    """
    global _continuous_active, _continuous_stopping, _continuous_recorder, _continuous_no_speech_count
    global _continuous_callbacks

    with _continuous_lock:
        if not _continuous_active:
            return
        _continuous_active = False
        rec = _continuous_recorder
        callbacks = _continuous_callbacks
        track_no_speech = force_transcribe and not _continuous_auto_restart
        _continuous_stopping = rec is not None
        _continuous_callbacks = _NO_CALLBACKS
        if not track_no_speech:
            _continuous_no_speech_count = 0

    on_transcript, on_status = callbacks[0], callbacks[1]
    if rec is not None:
        if force_transcribe and on_transcript:
            _safe_call(on_status, "transcribing")
            try:
                wav_path = rec.stop()
            except Exception as e:
                logger.warning("failed to stop recorder: %s", e)
                _safe_call(rec.cancel, warn="failed to cancel recorder: %s")
                wav_path = None
            threading.Thread(
                target=_finish_forced_stop, args=(wav_path, callbacks, track_no_speech), daemon=True
            ).start()
            return
        # cancel() (not stop()) discards buffered frames — the loop is over, we don't want to
        # transcribe a half-captured turn.
        _safe_call(rec.cancel, warn="failed to cancel recorder: %s")
    _finish_stop(on_status)


def _finish_forced_stop(wav_path: Optional[str], callbacks: tuple, track_no_speech: bool) -> None:
    """Background tail of ``stop_continuous(force_transcribe=True)``: transcribe, deliver, tally."""
    on_transcript, on_status, on_silent_limit, on_stop_phrase = callbacks
    # With auto_restart=False the CLIENT drives the loop, so a stop phrase must fire the stop
    # signal — discarding the transcript alone would leave the conversation running forever.
    transcript, stop_phrase, stop_text = _turn_transcript(
        wav_path, "failed to stop/transcribe recorder: %s", "stop_continuous", "ending voice chat"
    )
    if stop_phrase:
        _signal_halt(True, stop_text, on_stop_phrase, on_silent_limit)
    if transcript:
        _safe_call(on_transcript, transcript, warn="on_transcript callback raised: %s")
    if track_no_speech:
        held = _voice_activity_held()
        with _continuous_lock:
            should_halt, _ = _tally_silence(bool(transcript) or stop_phrase, held, "stop_continuous")
        if should_halt:
            _safe_call(on_silent_limit)
    _finish_stop(on_status)


def _finish_stop(on_status) -> None:
    """Clear the stopping flag, play the CLI-parity 660 Hz × 2 "stopped" cue, report idle."""
    global _continuous_stopping
    with _continuous_lock:
        _continuous_stopping = False
    _play_beep(frequency=660, count=2)
    _safe_call(on_status, "idle")


def is_continuous_active() -> bool:
    """Whether a continuous voice loop is currently running."""
    with _continuous_lock:
        return _continuous_active


def _continuous_on_silence() -> None:
    """AudioRecorder silence callback — runs in a daemon thread.

    Stops the current capture, transcribes, delivers text via ``on_transcript``, and — if the
    loop is still active — starts the next capture. Three consecutive silent cycles end the loop.
    """
    global _continuous_active, _continuous_no_speech_count

    _debug("_continuous_on_silence: fired")
    with _continuous_lock:
        if not _continuous_active:
            _debug("_continuous_on_silence: loop inactive — abort")
            return
        rec = _continuous_recorder
        on_transcript, on_status, on_silent_limit, on_stop_phrase = _continuous_callbacks
    if rec is None:
        _debug("_continuous_on_silence: no recorder — abort")
        return

    _safe_call(on_status, "transcribing")
    wav_path = rec.stop()
    # Peak RMS tells at a glance whether the mic was too quiet for SILENCE_RMS_THRESHOLD (200)
    # when stop() returns None despite the VAD firing.
    _debug(f"_continuous_on_silence: rec.stop -> {wav_path!r} (peak_rms={getattr(rec, '_peak_rms', -1)})")
    # CLI parity: double beep after the stream stops (safe from the CoreAudio conflict).
    _play_beep(frequency=660, count=2)

    transcript, stop_phrase, stop_text = _turn_transcript(
        wav_path, "continuous transcription failed: %s", "_continuous_on_silence", "ending loop", trace=True
    )
    # Held check runs outside the lock (the probe may call into the host surface).
    held = transcript is None and not stop_phrase and _voice_activity_held()
    with _continuous_lock:
        if not _continuous_active:
            _debug("_continuous_on_silence: stopped during transcribe — no restart")
            return
        limit_hit, no_speech = _tally_silence(bool(transcript) or stop_phrase, held, "_continuous_on_silence")

    if transcript:
        _safe_call(on_transcript, transcript, warn="on_transcript callback raised: %s")
    if stop_phrase or limit_hit:
        _debug(f"_continuous_on_silence: halting ({'stop phrase' if stop_phrase else f'{no_speech} silent cycles'})")
        with _continuous_lock:
            _continuous_active = False
            _continuous_no_speech_count = 0
        _signal_halt(stop_phrase, stop_text, on_stop_phrase, on_silent_limit)
        _safe_call(rec.cancel)
        _safe_call(on_status, "idle")
        return
    _rearm_after_turn(rec, on_status, no_speech)


def _rearm_after_turn(rec: Any, on_status, no_speech: int) -> None:
    """Wait out in-flight TTS, then restart capture (auto_restart) or stop (client-driven loop).

    CLI parity: the mic waits for TTS and then leaves a small gap so the speaker tail isn't
    captured — otherwise the agent's spoken reply lands back in the mic and gets re-submitted.
    """
    if not _tts_playing.is_set():
        _debug("_continuous_on_silence: waiting for TTS to finish")
        _tts_playing.wait(timeout=60)
        time.sleep(0.3)
        with _continuous_lock:
            if not _continuous_active:
                _debug("_continuous_on_silence: stopped while waiting for TTS")
                return
    if not _continuous_auto_restart:
        _debug("_continuous_on_silence: auto_restart=False, stopping loop")
        _deactivate(on_status)
        return
    _debug(f"_continuous_on_silence: restarting loop (no_speech={no_speech})")
    _play_beep(frequency=880, count=1)
    try:
        rec.start(on_silence_stop=_continuous_on_silence)
    except Exception as e:
        logger.error("failed to restart continuous recording: %s", e)
        _debug(f"_continuous_on_silence: restart raised {type(e).__name__}: {e}")
        _deactivate(on_status)
        return
    _safe_call(on_status, "listening")


# ── TTS API ──────────────────────────────────────────────────────────

# Legacy markdown stripper used only when tools.tts_text_normalize is unavailable.
_LEGACY_TTS_STRIP = [
    (re.compile(r'```[\s\S]*?```'), ' '),                       # fenced code blocks
    (re.compile(r'\[([^\]]+)\]\([^)]+\)'), r'\1'),               # [text](url) → text
    (re.compile(r'https?://\S+'), ''),                           # bare URLs
    (re.compile(r'\*\*(.+?)\*\*'), r'\1'),                       # bold
    (re.compile(r'\*(.+?)\*'), r'\1'),                           # italic
    (re.compile(r'`(.+?)`'), r'\1'),                             # inline code
    (re.compile(r'^#+\s*', re.MULTILINE), ''),                   # headers
    (re.compile(r'^\s*[-*]\s+', re.MULTILINE), ''),              # list bullets
    (re.compile(r'---+'), ''),                                   # horizontal rules
    (re.compile(r'\n{3,}'), '\n\n'),                             # excess newlines
]


def _speak_streaming(text: str, stop_event: Optional[threading.Event]) -> bool:
    """Speak via the CLI's ``stream_tts_to_speaker`` pipeline when the configured provider has a
    chunked streamer (audio starts on sentence one); False → caller uses the whole-file path.

    The full reply is fed as one delta + end-of-text sentinel and we block until the done event
    fires — same blocking contract as the sync path, just earlier first audio. ``stop_event`` is
    wired into the pipeline so external barge-in / stop paths can cut playback.
    """
    import queue

    # One dispatcher, zero parallel streaming implementations (#58930): when the configured provider has a
    # chunked streamer registered in tools.tts_streaming, route the whole reply through the same
    # stream_tts_to_speaker pipeline the CLI voice mode uses — audio starts on sentence one instead of after
    # full synthesis. Falls through to the legacy whole-file path when no streamer resolves.
    from tools.tts_streaming import resolve_streaming_provider
    from tools.tts_tool import _load_tts_config
    from tools.tts_tool_speaker import stream_tts_to_speaker

    if resolve_streaming_provider(_load_tts_config()) is None:
        return False
    text_queue: "queue.Queue" = queue.Queue()
    text_queue.put(text)
    text_queue.put(None)  # end-of-text sentinel
    done_event = threading.Event()
    stream_tts_to_speaker(text_queue, stop_event or threading.Event(), done_event)
    return done_event.is_set()


def _speak_whole_file(text: str) -> None:
    """Sync path: clean, synthesize to a temp MP3 via ``text_to_speech_tool``, play, unlink."""
    from tools.tts_tool import text_to_speech_tool

    # Shared cleaner (markdown, emoji, ⋗ blocks, verifier footer, units); the TTS tool owns
    # provider request limits and long-form chunking.
    try:
        from tools.tts_text_normalize import prepare_spoken_text
        tts_text = prepare_spoken_text(text, max_chars=None)
    except Exception:
        tts_text = text
        for pattern, repl in _LEGACY_TTS_STRIP:
            tts_text = pattern.sub(repl, tts_text)
        tts_text = tts_text.strip()
    if not tts_text:
        return

    # Pre-chosen MP3 path so we can play MP3 even when text_to_speech_tool auto-converts to OGG
    # for messaging platforms (afplay's OGG is flaky).
    os.makedirs(os.path.join(tempfile.gettempdir(), "hermes_voice"), exist_ok=True)
    mp3_path = os.path.join(tempfile.gettempdir(), "hermes_voice", f"tts_{time.strftime('%Y%m%d_%H%M%S')}.mp3")
    _debug(f"speak_text: synthesizing {len(tts_text)} chars -> {mp3_path}")
    raw_result = text_to_speech_tool(text=tts_text, output_path=mp3_path)
    try:
        tts_result = json.loads(raw_result) if isinstance(raw_result, str) else {}
    except Exception:
        tts_result = {}

    # The tool result is authoritative — long-form output may be several files.
    play_paths = tts_result.get("file_paths") or [tts_result.get("file_path") or mp3_path]
    played_any = False
    for play_path in play_paths if tts_result.get("success") else []:
        if os.path.isfile(play_path) and os.path.getsize(play_path) > 0:
            _debug(f"speak_text: playing {play_path} ({os.path.getsize(play_path)} bytes)")
            play_audio_file(play_path)
            played_any = True
    for path in set(play_paths + [mp3_path, mp3_path.rsplit(".", 1)[0] + ".ogg"]):
        if os.path.isfile(path):
            with contextlib.suppress(OSError):
                os.unlink(path)
    if not played_any:
        _debug(f"speak_text: TTS tool produced no audio at {mp3_path}")


def speak_text(text: str, stop_event: Optional[threading.Event] = None) -> None:
    """Synthesize ``text`` with the configured TTS provider and play it.

    While playback is in flight ``_tts_playing`` is cleared so the continuous loop waits before
    re-arming the mic (otherwise the agent's reply feedback-loops through the microphone). Live
    capture is cancelled first — otherwise the user's turn tail + our first syllables both land
    in the next recording window — and resumed afterwards so the user can answer without
    pressing the record key.
    """
    if not text or not text.strip():
        return

    paused_recording = False
    with _continuous_lock:
        if _continuous_active and getattr(_continuous_recorder, "is_recording", False):
            try:
                _continuous_recorder.cancel()
                paused_recording = True
            except Exception as e:
                logger.warning("failed to pause recorder for TTS: %s", e)

    _tts_playing.clear()
    _debug(f"speak_text: TTS begin (paused_recording={paused_recording})")
    try:
        # One dispatcher: streaming when a chunked streamer resolves, else the whole-file path.
        try:
            if _speak_streaming(text, stop_event):
                return
        except Exception as e:
            _debug(f"speak_text: streaming dispatch unavailable ({e}); using sync path")
        _speak_whole_file(text)
    except Exception as e:
        logger.warning("Voice TTS playback failed: %s", e)
        _debug(f"speak_text raised {type(e).__name__}: {e}")
    finally:
        _tts_playing.set()
        _debug("speak_text: TTS done")
        # The delay lets afplay release the audio device before sounddevice re-opens.
        if paused_recording:
            time.sleep(0.3)
            with _continuous_lock:
                if _continuous_active and _continuous_recorder is not None:
                    try:
                        _continuous_recorder.start(on_silence_stop=_continuous_on_silence)
                        _debug("speak_text: recording resumed after TTS")
                    except Exception as e:
                        logger.warning("failed to resume recorder after TTS: %s", e)

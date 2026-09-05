"""Audio dashboard routes: transcription upload, voice config, ElevenLabs voices, TTS speak/lease and the speak-stream WebSocket.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are late-bound (cycle-safe).
"""

import base64
import binascii
import contextlib
import logging
import queue
import tempfile
import threading
import asyncio
import json
import os
import urllib.parse
import urllib.request
from fastapi import APIRouter
from hermes_cli.web_routers._common import http_failure
from hermes_cli.web_deps import late
from hermes_cli.web_server_chat import _ws_auth_ok, _ws_request_is_allowed
from hermes_cli.web_server_gateway import _split_text_for_speak_stream
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from hermes_cli.web_models import AudioTranscriptionRequest, TTSSpeakRequest, TTSLeaseRequest
from typing import Any, Dict, Optional

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")
_voice_list_error_logged_once = late("_voice_list_error_logged_once")
load_env = late("load_env", "hermes_cli.config")

_AUDIO_MIME_EXTENSIONS: Dict[str, str] = {
    "audio/aac": ".aac", "audio/flac": ".flac", "audio/m4a": ".m4a", "audio/mp3": ".mp3",
    "audio/mp4": ".mp4", "audio/mpeg": ".mp3", "audio/ogg": ".ogg", "audio/wav": ".wav",
    "audio/wave": ".wav", "audio/webm": ".webm", "audio/x-m4a": ".m4a", "audio/x-wav": ".wav",
    "video/webm": ".webm",
}

_MAX_TRANSCRIPTION_UPLOAD_BYTES = 25 * 1024 * 1024

_SPEAK_MIME_BY_EXT = {
    ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".opus": "audio/ogg", ".wav": "audio/wav",
    ".flac": "audio/flac",
}


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


async def _run_config_scoped(profile: Optional[str], fn):
    """Run ``fn()`` on a worker thread under the config-only profile scope.

    Home-only contextvar scope, NOT ``_profile_scope``: these calls block for a
    provider round-trip and only need config/.env resolution, while
    ``_profile_scope`` holds a process-global skills lock for its entire body.
    """
    def _scoped():
        with _config_profile_scope(profile):
            return fn()

    return await asyncio.get_running_loop().run_in_executor(None, _scoped)


def _audio_extension_for_mime(mime_type: str) -> str:
    normalized = (mime_type or "").split(";", 1)[0].strip().lower()
    return _AUDIO_MIME_EXTENSIONS.get(normalized, ".webm")


@router.post("/api/audio/transcribe")
async def transcribe_audio_upload(
    payload: AudioTranscriptionRequest, profile: Optional[str] = None
):
    data_url = (payload.data_url or "").strip()
    if not data_url.startswith("data:") or "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid audio payload")

    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="Audio payload must be base64 encoded")

    mime_type = (payload.mime_type or header[5:].split(";", 1)[0] or "audio/webm").strip()
    normalized_mime_type = mime_type.split(";", 1)[0].lower()
    if not (normalized_mime_type.startswith("audio/") or normalized_mime_type == "video/webm"):
        raise HTTPException(status_code=400, detail="Payload must be an audio recording")

    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Audio payload is not valid base64")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio recording is empty")
    if len(audio_bytes) > _MAX_TRANSCRIPTION_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio recording is too large")

    temp_path = ""
    try:
        with http_failure("Desktop voice transcription failed", 500, "Transcription failed"):
            with tempfile.NamedTemporaryFile(
                prefix="hermes-desktop-voice-", suffix=_audio_extension_for_mime(mime_type),
                delete=False,
            ) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            # transcribe_recording (not raw transcribe_audio): filters Whisper
            # hallucinations and maps provider "empty transcript" errors to a
            # successful empty result — the live voice loop treats "" as silence
            # and re-listens instead of surfacing a 400 on every quiet turn.
            from tools.voice_mode import transcribe_recording

            result = await _run_config_scoped(profile, lambda: transcribe_recording(temp_path))
    finally:
        if temp_path:
            _unlink_quietly(temp_path)

    if not result.get("success"):
        err = result.get("error") or "Transcription failed"
        # No speech detected is a normal outcome for VAD/continuous voice loops
        # (re-listening on silence), not an error: return an empty transcript so
        # the client quietly re-listens instead of showing a failure toast.
        if "empty transcript" in err.lower():
            return {"ok": True, "transcript": "", "provider": result.get("provider")}
        raise HTTPException(status_code=400, detail=err)

    return {
        "ok": True, "transcript": str(result.get("transcript") or "").strip(),
        "provider": result.get("provider"),
    }


@router.get("/api/audio/voice-config")
async def get_client_voice_config(profile: Optional[str] = None):
    """The active profile's STT/TTS config for CLIENT-DIRECT voice.

    Lets the desktop cut the audio relay hop: mic audio goes straight to the
    profile's STT provider and reply text is synthesized on the client with
    the profile's TTS provider — the desktop↔gateway link carries only text.
    Providers that can only run on this host (local whisper, edge-tts,
    command/plugin providers) resolve to ``{"mode": "relay"}`` and the
    desktop keeps using the /api/audio/* relay endpoints.

    Same trust boundary as every profile-scoped route: the caller is an
    authenticated client that can already drive the agent. Keys in the
    response are held in client memory only, never persisted client-side.
    Gate: ``voice.client_direct`` in config.yaml (default true).
    """
    from tools.voice_client_config import resolve_client_voice_config
    try:
        result = await _run_config_scoped(profile, resolve_client_voice_config)
    except Exception:
        _log.exception("Client voice-config resolution failed")
        fallback = {"mode": "relay", "reason": "resolution error"}
        return {"ok": True, "stt": fallback, "tts": dict(fallback)}

    return {"ok": True, **result}


def _elevenlabs_voice_label(voice: Dict[str, Any]) -> str:
    name = str(voice.get("name") or voice.get("voice_id") or "Voice").strip()
    category = str(voice.get("category") or "").strip()

    return f"{name} ({category})" if category else name


@router.get("/api/audio/elevenlabs/voices")
async def get_elevenlabs_voices(profile: Optional[str] = None):
    """Return ElevenLabs voices when an API key is configured.

    The desktop UI uses this for the ``tts.elevenlabs.voice_id`` dropdown.
    Only non-secret voice metadata is returned; the API key stays server-side.
    """
    # Config-only scope (await-safe): the key lookup reads the requested
    # profile's .env, matching the profile the settings UI writes to.
    with _config_profile_scope(profile):
        api_key = (load_env().get("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        # Fallback for env-only deployments — scope-aware: under multiplex
        # os.environ may hold another profile's key, so honor the installed
        # scope's verdict before touching the env.
        try:
            from agent.secret_scope import UnscopedSecretError, get_secret

            try:
                api_key = (get_secret("ELEVENLABS_API_KEY") or "").strip()
            except UnscopedSecretError:
                api_key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
        except Exception:
            api_key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        return {"available": False, "voices": []}

    request = urllib.request.Request(
        "https://api.elevenlabs.io/v1/voices",
        headers={"Accept": "application/json", "xi-api-key": api_key},
    )

    try:
        loop = asyncio.get_running_loop()

        def _fetch() -> Dict[str, Any]:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))

        payload = await loop.run_in_executor(None, _fetch)
    except urllib.error.HTTPError as exc:
        # An auth failure (bad/expired/scoped key) is a persistent, user-fixable
        # state and the desktop polls this on every settings open/focus, so
        # treat 401/403 as "integration unavailable": 200 to the UI and log at
        # most once until the error signature changes.
        if exc.code in (401, 403):
            if _voice_list_error_logged_once(f"http-{exc.code}"):
                _log.info("ElevenLabs voices unavailable: %s — check ELEVENLABS_API_KEY", exc)
            return {"available": False, "voices": [], "error": "unauthorized"}
        if _voice_list_error_logged_once(f"http-{exc.code}"):
            _log.warning("ElevenLabs voice list failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not load ElevenLabs voices")
    except Exception as exc:
        if _voice_list_error_logged_once(str(exc)):
            _log.warning("ElevenLabs voice list failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not load ElevenLabs voices")
    _voice_list_error_logged_once(None)  # success — re-arm logging for next failure

    voices = []
    for voice in payload.get("voices") or []:
        if not isinstance(voice, dict):
            continue

        voice_id = str(voice.get("voice_id") or "").strip()
        if not voice_id:
            continue

        voices.append({
            "voice_id": voice_id, "name": str(voice.get("name") or voice_id),
            "label": _elevenlabs_voice_label(voice),
        })

    voices.sort(key=lambda item: str(item.get("label") or "").lower())
    return {"available": True, "voices": voices}


@router.post("/api/audio/speak")
async def speak_text(payload: TTSSpeakRequest, profile: Optional[str] = None):
    """Synthesize speech and return audio as base64 data URL.

    Used by the desktop voice-conversation mode to play back assistant
    responses without exposing the on-disk file path; reuses the TTS provider
    chain configured under ``tts.`` in config.yaml.
    """
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # _config_profile_scope raises 400/404 for a bad profile — pass it
    # through instead of masking it as a 500 synthesis failure.
    with http_failure("Desktop voice TTS failed", 500, "Speech synthesis failed"):
        from tools.tts_tool import text_to_speech_tool

        result_json = await _run_config_scoped(profile, lambda: text_to_speech_tool(text))

    try:
        result = json.loads(result_json) if isinstance(result_json, str) else result_json
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid TTS response")

    if not result.get("success"):
        raise HTTPException(
            status_code=400, detail=result.get("error") or "Speech synthesis failed",
        )

    file_path = result.get("file_path")
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=500, detail="Audio file missing")

    mime_type = _SPEAK_MIME_BY_EXT.get(os.path.splitext(file_path)[1].lower(), "audio/mpeg")

    def _read_and_unlink() -> bytes:
        # Off-loop: synthesized audio can be several MB; reading it inline
        # blocks the uvicorn event loop. Unlink rides the same thread hop so
        # the temp file cannot outlive an early return.
        try:
            with open(file_path, "rb") as fh:
                return fh.read()
        finally:
            _unlink_quietly(file_path)

    try:
        audio_bytes = await asyncio.to_thread(_read_and_unlink)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read audio: {exc}")

    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return {
        "ok": True, "data_url": f"data:{mime_type};base64,{encoded}", "mime_type": mime_type,
        "provider": result.get("provider"),
    }


@router.post("/api/audio/tts-lease")
async def tts_lease(payload: TTSLeaseRequest, profile: Optional[str] = None):
    """Desktop TTS-output toggles as warm-up / release signals.

    ``active: true`` registers a lease on the TTS engine and pre-loads the
    configured provider (local model, lazily-installed SDK) so the first spoken
    reply doesn't pay the load as dead air; ``active: false`` drops the lease
    and, once no surface holds one, unloads resident local models. Blocking
    work runs off the event loop. Warm-up failures are reported in the body,
    never as an HTTP error — the toggle must succeed even when preload fails.
    """
    lease = (payload.lease or "").strip()
    if not lease:
        raise HTTPException(status_code=400, detail="lease is required")

    def _apply():
        from tools.tts_tool_lifecycle import acquire_tts_lease, release_tts_lease
        if payload.active:
            with _config_profile_scope(profile):
                return acquire_tts_lease(lease)
        return release_tts_lease(lease)

    try:
        result = await asyncio.get_running_loop().run_in_executor(None, _apply)
    except HTTPException:
        raise
    except Exception as exc:
        _log.warning("TTS lease %s (%s) failed: %s", lease, payload.active, exc)
        result = {"leases": None, "action": "error", "error": str(exc)}
    return {"ok": True, "lease": lease, "active": payload.active, **result}


@router.websocket("/api/audio/speak-stream")
async def speak_stream_ws(ws: "WebSocket") -> None:
    """Streaming TTS for the desktop: text in, raw int16 PCM frames out.

    The socket is a per-reply speech *session*: the client feeds text
    incrementally as LLM deltas arrive, the server cuts sentences
    (``SentenceChunker`` — same cutter as the CLI/TUI speaker pipeline) and
    streams each one's PCM the moment it's ready, so speech overlaps generation.

    Protocol:
      client → ``{"text": "..."}`` frames (incremental; may combine with done),
               ``{"done": true}`` when the reply is complete,
               ``{"stop": true}`` or disconnect = barge-in
      server → ``{"type": "start", "sample_rate": N, "channels": 1}``,
               binary PCM frames, then ``{"type": "end"}``
      server → ``{"type": "fallback"}`` when the configured provider has no
               chunked API — the client uses the POST endpoint instead.
    """
    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return
    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return
    await ws.accept()

    # Profile via query param, like /api/pty and /api/console: the provider
    # chain + API keys must resolve from the requesting profile's config, not
    # the dashboard's own. The streamer captures its config at resolve time,
    # so scoping resolution scopes the whole session.
    profile = (ws.query_params.get("profile") or "").strip() or None

    loop = asyncio.get_running_loop()

    def _resolve():
        from tools.tts_streaming import resolve_streaming_provider
        from tools.tts_tool import _get_provider, _load_tts_config, _resolve_max_text_length
        with _config_profile_scope(profile):
            cfg = _load_tts_config()
            streamer = resolve_streaming_provider(cfg)
            cap = _resolve_max_text_length(_get_provider(cfg), cfg) if streamer else 0
        return streamer, cap

    try:
        streamer, cap = await loop.run_in_executor(None, _resolve)
    except Exception:
        _log.exception("speak-stream provider resolution failed")
        streamer, cap = None, 0
    if streamer is None:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "fallback"})
            await ws.close()
        return

    await ws.send_json(
        {"type": "start", "sample_rate": streamer.sample_rate, "channels": streamer.channels}
    )

    stop = threading.Event()
    text_q: queue.Queue = queue.Queue()  # str deltas; None = end-of-text
    chunks: asyncio.Queue = asyncio.Queue()  # PCM out; None = synthesis done

    def _produce():
        from tools.tts_streaming import SentenceChunker
        from tools.tts_text_normalize import _strip_markdown_for_tts

        chunker = SentenceChunker()

        # The session stays open for a whole agent turn and no text arrives
        # during tool execution, so without an idle flush a narration line with
        # no trailing whitespace ("Let me check.") sits in the chunker until
        # end-of-turn. Mirror the CLI speaker pipeline: poll with a timeout and
        # flush when the producer goes idle — immediately when the buffer ends
        # on sentence punctuation, after a longer quiet spell otherwise.
        idle_poll_seconds = 0.5
        idle_polls_before_force_flush = 4  # ~2s of silence

        def _sentences():
            idle_polls = 0
            while not stop.is_set():
                try:
                    delta = text_q.get(timeout=idle_poll_seconds)
                except queue.Empty:
                    idle_polls += 1
                    buffered = chunker.buf.strip()
                    if not buffered or ("<think" in chunker.buf and "</think>" not in chunker.buf):
                        continue
                    if buffered.endswith((".", "!", "?", "…", ":")) or idle_polls >= idle_polls_before_force_flush:
                        yield from chunker.flush()
                    continue
                idle_polls = 0
                if delta is None:
                    yield from chunker.flush()
                    return
                yield from chunker.feed(delta)

        try:
            for sentence in _sentences():
                cleaned = _strip_markdown_for_tts(sentence)
                if not cleaned:
                    continue
                for piece in _split_text_for_speak_stream(cleaned, cap):
                    for chunk in streamer.stream(piece):
                        if stop.is_set():
                            return
                        loop.call_soon_threadsafe(chunks.put_nowait, chunk)
        except Exception as exc:
            _log.warning("speak-stream synthesis failed: %s", exc)
        finally:
            loop.call_soon_threadsafe(chunks.put_nowait, None)

    threading.Thread(target=_produce, daemon=True).start()

    async def _pump_client():
        # Text frames feed synthesis; done ends the text; stop/disconnect
        # (or any unparseable frame) is barge-in.
        try:
            while True:
                frame = json.loads(await ws.receive_text())
                if frame.get("text"):
                    text_q.put(str(frame["text"]))
                if frame.get("stop"):
                    break
                if frame.get("done"):
                    text_q.put(None)
        except Exception:
            pass
        stop.set()
        text_q.put(None)  # unblock the producer

    pump = asyncio.ensure_future(_pump_client())
    try:
        while True:
            chunk = await chunks.get()
            if chunk is None:
                break
            await ws.send_bytes(chunk)
        if not stop.is_set():
            await ws.send_json({"type": "end"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop.set()
        text_q.put(None)
        pump.cancel()
        with contextlib.suppress(Exception):
            await ws.close()


def _converse_stt_model(profile: Optional[str]) -> Optional[str]:
    """STT model override for the converse loop, mirroring cli_voice_mixin._voice_stt_model.

    Local provider prefers ``stt.local.model`` (default ``base``); every other
    provider uses ``stt.model`` (or the provider default when unset). Resolved
    under the requesting profile's config scope.
    """
    with _config_profile_scope(profile):
        from hermes_cli.config import load_config

        stt = (load_config().get("stt") or {})
        if str(stt.get("provider") or "").strip().lower() == "local":
            local = stt.get("local") if isinstance(stt.get("local"), dict) else {}
            return (local or {}).get("model") or "base"
        return stt.get("model")


# Start-frame deadline: the client must send its {"type":"start", ...} within this window.
_CONVERSE_FIRST_FRAME_TIMEOUT = 5.0


async def _read_converse_start_frame(ws: "WebSocket") -> Optional[dict]:
    """Read the client's FIRST frame; return the parsed ``{"type":"start", ...}`` dict.

    Returns ``None`` (the caller sends ``bad_start`` + close 4400) on timeout, a
    disconnect/binary first frame, non-JSON, or any frame whose ``type`` isn't ``"start"``.
    Dashboard auth is the pre-accept token check, so no key is validated here.
    """
    try:
        message = await asyncio.wait_for(ws.receive(), timeout=_CONVERSE_FIRST_FRAME_TIMEOUT)
    except (asyncio.TimeoutError, Exception):
        return None
    if message.get("type") != "websocket.receive":
        return None
    text = message.get("text")
    if text is None:
        return None
    try:
        frame = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(frame, dict) or frame.get("type") != "start":
        return None
    return frame


@router.websocket("/api/audio/converse")
async def converse_ws(ws: "WebSocket") -> None:
    """Off-device realtime voice loop: mic PCM in, agent speech out, over one WS.

    The client sends a single ``{"type":"start", ...}`` frame first (carrying session
    config: input_rate/output_rate/idle_interval/name/profile; auth is the pre-accept
    dashboard token, so start.key is not consulted here), then streams PCM16 mono at
    ``input_rate`` as binary frames; the server does VAD → STT → a REAL agent turn
    (``prompt.submit`` in-process, so the spoken conversation persists) → streaming
    TTS, sending PCM16 (at ``output_rate``) back as binary frames. JSON text frames
    carry control. Barge-in is supported: speech during playback cuts the reply.

    Protocol:
      client → ``{"type":"start", ...}`` (FIRST frame; all config fields optional),
               then binary PCM16 mono frames at ``input_rate`` (30 ms blocks preferred),
               ``{"stop": true}`` to end, ``{"commit": true}`` to force endpoint
      server → ``{"type": "ready", "input": {...}, "output": {...}}``,
               ``{"type": "transcript", "text": ...}``,
               ``{"type": "speaking"}`` then binary PCM frames,
               ``{"type": "interrupted"}`` on barge-in,
               ``{"type": "turn_done"}`` after each reply,
               ``{"type": "error", "error": ...}`` on failure.

    The heavy lifting lives in the shared driver
    :func:`tools.voice_converse_loop.drive_converse_turns` (plus the dashboard turn
    glue in :mod:`hermes_cli.web_routers._converse_loop`); this handler only gates
    auth, pre-creates the session, wires the client pump, and adapts ``run_voice_turn``
    onto the driver.

    BARGE-IN (v1 limitation): a VAD trip while playing stops TTS PLAYBACK (tts_stop +
    mark_speech_interrupted) and emits ``{"type":"interrupted"}``, but the in-flight
    agent turn (``run_voice_turn`` / ``prompt.submit``) is NOT cancelled in v1 — it
    runs to completion, so a barged turn may still finish and fire tools, and the next
    utterance queues behind it. Cancelling the in-flight turn is a deliberate follow-up.
    """
    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return
    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return
    await ws.accept()

    # The client's FIRST frame is the single {"type":"start", ...} carrying all session
    # config (rates, idle_interval, name, profile). Dashboard auth is the pre-accept token
    # check above, so start.key is not consulted here — the frame need only be a valid start.
    frame = await _read_converse_start_frame(ws)
    if frame is None:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "error": "bad_start"})
            await ws.close(code=4400)
        return

    from tools.voice_converse_loop import parse_start_config

    input_rate, output_rate, idle_interval, name, profile = parse_start_config(frame)
    loop = asyncio.get_running_loop()

    def _resolve():
        import numpy as np
        from tools.tts_tool import _get_provider, _load_tts_config, _resolve_max_text_length
        from tools.voice_converse_loop import resample_synth, resolve_converse_synthesizer
        from hermes_cli.web_routers._converse_loop import ConverseSession, create_voice_session

        stt_model = _converse_stt_model(profile)
        with _config_profile_scope(profile):
            cfg = _load_tts_config()
            # Always resolves a synthesizer (streaming when available, else the one-shot
            # fallback) — works with any provider, incl. edge — wrapped to emit at output_rate.
            synth = resample_synth(resolve_converse_synthesizer(cfg), output_rate)
            cap = _resolve_max_text_length(_get_provider(cfg), cfg)
        session = ConverseSession(np, stt_model=stt_model, input_rate=input_rate)
        sid = create_voice_session()
        return synth, cap, session, sid

    try:
        synth, cap, session, sid = await loop.run_in_executor(None, _resolve)
    except Exception:
        _log.exception("converse setup failed")
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "error": "converse setup failed"})
            await ws.close()
        return

    await ws.send_json({
        "type": "ready",
        "input": {"sample_rate": input_rate, "format": "pcm16", "block_ms": 30},
        "output": {"sample_rate": output_rate, "format": "pcm16"},
    })

    session.start()

    # The tui_gateway session (``sid``) owns the durable conversation state used by
    # each ``run_voice_turn`` (via prompt.submit), so this list is NOT fed to the
    # agent; it exists only so the shared driver can record the turn transcript/reply
    # under a bounded tail cap (keeping a long-lived socket's memory in check).
    conversation_history: list = []

    async def _pump_client():
        # Binary frames feed the mic shim; {"stop"}/disconnect ends; {"commit"}
        # forces the current utterance to endpoint.
        try:
            while True:
                frame = await ws.receive()
                if frame.get("bytes") is not None:
                    session.stream.feed(frame["bytes"])
                    continue
                text = frame.get("text")
                if text is None:
                    break  # websocket.disconnect
                try:
                    msg = json.loads(text)
                except (ValueError, TypeError):
                    continue
                if msg.get("stop"):
                    break
                if msg.get("commit"):
                    session.commit()
        except Exception:
            _log.debug("converse client pump ended", exc_info=True)
        session.stop()

    # NAME (v1 dashboard limitation): run_voice_turn -> prompt.submit has no clean per-turn
    # ephemeral-system-prompt seam (unlike the gateway's _run_agent), so the name-aware voice
    # prompt can't be injected as a system prompt here. As a fallback we prepend a one-line
    # identity note to the FIRST turn's transcript only, so the model learns its name once
    # without polluting every subsequent turn's history.
    _identity_pending = [bool(name)]

    async def _run_turn(transcript, on_delta, *, interrupted: bool):
        # Dashboard turn adapter for drive_converse_turns: run the real agent turn via
        # run_voice_turn (prompt.submit in-process), streaming deltas out through
        # on_delta. Returns (reply_text, err): run_voice_turn already returns the error
        # (or None), and the reply is the joined deltas (computed by the driver, so we
        # return "" here). The per-turn barge-in note is plumbed via `interrupted`.
        # run_voice_turn is sync (blocks until the turn ends), so run it off the loop.
        from hermes_cli.web_routers._converse_loop import run_voice_turn

        text = transcript
        if _identity_pending[0]:
            _identity_pending[0] = False
            text = (f"(You are {name}. This is a live voice conversation — keep replies to a "
                    f"couple of short spoken sentences.)\n\n{transcript}")
        err = await loop.run_in_executor(
            None, lambda: run_voice_turn(sid, text, on_delta, interrupted=interrupted))
        return "", err

    async def _drive_turns():
        from tools.voice_converse_loop import drive_converse_turns

        # start.idle_interval: session mode — periodic {"type":"idle"} during quiet (socket
        # stays open) + spoken stop phrases become {"type":"stop_word"}. 0 = continuous.
        await drive_converse_turns(
            session=session, synth=synth, cap=cap, loop=loop,
            send_json=ws.send_json, send_bytes=ws.send_bytes,
            run_turn=_run_turn, history=conversation_history,
            idle_interval=idle_interval)

    pump = asyncio.ensure_future(_pump_client())
    driver = asyncio.ensure_future(_drive_turns())
    try:
        await driver
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        _log.exception("converse loop crashed")
    finally:
        session.stop()
        pump.cancel()
        driver.cancel()
        with contextlib.suppress(Exception):
            await ws.close()

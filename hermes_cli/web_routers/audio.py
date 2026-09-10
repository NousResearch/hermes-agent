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

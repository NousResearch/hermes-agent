"""``hermes tts serve`` — expose the configured TTS provider over an OpenAI-compatible HTTP API.

Why this exists
---------------
Hermes' TTS providers (Piper, KittenTTS, NeuTTS, Edge, ...) are in-process Python
libraries. The Hermes WebUI's "OpenAI TTS (server)" engine speaks OpenAI's
``/v1/audio/speech`` wire format and honours ``tts.openai.base_url``, but it can
only reach a *server* — there is no way to point it at a local engine directly.

This module is that server: it listens on loopback and answers
``POST /v1/audio/speech`` by calling Hermes' own TTS stack, so a local voice
becomes reachable by any OpenAI-compatible client (the WebUI included) with no
third-party server in the loop.

Design notes
------------
* Synthesis is delegated to :func:`tools.tts_tool.text_to_speech_tool` — the same
  entry point the agent's TTS tool uses — so chunking, text normalisation,
  provider dispatch and container repair behave identically whether the audio
  came from the chat loop or from this server. The tool returns a JSON envelope
  naming the file it wrote; the bytes are read back and the file removed.
* The provider is **not** selectable per request. ``tts.provider`` in
  ``config.yaml`` is the authoritative backend selector (see #90109): a
  per-call override that disagrees with it is ignored upstream, and routing
  speech to a different vendor than the operator configured is exactly what
  that guard exists to prevent. The configured provider is therefore read once
  at startup and reported by ``/health``.
* Voice selection follows the same rule: ``voice_id`` is an ElevenLabs-only
  concept upstream, so the request's ``voice`` field is accepted and ignored
  rather than silently rerouting the voice.
* The server binds to loopback and has **no authentication**; it is a local
  convenience, not an internet-facing service. Binding elsewhere requires an
  explicit opt-in.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("hermes_cli.tts_server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5099

# OpenAI response_format -> the container the synthesis path must produce.
# Upstream derives the format from the output file's extension
# (tools/tts_tool_providers._tts_response_format_from_path) and ffmpeg-converts
# WAV-native engines into it, so the extension is what we steer. Formats the
# shared path cannot produce are left out rather than answered with the wrong
# bytes: upstream maps an unknown format to mp3, and so do we.
#
# "opus" is the one that needs care: ffmpeg's default codec for an .ogg target
# is Vorbis, and OpenAI's "opus" means Opus-in-Ogg. Upstream's own fix for the
# same mismatch is tools.tts_tool_delivery._convert_to_opus, which we reuse.
_RESPONSE_FORMAT_EXT = {
    "mp3": ".mp3",
    "opus": ".ogg",
    "wav": ".wav",
    "flac": ".flac",
}

_RESPONSE_CONTENT_TYPE = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "wav": "audio/wav",
    "flac": "audio/flac",
}

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class TTSRequestError(Exception):
    """A client-side error (400) whose message is safe to return verbatim."""


def _normalize_response_format(raw: Any) -> str:
    """OpenAI ``response_format`` -> one of mp3/opus/wav/flac/pcm (default mp3)."""
    value = str(raw or "mp3").strip().lower()
    return value if value in _RESPONSE_FORMAT_EXT else "mp3"


def _resolve_bind_address(host: str, port: int, *, allow_remote: bool) -> tuple[str, int]:
    """Bind address, refusing a non-loopback host unless explicitly allowed.

    This server has no authentication, so exposing it on a LAN interface would
    let anyone on the network drive the host's voice — and, for a paid
    provider, spend the owner's credits.
    """
    if host not in _LOOPBACK_HOSTS and not allow_remote:
        raise ValueError(
            f"refusing to bind {host!r}: this server has no authentication. "
            f"Use a loopback address, or pass --allow-remote if you understand the risk."
        )
    return host, port


def _configured_tts_section() -> dict[str, Any]:
    """The active profile's ``tts`` config section ({} when unavailable)."""
    from hermes_cli.config import load_config

    section = load_config().get("tts")
    return section if isinstance(section, dict) else {}


def _configured_provider() -> str:
    """The configured TTS provider name (``tts.provider``)."""
    from tools.tts_tool import _get_provider

    return _get_provider(_configured_tts_section())


def _configured_voice() -> str:
    """The provider's configured voice, for display only."""
    tts_config = _configured_tts_section()
    provider = _configured_provider()
    section = tts_config.get(provider)
    if isinstance(section, dict):
        for key in ("voice", "voice_id"):
            value = section.get(key)
            if value:
                return str(value)
    return ""


def synthesize_to_bytes(
    text: str,
    *,
    response_format: str = "mp3",
    speed: Optional[float] = None,
) -> tuple[bytes, str]:
    """Synthesize *text* with the configured provider -> ``(audio_bytes, content_type)``.

    Raises :class:`TTSRequestError` for a client-side problem and ``RuntimeError``
    when the synthesis itself fails.
    """
    from tools.tts_tool import text_to_speech_tool

    fmt = _normalize_response_format(response_format)
    ext = _RESPONSE_FORMAT_EXT[fmt]
    content_type = _RESPONSE_CONTENT_TYPE[fmt]

    # A private temp dir per request: the synthesis path writes a file and
    # returns its path; we read the bytes back and delete them.
    with tempfile.TemporaryDirectory(prefix="hermes-tts-") as tmpdir:
        output_path = str(Path(tmpdir) / f"speech{ext}")
        envelope = text_to_speech_tool(text, output_path=output_path, speed=speed)
        try:
            result = json.loads(envelope)
        except (TypeError, ValueError):
            raise RuntimeError(f"TTS returned a non-JSON envelope: {envelope[:200]!r}")
        if not result.get("success"):
            raise TTSRequestError(str(result.get("error") or "TTS synthesis failed"))

        file_path = result.get("file_path") or ""
        if not file_path or not os.path.isfile(file_path):
            raise RuntimeError("TTS synthesis produced no output file")

        if fmt == "opus":
            file_path = _ensure_opus(file_path)

        with open(file_path, "rb") as fh:
            return fh.read(), content_type


def _ensure_opus(file_path: str) -> str:
    """Return a path holding real Opus-in-Ogg, converting *file_path* if needed.

    The synthesis path converts a WAV-native engine into whatever the target
    extension implies, and ffmpeg's default codec for ``.ogg`` is Vorbis — while
    OpenAI's ``opus`` format means Opus. Upstream already solves this for voice
    bubbles (tools.tts_tool_delivery._convert_to_opus); reuse it rather than
    growing a second transcoder.

    A native-Opus provider (OpenAI, ElevenLabs, ...) already writes real Opus,
    so the codec header is checked first and the file is left alone. A failed
    conversion keeps the original: the bytes still play.
    """
    from tools.tts_tool_delivery import _convert_to_opus

    try:
        with open(file_path, "rb") as fh:
            header = fh.read(64)
    except OSError:
        return file_path
    if b"OpusHead" in header:
        return file_path
    converted = _convert_to_opus(file_path)
    return converted or file_path


def _build_handler(provider: str, voice: str, home: str) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to the resolved provider/voice and profile home.

    The handler runs in a ``ThreadingHTTPServer`` worker thread, and a bare
    thread starts with an empty context: the ContextVar-scoped Hermes home
    override does not cross the hop, so an unbound read would silently land on
    the launch profile instead of the one this server was started for. Each
    request therefore re-binds *home* — the profile resolved on the serving
    thread — around the work.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "HermesTTSServer/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            logger.info("[tts-serve] " + format, *args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/health":
                self._send_json(200, {"ok": True, "provider": provider, "voice": voice})
            elif path in ("/v1/models", "/models"):
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": provider or "hermes-tts", "object": "model",
                              "owned_by": "hermes"}],
                })
            else:
                self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/")
            if path not in ("/v1/audio/speech", "/audio/speech"):
                self._send_json(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, TypeError):
                self._send_json(400, {"error": {"message": "invalid JSON body"}})
                return
            if not isinstance(payload, dict):
                self._send_json(400, {"error": {"message": "body must be a JSON object"}})
                return

            text = str(payload.get("input") or payload.get("text") or "").strip()
            if not text:
                self._send_json(400, {"error": {"message": "input is required"}})
                return

            response_format = _normalize_response_format(payload.get("response_format"))
            speed = payload.get("speed")
            if speed is not None:
                try:
                    speed = float(speed)
                except (TypeError, ValueError):
                    self._send_json(400, {"error": {"message": "speed must be a number"}})
                    return

            try:
                audio, content_type = _synthesize_in_profile(
                    text, home, response_format=response_format, speed=speed,
                )
            except TTSRequestError as exc:
                self._send_json(400, {"error": {"message": str(exc)}})
                return
            except Exception as exc:
                logger.exception("TTS synthesis failed")
                self._send_json(500, {"error": {"message": f"TTS synthesis failed: {exc}"}})
                return

            self._send(200, audio, content_type)

    return Handler


def _synthesize_in_profile(
    text: str, home: str, *, response_format: str, speed: Optional[float],
) -> tuple[bytes, str]:
    """Synthesize under the serving profile's scope (see :func:`_build_handler`)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(home)
    try:
        return synthesize_to_bytes(text, response_format=response_format, speed=speed)
    finally:
        reset_hermes_home_override(token)


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    allow_remote: bool = False,
) -> int:
    """Run the TTS server until interrupted. Returns a process exit code."""
    try:
        host, port = _resolve_bind_address(host, port, allow_remote=allow_remote)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    provider = _configured_provider()
    voice = _configured_voice()

    # Resolve the serving profile's home on THIS thread (the caller's scope),
    # then hand it to the handler: the handler runs in a worker thread whose
    # context is empty, so the scope must be bound explicitly per request.
    from hermes_constants import get_hermes_home

    home = str(get_hermes_home())

    try:
        server = ThreadingHTTPServer((host, port), _build_handler(provider, voice, home))
    except OSError as exc:
        print(f"Could not bind {host}:{port}: {exc}", file=sys.stderr)
        return 1

    print(f"Hermes TTS server on http://{host}:{port}")
    print(f"  provider: {provider}")
    print(f"  voice:    {voice or '(provider default)'}")
    print(f"  profile:  {home}")
    print("  endpoint: POST /v1/audio/speech   (OpenAI-compatible)")
    print("  health:   GET  /health")
    print("Press Ctrl+C to stop.")

    # serve_forever() runs on the MAIN thread so Ctrl+C lands here directly;
    # a daemon thread would swallow the signal and the shutdown path would
    # depend on a cross-thread join.
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
    return 0

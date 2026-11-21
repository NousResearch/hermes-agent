"""Contract tests for ``hermes tts serve`` — the OpenAI-compatible TTS server.

The server's whole job is to be reachable by an OpenAI-compatible client (the
Hermes WebUI's "OpenAI TTS (server)" engine is the motivating one), so these
tests pin the wire contract rather than the implementation: the routes, the
request body shape, the response headers and the bytes on the wire.
"""

from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request

import pytest

from hermes_cli import tts_server


@pytest.fixture
def server(monkeypatch):
    """Run the handler on an ephemeral port with synthesis stubbed out.

    The stub replaces only the byte-producing step; routing, body parsing and
    the response envelope are the real code paths.
    """
    from hermes_constants import get_hermes_home

    home = str(get_hermes_home())
    handler = tts_server._build_handler("piper", "en_US-lessac-medium", home)
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post(base: str, path: str, payload: dict, *, raw: bytes | None = None):
    """POST to the server and return ``(status, headers, body_bytes)``."""
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def test_health_reports_the_configured_provider(server):
    """``GET /health`` must name the provider the server is actually speaking with."""
    with urllib.request.urlopen(f"{server}/health", timeout=10) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read())
    assert payload == {
        "ok": True, "provider": "piper", "voice": "en_US-lessac-medium"}


def test_audio_speech_returns_the_synthesized_bytes(server, monkeypatch):
    """``POST /v1/audio/speech`` returns audio bytes, not a JSON envelope."""
    monkeypatch.setattr(
        tts_server, "synthesize_to_bytes",
        lambda text, **kwargs: (b"ID3fake-audio", "audio/mpeg"))

    status, headers, body = _post(
        server, "/v1/audio/speech", {"model": "tts-1", "input": "hello"})

    assert status == 200
    assert headers["Content-Type"] == "audio/mpeg"
    assert body == b"ID3fake-audio"


def test_audio_speech_accepts_the_unversioned_openai_path(server, monkeypatch):
    """OpenAI-compatible clients post to ``/v1/audio/speech``; some use ``/audio/speech``."""
    monkeypatch.setattr(
        tts_server, "synthesize_to_bytes",
        lambda text, **kwargs: (b"audio", "audio/mpeg"))

    status, _, body = _post(server, "/audio/speech", {"input": "hello"})

    assert status == 200
    assert body == b"audio"


def test_requested_response_format_reaches_the_synthesizer(server, monkeypatch):
    """``response_format`` must steer the container the client asked for."""
    seen = {}

    def _capture(text, **kwargs):
        seen.update(kwargs)
        return b"audio", "audio/ogg"

    monkeypatch.setattr(tts_server, "synthesize_to_bytes", _capture)

    status, headers, _ = _post(
        server, "/v1/audio/speech", {"input": "hello", "response_format": "opus"})

    assert status == 200
    assert headers["Content-Type"] == "audio/ogg"
    assert seen["response_format"] == "opus"


def test_speed_is_forwarded_to_the_synthesizer(server, monkeypatch):
    """``speed`` is part of OpenAI's body and must reach the synthesis call."""
    seen = {}

    def _capture(text, **kwargs):
        seen.update(kwargs)
        return b"audio", "audio/mpeg"

    monkeypatch.setattr(tts_server, "synthesize_to_bytes", _capture)

    status, _, _ = _post(server, "/v1/audio/speech", {"input": "hello", "speed": 1.5})

    assert status == 200
    assert seen["speed"] == 1.5


def test_missing_input_is_a_client_error(server):
    """An empty body must not synthesize silence — it is a 400."""
    status, _, body = _post(server, "/v1/audio/speech", {"model": "tts-1"})

    assert status == 400
    assert "input" in json.loads(body)["error"]["message"]


def test_malformed_json_is_a_client_error(server):
    """A broken body is a 400, not a 500."""
    status, _, body = _post(server, "/v1/audio/speech", {}, raw=b"{not json")

    assert status == 400
    assert json.loads(body)["error"]["message"] == "invalid JSON body"


def test_unknown_route_is_not_found(server):
    """Unknown paths 404 rather than falling through to synthesis."""
    status, _, body = _post(server, "/v1/nope", {"input": "hello"})

    assert status == 404
    assert json.loads(body)["error"]["message"] == "not found"


def test_synthesis_failure_is_a_server_error_not_a_crash(server, monkeypatch):
    """A provider failure must surface as a 500 and leave the server serving."""

    def _boom(text, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(tts_server, "synthesize_to_bytes", _boom)

    status, _, body = _post(server, "/v1/audio/speech", {"input": "hello"})
    assert status == 500
    assert "engine exploded" in json.loads(body)["error"]["message"]

    # The server must still answer afterwards.
    monkeypatch.setattr(
        tts_server, "synthesize_to_bytes",
        lambda text, **kwargs: (b"audio", "audio/mpeg"))
    status, _, _ = _post(server, "/v1/audio/speech", {"input": "hello again"})
    assert status == 200


def test_bind_refuses_a_non_loopback_host_by_default():
    """The server has no authentication, so a LAN bind must be an explicit opt-in."""
    with pytest.raises(ValueError, match="no authentication"):
        tts_server._resolve_bind_address("0.0.0.0", 5099, allow_remote=False)


def test_bind_allows_a_non_loopback_host_with_the_opt_in():
    """``--allow-remote`` is the deliberate escape hatch."""
    assert tts_server._resolve_bind_address(
        "0.0.0.0", 5099, allow_remote=True) == ("0.0.0.0", 5099)


def test_bind_allows_loopback_by_default():
    """The default bind is loopback on both spellings of the address."""
    for host in ("127.0.0.1", "::1", "localhost"):
        assert tts_server._resolve_bind_address(host, 5099, allow_remote=False)[0] == host


def test_tts_without_a_subcommand_reports_usage(capsys):
    """``hermes tts`` with no sub-action must say what to run, not print the top-level usage."""
    from hermes_cli.tts_cli import cmd_tts

    code = cmd_tts(argparse.Namespace(tts_command=None))
    out = capsys.readouterr().out

    assert code == 1
    assert "hermes tts serve" in out
    assert "hermes tts --help" in out


def test_unknown_tts_subcommand_is_rejected(capsys):
    """An unknown sub-action must fail loudly rather than silently doing nothing."""
    from hermes_cli.tts_cli import cmd_tts

    code = cmd_tts(argparse.Namespace(tts_command="bogus"))
    out = capsys.readouterr().out

    assert code == 1
    assert "bogus" in out


def test_synthesis_failure_envelope_becomes_a_client_error(monkeypatch):
    """A tool-level failure envelope maps to a 400, not a 500.

    The TTS tool reports a configuration/voice problem as a failed envelope; that
    is the caller's fault (their provider is not set up), so it must not be
    reported as a server fault.
    """
    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        lambda *a, **k: json.dumps({"success": False, "error": "no voice configured"}))

    with pytest.raises(tts_server.TTSRequestError, match="no voice configured"):
        tts_server.synthesize_to_bytes("hello")


def test_opus_request_produces_real_opus_not_vorbis(tmp_path, monkeypatch):
    """``response_format=opus`` must return Opus-in-Ogg, not ffmpeg's default Vorbis.

    The synthesis path converts a WAV-native engine into whatever the target
    extension implies, and ffmpeg's default codec for ``.ogg`` is Vorbis — while
    OpenAI's "opus" means Opus. Regression: the server answered "opus" requests
    with a Vorbis stream.
    """
    ogg_path = tmp_path / "speech.ogg"
    # A minimal Ogg page whose codec header says Vorbis, as ffmpeg would write.
    ogg_path.write_bytes(b"OggS\x00\x02" + b"\x00" * 24 + b"\x01vorbis" + b"\x00" * 16)

    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        lambda *a, **k: json.dumps({"success": True, "file_path": str(ogg_path)}))

    converted = []
    monkeypatch.setattr(
        "tools.tts_tool_delivery._convert_to_opus",
        lambda path: converted.append(path) or str(path))

    tts_server.synthesize_to_bytes("hello", response_format="opus")

    assert converted == [str(ogg_path)], "a Vorbis stream must be transcoded to Opus"


def test_native_opus_output_is_not_re_encoded(tmp_path, monkeypatch):
    """A provider that already wrote Opus must be left alone (no lossy re-encode)."""
    ogg_path = tmp_path / "speech.ogg"
    ogg_path.write_bytes(b"OggS\x00\x02" + b"\x00" * 24 + b"OpusHead" + b"\x00" * 16)

    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        lambda *a, **k: json.dumps({"success": True, "file_path": str(ogg_path)}))

    converted = []
    monkeypatch.setattr(
        "tools.tts_tool_delivery._convert_to_opus",
        lambda path: converted.append(path) or str(path))

    tts_server.synthesize_to_bytes("hello", response_format="opus")

    assert converted == [], "an already-Opus stream must not be transcoded again"


def test_synthesis_runs_bound_to_the_serving_profile(tmp_path, monkeypatch):
    """The request thread must see the serving profile's home, not the launch one.

    The handler runs in a ``ThreadingHTTPServer`` worker thread, whose context
    starts empty. Without an explicit re-bind, ``get_hermes_home()`` inside the
    synthesis would fall back to the launch profile's home — a silent
    wrong-profile read, the exact leak HX012 guards against.
    """
    from hermes_constants import get_hermes_home

    serving_home = str(tmp_path / "serving-profile")
    seen = {}

    def _capture(text, **kwargs):
        seen["home"] = str(get_hermes_home())
        return b"audio", "audio/mpeg"

    monkeypatch.setattr(tts_server, "synthesize_to_bytes", _capture)

    handler = tts_server._build_handler("piper", "en_US-lessac-medium", serving_home)
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        status, _, _ = _post(
            f"http://{host}:{port}", "/v1/audio/speech", {"input": "hello"})
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert status == 200
    assert seen["home"] == serving_home, (
        "synthesis must run under the serving profile's scope, "
        f"not {seen['home']!r}")


def test_profile_scope_is_released_after_a_request(server, monkeypatch):
    """The scope binding must be undone even when synthesis raises.

    A leaked ContextVar override would make every later request in the same
    process read a home nobody asked for.
    """
    from hermes_constants import get_hermes_home_override

    def _boom(text, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(tts_server, "synthesize_to_bytes", _boom)

    status, _, _ = _post(server, "/v1/audio/speech", {"input": "hello"})
    assert status == 500
    assert get_hermes_home_override() is None, (
        "the profile scope override must not leak past the request")

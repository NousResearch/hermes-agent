"""Regression tests for the /api/audio/{filename} TTS playback route.

Covers the three issues teknium1 raised on PR #16717:
  * the route must not be exempted wholesale from auth — only the file-
    serving GET accepts the ?token= query-auth fallback, and unrelated
    /api/audio/* endpoints (speak, transcribe, elevenlabs/voices) still
    require normal auth.
  * the cache dir must resolve through get_hermes_dir("cache/audio",
    "audio_cache"), matching tools/tts_tool.py, not a hardcoded legacy path.
  * path traversal outside the cache dir is rejected.
"""

from starlette.testclient import TestClient

from hermes_cli import web_server


def _client():
    prev_auth_required = getattr(web_server.app.state, "auth_required", None)
    prev_bound_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None

    client = TestClient(web_server.app)
    try:
        yield client
    finally:
        client.close()
        if prev_auth_required is None:
            delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = prev_auth_required
        if prev_bound_host is None:
            if hasattr(web_server.app.state, "bound_host"):
                delattr(web_server.app.state, "bound_host")
        else:
            web_server.app.state.bound_host = prev_bound_host


def _authed_client():
    gen = _client()
    client = next(gen)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client, gen


def _anon_client():
    gen = _client()
    return next(gen), gen


def test_audio_file_route_resolves_via_get_hermes_dir(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / "cache" / "audio").mkdir(parents=True)
    (home / "cache" / "audio" / "clip.mp3").write_bytes(b"id3-fake-mp3-bytes")
    monkeypatch.delenv("HERMES_AUDIO_CACHE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))

    client, gen = _authed_client()
    try:
        resp = client.get("/api/audio/clip.mp3")
        assert resp.status_code == 200
        assert resp.content == b"id3-fake-mp3-bytes"
        assert resp.headers["content-type"] == "audio/mpeg"
    finally:
        next(gen, None)


def test_audio_file_route_prefers_legacy_dir_if_present(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / "audio_cache").mkdir(parents=True)
    (home / "audio_cache" / "old.mp3").write_bytes(b"legacy-bytes")
    monkeypatch.delenv("HERMES_AUDIO_CACHE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))

    client, gen = _authed_client()
    try:
        resp = client.get("/api/audio/old.mp3")
        assert resp.status_code == 200
        assert resp.content == b"legacy-bytes"
    finally:
        next(gen, None)


def test_audio_file_route_rejects_path_traversal(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / "cache" / "audio").mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me")
    monkeypatch.delenv("HERMES_AUDIO_CACHE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))

    client, gen = _authed_client()
    try:
        # %2f is decoded to a literal "/" by httpx's own URL parsing before
        # the request is even sent, which then fails to match the
        # single-segment {filename} route entirely (a different 404, not the
        # handler's traversal guard). Backslashes aren't a URL meta-character
        # so they survive intact — and pathlib treats them as a separator on
        # Windows, giving the same traversal shape without that client-side
        # normalization getting in the way.
        resp = client.get("/api/audio/..%5C..%5Csecret.txt")
        assert resp.status_code == 403
    finally:
        next(gen, None)


def test_audio_file_route_requires_auth_without_token(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / "cache" / "audio").mkdir(parents=True)
    (home / "cache" / "audio" / "clip.mp3").write_bytes(b"bytes")
    monkeypatch.delenv("HERMES_AUDIO_CACHE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))

    client, gen = _anon_client()
    try:
        resp = client.get("/api/audio/clip.mp3")
        assert resp.status_code == 401
    finally:
        next(gen, None)


def test_audio_file_route_accepts_query_token(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / "cache" / "audio").mkdir(parents=True)
    (home / "cache" / "audio" / "clip.mp3").write_bytes(b"bytes")
    monkeypatch.delenv("HERMES_AUDIO_CACHE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))

    client, gen = _anon_client()
    try:
        resp = client.get(f"/api/audio/clip.mp3?token={web_server._SESSION_TOKEN}")
        assert resp.status_code == 200
        assert resp.content == b"bytes"
    finally:
        next(gen, None)


def test_query_token_does_not_extend_to_other_audio_endpoints():
    # /api/audio/speak, /api/audio/transcribe, /api/audio/elevenlabs/voices
    # are unrelated sensitive endpoints under the same /api/audio prefix —
    # the query-token allowance must stay scoped to the file-serving GET and
    # not widen to a blanket "/api/audio" prefix match.
    assert web_server._has_valid_query_token(
        _FakeRequest(f"/api/audio/speak?token={web_server._SESSION_TOKEN}"),
        "/api/audio/speak",
    ) is False


class _FakeRequest:
    def __init__(self, url_with_query: str):
        path, _, query = url_with_query.partition("?")
        from urllib.parse import parse_qs

        parsed = parse_qs(query)
        self.query_params = {k: v[0] for k, v in parsed.items()}

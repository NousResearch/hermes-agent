"""/api/audio/scribe-token — scoped ElevenLabs realtime credentials."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        test_client.close()
        if previous_auth_required is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous_auth_required


class _Response:
    status_code = 200
    text = '{"token":"sutkn_secret"}'

    def raise_for_status(self):
        return None

    def json(self):
        return {"token": "sutkn_secret"}


def test_mints_realtime_token_without_returning_long_lived_key(client, monkeypatch):
    monkeypatch.setattr(
        "tools.voice_client_config.resolve_client_voice_config",
        lambda: {
            "stt": {
                "mode": "direct",
                "wire": "elevenlabs-stt",
                "provider": "elevenlabs",
                "base_url": "https://api.elevenlabs.io/v1",
                "api_key": "el_long_lived",
                "model": "scribe_v2",
                "language": "en",
            },
            "tts": {"mode": "relay"},
        },
    )
    calls = []

    def fake_post(url, *, headers, timeout):
        calls.append((url, headers, timeout))
        return _Response()

    monkeypatch.setattr("requests.post", fake_post)

    response = client.post("/api/audio/scribe-token")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "ok": True,
        "token": "sutkn_secret",
        "websocket_url": "wss://api.elevenlabs.io",
        "model": "scribe_v2_realtime",
        "language": "en",
    }
    assert "el_long_lived" not in response.text
    assert calls == [
        (
            "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe",
            {"xi-api-key": "el_long_lived"},
            15,
        )
    ]


def test_non_elevenlabs_provider_returns_409(client, monkeypatch):
    monkeypatch.setattr(
        "tools.voice_client_config.resolve_client_voice_config",
        lambda: {"stt": {"mode": "direct", "provider": "groq"}, "tts": {"mode": "relay"}},
    )

    response = client.post("/api/audio/scribe-token")

    assert response.status_code == 409
    assert "ElevenLabs" in response.json()["detail"]


def test_upstream_failure_does_not_leak_api_key(client, monkeypatch):
    monkeypatch.setattr(
        "tools.voice_client_config.resolve_client_voice_config",
        lambda: {
            "stt": {
                "mode": "direct",
                "provider": "elevenlabs",
                "base_url": "https://api.elevenlabs.io/v1",
                "api_key": "el_do_not_leak",
            },
            "tts": {"mode": "relay"},
        },
    )

    def fail(*args, **kwargs):
        raise RuntimeError("upstream failed for el_do_not_leak")

    monkeypatch.setattr("requests.post", fail)

    response = client.post("/api/audio/scribe-token")

    assert response.status_code == 502
    assert "el_do_not_leak" not in response.text

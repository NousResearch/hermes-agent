"""``/api/audio/voice-live/status`` and ``POST /api/audio/grok-voice/session``.

The desktop mode switch stays the status route. When the saved mode is grok-live the
availability it returns is Grok's, and the session route hands back only the minted secret.
"""

from __future__ import annotations

import pytest

from tools.grok_voice import GrokVoiceError


@pytest.fixture
def client(monkeypatch, tmp_path):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_state
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    http = TestClient(app)
    http.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return http


def test_status_reports_grok_availability_when_that_mode_is_saved(client, monkeypatch):
    monkeypatch.setattr("tools.voice_live.resolve_voice_chat_status", lambda: {
        "mode": "grok-live",
        "available": True,
        "reason": None,
        "model": "grok-voice-latest",
        "voice": "eve",
        "auth": "oauth",
        "gpt_live_available": False,
        "gpt_live_reason": "no OpenAI API key (set OPENAI_API_KEY or voice.gpt_live.api_key)",
        "grok_available": True,
        "grok_reason": None,
        "grok_auth": "oauth",
    })

    body = client.get("/api/audio/voice-live/status").json()

    assert body["ok"] is True
    assert body["mode"] == "grok-live"
    assert body["available"] is True
    assert body["auth"] == "oauth"
    assert body["model"] == "grok-voice-latest"
    assert "bearer" not in body
    assert "client_secret" not in body


def test_grok_session_503_when_no_credential(client, monkeypatch):
    def refuse():
        raise GrokVoiceError("no_credential", "Grok Voice needs a Grok sign-in or XAI_API_KEY.", 503)

    monkeypatch.setattr("tools.grok_voice.mint_grok_voice_session", refuse)

    response = client.post("/api/audio/grok-voice/session")

    assert response.status_code == 503
    assert "Bearer" not in response.text


def test_grok_session_502_when_xai_rejects_and_returns_only_the_secret(client, monkeypatch):
    def reject():
        raise GrokVoiceError("rejected", "xAI refused the client secret request (401).", 502)

    monkeypatch.setattr("tools.grok_voice.mint_grok_voice_session", reject)
    refused = client.post("/api/audio/grok-voice/session")
    assert refused.status_code == 502
    assert "401" in refused.text

    monkeypatch.setattr("tools.grok_voice.mint_grok_voice_session", lambda: {
        "url": "wss://api.x.ai/v1/realtime?model=grok-voice-latest",
        "protocol_prefix": "xai-client-secret.",
        "client_secret": {"value": "ephemeral-client-secret", "expires_at": 9},
        "model": "grok-voice-latest",
        "session": {"voice": "eve"},
    })
    minted = client.post("/api/audio/grok-voice/session").json()
    assert minted["ok"] is True
    assert minted["client_secret"]["value"] == "ephemeral-client-secret"
    assert "api_key" not in minted

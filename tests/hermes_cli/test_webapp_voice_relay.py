"""The Webapp relays voice instead of handing its tab provider keys.

``/api/audio/voice-config`` lets native Desktop call STT/TTS providers itself. A browser tab cannot:
a provider without CORS (the Nous tool gateway Hermes Cloud routes speech through) refuses the
tab's fetch, so dictation failed with "Failed to fetch", and the response put the keys in the page.
On the Webapp surface both directions resolve to relay, with no key in the body; native surfaces
keep client-direct.
"""
import importlib
import sys
from pathlib import Path

import pytest
import hermes_yaml as yaml
from starlette.testclient import TestClient

_TOKEN = "a" * 43
_KEY = "gsk_voice_relay_test"


@pytest.fixture
def voice_config(tmp_path, monkeypatch):
    """GET /api/audio/voice-config on ``web_server.app`` serving *surface*, for a profile whose STT
    resolves client-direct (Groq, with its key)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"stt": {"provider": "groq"}}), encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", _KEY)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("hermes_cli.config", "tools.transcription_tools", "tools.tts_tool",
                 "tools.voice_client_config"):
        if name in sys.modules:
            importlib.reload(sys.modules[name])

    from hermes_cli import web_server

    monkeypatch.setattr(web_server.app.state, "bound_host", "localhost", raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", _TOKEN)
    client = TestClient(web_server.app, base_url="http://localhost")

    def get(surface):
        monkeypatch.setattr(web_server.app.state, "ui_surface", surface, raising=False)
        response = client.get("/api/audio/voice-config", headers={"X-Hermes-Session-Token": _TOKEN})
        assert response.status_code == 200, response.text
        return response

    return get


def test_the_webapp_relays_voice_and_its_tab_gets_no_key(voice_config):
    response = voice_config("webapp")

    body = response.json()
    assert body["stt"]["mode"] == "relay"
    assert body["tts"]["mode"] == "relay"
    assert _KEY not in response.text


def test_native_desktop_still_calls_the_provider_directly(voice_config):
    stt = voice_config("dashboard").json()["stt"]

    assert stt["mode"] == "direct"
    assert stt["api_key"] == _KEY

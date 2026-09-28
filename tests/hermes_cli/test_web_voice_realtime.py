"""Live voice session minting (OpenAI Realtime and Gemini Live) (``/api/voice/realtime/*``): the OpenAI key never leaves the
backend, and the minted session carries the configured model plus the ``ask_jarvis`` bridge."""
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_cli.web_routers import voice_realtime


def test_new_install_defaults_to_gemini_live():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    voice = DEFAULT_CONFIG["voice"]
    assert voice["engine"] == "realtime"
    assert voice["realtime"]["provider"] == "gemini"
    assert voice["realtime"]["gemini"]["model"] == "gemini-3.8-live"
    assert voice["realtime"]["gemini"]["voice"] == voice_realtime.DEFAULT_GEMINI_VOICE == "Puck"


def test_realtime_voice_is_czesiek_and_keeps_main_session_bridge():
    instructions = voice_realtime.realtime_instructions("pl")

    assert "You are Czesiek" in instructions
    assert "Speak Polish" in instructions
    assert "background agents" in instructions
    assert "ask_jarvis" in instructions
    assert "delegate_to_hermes" in instructions
    assert voice_realtime.ASK_JARVIS_TOOL["name"] == "ask_jarvis"
    assert voice_realtime.DELEGATE_TO_HERMES_TOOL["name"] == "delegate_to_hermes"
    assert "Czesiek's main Hermes session" in voice_realtime.ASK_JARVIS_TOOL["description"]


@pytest.fixture
def client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_session_returns_only_the_ephemeral_secret(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-never-leaves")
    monkeypatch.setattr(voice_realtime, "load_config", lambda: {
        "voice": {"realtime": {"model": "gpt-realtime-2.1-mini", "voice": "cedar"}}})
    sent = {}

    async def fake_mint(base_url, api_key, session):
        sent.update(base_url=base_url, api_key=api_key, session=session)
        return {"value": "ek_short_lived", "expires_at": 1_900_000_000}

    monkeypatch.setattr(voice_realtime, "_mint_client_secret", fake_mint)

    resp = client.post("/api/voice/realtime/session")
    assert resp.status_code == 200
    body = resp.json()
    assert body["client_secret"] == "ek_short_lived"
    assert "sk-live-never-leaves" not in resp.text
    # The configured model and voice are what the minted session is built with.
    assert sent["api_key"] == "sk-live-never-leaves"
    assert sent["session"]["model"] == body["model"] == "gpt-realtime-2.1-mini"
    assert sent["session"]["audio"]["output"]["voice"] == body["voice"] == "cedar"
    assert [t["name"] for t in sent["session"]["tools"]] == ["ask_jarvis", "delegate_to_hermes"]
    assert body["calls_url"].startswith(sent["base_url"])


def test_session_without_a_key_is_a_clear_400(client, monkeypatch):
    monkeypatch.setattr(voice_realtime, "load_config", lambda: {"voice": {"realtime": {"provider": "openai"}}})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.setattr(voice_realtime, "_resolve_key", lambda _provider: "")

    async def never(*_a, **_k):
        raise AssertionError("must not call OpenAI without a key")

    monkeypatch.setattr(voice_realtime, "_mint_client_secret", never)

    resp = client.post("/api/voice/realtime/session")
    assert resp.status_code == 400
    assert "OPENAI_API_KEY" in resp.json()["detail"]
    assert client.get("/api/voice/realtime/status").json()["available"] is False


def test_settings_fall_back_to_defaults_for_malformed_config():
    settings = voice_realtime.realtime_settings({"voice": {"realtime": {"model": "  ", "voice": 7}}})
    assert settings["model"] == voice_realtime.DEFAULT_REALTIME_MODEL
    assert settings["voice"] == voice_realtime.DEFAULT_REALTIME_VOICE
    assert voice_realtime.realtime_settings(None)["base_url"] == voice_realtime.DEFAULT_REALTIME_BASE_URL


@pytest.fixture
def fake_google():
    """A local stand-in for generativelanguage.googleapis.com's token endpoint."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — http.server API
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            received.append({"path": self.path, "key": self.headers.get("x-goog-api-key"), "body": body})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"name": "auth_tokens/one-use-token"}')

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", received
    server.shutdown()


def test_gemini_session_hands_out_a_one_use_token_with_the_setup_locked_in(client, monkeypatch, fake_google):
    base, received = fake_google
    monkeypatch.setattr(voice_realtime, "GEMINI_API_BASE", base)
    real_client = voice_realtime.httpx.AsyncClient
    transport_contexts = []

    def client_with_transport(*args, **kwargs):
        transport_contexts.append(kwargs.get("verify"))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(voice_realtime.httpx, "AsyncClient", client_with_transport)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-never-leaves")
    monkeypatch.setattr(voice_realtime, "load_config", lambda: {
        "voice": {"realtime": {"provider": "gemini", "gemini": {"voice": "Kore"}}}})

    resp = client.post("/api/voice/realtime/session")
    assert isinstance(transport_contexts[-1], ssl.SSLContext)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"] == "gemini"
    assert body["token"] == "auth_tokens/one-use-token"
    assert "AIza-never-leaves" not in resp.text
    assert body["ws_url"].endswith("BidiGenerateContentConstrained")
    [minted] = received
    assert minted["path"] == "/v1alpha/auth_tokens" and minted["key"] == "AIza-never-leaves"
    locked = minted["body"]["bidiGenerateContentSetup"]
    # What the renderer is told to send is exactly what the token locks.
    assert locked == body["setup"]
    assert locked["model"] == f"models/{voice_realtime.DEFAULT_GEMINI_MODEL}" and body["model"] == voice_realtime.DEFAULT_GEMINI_MODEL
    assert locked["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"
    assert [f["name"] for f in locked["tools"][0]["functionDeclarations"]] == ["ask_jarvis", "delegate_to_hermes"]
    assert minted["body"]["uses"] == 1
    for field in ("model", "systemInstruction", "tools", "generationConfig"):
        assert field in minted["body"]["fieldMask"].split(",")


def test_gemini_without_a_google_key_says_which_key(client, monkeypatch):
    monkeypatch.setattr(voice_realtime, "_resolve_key", lambda _provider: "")
    monkeypatch.setattr(voice_realtime, "load_config", lambda: {"voice": {"realtime": {"provider": "gemini"}}})

    resp = client.post("/api/voice/realtime/session")

    assert resp.status_code == 400
    assert "GEMINI_API_KEY" in resp.json()["detail"]
    status = client.get("/api/voice/realtime/status").json()
    assert status["provider"] == "gemini" and status["available"] is False


def test_status_can_ask_about_a_provider_before_it_is_chosen(client, monkeypatch):
    monkeypatch.setattr(voice_realtime, "_resolve_key", lambda provider: "k" if provider == "gemini" else "")
    monkeypatch.setattr(voice_realtime, "load_config", lambda: {})

    assert client.get("/api/voice/realtime/status").json()["available"] is False
    gemini = client.get("/api/voice/realtime/status", params={"provider": "gemini"}).json()
    assert gemini["available"] is True and gemini["model"] == voice_realtime.DEFAULT_GEMINI_MODEL


def test_unknown_provider_falls_back_to_openai():
    assert voice_realtime.realtime_settings({"voice": {"realtime": {"provider": "skype"}}})["provider"] == "openai"

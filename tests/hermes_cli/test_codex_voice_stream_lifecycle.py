"""Exercise Desktop speech termination through the real authenticated WebSocket route."""

import json
import shutil
import subprocess
import threading
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from tools import tts_streaming


@pytest.fixture
def speech_session(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("tts:\n  provider: openai-codex\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)

    class Provider(tts_streaming.StreamingTTSProvider):
        instances = []
        mode = "success"

        @staticmethod
        def available():
            return True

        def __init__(self, config, section):
            super().__init__(config, section)
            self.cancelled = threading.Event()
            self.blocked = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()
            self.requests = []
            self.instances.append(self)

        def cancel(self):
            self.cancelled.set()

        def stream(self, text):
            self.requests.append(text)
            try:
                yield b"\x01\x00\x02\x00"
                if self.mode == "blocked":
                    self.blocked.set()
                    # Deliberately uncooperative network read: router teardown
                    # must finish before this external operation returns.
                    self.release.wait(10)
                    yield b"\x03\x00"
                elif self.mode == "error":
                    raise RuntimeError("provider failed after partial audio")
            finally:
                self.finished.set()

    monkeypatch.setitem(tts_streaming._REGISTRY, "openai-codex", Provider)
    returned = threading.Event()

    async def app(scope, receive, send):
        try:
            await web_server.app(scope, receive, send)
        finally:
            if scope["type"] == "websocket":
                returned.set()

    client = TestClient(app)
    url = "/api/audio/speak-stream?" + urlencode({"token": web_server._SESSION_TOKEN})
    try:
        yield client, url, Provider, returned
    finally:
        for provider in Provider.instances:
            provider.release.set()
        client.close()


@pytest.mark.parametrize("terminal", ["stop", "disconnect"])
def test_barge_in_closes_socket_before_blocked_provider_returns(speech_session, terminal):
    client, url, provider_class, returned = speech_session
    provider_class.mode = "blocked"
    with client.websocket_connect(url) as socket:
        assert socket.receive_json()["type"] == "start"
        socket.send_json({"text": "This sentence is ready to speak. Another sentence stays queued. "})
        assert socket.receive_bytes() == b"\x01\x00\x02\x00"
        provider = provider_class.instances[0]
        assert provider.blocked.wait(2)
        if terminal == "stop":
            socket.send_json({"stop": True})
        else:
            socket.close()
        try:
            assert returned.wait(2), "socket handler waited for blocked provider"
            assert provider.cancelled.is_set()
            assert not provider.finished.is_set()
        finally:
            provider.release.set()
        assert provider.finished.wait(2)
    assert provider.requests == ["This sentence is ready to speak."]


@pytest.mark.parametrize("outcome", ["success", "error"])
def test_audio_starts_before_done_and_failure_never_reports_success(speech_session, outcome):
    client, url, provider_class, returned = speech_session
    provider_class.mode = outcome
    with client.websocket_connect(url) as socket:
        assert socket.receive_json() == {"type": "start", "sample_rate": 24000, "channels": 1}
        socket.send_json({"text": "This complete sentence can speak immediately. "})
        assert socket.receive_bytes() == b"\x01\x00\x02\x00"
        if outcome == "success":
            socket.send_json({"done": True})
        terminal = socket.receive_json()
        assert terminal["type"] == ("end" if outcome == "success" else "error")
        assert socket.receive()["type"] == "websocket.close"
        assert returned.wait(2)
    assert provider_class.instances[0].requests == ["This complete sentence can speak immediately."]


def test_selected_codex_provider_decodes_mp3_to_live_pcm(tmp_path, monkeypatch):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for Codex live voice")
    mp3 = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=0.1", "-f", "mp3", "pipe:1"],
        check=True, capture_output=True, timeout=10,
    ).stdout
    (tmp_path / "config.yaml").write_text(
        "tts:\n  provider: openai-codex\n  openai_codex:\n    voice: ember\n    timeout: 30\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    from hermes_constants import get_hermes_home
    from tools import tts_tool_codex

    def credentials():
        assert get_hermes_home() == tmp_path
        return object(), {"api_key": "profile-subscription-token"}

    requests = []

    def synthesize(text, api_key, **kwargs):
        requests.append((text, api_key, kwargs))
        return SimpleNamespace(audio=mp3)

    monkeypatch.setattr(tts_tool_codex, "_has_codex_tts_backend", lambda: True)
    monkeypatch.setattr(tts_tool_codex, "_codex_tts_credentials", credentials)
    monkeypatch.setattr(tts_tool_codex, "synthesize_codex_speech", synthesize)
    url = "/api/audio/speak-stream?" + urlencode({"token": web_server._SESSION_TOKEN})
    client = TestClient(web_server.app)
    with client.websocket_connect(url) as socket:
        assert socket.receive_json() == {"type": "start", "sample_rate": 24000, "channels": 1}
        socket.send_json({"text": "This sentence must speak before the agent finishes. "})
        pcm = socket.receive_bytes()
        assert pcm and len(pcm) % 2 == 0
        assert any(pcm), "decoded sine wave must contain audible samples"
        socket.send_json({"done": True})
        while True:
            frame = socket.receive()
            if "bytes" in frame:
                assert len(frame["bytes"]) % 2 == 0
                continue
            assert json.loads(frame["text"]) == {"type": "end"}
            break
    client.close()
    assert len(requests) == 1
    text, key, options = requests[0]
    assert text == "This sentence must speak before the agent finishes."
    assert key == "profile-subscription-token"
    assert options["voice"] == "ember"
    assert 0 < options["timeout"] <= 30

"""A local Desktop's voice sockets ride the grant of its ``/api/ws`` dial (RT-71, RT-80).

Desktop dials ``/api/audio/speak-stream`` and ``/api/audio/transcribe-stream`` beside its
``/api/ws`` socket with the same one-use ``interactive`` grant in the gateway subprotocol. Those
upgrades were refused on a local gateway (it holds no static token), so spoken replies and live
dictation never streamed. The grant's home, not ``?profile=``, decides whose voice config runs.
"""
from __future__ import annotations

import pytest
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server
from tests.hermes_cli import test_plugin_api_consent_gate as consent_gate

daemon = consent_gate.daemon  # the multiplexed daemon stand-in fixture

SPEAK, TRANSCRIBE = "/api/audio/speak-stream", "/api/audio/transcribe-stream"


def _dial(home, purpose="interactive"):
    ticket = web_server.app.state.gateway_runner.session_ticket_store.mint(
        profile_id=str(home), subject="desktop", purpose=purpose)
    return ["hermes-gateway-v1", f"hermes-gateway-ticket.{ticket}"]


@pytest.fixture
def scopes(daemon, monkeypatch):
    """Fake live providers recording (stage, home, profile secret) at each provider stage."""
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home
    from tools import transcription_streaming, tts_streaming
    (daemon.beta / ".env").write_text("VOICE_KEY=beta\n")
    seen = []

    def record(stage):
        seen.append((stage, get_hermes_home(), get_secret("VOICE_KEY")))

    class Session:
        set_input_rate = push_audio = cancel = lambda self, *_: None

        def finalize(self):
            record("stt.finalize")
            return {"success": True, "transcript": "hi", "provider": "fake"}

    class Streamer:
        sample_rate, channels = 24000, 1

        def stream(self, text):
            record("tts.synthesize")
            yield b"\0\0"

    def open_streaming_session(**_):
        record("stt.open")
        return Session()

    def resolve_streaming_provider(*_):
        record("tts.resolve")
        return Streamer()

    monkeypatch.setattr(transcription_streaming, "open_streaming_session", open_streaming_session)
    monkeypatch.setattr(tts_streaming, "resolve_streaming_provider", resolve_streaming_provider)
    return seen


def test_spoken_reply_runs_in_the_grant_profile(daemon, scopes):
    with daemon.client.websocket_connect(SPEAK, subprotocols=_dial(daemon.beta)) as ws:
        assert ws.accepted_subprotocol == "hermes-gateway-v1"  # the browser fails the handshake otherwise
        ws.send_json({"text": "Hello there.", "done": True})
        assert ws.receive_json()["type"] == "start"
        assert ws.receive_bytes() == b"\0\0"
        assert ws.receive_json() == {"type": "end"}
    # the grant's profile, with no ?profile= selector
    assert scopes == [(stage, daemon.beta, "beta") for stage in ("tts.resolve", "tts.synthesize")]


def test_dictation_runs_in_the_grant_profile(daemon, scopes):
    with daemon.client.websocket_connect(TRANSCRIBE, subprotocols=_dial(daemon.beta)) as ws:
        assert ws.accepted_subprotocol == "hermes-gateway-v1"
        ws.send_json({"sample_rate": 16000})
        ws.send_bytes(b"\0\0")
        ws.send_json({"eos": True})
        assert ws.receive_json() == {"type": "final", "transcript": "hi", "provider": "fake"}
    assert scopes == [(stage, daemon.beta, "beta") for stage in ("stt.open", "stt.finalize")]


@pytest.mark.parametrize("case", ["no_credential", "other_profile", "browser_origin", "wrong_purpose"])
def test_other_upgrades_are_still_refused(daemon, scopes, case):
    (daemon.launch / "profiles" / "gamma").mkdir()
    path = TRANSCRIBE
    kwargs: dict = {"subprotocols": _dial(daemon.beta)}
    if case == "no_credential":
        kwargs = {}
    elif case == "other_profile":
        path += "?profile=gamma"
    elif case == "browser_origin":
        kwargs["headers"] = {"origin": "http://evil.example"}
    else:
        kwargs["subprotocols"] = _dial(daemon.beta, purpose="native-http")
    with pytest.raises(WebSocketDisconnect) as refused, daemon.client.websocket_connect(path, **kwargs):
        pass
    assert refused.value.code == 4401
    assert scopes == []

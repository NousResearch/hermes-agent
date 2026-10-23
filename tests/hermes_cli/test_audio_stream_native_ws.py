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
def stt_homes(monkeypatch):
    from hermes_constants import get_hermes_home
    from tools import transcription_streaming
    homes = []

    def open_streaming_session(**_):  # no live STT: the socket answers an error frame and closes
        homes.append(get_hermes_home())

    monkeypatch.setattr(transcription_streaming, "open_streaming_session", open_streaming_session)
    return homes


@pytest.mark.parametrize("path", [SPEAK, TRANSCRIBE])
def test_desktop_grant_opens_the_voice_sockets(daemon, stt_homes, path):
    with daemon.client.websocket_connect(path, subprotocols=_dial(daemon.beta)) as ws:
        assert ws.accepted_subprotocol == "hermes-gateway-v1"  # the browser fails the handshake otherwise
        if path == TRANSCRIBE:
            assert ws.receive_json()["type"] == "error"
    if path == TRANSCRIBE:
        assert stt_homes == [daemon.beta]  # the grant's profile, with no ?profile= selector


@pytest.mark.parametrize("case", ["no_credential", "other_profile", "browser_origin", "wrong_purpose"])
def test_other_upgrades_are_still_refused(daemon, stt_homes, case):
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
    assert stt_homes == []

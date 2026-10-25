"""Each Webapp socket ticket opens only the socket it was minted for.

The browser never puts its session token in a socket URL. An authenticated request mints a one-use,
short-lived ticket instead, from one route, ``POST /api/webapp/socket-ticket?socket=``: ``gateway`` for
the chat socket (``/api/ws``), ``host-terminal`` for the host terminal (``/api/host-terminal``),
``speak-stream`` and ``transcribe-stream`` for the voice streams (``/api/audio/speak-stream``,
``/api/audio/transcribe-stream``). A host shell opens only with a terminal ticket, a chat ticket opens
no host shell, a terminal ticket opens no chat socket, a voice stream opens with a ticket minted for
that stream and no other, and a ticket minted with a session token that has since been replaced opens
nothing.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.hermes_cli.test_web_host_terminal_persistence import IdleBridge, metadata

_TOKEN = "a" * 43
_SESSION = {"X-Hermes-Session-Token": _TOKEN}


def serve_webapp(tmp_path, monkeypatch):
    """The loopback Webapp: ``web_server.app`` mints tickets, a small app serves the host terminal,
    and ``shells`` lists the idle stand-ins it spawns for host shells."""
    from hermes_cli import web_host_terminal, web_server, web_server_chat
    from hermes_cli.web_host_terminal_sessions import host_terminal_lifespan
    from hermes_cli.web_routers.webapp import router

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", "localhost", raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", _TOKEN)
    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(web_host_terminal, "shell_spec", lambda: (["/bin/sh", "-i"], "sh"))
    shells = []

    def spawn(*_args, **_kwargs):
        shells.append(IdleBridge())
        return shells[-1]

    monkeypatch.setattr(web_server_chat.PtyBridge, "spawn", spawn)
    host = FastAPI(lifespan=host_terminal_lifespan)
    host.include_router(router)
    return SimpleNamespace(server=web_server, client=TestClient(web_server.app, base_url="http://localhost"),
                           host=host, shells=shells)


@pytest.fixture
def webapp(tmp_path, monkeypatch):
    return serve_webapp(tmp_path, monkeypatch)


def _mint(client, socket, headers=_SESSION):
    response = client.post("/api/webapp/socket-ticket", params={"socket": socket}, headers=headers)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert set(body) == {"ticket", "ttl_seconds"} and 0 < body["ttl_seconds"] <= 60
    return body["ticket"]


def _refused(client, url, code=4401, **kwargs):
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect(url, **kwargs) as ws:
            ws.receive_text()
    assert closed.value.code == code


def test_a_terminal_ticket_opens_one_host_shell_once(webapp):
    client, host = webapp.client, webapp.host
    assert client.post("/api/webapp/socket-ticket", params={"socket": "host-terminal"}).status_code == 401
    ticket = _mint(client, "host-terminal")
    with TestClient(host) as shells:
        with shells.websocket_connect(f"ws://localhost/api/host-terminal?ticket={ticket}") as ws:
            assert metadata(ws)["terminalId"]
        _refused(shells, f"ws://localhost/api/host-terminal?ticket={ticket}")
    assert len(webapp.shells) == 1


def test_the_host_terminal_refuses_the_session_token_and_every_other_ticket(webapp):
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket

    client, host = webapp.client, webapp.host
    others = [_mint(client, "gateway"), mint_ticket(user_id="legacy-token-owner", provider="session-token")]
    with TestClient(host) as shells:
        _refused(shells, f"ws://localhost/api/host-terminal?token={_TOKEN}")
        for ticket in others:
            _refused(shells, f"ws://localhost/api/host-terminal?ticket={ticket}")
        protocols = ["hermes-gateway-v1", f"hermes-gateway-ticket.{_mint(client, 'gateway')}"]
        _refused(shells, "ws://localhost/api/host-terminal", subprotocols=protocols)
    assert webapp.shells == []


def test_a_terminal_ticket_does_not_open_the_chat_socket(webapp):
    client = webapp.client
    protocols = ["hermes-gateway-v1", f"hermes-gateway-ticket.{_mint(client, 'host-terminal')}"]
    _refused(client, "ws://localhost/api/ws?native_dial=browser", subprotocols=protocols)
    _refused(client, f"ws://localhost/api/ws?ticket={_mint(client, 'host-terminal')}")


def test_tickets_minted_before_the_session_token_changed_open_nothing(webapp, monkeypatch):
    server, client, host = webapp.server, webapp.client, webapp.host
    terminal, chat = _mint(client, "host-terminal"), _mint(client, "gateway")
    monkeypatch.setattr(server, "_SESSION_TOKEN", "b" * 43)
    with TestClient(host) as shells:
        _refused(shells, f"ws://localhost/api/host-terminal?ticket={terminal}")
    _refused(client, f"ws://localhost/api/ws?ticket={chat}")
    assert webapp.shells == []


@pytest.mark.parametrize("surface", ["dashboard", "serve"])
def test_other_surfaces_do_not_mint_terminal_tickets(webapp, monkeypatch, surface):
    server, client = webapp.server, webapp.client
    monkeypatch.setattr(server.app.state, "ui_surface", surface)
    minted = client.post("/api/webapp/socket-ticket", params={"socket": "host-terminal"}, headers=_SESSION)
    assert minted.status_code == 404


def test_a_signed_in_browser_opens_its_host_shell_with_a_terminal_ticket(webapp, monkeypatch):
    """Wildcard host: the dashboard's cookie sign-in mints the ticket, and the shell is that login's."""
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider
    from tests.hermes_cli.test_dashboard_auth_ws_auth import _logged_in

    server, host = webapp.server, webapp.host
    _reset_for_tests()
    clear_providers()
    register_provider(StubAuthProvider())
    monkeypatch.setattr(server.app.state, "bound_host", "fly-app.fly.dev")
    monkeypatch.setattr(server.app.state, "bound_port", 443, raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", True)
    try:
        anonymous = TestClient(server.app, base_url="https://fly-app.fly.dev")
        assert anonymous.post("/api/webapp/socket-ticket", params={"socket": "host-terminal"}, headers=_SESSION,
                              follow_redirects=False).status_code in (302, 401)
        gated = TestClient(server.app, base_url="https://fly-app.fly.dev")
        _logged_in(gated)
        ticket = _mint(gated, "host-terminal", headers={})
        generic = gated.post("/api/auth/ws-ticket").json()["ticket"]
        with TestClient(host, base_url="https://fly-app.fly.dev") as shells:
            url = "wss://fly-app.fly.dev/api/host-terminal?ticket="
            with shells.websocket_connect(url + ticket) as ws:
                shell = metadata(ws)["terminalId"]
                assert host.state.host_terminals.owners[shell].identity[0] == "stub"
            _refused(shells, url + generic)
        assert len(webapp.shells) == 1
    finally:
        clear_providers()
        _reset_for_tests()


_SPEAK = "/api/audio/speak-stream"
_TRANSCRIBE = "/api/audio/transcribe-stream"


@pytest.fixture
def voice(monkeypatch):
    """Voice streams that answer once they admit a socket: speech yields one PCM frame, and live
    dictation reports that no live STT is configured."""
    class Streamer:
        sample_rate = 24000
        channels = 1

        def stream(self, _text):
            yield b"\x01\x02"

    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: Streamer())
    monkeypatch.setattr("tools.tts_tool._load_tts_config", dict)
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda cfg: "fake")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda provider, cfg: 4000)
    monkeypatch.setattr("tools.transcription_streaming.open_streaming_session", lambda **_kwargs: None)


def _speaks(client, url):
    with client.websocket_connect(url) as ws:
        ws.send_text(json.dumps({"text": "Hello there.", "done": True}))
        assert ws.receive_json() == {"type": "start", "sample_rate": 24000, "channels": 1}


def _transcribes(client, url):
    with client.websocket_connect(url) as ws:
        # Admitted: the stream answers, here that no live STT is configured.
        assert ws.receive_json()["type"] == "error"


def test_each_voice_stream_opens_once_with_a_ticket_minted_for_it(webapp, voice):
    client = webapp.client
    assert client.post("/api/webapp/socket-ticket", params={"socket": "speak-stream"}).status_code == 401
    speak = _mint(client, "speak-stream")
    _speaks(client, f"ws://localhost{_SPEAK}?ticket={speak}")
    _refused(client, f"ws://localhost{_SPEAK}?ticket={speak}")
    transcribe = _mint(client, "transcribe-stream")
    _transcribes(client, f"ws://localhost{_TRANSCRIBE}?ticket={transcribe}")
    _refused(client, f"ws://localhost{_TRANSCRIBE}?ticket={transcribe}")


def test_a_voice_stream_refuses_a_ticket_minted_for_another_socket(webapp, voice):
    """The chat socket's ticket, the host terminal's and the other voice stream's open neither voice
    stream; a voice stream's ticket opens no chat socket and no host shell."""
    client, host = webapp.client, webapp.host
    for url, other in ((_SPEAK, "transcribe-stream"), (_TRANSCRIBE, "speak-stream")):
        for ticket in (_mint(client, "gateway"), _mint(client, "host-terminal"), _mint(client, other)):
            _refused(client, f"ws://localhost{url}?ticket={ticket}")
    _refused(client, f"ws://localhost/api/ws?ticket={_mint(client, 'speak-stream')}")
    protocols = ["hermes-gateway-v1", f"hermes-gateway-ticket.{_mint(client, 'transcribe-stream')}"]
    _refused(client, "ws://localhost/api/ws?native_dial=browser", subprotocols=protocols)
    with TestClient(host) as shells:
        _refused(shells, f"ws://localhost/api/host-terminal?ticket={_mint(client, 'speak-stream')}")
    assert webapp.shells == []


def test_voice_tickets_minted_before_the_session_token_changed_open_nothing(webapp, voice, monkeypatch):
    server, client = webapp.server, webapp.client
    speak, transcribe = _mint(client, "speak-stream"), _mint(client, "transcribe-stream")
    monkeypatch.setattr(server, "_SESSION_TOKEN", "b" * 43)
    _refused(client, f"ws://localhost{_SPEAK}?ticket={speak}")
    _refused(client, f"ws://localhost{_TRANSCRIBE}?ticket={transcribe}")


def test_socket_tickets_are_minted_only_for_the_webapps_sockets_on_the_webapp(webapp, monkeypatch):
    """One route mints every socket ticket; it knows only the Webapp's four sockets, and the
    standalone dashboard and ``hermes serve`` mint none."""
    server, client = webapp.server, webapp.client
    for socket in ("display", "events", "speak", ""):
        response = client.post("/api/webapp/socket-ticket", params={"socket": socket}, headers=_SESSION)
        assert response.status_code == 400, socket
    for surface in ("dashboard", "serve"):
        monkeypatch.setattr(server.app.state, "ui_surface", surface)
        for socket in ("gateway", "host-terminal", "speak-stream", "transcribe-stream"):
            response = client.post("/api/webapp/socket-ticket", params={"socket": socket}, headers=_SESSION)
            assert response.status_code == 404, (surface, socket)


def test_a_native_clients_unnamed_ticket_still_opens_a_voice_stream(webapp, voice):
    """A ticket named for no socket, as ``/api/auth/ws-ticket`` mints for native Desktop, opens a
    voice stream as it did before the streams had names."""
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket

    client = webapp.client
    _speaks(client, f"ws://localhost{_SPEAK}?ticket={mint_ticket(user_id='native', provider='stub')}")
    _transcribes(client, f"ws://localhost{_TRANSCRIBE}?ticket={mint_ticket(user_id='native', provider='stub')}")


def test_a_signed_in_browser_opens_its_voice_streams_with_their_own_tickets(webapp, voice, monkeypatch):
    """Wildcard host: the sign-in cookie mints each voice stream's ticket; the chat socket's does not
    open one."""
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider
    from tests.hermes_cli.test_dashboard_auth_ws_auth import _logged_in

    server = webapp.server
    _reset_for_tests()
    clear_providers()
    register_provider(StubAuthProvider())
    monkeypatch.setattr(server.app.state, "bound_host", "fly-app.fly.dev")
    monkeypatch.setattr(server.app.state, "bound_port", 443, raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", True)
    try:
        anonymous = TestClient(server.app, base_url="https://fly-app.fly.dev")
        assert anonymous.post("/api/webapp/socket-ticket", params={"socket": "speak-stream"},
                              follow_redirects=False).status_code in (302, 401)
        gated = TestClient(server.app, base_url="https://fly-app.fly.dev")
        _logged_in(gated)
        _speaks(gated, f"wss://fly-app.fly.dev{_SPEAK}?ticket={_mint(gated, 'speak-stream', headers={})}")
        _transcribes(
            gated, f"wss://fly-app.fly.dev{_TRANSCRIBE}?ticket={_mint(gated, 'transcribe-stream', headers={})}")
        _refused(gated, f"wss://fly-app.fly.dev{_SPEAK}?ticket={_mint(gated, 'gateway', headers={})}")
    finally:
        clear_providers()
        _reset_for_tests()

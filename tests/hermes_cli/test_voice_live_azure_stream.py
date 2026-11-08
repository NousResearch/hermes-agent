"""The authenticated server-side relay ``/api/audio/voice-live-stream`` (Azure realtime gpt-live).

The browser (or desktop) dials this WebSocket and sends ``{"type": "session.start", "history":
[...]}``; the backend opens the Azure realtime WebSocket itself (preview URL + ``api-key``
header), runs the ``session.created`` → ``session.update`` → ``session.updated`` handshake, then
emits a synthetic ``session.started`` and pumps vendor events. The Azure URL and resource key
never reach the client. Model-driven delegation rides a ``delegate_to_hermes`` function tool:
``response.function_call_arguments.done`` is forwarded (its ``call_id`` drives the desktop's
``onDelegation``/``prompt.submit``), browser ``session.commentary.append`` frames accumulate
server-side, and ``session.commentary.complete`` yields ONE ``conversation.item.create``
function_call_output + ``response.create`` (``session.thinking.append`` is the fallback text).
``session.close`` closes everything.

Driven through the real app with a scripted fake vendor socket (``voice_live._azure_ws_connect``
patched) — no network. Auth, profile scoping and secret hygiene are pinned here; the pure
translation is pinned in ``tests/tools/test_voice_live_relay.py``.
"""

import json
import queue
from pathlib import Path

import pytest
import hermes_yaml as yaml
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.waiting import wait_until
from tools import voice_live

# The multiplexed daemon stand-in (launch home + profile beta, real TicketStore), shared with
# the other native-grant socket suites.
from tests.hermes_cli import test_plugin_api_consent_gate as consent_gate

daemon = consent_gate.daemon

_TOKEN = "c" * 43
AZURE_KEY = "azure-stream-test-key-0777"
AZURE_BASE = "https://stream-test.openai.azure.com"
DEPLOYMENT = "gpt-4o-realtime-preview"
WS_URL = (
    f"wss://stream-test.openai.azure.com/openai/realtime"
    f"?api-version={voice_live.AZURE_REALTIME_API_VERSION}&deployment={DEPLOYMENT}"
)
ROUTE = "/api/audio/voice-live-stream"


class FakeVendorWS:
    """Scripted vendor socket; ``recv`` blocks on a queue (no sleeps, no network)."""

    def __init__(self, events=()):
        self._q: queue.Queue = queue.Queue()
        for event in events:
            self._q.put(event)
        self.sent = []
        self.closed = False

    def push(self, event):
        self._q.put(event)

    def send(self, message):
        if self.closed:
            raise ConnectionError("send on closed socket")
        self.sent.append(json.loads(message))

    def recv(self, timeout=None):
        try:
            event = self._q.get(timeout=timeout if timeout else 0.2)
        except queue.Empty:
            raise TimeoutError("poll")
        if isinstance(event, Exception):
            raise event
        return event if isinstance(event, str) else json.dumps(event)

    def close(self):
        self.closed = True

    def sent_types(self):
        return [m.get("type") for m in self.sent]


def _voice_config(home, *, base_url=AZURE_BASE, api_key=AZURE_KEY):
    (home / "config.yaml").write_text(yaml.safe_dump({"voice": {
        "voice_chat_mode": "gpt-live",
        "gpt_live": {"base_url": base_url, "model": DEPLOYMENT, "voice": "echo", "api_key": api_key},
    }}), encoding="utf-8")


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """The dashboard app in loopback token mode serving a temp home with an Azure voice config."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _voice_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import web_server
    monkeypatch.setattr(web_server.app.state, "bound_host", "localhost", raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", _TOKEN)
    return TestClient(web_server.app, base_url="http://localhost"), home


class _VendorFactory:
    """Patch ``_azure_ws_connect``; each ``install()`` returns a fresh scripted socket."""

    def __init__(self, monkeypatch):
        self.created = {}
        self._monkeypatch = monkeypatch

    def install(self, events=()):
        ws = FakeVendorWS(events)
        created = self.created

        def fake_connect(url, api_key, timeout):
            created.update(url=url, api_key=api_key, timeout=timeout, ws=ws)
            return ws

        self._monkeypatch.setattr(voice_live, "_azure_ws_connect", fake_connect)
        return ws


@pytest.fixture
def vendor(monkeypatch):
    return _VendorFactory(monkeypatch)


def _dial(client, query=""):
    return client.websocket_connect(f"{ROUTE}?token={_TOKEN}{query}", headers={"host": "localhost"})


def _handshake(ws):
    ws.send_json({"type": "session.start", "history": [{"type": "message", "role": "user", "content": "hi"}]})
    return ws.receive_json()


class TestRouteAuth:
    def test_no_credential_is_refused_4401(self, app_client, vendor):
        client, _home = app_client
        vendor.install([{"type": "session.created", "session": {"id": "s"}}, {"type": "session.updated"}])
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(ROUTE, headers={"host": "localhost"}):
                pass
        assert refused.value.code == 4401
        assert vendor.created == {}

    def test_wrong_token_is_refused_4401(self, app_client, vendor):
        client, _home = app_client
        vendor.install([{"type": "session.created", "session": {"id": "s"}}, {"type": "session.updated"}])
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(f"{ROUTE}?token={'x' * 43}", headers={"host": "localhost"}):
                pass
        assert refused.value.code == 4401
        assert vendor.created == {}

    def test_foreign_origin_is_refused_4403(self, app_client, vendor):
        client, _home = app_client
        vendor.install([{"type": "session.created", "session": {"id": "s"}}, {"type": "session.updated"}])
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(
                    f"{ROUTE}?token={_TOKEN}",
                    headers={"host": "localhost", "origin": "http://evil.example"}):
                pass
        assert refused.value.code == 4403
        assert vendor.created == {}


class TestOpenHandshakeOverTheRoute:
    def test_webapp_mints_a_ticket_for_the_azure_voice_socket(self, app_client):
        client, _home = app_client
        client.app.state.ui_surface = "webapp"
        response = client.post(
            "/api/webapp/socket-ticket?socket=voice-live-stream",
            headers={"X-Hermes-Session-Token": _TOKEN},
        )
        assert response.status_code == 200
        assert response.json()["ticket"]

    def test_session_start_opens_azure_and_emits_synthetic_started(self, app_client, vendor):
        client, _home = app_client
        ws = vendor.install([
            {"type": "session.created", "session": {"id": "sess_rt"}},
            {"type": "session.updated", "session": {"id": "sess_rt"}},
        ])
        with _dial(client) as browser:
            started = _handshake(browser)
            assert started == {"type": "session.started", "session": {"id": "sess_rt"}}
            blob = json.dumps(started)
            assert AZURE_KEY not in blob and WS_URL not in blob

            wait_until(lambda: ws.sent_types() == ["session.update", "conversation.item.create"], what="session configured and history seeded")
            session = ws.sent[0]["session"]
            assert session["tools"][0]["name"] == "delegate_to_hermes"
            assert session["input_audio_transcription"] == {"model": "whisper-1"}
            assert "input" not in session
            assert ws.sent[1] == {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": "hi"}}
            # the vendor socket was opened with the api-key header arg; the URL carries no key
            assert vendor.created["api_key"] == AZURE_KEY
            assert vendor.created["url"] == WS_URL
            assert AZURE_KEY not in vendor.created["url"]
            assert ws.closed is False  # the relay keeps it open for the pump

            browser.send_json({"type": "session.close"})
            closed = browser.receive_json()
            assert closed["type"] == "session.closed"
        wait_until(lambda: ws.closed, what="vendor socket closed on session.close")

    def test_first_frame_must_be_session_start(self, app_client, vendor):
        client, _home = app_client
        vendor.install([{"type": "session.created", "session": {"id": "s"}}, {"type": "session.updated"}])
        with _dial(client) as browser:
            browser.send_json({"type": "input_audio_buffer.append", "audio": "AAAA"})
            reply = browser.receive_json()
            assert reply["type"] == "error"
            assert "session.start" in reply["message"]
        assert vendor.created == {}  # no vendor connection was opened

    def test_vendor_error_during_open_is_reported_without_secrets(self, app_client, vendor, caplog):
        import logging

        client, _home = app_client
        vendor.install([{"type": "error", "error": {"message": f"invalid api-key {AZURE_KEY}"}}])
        with caplog.at_level(logging.DEBUG):
            with _dial(client) as browser:
                browser.send_json({"type": "session.start"})
                reply = browser.receive_json()
                assert reply["type"] == "error"
                assert "invalid api-key" in reply["message"]
                assert AZURE_KEY not in json.dumps(reply)
        assert AZURE_KEY not in caplog.text

    def test_non_azure_backend_is_refused_without_connecting(self, app_client, vendor):
        client, home = app_client
        _voice_config(home, base_url="https://api.openai.com/v1")
        with _dial(client) as browser:
            browser.send_json({"type": "session.start"})
            reply = browser.receive_json()
            assert reply["type"] == "error"
            assert "Azure" in reply["message"]
        assert vendor.created == {}

    def test_missing_key_is_refused_without_connecting(self, app_client, vendor, monkeypatch):
        client, home = app_client
        _voice_config(home, api_key="")
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with _dial(client) as browser:
            browser.send_json({"type": "session.start"})
            reply = browser.receive_json()
            assert reply["type"] == "error"
            assert "API key" in reply["message"]
        assert vendor.created == {}

    def test_unknown_profile_is_refused_and_never_connects(self, app_client, vendor):
        client, _home = app_client
        vendor.install([{"type": "session.created", "session": {"id": "s"}}, {"type": "session.updated"}])
        with _dial(client, "&profile=no-such-profile") as browser:
            browser.send_json({"type": "session.start"})
            reply = browser.receive_json()
            assert reply["type"] == "error"
        assert vendor.created == {}


class TestDelegationRoundTripOverTheRoute:
    def test_function_call_done_forwarded_and_complete_answers_once(self, app_client, vendor):
        client, _home = app_client
        ws = vendor.install([
            {"type": "session.created", "session": {"id": "sess_d"}},
            {"type": "session.updated"},
        ])
        with _dial(client) as browser:
            assert _handshake(browser)["type"] == "session.started"

            ws.push({"type": "response.function_call_arguments.done", "call_id": "call_1",
                     "name": "delegate_to_hermes", "arguments": '{"request": "weather?"}'})
            done = browser.receive_json()
            assert done["type"] == "response.function_call_arguments.done"
            assert done["call_id"] == "call_1"

            browser.send_json({"type": "session.commentary.append", "delegation_id": "call_1", "content": "It is"})
            browser.send_json({"type": "session.commentary.append", "delegation_id": "call_1", "content": "sunny."})
            browser.send_json({"type": "session.commentary.complete", "delegation_id": "call_1"})
            wait_until(lambda: ws.sent_types()[-2:] == ["conversation.item.create", "response.create"],
                       what="one function_call_output + response.create")
            output_events = [m for m in ws.sent if m.get("item", {}).get("type") == "function_call_output"]
            assert len(output_events) == 1  # ONE output for the whole delegation
            item = output_events[0]["item"]
            assert item == {"type": "function_call_output", "call_id": "call_1", "output": "It is sunny."}

            # vendor chatter keeps flowing to the browser
            ws.push({"type": "response.audio.delta", "delta": "AAAA"})
            assert browser.receive_json() == {"type": "response.audio.delta", "delta": "AAAA"}

            browser.send_json({"type": "session.close"})
            assert browser.receive_json()["type"] == "session.closed"
        assert ws.closed is True

    def test_think_is_fallback_when_no_commentary_and_audio_passes_through(self, app_client, vendor):
        client, _home = app_client
        ws = vendor.install([
            {"type": "session.created", "session": {"id": "s"}},
            {"type": "session.updated"},
        ])
        with _dial(client) as browser:
            assert _handshake(browser)["type"] == "session.started"
            ws.push({"type": "response.function_call_arguments.done", "call_id": "call_2",
                     "name": "delegate_to_hermes", "arguments": "{}"})
            assert browser.receive_json()["call_id"] == "call_2"

            browser.send_json({"type": "input_audio_buffer.append", "audio": "AAAA"})
            browser.send_json({"type": "session.thinking.append", "delegation_id": "call_2", "content": "checking…"})
            browser.send_json({"type": "session.commentary.complete", "delegation_id": "call_2"})
            wait_until(lambda: "response.create" in ws.sent_types(), what="response.create sent")
            item = next(m for m in ws.sent if m.get("item", {}).get("type") == "function_call_output")["item"]
            assert item["output"] == "checking…"  # think fallback, no commentary arrived
            # the audio frame passed straight through to the vendor
            assert {"type": "input_audio_buffer.append", "audio": "AAAA"} in ws.sent

            browser.send_json({"type": "session.close"})
            assert browser.receive_json()["type"] == "session.closed"

    def test_vendor_event_carrying_secrets_is_redacted_before_the_browser(self, app_client, vendor, caplog):
        import logging

        client, _home = app_client
        ws = vendor.install([
            {"type": "session.created", "session": {"id": "s"}},
            {"type": "session.updated"},
        ])
        with caplog.at_level(logging.DEBUG):
            with _dial(client) as browser:
                assert _handshake(browser)["type"] == "session.started"
                ws.push({"type": "error", "error": {"message": f"boom {AZURE_KEY} at {WS_URL}"}})
                reply = browser.receive_json()
                assert reply["type"] == "error"
                blob = json.dumps(reply)
                assert AZURE_KEY not in blob and WS_URL not in blob
                assert "[redacted]" in blob
                browser.send_json({"type": "session.close"})
                assert browser.receive_json()["type"] == "session.closed"
        assert AZURE_KEY not in caplog.text and WS_URL not in caplog.text

    def test_vendor_disconnect_closes_the_browser_session(self, app_client, vendor):
        client, _home = app_client
        ws = vendor.install([
            {"type": "session.created", "session": {"id": "s"}},
            {"type": "session.updated"},
            ConnectionError("vendor dropped"),
        ])
        with _dial(client) as browser:
            assert _handshake(browser)["type"] == "session.started"
            closed = browser.receive_json()
            assert closed["type"] == "session.closed"
        wait_until(lambda: ws.closed, what="vendor socket closed")

    def test_client_disconnect_closes_the_vendor_socket(self, app_client, vendor):
        client, _home = app_client
        ws = vendor.install([
            {"type": "session.created", "session": {"id": "s"}},
            {"type": "session.updated"},
        ])
        with _dial(client) as browser:
            assert _handshake(browser)["type"] == "session.started"
        wait_until(lambda: ws.closed, what="vendor socket closed when browser left")


@pytest.fixture(autouse=True)
def _restore_process_scope_state():
    """``_config_profile_scope``/``launch_secret_scope`` can freeze process state; restore it."""
    from agent import secret_scope
    from tui_gateway import launch_profile_policy

    was_active = secret_scope.is_multiplex_active()
    snapshot = launch_profile_policy._snapshot
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(was_active)
        launch_profile_policy._snapshot = snapshot


class TestProfileScope:
    """The relay resolves voice config + key from the REQUESTING profile, not the launch one."""

    def test_query_profile_selects_whose_azure_config_and_key_open_the_socket(
            self, tmp_path, monkeypatch, vendor):
        from hermes_cli import profiles, web_server

        launch = tmp_path / ".hermes"
        beta = launch / "profiles" / "beta"
        beta.mkdir(parents=True)
        _voice_config(launch, base_url="https://launch.openai.azure.com", api_key="launch-key")
        _voice_config(beta, base_url="https://beta.openai.azure.com", api_key="beta-key")
        (beta / ".env").write_text("VOICE_TOOLS_OPENAI_KEY=beta-env-key\n")
        monkeypatch.setenv("HERMES_HOME", str(launch))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: launch)
        monkeypatch.setattr(profiles, "_get_profiles_root", lambda: launch / "profiles")
        monkeypatch.setattr(web_server.app.state, "bound_host", "localhost", raising=False)
        monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(web_server, "_SESSION_TOKEN", _TOKEN)
        client = TestClient(web_server.app, base_url="http://localhost")

        seen = {}

        def fake_connect(url, api_key, timeout):
            from agent.secret_scope import get_secret
            from hermes_constants import get_hermes_home
            seen.update(url=url, api_key=api_key, home=str(get_hermes_home()),
                        env_key=get_secret("VOICE_TOOLS_OPENAI_KEY"))
            return vendor.install([{"type": "session.created", "session": {"id": "s"}},
                           {"type": "session.updated"}])

        monkeypatch.setattr(voice_live, "_azure_ws_connect", fake_connect)
        with _dial(client, "&profile=beta") as browser:
            assert _handshake(browser)["type"] == "session.started"
            browser.send_json({"type": "session.close"})
            assert browser.receive_json()["type"] == "session.closed"

        assert Path(seen["home"]).resolve() == beta.resolve()
        assert "beta.openai.azure.com" in seen["url"]
        assert seen["api_key"] == "beta-key"  # beta's config key, never the launch profile's
        assert seen["env_key"] == "beta-env-key"
        assert "launch-key" not in json.dumps(seen)

    def test_native_grant_home_owns_the_relay(self, daemon, monkeypatch, vendor):
        """A desktop's ``interactive`` grant for profile beta: the grant's home — not the
        launch profile's env and not ``?profile=`` — resolves the Azure config and key."""
        from agent.secret_scope import get_secret
        from hermes_constants import get_hermes_home
        from hermes_cli import web_server

        _voice_config(daemon.beta, base_url="https://beta.openai.azure.com", api_key="")
        (daemon.beta / ".env").write_text("VOICE_TOOLS_OPENAI_KEY=beta-env-key\n")
        seen = {}

        def fake_connect(url, api_key, timeout):
            seen.update(url=url, api_key=api_key, home=str(get_hermes_home()),
                        env_key=get_secret("VOICE_TOOLS_OPENAI_KEY"))
            return vendor.install([{"type": "session.created", "session": {"id": "s"}},
                                   {"type": "session.updated"}])

        monkeypatch.setattr(voice_live, "_azure_ws_connect", fake_connect)
        ticket = web_server.app.state.gateway_runner.session_ticket_store.mint(
            profile_id=str(daemon.beta), subject="desktop", purpose="interactive")
        with daemon.client.websocket_connect(
                ROUTE, subprotocols=["hermes-gateway-v1", f"hermes-gateway-ticket.{ticket}"]) as browser:
            assert browser.accepted_subprotocol == "hermes-gateway-v1"
            browser.send_json({"type": "session.start"})
            assert browser.receive_json()["type"] == "session.started"
            browser.send_json({"type": "session.close"})
            assert browser.receive_json()["type"] == "session.closed"

        assert Path(seen["home"]).resolve() == daemon.beta.resolve()
        assert "beta.openai.azure.com" in seen["url"]
        assert seen["api_key"] == "beta-env-key"  # the grant profile's .env, via its secret scope
        assert seen["env_key"] == "beta-env-key"

    def test_grant_with_conflicting_profile_query_is_refused_4401(self, daemon, monkeypatch, vendor):
        from hermes_cli import web_server

        _voice_config(daemon.beta, base_url="https://beta.openai.azure.com", api_key="beta-key")
        monkeypatch.setattr(voice_live, "_azure_ws_connect",
                            lambda *a, **k: pytest.fail("must not connect"))
        (daemon.launch / "profiles" / "gamma").mkdir(parents=True)
        ticket = web_server.app.state.gateway_runner.session_ticket_store.mint(
            profile_id=str(daemon.beta), subject="desktop", purpose="interactive")
        with pytest.raises(WebSocketDisconnect) as refused:
            with daemon.client.websocket_connect(
                    f"{ROUTE}?profile=gamma",
                    subprotocols=["hermes-gateway-v1", f"hermes-gateway-ticket.{ticket}"]):
                pass
        assert refused.value.code == 4401

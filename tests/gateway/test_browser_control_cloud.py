import threading
from types import SimpleNamespace

import pytest

from gateway.browser_control_broker import (
    BROWSER_CONTROL_CAPABILITIES,
    ControllerRejected,
    get_browser_control_broker,
)
from hermes_cli import web_server
import hermes_cli.web_server_chat as _web_server_chat
from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests, mint_ticket
from tui_gateway import server
from tui_gateway.ws import WSTransport
from tui_gateway.methods_browser_control import _broker_event_writer, _principal_digest


def _fake_ticket_ws(ticket):
    return SimpleNamespace(
        query_params={"ticket": ticket},
        headers={},
        client=SimpleNamespace(host="203.0.113.7"),
        url=SimpleNamespace(path="/api/ws"),
    )


def _fake_ticket_subprotocol_ws(ticket):
    return SimpleNamespace(
        query_params={},
        headers={
            "sec-websocket-protocol": (
                f"{_web_server_chat._GATEWAY_WS_PROTOCOL}, "
                f"{_web_server_chat._GATEWAY_WS_TICKET_PROTOCOL_PREFIX}{ticket}"
            )
        },
        client=SimpleNamespace(host="203.0.113.7"),
        url=SimpleNamespace(path="/api/ws"),
    )


@pytest.fixture
def gated_dashboard():
    previous = getattr(web_server.app.state, "auth_required", False)
    web_server.app.state.auth_required = True
    try:
        yield
    finally:
        web_server.app.state.auth_required = previous
        _reset_for_tests()


def test_dashboard_ticket_identity_is_carried_forward_without_trusting_rpc_params(gated_dashboard):
    _reset_for_tests()
    ticket = mint_ticket(user_id="user-fixture", provider="provider-fixture")
    ws = _fake_ticket_ws(ticket)

    assert _web_server_chat._ws_auth_ok(ws) is True
    assert ws._hermes_auth_identity == {
        "user_id": "user-fixture",
        "provider": "provider-fixture",
    }
    assert _web_server_chat._ws_auth_ok(_fake_ticket_ws(ticket)) is False


def test_dashboard_ticket_subprotocol_carries_the_same_server_identity(gated_dashboard):
    _reset_for_tests()
    ticket = mint_ticket(user_id="subprotocol-user", provider="provider-fixture")
    ws = _fake_ticket_subprotocol_ws(ticket)

    assert _web_server_chat._ws_auth_ok(ws) is True
    assert ws._hermes_auth_identity == {
        "user_id": "subprotocol-user",
        "provider": "provider-fixture",
    }
    assert ws._hermes_ws_subprotocol == _web_server_chat._GATEWAY_WS_PROTOCOL


def test_ws_transport_records_only_server_authenticated_identity():
    loop = SimpleNamespace()
    identity = {"user_id": "user-fixture", "provider": "provider-fixture"}
    transport = WSTransport(
        SimpleNamespace(),
        loop,
        peer="identity-test",
        auth_identity=identity,
    )
    assert transport.auth_identity == identity


def test_cloud_agent_context_binds_registration_principal_and_transport_family():
    from gateway.session_context import clear_session_vars, get_session_env

    identity = {"user_id": "user-fixture", "provider": "provider-fixture"}
    transport = SimpleNamespace(auth_identity=identity)
    server._sessions["context-session-fixture"] = {
        "transport": transport,
        "session_key": "stored-context-session",
        "profile": "default",
        "agent": SimpleNamespace(session_id="context-session-fixture"),
    }
    tokens = []
    try:
        tokens = server._set_session_context("stored-context-session")
        assert get_session_env("HERMES_BROWSER_CONTROL_PRINCIPAL") == _principal_digest(
            identity
        )
        assert (
            get_session_env("HERMES_BROWSER_CONTROL_TRANSPORT_FAMILY")
            == "cloud-ticket-ws"
        )
    finally:
        clear_session_vars(tokens)
        server._sessions.pop("context-session-fixture", None)


def test_cloud_event_writer_surfaces_closed_or_failed_transport_immediately():
    class RaisingTransport:
        def write(self, _frame):
            raise ConnectionError("fixture transport closed")

    class FalseTransport:
        def write(self, _frame):
            return False

    frame = {"method": "browser.controller.command", "params": {"command_id": "fixture"}}
    with pytest.raises(ConnectionError, match="fixture transport closed"):
        _broker_event_writer(RaisingTransport(), "session-fixture")(frame)
    with pytest.raises(ConnectionError, match="failed"):
        _broker_event_writer(FalseTransport(), "session-fixture")(frame)


def test_cloud_principal_digest_is_unambiguous_across_identity_components():
    assert _principal_digest({"user_id": "a:b", "provider": "c"}) != _principal_digest(
        {"user_id": "a", "provider": "b:c"}
    )


@pytest.mark.parametrize(
    "identity",
    [None, {}, {"user_id": "server-internal", "provider": "server-internal"}],
)
def test_cloud_controller_registration_rejects_missing_or_internal_identity(monkeypatch, identity):
    monkeypatch.setattr(
        "gateway.browser_control_broker.browser_control_enabled", lambda: True
    )

    class Transport:
        auth_identity = identity

        def write(self, _frame):
            return True

    transport = Transport()
    server._sessions["session-fixture"] = {
        "transport": transport,
        "session_key": "stored-session-fixture",
        "profile": "default",
    }
    try:
        response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "browser.controller.register",
                "params": {
                    "session_id": "session-fixture",
                    "controller_id": "controller-fixture",
                    "browser_profile_id": "browser-profile-fixture",
                    "capabilities": ["controller.noop"],
                    "principal_id": "spoofed-client-principal",
                },
            },
            transport,
        )
        assert response["error"]["code"] == 4403
    finally:
        server._sessions.pop("session-fixture", None)


@pytest.mark.parametrize(
    "params",
    [
        {"protocol_version": 2, "capabilities": ["browser_navigate"]},
        {"protocol_version": True, "capabilities": ["browser_navigate"]},
        {
            "protocol_version": 1,
            "capabilities": ["browser_cdp", "arbitrary.capability"],
        },
    ],
)
def test_cloud_registration_rejects_unsupported_protocol_or_empty_capabilities(
    monkeypatch, params
):
    monkeypatch.setattr(
        "gateway.browser_control_broker.browser_control_enabled", lambda: True
    )

    class Transport:
        auth_identity = {
            "user_id": "user-fixture",
            "provider": "provider-fixture",
        }

        def write(self, _frame):
            return True

    transport = Transport()
    server._sessions["registration-session-fixture"] = {
        "transport": transport,
        "session_key": "stored-registration-session",
        "profile": "default",
    }
    try:
        response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "browser.controller.register",
                "params": {
                    "session_id": "registration-session-fixture",
                    "controller_id": "controller-fixture",
                    "browser_profile_id": "browser-profile-fixture",
                    **params,
                },
            },
            transport,
        )
        assert response["error"]["code"] == 4403
    finally:
        server._sessions.pop("registration-session-fixture", None)


def test_cloud_gateway_real_action_round_trip_is_bound_to_identity_and_transport(monkeypatch):
    monkeypatch.setattr(
        "gateway.browser_control_broker.browser_control_enabled", lambda: True
    )
    broker = get_browser_control_broker()
    broker.reset()
    frames = []
    ready = threading.Event()

    class Transport:
        auth_identity = {
            "user_id": "user-fixture",
            "provider": "provider-fixture",
        }

        def write(self, frame):
            frames.append(frame)
            if frame.get("method") == "event":
                ready.set()
            return True

    transport = Transport()
    server._sessions["session-fixture"] = {
        "transport": transport,
        "session_key": "stored-session-fixture",
        "profile": "default",
    }
    try:
        registration = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "browser.controller.register",
                "params": {
                    "protocol_version": 1,
                    "session_id": "session-fixture",
                    "controller_id": "controller-fixture",
                    "browser_profile_id": "browser-profile-fixture",
                    "capabilities": ["browser_navigate", "browser_cdp"],
                    "principal_id": "spoofed-client-principal",
                },
            },
            transport,
        )
        scope_payload = registration["result"]["scope"]
        assert scope_payload["principal_id"] != "spoofed-client-principal"
        assert scope_payload["transport_family"] == "cloud-ticket-ws"
        assert scope_payload["capabilities"] == ["browser_navigate"]
        assert "browser_cdp" not in BROWSER_CONTROL_CAPABILITIES

        missing_identity = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 41,
                "method": "browser.controller.register",
                "params": {
                    "session_id": "session-fixture",
                    "controller_id": "",
                    "browser_profile_id": "",
                    "capabilities": ["controller.noop"],
                },
            },
            transport=transport,
        )
        assert missing_identity["error"]["code"] == 4403

        scope = broker.scope_for_session(
            session_id="session-fixture",
            principal_id=scope_payload["principal_id"],
            transport_family="cloud-ticket-ws",
        )
        assert scope is not None
        heartbeat_response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 42,
                "method": "browser.controller.heartbeat",
                "params": {"session_id": "session-fixture"},
            },
            transport,
        )
        assert heartbeat_response["result"] == {"ok": True}

        foreign_transport = Transport()
        foreign_heartbeat = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 43,
                "method": "browser.controller.heartbeat",
                "params": {"session_id": "session-fixture"},
            },
            foreign_transport,
        )
        assert foreign_heartbeat["error"]["code"] == 4403
        outcome = {}

        def dispatch_navigate():
            outcome["result"] = broker.dispatch(
                scope,
                action="browser_navigate",
                arguments={"url": "https://example.test"},
                tool_call_id="tool-call-cloud",
            )

        thread = threading.Thread(target=dispatch_navigate)
        thread.start()
        assert ready.wait(timeout=1.0)
        command_event = frames[-1]
        assert command_event["method"] == "event"
        assert command_event["params"]["type"] == "browser.controller.command"
        command_id = command_event["params"]["payload"]["command_id"]

        result_response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "browser.controller.result",
                "params": {
                    "session_id": "session-fixture",
                    "command_id": command_id,
                    "ok": True,
                    "result": {"echo": "cloud"},
                },
            },
            transport,
        )
        assert result_response["result"]["accepted"] is True
        thread.join(timeout=1.0)
        assert outcome["result"] == {"echo": "cloud"}

        rejected_outcome = {}
        ready.clear()

        def dispatch_rejected():
            try:
                rejected_outcome["result"] = broker.dispatch(
                    scope,
                    action="browser_navigate",
                    arguments={"url": "https://reject.example.test"},
                    tool_call_id="tool-call-cloud-rejected",
                )
            except Exception as exc:  # asserted below
                rejected_outcome["error"] = exc

        rejected_thread = threading.Thread(target=dispatch_rejected, daemon=True)
        rejected_thread.start()
        assert ready.wait(timeout=1.0)
        rejected_event = frames[-1]
        rejected_response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "browser.controller.result",
                "params": {
                    "session_id": "session-fixture",
                    "command_id": rejected_event["params"]["payload"]["command_id"],
                    "ok": "false",
                    "error": {"code": "controller_rejected", "message": "fixture rejection"},
                },
            },
            transport=transport,
        )
        assert rejected_response["result"]["accepted"] is True
        rejected_thread.join(timeout=1.0)
        assert isinstance(rejected_outcome.get("error"), ControllerRejected)
        assert "controller_rejected" in str(rejected_outcome["error"])
        assert broker.pending_count == 0
    finally:
        broker.reset()
        server._sessions.pop("session-fixture", None)


def test_cloud_same_identity_reconnect_refreshes_transport_and_completes_pending(monkeypatch):
    monkeypatch.setattr(
        "gateway.browser_control_broker.browser_control_enabled", lambda: True
    )
    broker = get_browser_control_broker()
    broker.reset()
    ready = threading.Event()
    first_frames = []
    second_frames = []

    class Transport:
        auth_identity = {
            "user_id": "reconnect-user",
            "provider": "provider-fixture",
        }

        def __init__(self, frames):
            self.frames = frames

        def write(self, frame):
            self.frames.append(frame)
            if frame.get("method") == "event":
                ready.set()
            return True

    first = Transport(first_frames)
    second = Transport(second_frames)
    session = {
        "transport": first,
        "session_key": "stored-reconnect-session",
        "profile": "default",
    }
    server._sessions["reconnect-session"] = session
    try:
        first_registration = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "browser.controller.register",
                "params": {
                    "protocol_version": 1,
                    "session_id": "reconnect-session",
                    "controller_id": "controller-fixture",
                    "browser_profile_id": "browser-profile-fixture",
                    "capabilities": ["browser_navigate"],
                },
            },
            first,
        )
        scope_payload = first_registration["result"]["scope"]
        scope = broker.scope_for_session(
            session_id="reconnect-session",
            principal_id=scope_payload["principal_id"],
            transport_family="cloud-ticket-ws",
        )
        outcome = {}

        def dispatch():
            outcome["result"] = broker.dispatch(
                scope,
                action="browser_navigate",
                arguments={"url": "https://example.test"},
                tool_call_id="tool-call-reconnect",
            )

        thread = threading.Thread(target=dispatch)
        thread.start()
        assert ready.wait(timeout=1.0)
        command_id = first_frames[-1]["params"]["payload"]["command_id"]

        assert broker.disconnect_owner(first) == 1
        assert thread.is_alive()
        session["transport"] = second
        second_registration = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "browser.controller.register",
                "params": {
                    "protocol_version": 1,
                    "session_id": "reconnect-session",
                    "controller_id": "controller-fixture",
                    "browser_profile_id": "browser-profile-fixture",
                    "capabilities": ["browser_navigate", "browser_snapshot"],
                },
            },
            second,
        )
        assert second_registration["result"]["scope"]["capabilities"] == [
            "browser_navigate",
            "browser_snapshot",
        ]
        result = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "browser.controller.result",
                "params": {
                    "session_id": "reconnect-session",
                    "command_id": command_id,
                    "ok": True,
                    "result": {"reconnected": True},
                },
            },
            second,
        )
        assert result["result"]["accepted"] is True
        thread.join(timeout=1.0)
        assert outcome.get("result") == {"reconnected": True}
    finally:
        broker.reset()
        server._sessions.pop("reconnect-session", None)


def test_cloud_explicit_detach_requires_current_authenticated_owner(monkeypatch):
    monkeypatch.setattr(
        "gateway.browser_control_broker.browser_control_enabled", lambda: True
    )
    broker = get_browser_control_broker()
    broker.reset()

    class Transport:
        auth_identity = {
            "user_id": "detach-user",
            "provider": "provider-fixture",
        }

        def write(self, _frame):
            return True

    owner = Transport()
    foreign = Transport()
    server._sessions["detach-session"] = {
        "transport": owner,
        "session_key": "stored-detach-session",
        "profile": "default",
    }
    try:
        registration = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "browser.controller.register",
                "params": {
                    "protocol_version": 1,
                    "session_id": "detach-session",
                    "controller_id": "controller-fixture",
                    "browser_profile_id": "browser-profile-fixture",
                    "capabilities": ["controller.noop"],
                },
            },
            owner,
        )
        assert "result" in registration

        foreign_result = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "browser.controller.detach",
                "params": {"session_id": "detach-session"},
            },
            foreign,
        )
        assert foreign_result["error"]["code"] == 4403

        detached = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "browser.controller.detach",
                "params": {"session_id": "detach-session"},
            },
            owner,
        )
        assert detached["result"] == {"detached": True}
        assert broker.scope_for_session(
            session_id="detach-session",
            principal_id=registration["result"]["scope"]["principal_id"],
            transport_family="cloud-ticket-ws",
        ) is None
    finally:
        broker.reset()
        server._sessions.pop("detach-session", None)

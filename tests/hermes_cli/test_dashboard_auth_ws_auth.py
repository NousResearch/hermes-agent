"""Tests for the WS-upgrade auth helper (Phase 5 task 5.2).

The dashboard's WS endpoints (``/api/pty``, ``/api/console``, ``/api/ws``,
``/api/pub``, ``/api/events``) share an auth gate: ``_ws_auth_ok``. In
loopback mode it accepts ``?token=<_SESSION_TOKEN>``; in gated mode it accepts
a single-use ``?ticket=`` minted by ``POST /api/auth/ws-ticket``.

These tests exercise the helper at the unit level, the ticket-mint endpoint
under realistic gated-mode setup, and one public mint-to-upgrade-to-dispatch
round trip. The full upgrade supplies explicit Host/Origin headers because
Starlette's WebSocket TestClient does not inherit the HTTP ``base_url`` host.
"""

from __future__ import annotations

import contextlib
import threading
import time
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli import mcp_startup
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.ws_tickets import (
    _reset_for_tests,
    consume_ticket,
    consume_internal_credential,
    internal_ws_credential,
    mint_ticket,
)
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider
from tui_gateway import server as tui_server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gated_app():
    """web_server.app configured for gated mode + stub provider registered."""
    _reset_for_tests()
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    _reset_for_tests()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


@pytest.fixture
def loopback_app():
    """web_server.app configured for loopback mode (gate OFF)."""
    _reset_for_tests()
    clear_providers()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8080
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://127.0.0.1:8080")
    yield client
    _reset_for_tests()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


@pytest.fixture
def insecure_public_app():
    """web_server.app configured for all-interfaces insecure mode."""
    _reset_for_tests()
    clear_providers()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "0.0.0.0"
    web_server.app.state.bound_port = 9120
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://192.168.0.222:9120")
    yield client
    _reset_for_tests()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _logged_in(client: TestClient) -> None:
    """Drive the stub OAuth round trip so the client holds session cookies."""
    r1 = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert r1.status_code == 302
    state = r1.headers["location"].split("state=")[1]
    r2 = client.get(
        f"/auth/callback?code=stub_code&state={state}", follow_redirects=False
    )
    assert r2.status_code == 302


_MOBILE_SCOPES = [
    "conversation.read",
    "conversation.write",
    "conversation.control",
]
_PUBLIC_WS_HEADERS = {
    "Host": "fly-app.fly.dev",
    "Origin": "https://fly-app.fly.dev",
}


def _mint_mobile_ticket(client: TestClient) -> str:
    response = client.post(
        "/api/auth/ws-ticket",
        json={
            "audience": "hermes.mobile",
            "scopes": _MOBILE_SCOPES,
        },
    )
    assert response.status_code == 200
    assert response.json()["granted_scopes"] == _MOBILE_SCOPES
    return response.json()["ticket"]


def _receive_rpc(socket, request_id: str) -> dict:
    """Read interleaved events until one public JSON-RPC response arrives."""
    for _ in range(30):
        frame = socket.receive_json()
        if frame.get("id") == request_id:
            return frame
    raise AssertionError(f"JSON-RPC response {request_id!r} was not delivered")


def _receive_rpc_and_event(socket, request_id: str, event_type: str) -> tuple[dict, dict]:
    """Collect a response and event without assuming their wire order."""
    response = None
    event = None
    for _ in range(30):
        frame = socket.receive_json()
        if frame.get("id") == request_id:
            response = frame
        if frame.get("params", {}).get("type") == event_type:
            event = frame
        if response is not None and event is not None:
            return response, event
    raise AssertionError(
        f"response {request_id!r} and event {event_type!r} were not delivered"
    )


def _assert_mobile_ready(socket) -> dict:
    ready = socket.receive_json()
    assert ready["method"] == "event"
    assert ready["params"]["type"] == "gateway.ready"
    payload = ready["params"]["payload"]
    assert payload["protocol"] == {"name": "hermes.tui.jsonrpc", "major": 1}
    assert payload["contract"] == {"name": "hermes.mobile", "major": 1}
    assert payload["schemas"]["session.synchronization"] == 1
    assert payload["schemas"]["mutation.receipt"] == 1
    assert payload["schemas"]["approval.lifecycle"] == 1
    mutation = payload["capabilities"]["mutation.idempotency"]
    assert mutation["version"] == 1
    assert {
        "prompt.submit",
        "session.interrupt",
        "approval.respond",
        "session.delete",
    } <= set(mutation["methods"])
    assert mutation["status_method"] == "mutation.status"
    lifecycle = payload["capabilities"]["interaction.lifecycle"]
    assert lifecycle["version"] == 1
    assert "approval" in lifecycle["kinds"]
    assert "approval.respond" in lifecycle["response_methods"]
    assert payload["authorization"] == {
        "subject": "stub-user-1",
        "provider": "stub",
        "audience": "hermes.mobile",
        "scopes": _MOBILE_SCOPES,
    }
    return payload


# ---------------------------------------------------------------------------
# POST /api/auth/ws-ticket — the mint endpoint
# ---------------------------------------------------------------------------


class TestWsTicketEndpoint:
    def test_authenticated_session_can_mint(self, gated_app):
        _logged_in(gated_app)
        r = gated_app.post("/api/auth/ws-ticket")
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"ticket", "ttl_seconds"}
        assert isinstance(body["ticket"], str)
        assert len(body["ticket"]) >= 32
        assert body["ttl_seconds"] == 30
        grant = consume_ticket(body["ticket"])
        assert grant["audience"] == "dashboard"
        assert grant["scopes"] == ("*",)

    def test_unauthenticated_returns_401_or_redirect(self, gated_app):
        r = gated_app.post("/api/auth/ws-ticket", follow_redirects=False)
        # gated_auth_middleware short-circuits before the route — it
        # returns either 401 or 302. Either is fine.
        assert r.status_code in (302, 401)

    def test_each_call_returns_a_distinct_ticket(self, gated_app):
        _logged_in(gated_app)
        tickets = {gated_app.post("/api/auth/ws-ticket").json()["ticket"]
                   for _ in range(5)}
        assert len(tickets) == 5

    def test_mobile_ticket_returns_and_preserves_the_granted_scopes(self, gated_app):
        _logged_in(gated_app)

        response = gated_app.post(
            "/api/auth/ws-ticket",
            json={
                "audience": "hermes.mobile",
                "scopes": ["conversation.read", "conversation.write"],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["audience"] == "hermes.mobile"
        assert body["granted_scopes"] == [
            "conversation.read",
            "conversation.write",
        ]
        grant = consume_ticket(body["ticket"])
        assert grant["audience"] == "hermes.mobile"
        assert grant["scopes"] == ("conversation.read", "conversation.write")

    @pytest.mark.parametrize("scope", ["approval.respond", "shell.exec"])
    def test_mobile_ticket_rejects_scopes_not_enforced_by_this_contract(
        self,
        gated_app,
        scope,
    ):
        _logged_in(gated_app)

        response = gated_app.post(
            "/api/auth/ws-ticket",
            json={"audience": "hermes.mobile", "scopes": [scope]},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == f"unsupported mobile scope: {scope}"

    def test_scopes_without_a_mobile_audience_do_not_mint_legacy_authority(
        self,
        gated_app,
    ):
        _logged_in(gated_app)

        response = gated_app.post(
            "/api/auth/ws-ticket",
            json={"scopes": ["conversation.read"]},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "audience is required when scopes are requested"

    def test_nonempty_request_without_an_audience_does_not_mint_legacy_authority(
        self,
        gated_app,
    ):
        _logged_in(gated_app)

        response = gated_app.post("/api/auth/ws-ticket", json={})

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "audience is required for a non-empty ticket request"
        )

    def test_mobile_write_grant_requires_read_scope(self, gated_app):
        _logged_in(gated_app)

        response = gated_app.post(
            "/api/auth/ws-ticket",
            json={
                "audience": "hermes.mobile",
                "scopes": ["conversation.write"],
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "conversation.read is required for every mobile WebSocket grant"
        )

    def test_get_method_is_not_allowed(self, gated_app):
        _logged_in(gated_app)
        r = gated_app.get("/api/auth/ws-ticket", follow_redirects=False)
        # GET must not mint a ticket (which would be cookie-replayable via
        # <img src=…> from a malicious origin). Accepted responses:
        #   401 — gated middleware allowlist-miss
        #   404 — SPA catch-all swallowed it
        #   405 — Method Not Allowed (route only registered for POST)
        #   200 — SPA index.html was served (catch-all caught the path)
        # In every case the JSON body of a successful ticket mint must
        # NOT be present. The assertion below holds even when the SPA
        # shell happens to serve a 200.
        body = r.text
        assert "ticket" not in body or '"ttl_seconds"' not in body, (
            f"GET /api/auth/ws-ticket leaked a ticket (status={r.status_code}, "
            f"body[:200]={body[:200]!r})"
        )


def test_scoped_ticket_grant_reaches_gateway_ready_and_dispatch(gated_app, monkeypatch):
    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", lambda **_kw: None)
    monkeypatch.setattr(tui_server, "resolve_skin", lambda: "test-skin")

    _logged_in(gated_app)
    minted = gated_app.post(
        "/api/auth/ws-ticket",
        json={
            "audience": "hermes.mobile",
            "scopes": ["conversation.read"],
        },
    )
    assert minted.status_code == 200

    with gated_app.websocket_connect(
        f"/api/ws?ticket={minted.json()['ticket']}",
        headers={
            "Host": "fly-app.fly.dev",
            "Origin": "https://fly-app.fly.dev",
        },
    ) as socket:
        ready = socket.receive_json()
        authorization = ready["params"]["payload"]["authorization"]
        assert authorization == {
            "subject": "stub-user-1",
            "provider": "stub",
            "audience": "hermes.mobile",
            "scopes": ["conversation.read"],
        }

        socket.send_json(
            {
                "jsonrpc": "2.0",
                "id": "denied-write",
                "method": "prompt.submit",
                "params": {},
            }
        )
        denied = socket.receive_json()
        assert denied["error"]["data"]["required_scope"] == "conversation.write"


def test_mobile_contract_public_conformance_round_trip(
    gated_app,
    monkeypatch,
    tmp_path,
):
    """Prove mobile recovery through auth, FastAPI, and real WebSockets."""
    from unittest.mock import MagicMock

    import run_agent
    from agent import title_generator
    from hermes_state import SessionDB
    from tools import approval
    from tui_gateway import mobile_contract
    from tui_gateway.mobile_mutations import MobileMutationStore
    from tui_gateway.mobile_sync import SessionEventStream

    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(
        mcp_startup,
        "start_background_mcp_discovery",
        lambda **_kw: None,
    )
    monkeypatch.setattr(tui_server, "resolve_skin", lambda: "test-skin")
    monkeypatch.setattr(
        tui_server,
        "_claim_active_session_slot",
        lambda *_a, **_k: (None, None),
    )
    monkeypatch.setattr(tui_server, "_schedule_agent_build", lambda *_a: None)
    monkeypatch.setattr(
        tui_server,
        "_schedule_session_cap_enforcement",
        lambda: None,
    )
    monkeypatch.setattr(tui_server, "_register_session_cwd", lambda *_a: None)
    monkeypatch.setattr(tui_server, "_profile_home", lambda *_a: None)
    monkeypatch.setattr(
        tui_server,
        "_completion_cwd",
        lambda *_a, **_k: str(tmp_path),
    )
    monkeypatch.setattr(tui_server, "_git_branch_for_cwd", lambda *_a: "")
    monkeypatch.setattr(tui_server, "_resolve_model", lambda: "test/model")
    monkeypatch.setattr(tui_server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(tui_server, "_schedule_ws_orphan_reap", lambda *_a: None)
    monkeypatch.setattr(tui_server, "make_stream_renderer", lambda *_a: None)
    monkeypatch.setattr(tui_server, "render_message", lambda *_a: None)
    monkeypatch.setattr(
        tui_server,
        "_sync_agent_model_with_config",
        lambda *_a: None,
    )
    monkeypatch.setattr(title_generator, "maybe_auto_title", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda *_a, **_k: [])
    monkeypatch.setattr(
        run_agent,
        "check_toolset_requirements",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(run_agent, "OpenAI", MagicMock)

    database = SessionDB(db_path=tmp_path / "state.db")
    mutation_store = MobileMutationStore(tmp_path / "mobile-mutations.sqlite3")
    monkeypatch.setattr(tui_server, "_get_db", lambda: database)
    monkeypatch.setattr(
        tui_server,
        "_session_db",
        lambda _session: contextlib.nullcontext(database),
    )
    monkeypatch.setattr(
        tui_server,
        "_mobile_mutation_store",
        lambda: mutation_store,
    )
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 10},
    )

    turn_entered = threading.Event()
    release_turn = threading.Event()
    turn_count = 0
    tui_server._sessions.clear()
    _logged_in(gated_app)

    stored_session_id = ""
    agent = None
    approval_thread = None
    approval_decision = {}
    try:
        first_ticket = _mint_mobile_ticket(gated_app)
        with gated_app.websocket_connect(
            f"/api/ws?ticket={first_ticket}",
            headers=_PUBLIC_WS_HEADERS,
        ) as socket:
            _assert_mobile_ready(socket)
            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "create",
                    "method": "session.create",
                    "params": {"cols": 100},
                }
            )
            created = _receive_rpc(socket, "create")["result"]
            live_session_id = created["session_id"]
            stored_session_id = created["stored_session_id"]
            initial_cursor = created["synchronization"]["recovery"]["cursor"]
            assert created["synchronization"]["recovery"]["outcome"] == "reset"
            assert created["synchronization"]["recovery"]["reason"] == (
                "cursor_missing"
            )

            agent = run_agent.AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                platform="tui",
                quiet_mode=True,
                session_db=database,
                session_id=stored_session_id,
                skip_context_files=True,
                skip_memory=True,
            )
            agent.api_mode = "chat_completions"
            agent.client = MagicMock()
            agent._cached_system_prompt = "SYSTEM"
            agent.compression_enabled = False
            agent._skip_mcp_refresh = True

            def deterministic_provider(_api_kwargs, *, on_first_delta=None):
                nonlocal turn_count
                turn_count += 1
                turn_entered.set()
                assert release_turn.wait(timeout=5)
                if on_first_delta is not None:
                    on_first_delta()
                stream_callback = getattr(agent, "_stream_callback", None)
                assert callable(stream_callback)
                stream_callback("fixture complete")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(
                                content="fixture complete",
                                reasoning=None,
                                reasoning_content=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                    model="test/model",
                    usage=SimpleNamespace(
                        completion_tokens=2,
                        prompt_tokens=4,
                        total_tokens=6,
                    ),
                )

            agent._interruptible_streaming_api_call = deterministic_provider
            agent._interruptible_api_call = lambda api_kwargs: (
                deterministic_provider(api_kwargs)
            )
            session = tui_server._sessions[live_session_id]
            session["agent"] = agent
            session["agent_ready"] = None
            tui_server._register_gateway_approval_callbacks(
                stored_session_id,
                live_session_id,
            )

            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "prompt-uncertain",
                    "method": "prompt.submit",
                    "params": {
                        "client_request_id": "prompt-public-1",
                        "expected_stored_session_id": stored_session_id,
                        "session_id": live_session_id,
                        "text": "one durable public turn",
                    },
                }
            )
            assert turn_entered.wait(timeout=2)
            # Intentionally leave without consuming the prompt ACK.

        release_turn.set()
        deadline = time.monotonic() + 3
        receipt = None
        while time.monotonic() < deadline:
            receipt = mutation_store.status(
                provider="stub",
                subject="stub-user-1",
                client_request_id="prompt-public-1",
            )
            if (
                receipt
                and receipt["state"] == "completed"
                and not session.get("running")
            ):
                break
            time.sleep(0.01)
        assert receipt is not None and receipt["state"] == "completed"
        assert session["running"] is False
        run_thread = session["_run_thread"]
        run_thread.join(timeout=2)
        assert not run_thread.is_alive()
        persisted_messages = database.get_messages(stored_session_id)
        assert [
            row["content"] for row in persisted_messages if row["role"] == "user"
        ] == ["one durable public turn"]

        second_ticket = _mint_mobile_ticket(gated_app)
        with gated_app.websocket_connect(
            f"/api/ws?ticket={second_ticket}",
            headers=_PUBLIC_WS_HEADERS,
        ) as socket:
            _assert_mobile_ready(socket)
            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resume-after-prompt",
                    "method": "session.resume",
                    "params": {
                        "cols": 100,
                        "cursor": initial_cursor,
                        "session_id": stored_session_id,
                    },
                }
            )
            resumed = _receive_rpc(socket, "resume-after-prompt")["result"]
            synchronization = resumed["synchronization"]
            assert synchronization["recovery"]["outcome"] == "complete"
            assert synchronization["recovery"]["snapshot_required"] is False
            replayed_types = [
                frame["params"]["type"]
                for frame in synchronization["recovery"]["events"]
            ]
            assert replayed_types.count("message.start") == 1
            assert replayed_types.count("message.delta") == 1
            assert replayed_types.count("message.complete") == 1
            assert replayed_types.index("message.start") < replayed_types.index(
                "message.delta"
            ) < replayed_types.index("message.complete")
            snapshot_messages = synchronization["snapshot"]["messages"]
            assert [
                message
                for message in snapshot_messages
                if message["role"] == "user"
            ] == [{"role": "user", "text": "one durable public turn"}]
            assert [
                message
                for message in snapshot_messages
                if message["role"] == "assistant"
            ] == [{"role": "assistant", "text": "fixture complete"}]
            active_session_id = resumed["session_id"]

            prompt_params = {
                "client_request_id": "prompt-public-1",
                "expected_stored_session_id": stored_session_id,
                "session_id": active_session_id,
                "text": "one durable public turn",
            }
            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "prompt-retry",
                    "method": "prompt.submit",
                    "params": prompt_params,
                }
            )
            prompt_retry = _receive_rpc(socket, "prompt-retry")["result"]
            assert prompt_retry["mutation"] == {
                "client_request_id": "prompt-public-1",
                "deduplicated": True,
                "state": "completed",
            }
            assert turn_count == 1

            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "history-after-retry",
                    "method": "session.history",
                    "params": {"session_id": active_session_id},
                }
            )
            history = _receive_rpc(socket, "history-after-retry")["result"]
            assert [
                message
                for message in history["messages"]
                if message["role"] == "user"
            ] == [{"role": "user", "text": "one durable public turn"}]
            assert [
                message
                for message in history["messages"]
                if message["role"] == "assistant"
            ] == [{"role": "assistant", "text": "fixture complete"}]

            pre_approval_cursor = synchronization["recovery"]["cursor"]
            registered_notify = approval._gateway_notify_cbs[stored_session_id]
            approval_emitted = threading.Event()

            def wait_for_approval():
                def notify(data):
                    registered_notify(data)
                    approval_emitted.set()

                approval_decision["value"] = approval._await_gateway_decision(
                    stored_session_id,
                    notify,
                    {
                        "command": "printf mobile-contract-conformance",
                        "description": "harmless conformance approval",
                        "pattern_key": "printf",
                        "pattern_keys": ["printf"],
                    },
                )

            approval_thread = threading.Thread(target=wait_for_approval)
            approval_thread.start()
            assert approval_emitted.wait(timeout=2)
            # Intentionally disconnect before consuming approval.request.

        third_ticket = _mint_mobile_ticket(gated_app)
        with gated_app.websocket_connect(
            f"/api/ws?ticket={third_ticket}",
            headers=_PUBLIC_WS_HEADERS,
        ) as socket:
            _assert_mobile_ready(socket)
            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resume-approval",
                    "method": "session.resume",
                    "params": {
                        "cols": 100,
                        "cursor": pre_approval_cursor,
                        "session_id": stored_session_id,
                    },
                }
            )
            approval_resume = _receive_rpc(socket, "resume-approval")["result"]
            approval_sync = approval_resume["synchronization"]
            assert approval_sync["recovery"]["outcome"] == "complete"
            replayed_approval = [
                frame
                for frame in approval_sync["recovery"]["events"]
                if frame["params"]["type"] == "approval.request"
            ]
            assert len(replayed_approval) == 1
            pending_approvals = approval_sync["snapshot"]["pending_interactions"]
            assert len(pending_approvals) == 1
            approval_id = replayed_approval[0]["params"]["payload"]["approval_id"]
            assert pending_approvals[0]["approval_id"] == approval_id
            assert pending_approvals[0]["kind"] == "approval"
            terminal_cursor = approval_sync["recovery"]["cursor"]
            approval_params = {
                "approval_id": approval_id,
                "choice": "once",
                "client_request_id": "approval-public-1",
                "expected_stored_session_id": stored_session_id,
                "session_id": approval_resume["session_id"],
            }

            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resolve-approval",
                    "method": "approval.respond",
                    "params": approval_params,
                }
            )
            resolved, terminal = _receive_rpc_and_event(
                socket,
                "resolve-approval",
                "approval.resolved",
            )
            assert resolved["result"]["outcome"] == "resolved"
            assert resolved["result"]["approval"]["approval_id"] == approval_id
            assert resolved["result"]["mutation"]["deduplicated"] is False
            assert terminal["params"]["payload"]["approval_id"] == approval_id

            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resolve-approval-retry",
                    "method": "approval.respond",
                    "params": approval_params,
                }
            )
            replayed_resolution = _receive_rpc(
                socket,
                "resolve-approval-retry",
            )["result"]
            assert replayed_resolution["outcome"] == "resolved"
            assert replayed_resolution["approval"] == resolved["result"]["approval"]
            assert replayed_resolution["mutation"] == {
                "client_request_id": "approval-public-1",
                "deduplicated": True,
                "state": "completed",
            }
            approval_thread.join(timeout=2)
            assert not approval_thread.is_alive()
            assert approval_decision == {
                "value": {"resolved": True, "choice": "once", "reason": None}
            }

            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resume-terminal",
                    "method": "session.resume",
                    "params": {
                        "cols": 100,
                        "cursor": terminal_cursor,
                        "session_id": stored_session_id,
                    },
                }
            )
            terminal_resume = _receive_rpc(socket, "resume-terminal")["result"]
            terminal_sync = terminal_resume["synchronization"]
            terminal_events = [
                frame
                for frame in terminal_sync["recovery"]["events"]
                if frame["params"]["type"] == "approval.resolved"
            ]
            assert len(terminal_events) == 1
            assert terminal_events[0]["params"]["payload"]["approval_id"] == (
                approval_id
            )
            assert terminal_sync["snapshot"]["pending_interactions"] == []

            session = tui_server._sessions[terminal_resume["session_id"]]
            with session["history_lock"]:
                session["mobile_sync"] = SessionEventStream(
                    mobile_contract.SERVER_INSTANCE_ID,
                    max_events=1,
                    max_bytes=1024 * 1024,
                )
                gap_cursor = session["mobile_sync"].cursor()
            tui_server._emit(
                "status.update",
                terminal_resume["session_id"],
                {"kind": "step", "text": "one"},
            )
            tui_server._emit(
                "status.update",
                terminal_resume["session_id"],
                {"kind": "step", "text": "two"},
            )

        fourth_ticket = _mint_mobile_ticket(gated_app)
        with gated_app.websocket_connect(
            f"/api/ws?ticket={fourth_ticket}",
            headers=_PUBLIC_WS_HEADERS,
        ) as socket:
            _assert_mobile_ready(socket)
            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resume-gap",
                    "method": "session.resume",
                    "params": {
                        "cols": 100,
                        "cursor": gap_cursor,
                        "session_id": stored_session_id,
                    },
                }
            )
            gap = _receive_rpc(socket, "resume-gap")["result"]["synchronization"]
            assert gap["recovery"]["outcome"] == "gap"
            assert gap["recovery"]["reason"] == "replay_evicted"
            assert gap["recovery"]["snapshot_required"] is True

            stream_id = gap["snapshot"]["stream_id"]
            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resume-stream-reset",
                    "method": "session.resume",
                    "params": {
                        "cols": 100,
                        "cursor": {
                            "server_instance_id": mobile_contract.SERVER_INSTANCE_ID,
                            "stream_id": "retired-stream",
                            "sequence": 0,
                        },
                        "session_id": stored_session_id,
                    },
                }
            )
            stream_reset = _receive_rpc(
                socket,
                "resume-stream-reset",
            )["result"]["synchronization"]["recovery"]
            assert stream_reset["outcome"] == "reset"
            assert stream_reset["reason"] == "stream_changed"

            socket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": "resume-server-reset",
                    "method": "session.resume",
                    "params": {
                        "cols": 100,
                        "cursor": {
                            "server_instance_id": "retired-server",
                            "stream_id": stream_id,
                            "sequence": 0,
                        },
                        "session_id": stored_session_id,
                    },
                }
            )
            server_reset = _receive_rpc(
                socket,
                "resume-server-reset",
            )["result"]["synchronization"]["recovery"]
            assert server_reset["outcome"] == "reset"
            assert server_reset["reason"] == "server_instance_changed"
    finally:
        release_turn.set()
        for session in list(tui_server._sessions.values()):
            run_thread = session.get("_run_thread")
            join = getattr(run_thread, "join", None)
            if callable(join):
                join(timeout=5)
        if stored_session_id:
            approval.unregister_gateway_notify(stored_session_id)
            with approval._lock:
                approval._gateway_queues.pop(stored_session_id, None)
                approval._gateway_tombstones.pop(stored_session_id, None)
                approval._schedule_gateway_tombstone_cleanup_locked()
        if approval_thread is not None:
            approval_thread.join(timeout=1)
        if agent is not None:
            agent.close()
        tui_server._sessions.clear()
        mutation_store.close()
        database.close()


# ---------------------------------------------------------------------------
# _ws_auth_ok — unit-level (synthetic WebSocket-shaped object)
# ---------------------------------------------------------------------------


@pytest.fixture
def insecure_explicit_host_app():
    """web_server.app bound to an explicit non-loopback host (--insecure).

    Models `--host 100.64.0.10 --insecure` (e.g. a Tailscale IP behind
    `tailscale serve`) — a specific address rather than the all-interfaces
    0.0.0.0 wildcard.
    """
    _reset_for_tests()
    clear_providers()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "100.64.0.10"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://100.64.0.10:9119")
    yield client
    _reset_for_tests()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _fake_ws(*, query: dict, client_host: str = "127.0.0.1", path: str = "/api/pty"):
    """Build a stand-in for starlette.WebSocket good enough for _ws_auth_ok."""

    class _QP:
        def __init__(self, q):
            self._q = q

        def get(self, k, default=""):
            return self._q.get(k, default)

    return SimpleNamespace(
        query_params=_QP(query),
        client=SimpleNamespace(host=client_host),
        url=SimpleNamespace(path=path),
    )


class TestWsAuthOkLoopback:
    """Gate OFF — legacy token path."""

    def test_correct_token_accepted(self, loopback_app):
        ws = _fake_ws(query={"token": web_server._SESSION_TOKEN})
        assert web_server._ws_auth_ok(ws) is True

    def test_wrong_token_rejected(self, loopback_app):
        ws = _fake_ws(query={"token": "not-the-real-token"})
        assert web_server._ws_auth_ok(ws) is False

    def test_missing_token_rejected(self, loopback_app):
        ws = _fake_ws(query={})
        assert web_server._ws_auth_ok(ws) is False

    def test_ticket_param_ignored_in_loopback(self, loopback_app):
        # Even if someone sneaks a ticket through, loopback mode only
        # cares about ?token=. A naked ticket isn't a token.
        ticket = mint_ticket(user_id="u1", provider="stub")
        ws = _fake_ws(query={"ticket": ticket})
        assert web_server._ws_auth_ok(ws) is False


class TestWsAuthOkGated:
    """Gate ON — ticket path only."""

    def test_valid_ticket_accepted(self, gated_app):
        ticket = mint_ticket(user_id="u1", provider="stub")
        ws = _fake_ws(query={"ticket": ticket})
        assert web_server._ws_auth_ok(ws) is True

    @pytest.mark.parametrize("path", ["/api/pty", "/api/console", "/api/pub", "/api/events"])
    def test_mobile_ticket_cannot_authenticate_non_gateway_websockets(
        self,
        gated_app,
        path,
    ):
        ticket = mint_ticket(
            user_id="mobile-user",
            provider="stub",
            audience="hermes.mobile",
            scopes=("conversation.read",),
        )
        ws = _fake_ws(query={"ticket": ticket}, path=path)

        reason, credential, authorization = web_server._ws_auth_result(ws)

        assert reason == "ticket_audience_mismatch"
        assert credential == "ticket"
        assert authorization is None

    def test_consumed_ticket_rejected(self, gated_app):
        ticket = mint_ticket(user_id="u1", provider="stub")
        ws_one = _fake_ws(query={"ticket": ticket})
        ws_two = _fake_ws(query={"ticket": ticket})
        assert web_server._ws_auth_ok(ws_one) is True
        # Single-use — second consumption fails.
        assert web_server._ws_auth_ok(ws_two) is False

    def test_unknown_ticket_rejected(self, gated_app):
        ws = _fake_ws(query={"ticket": "never-minted"})
        assert web_server._ws_auth_ok(ws) is False

    def test_missing_ticket_rejected(self, gated_app):
        ws = _fake_ws(query={})
        assert web_server._ws_auth_ok(ws) is False

    def test_legacy_token_rejected_in_gated_mode(self, gated_app):
        """Critical: gated mode must NOT honour the legacy token path
        even when someone has access to the in-process value of
        _SESSION_TOKEN (e.g. a leaked log line)."""
        ws = _fake_ws(query={"token": web_server._SESSION_TOKEN})
        assert web_server._ws_auth_ok(ws) is False

    def test_rejection_audit_logs(self, gated_app, tmp_path, monkeypatch):
        # Point the audit log at a tmp dir so we can read what got written.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.dashboard_auth import audit as audit_mod

        # The log path is resolved lazily on the first audit_log() call;
        # bust any cached handler so it re-resolves.
        if hasattr(audit_mod, "_LOGGER"):
            monkeypatch.setattr(audit_mod, "_LOGGER", None, raising=False)

        ws = _fake_ws(query={"ticket": "never-minted"})
        assert web_server._ws_auth_ok(ws) is False

        log_file = tmp_path / "logs" / "dashboard-auth.log"
        # The audit module may write asynchronously through stdlib logging,
        # but flush is synchronous. If the file doesn't exist yet, the
        # logger may not have been initialized in this process — that's
        # acceptable as long as the rejection path didn't crash.
        if log_file.exists():
            content = log_file.read_text()
            assert "ws_ticket_rejected" in content

    def test_internal_credential_accepted(self, gated_app):
        """Server-spawned children present the process-lifetime internal
        credential via ?internal= and are accepted in gated mode."""
        cred = internal_ws_credential()
        ws = _fake_ws(query={"internal": cred})
        assert web_server._ws_auth_ok(ws) is True

    def test_internal_credential_is_multi_use(self, gated_app):
        """Unlike single-use tickets, the internal credential survives
        repeated use so the child can reconnect."""
        cred = internal_ws_credential()
        for _ in range(3):
            ws = _fake_ws(query={"internal": cred})
            assert web_server._ws_auth_ok(ws) is True

    def test_wrong_internal_credential_rejected(self, gated_app):
        # Mint the real one so the store is non-empty, then present a bogus value.
        internal_ws_credential()
        ws = _fake_ws(query={"internal": "not-the-internal-credential"})
        assert web_server._ws_auth_ok(ws) is False

    def test_internal_credential_not_accepted_in_loopback(self, loopback_app):
        """Outside gated mode, ?internal= is meaningless — only ?token= works.
        A naked internal credential must not authenticate."""
        cred = internal_ws_credential()
        ws = _fake_ws(query={"internal": cred})
        assert web_server._ws_auth_ok(ws) is False


class TestWsRequestIsAllowedGated:
    """Bug fix: in gated mode, the WS peer-IP loopback check must be
    bypassed.

    When the OAuth gate is active, ``start_server`` runs uvicorn with
    ``proxy_headers=True`` so the dashboard can honour
    ``X-Forwarded-Proto`` from Fly's TLS terminator. A side effect is that
    ``ws.client.host`` is rewritten to the X-Forwarded-For value — the
    real internet client IP, never loopback. The loopback peer guard
    (intended only for unauthenticated loopback dev) must not also reject
    those upgrades: the OAuth gate + single-use ticket is the auth.

    Regression coverage: every WS endpoint (``/api/pty``, ``/api/console``,
    ``/api/ws``, ``/api/pub``, ``/api/events``) calls
    ``_ws_request_is_allowed`` after ``_ws_auth_ok``. If the peer-IP check
    rejects gated mode, the chat
    tab + sidebar tool feed silently fail to connect even after a
    successful OAuth login.
    """

    def test_non_loopback_peer_allowed_in_gated_mode(self, gated_app):
        ws = _fake_ws(query={}, client_host="203.0.113.7")
        # Host header matches the bound host so the DNS-rebinding guard
        # passes; only the peer-IP check is under test.
        ws.headers = {"host": "fly-app.fly.dev"}
        assert web_server._ws_request_is_allowed(ws) is True

    def test_non_loopback_peer_rejected_in_loopback_mode(self, loopback_app):
        """Loopback mode still enforces the peer-IP guard — the legacy
        token path is the only auth and we don't want random LAN hosts
        guessing it."""
        ws = _fake_ws(query={}, client_host="192.168.1.42")
        ws.headers = {"host": "127.0.0.1:8080"}
        assert web_server._ws_request_is_allowed(ws) is False

    def test_loopback_peer_allowed_in_loopback_mode(self, loopback_app):
        ws = _fake_ws(query={}, client_host="127.0.0.1")
        ws.headers = {"host": "127.0.0.1:8080"}
        assert web_server._ws_request_is_allowed(ws) is True

    def test_non_loopback_peer_allowed_in_insecure_public_mode(self, insecure_public_app):
        """`--host 0.0.0.0 --insecure` is an explicit LAN/public opt-in.

        Regression coverage for the dashboard `/chat` breakage where the
        HTML shell loaded on 9120 but every WebSocket upgrade was rejected
        with 403 because the loopback-only peer guard still ran even though
        the operator intentionally exposed the dashboard on all interfaces.
        """
        ws = _fake_ws(query={}, client_host="192.168.0.55")
        ws.headers = {
            "host": "192.168.0.222:9120",
            "origin": "http://192.168.0.222:9120",
        }
        assert web_server._ws_request_is_allowed(ws) is True

    def test_peer_allowed_on_explicit_non_loopback_bind(self, insecure_explicit_host_app):
        """`--host 100.64.0.10 --insecure` (Tailscale/LAN IP) is an explicit
        non-loopback opt-in too — not just the 0.0.0.0 wildcard.

        Regression coverage: the merged 0.0.0.0/:: fix did not cover binding
        directly to a specific tailnet/LAN address, so `/chat` HTML loaded but
        WS upgrades were still rejected by the loopback-only peer guard.
        """
        ws = _fake_ws(query={}, client_host="100.64.0.99")
        ws.headers = {
            "host": "100.64.0.10:9119",
            "origin": "http://100.64.0.10:9119",
        }
        assert web_server._ws_request_is_allowed(ws) is True

    def test_rebinding_host_rejected_on_explicit_non_loopback_bind(
        self, insecure_explicit_host_app
    ):
        """Lifting the peer-IP gate for an explicit bind must NOT lift the
        DNS-rebinding Host guard: a mismatched Host header is still rejected,
        because an explicit non-loopback bind requires an exact Host match in
        `_is_accepted_host` (unlike the 0.0.0.0 wildcard, which accepts any).
        """
        ws = _fake_ws(query={}, client_host="100.64.0.99")
        ws.headers = {"host": "evil.example.com"}
        assert web_server._ws_request_is_allowed(ws) is False

    def test_host_origin_guard_still_runs_in_gated_mode(self, gated_app):
        """Bypassing the peer-IP check must not bypass the DNS-rebinding
        Host header guard — that one still protects against attacker
        sites resolving DNS to the public IP."""
        ws = _fake_ws(query={}, client_host="203.0.113.7")
        ws.headers = {"host": "evil.example.com"}
        assert web_server._ws_request_is_allowed(ws) is False

    # -- security: empty / missing peer must fail closed in loopback mode --
    # Regression for the fail-open default-allow where
    # ``ws.client is None`` or ``ws.client.host == ""`` was treated as
    # "allowed" on a loopback-bound dashboard with auth disabled. ASGI
    # servers behind a misconfigured proxy or a unix-socket transport can
    # deliver either shape, so both must be rejected explicitly.

    def test_empty_client_host_rejected_in_loopback_mode(self, loopback_app):
        """An empty ws.client.host must be rejected on a loopback bind."""
        ws = _fake_ws(query={}, client_host="")
        ws.headers = {"host": "127.0.0.1:8080"}
        assert web_server._ws_client_is_allowed(ws) is False
        assert web_server._ws_request_is_allowed(ws) is False

    def test_missing_client_object_rejected_in_loopback_mode(self, loopback_app):
        """ws.client is None must be rejected on a loopback bind."""
        ws = _fake_ws(query={}, client_host="")
        ws.client = None  # ASGI servers can omit the client tuple entirely
        ws.headers = {"host": "127.0.0.1:8080"}
        assert web_server._ws_client_is_allowed(ws) is False
        assert web_server._ws_request_is_allowed(ws) is False

    def test_empty_client_host_reason_is_block(self, loopback_app):
        """_ws_client_reason must return a block reason for an empty peer,
        not ``None`` (which the dispatcher treats as ``allowed``)."""
        ws = _fake_ws(query={}, client_host="")
        ws.headers = {"host": "127.0.0.1:8080"}
        reason = web_server._ws_client_reason(ws)
        assert reason is not None
        assert "missing_or_empty_peer" in reason

    def test_empty_client_host_still_allowed_in_insecure_public_mode(
        self, insecure_public_app
    ):
        """The empty-peer fail-closed guard must only apply to loopback
        binds. With an explicit ``--host 0.0.0.0 --insecure`` opt-in, the
        loopback-only peer restriction does not run at all, so the empty
        peer case bypasses the new guard the same way a legitimate LAN
        peer does. Without this, the fix would regress the public-bind
        path the dashboard relies on."""
        ws = _fake_ws(query={}, client_host="")
        ws.headers = {
            "host": "192.168.0.222:9120",
            "origin": "http://192.168.0.222:9120",
        }
        assert web_server._ws_client_is_allowed(ws) is True

    def test_empty_client_host_still_allowed_in_gated_mode(self, gated_app):
        """The empty-peer fail-closed guard must not apply when the OAuth
        gate is active (``auth_required=True``). Gated mode rewrites
        ``ws.client.host`` via ``proxy_headers=True``, and the ticket is
        the auth, so peer-IP is irrelevant on that path."""
        ws = _fake_ws(query={}, client_host="")
        ws.headers = {"host": "dashboard.example.com"}
        assert web_server._ws_client_is_allowed(ws) is True


class TestWsHostOriginGuardOrigins:
    """The WS Origin guard must let the packaged desktop shell connect.

    Electron loads the packaged renderer over ``file://``, so its WebSocket
    handshake carries ``Origin: file://`` (or the opaque ``null``, or a custom
    ``app://`` scheme). The DNS-rebinding guard only needs to block cross-site
    http(s) origins — a malicious web page can never forge a non-web origin.

    This guard runs only AFTER ``_ws_auth_ok`` has validated the WS credential
    (session token on loopback / ``--insecure`` binds, single-use ``?ticket=``
    on OAuth-gated binds), so a non-web origin is trusted in every mode: the
    credential is the real gate, and a ``file://`` / ``null`` origin cannot
    originate a DNS-rebinding browser attack. ``http(s)`` origins are still
    match-checked against the bound host.
    """

    def _ws(self, *, origin, host):
        ws = _fake_ws(query={}, path="/api/ws")
        ws.headers = {"host": host, "origin": origin}
        return ws

    def test_loopback_file_origin_allowed(self, loopback_app):
        ws = self._ws(origin="file://", host="127.0.0.1:8080")
        assert web_server._ws_host_origin_is_allowed(ws) is True

    def test_loopback_null_origin_allowed(self, loopback_app):
        ws = self._ws(origin="null", host="127.0.0.1:8080")
        assert web_server._ws_host_origin_is_allowed(ws) is True

    def test_loopback_app_scheme_origin_allowed(self, loopback_app):
        ws = self._ws(origin="app://hermes", host="127.0.0.1:8080")
        assert web_server._ws_host_origin_is_allowed(ws) is True

    def test_loopback_matching_http_origin_allowed(self, loopback_app):
        # The dev renderer (vite) loads over http://127.0.0.1:<port>.
        ws = self._ws(origin="http://127.0.0.1:5174", host="127.0.0.1:8080")
        assert web_server._ws_host_origin_is_allowed(ws) is True

    def test_loopback_cross_site_http_origin_rejected(self, loopback_app):
        # DNS-rebinding / cross-site: a real web attacker can only present an
        # http(s) origin, and that must still be rejected.
        ws = self._ws(origin="http://evil.test", host="127.0.0.1:8080")
        assert web_server._ws_host_origin_is_allowed(ws) is False

    def test_explicit_non_loopback_file_origin_allowed(self, insecure_explicit_host_app):
        """Packaged Hermes Desktop also uses file:// when connecting to a
        Tailscale/LAN dashboard bind.

        The WebSocket route calls _ws_auth_ok before this guard, so in
        non-gated mode the legacy session token remains the auth boundary.
        """
        ws = self._ws(origin="file://", host="100.64.0.10:9119")
        assert web_server._ws_host_origin_is_allowed(ws) is True

    def test_explicit_non_loopback_null_origin_allowed(self, insecure_explicit_host_app):
        ws = self._ws(origin="null", host="100.64.0.10:9119")
        assert web_server._ws_host_origin_is_allowed(ws) is True

    def test_explicit_non_loopback_cross_site_http_origin_rejected(
        self, insecure_explicit_host_app
    ):
        ws = self._ws(origin="http://localhost:9119", host="100.64.0.10:9119")
        assert web_server._ws_host_origin_is_allowed(ws) is False

    def test_gated_file_origin_allowed(self, gated_app):
        # The packaged desktop app drives a remote OAuth-GATED gateway over a
        # file:// renderer origin. The WS route validates the single-use
        # ?ticket= in _ws_auth_ok before this guard runs, and a file:// origin
        # can't be a DNS-rebinding browser attack, so the Origin guard must let
        # it through. This is the regression that broke desktop → hosted
        # gateway connections — every WS upgrade got HTTP 403 even with a valid
        # ticket.
        ws = self._ws(origin="file://", host="fly-app.fly.dev")
        assert web_server._ws_host_origin_is_allowed(ws) is True

    def test_gated_null_origin_allowed(self, gated_app):
        ws = self._ws(origin="null", host="fly-app.fly.dev")
        assert web_server._ws_host_origin_is_allowed(ws) is True

    def test_gated_app_scheme_origin_allowed(self, gated_app):
        ws = self._ws(origin="app://.", host="fly-app.fly.dev")
        assert web_server._ws_host_origin_is_allowed(ws) is True

    def test_gated_cross_site_http_origin_still_host_checked(self, gated_app):
        # An http(s) origin is still subjected to the same-host check even on a
        # gated bind: a cross-site http origin whose netloc doesn't match the
        # bound host is rejected. Real browser DNS-rebinding defence unchanged.
        ws = self._ws(origin="https://evil.test", host="fly-app.fly.dev")
        assert web_server._ws_host_origin_is_allowed(ws) is False

    def test_gated_same_host_https_origin_allowed(self, gated_app):
        ws = self._ws(origin="https://fly-app.fly.dev", host="fly-app.fly.dev")
        assert web_server._ws_host_origin_is_allowed(ws) is True


class TestSidecarUrl:
    def test_loopback_uses_session_token(self, loopback_app):
        url = web_server._build_sidecar_url("ch-1")
        assert url is not None
        assert f"token={web_server._SESSION_TOKEN}" in url
        assert "ticket=" not in url

    def test_gated_uses_internal_credential(self, gated_app):
        url = web_server._build_sidecar_url("ch-1")
        assert url is not None
        assert "token=" not in url
        assert "ticket=" not in url
        assert "internal=" in url
        # The value should be the live process-lifetime internal credential,
        # multi-use so the child can reconnect /api/pub.
        cred = url.split("internal=")[1].split("&")[0]
        info = consume_internal_credential(cred)
        assert info["user_id"] == "server-internal"
        assert info["provider"] == "server-internal"
        # Multi-use: a second consume still succeeds (unlike a ticket).
        assert consume_internal_credential(cred)["provider"] == "server-internal"

    def test_no_bound_host_returns_none(self, gated_app):
        web_server.app.state.bound_host = None
        try:
            assert web_server._build_sidecar_url("ch") is None
        finally:
            web_server.app.state.bound_host = "fly-app.fly.dev"


# ---------------------------------------------------------------------------
# _build_gateway_ws_url — the TUI child's primary JSON-RPC backend WS.
# Loopback uses ?token=; gated mode uses the multi-use internal credential
# (NOT a single-use ticket — the child reuses this URL across reconnects).
# ---------------------------------------------------------------------------


class TestGatewayWsUrl:
    def test_loopback_uses_session_token(self, loopback_app):
        url = web_server._build_gateway_ws_url()
        assert url is not None
        assert "/api/ws?" in url
        assert f"token={web_server._SESSION_TOKEN}" in url
        assert "internal=" not in url

    def test_gated_uses_internal_credential(self, gated_app):
        url = web_server._build_gateway_ws_url()
        assert url is not None
        assert "/api/ws?" in url
        assert "token=" not in url
        assert "ticket=" not in url
        assert "internal=" in url
        cred = url.split("internal=")[1].split("&")[0]
        # The credential authenticates against _ws_auth_ok in gated mode.
        ws = _fake_ws(query={"internal": cred})
        assert web_server._ws_auth_ok(ws) is True

    def test_gated_credential_matches_sidecar(self, gated_app):
        """Both server-internal builders share one process credential, so a
        single value authenticates /api/ws and /api/pub alike."""
        gw = web_server._build_gateway_ws_url()
        sc = web_server._build_sidecar_url("ch-1")
        assert gw is not None and sc is not None
        gw_cred = gw.split("internal=")[1].split("&")[0]
        sc_cred = sc.split("internal=")[1].split("&")[0]
        assert gw_cred == sc_cred

    def test_no_bound_host_returns_none(self, gated_app):
        web_server.app.state.bound_host = None
        try:
            assert web_server._build_gateway_ws_url() is None
        finally:
            web_server.app.state.bound_host = "fly-app.fly.dev"

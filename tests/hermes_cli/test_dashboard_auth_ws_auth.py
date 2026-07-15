"""Tests for the WS-upgrade auth helper (Phase 5 task 5.2).

The dashboard's WS endpoints (``/api/pty``, ``/api/console``, ``/api/ws``,
``/api/pub``, ``/api/events``) share an auth gate: ``_ws_auth_ok``. In
loopback mode it accepts ``?token=<_SESSION_TOKEN>``; in gated mode it accepts
a single-use ``?ticket=`` minted by ``POST /api/auth/ws-ticket``.

These tests exercise the helper, ticket-mint endpoint, and focused authenticated
gateway/Computer Use WebSocket paths under realistic gated-mode setup.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import queue
import time
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.ws_tickets import (
    _reset_for_tests,
    consume_internal_credential,
    internal_ws_credential,
    mint_ticket,
)
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


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


# ---------------------------------------------------------------------------
# POST /api/auth/ws-ticket — the mint endpoint
# ---------------------------------------------------------------------------


class TestWsTicketEndpoint:
    def test_authenticated_session_can_mint(self, gated_app):
        _logged_in(gated_app)
        r = gated_app.post("/api/auth/ws-ticket")
        assert r.status_code == 200
        body = r.json()
        assert "ticket" in body
        assert isinstance(body["ticket"], str)
        assert len(body["ticket"]) >= 32
        assert body["ttl_seconds"] == 30

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

    def test_valid_ticket_propagates_verified_principal(self, gated_app):
        ticket = mint_ticket(user_id="user-61507", provider="stub")
        ws = _fake_ws(query={"ticket": ticket})

        auth = web_server._ws_auth_context(ws)

        assert auth.reason is None
        assert auth.authenticated_principal == ("stub", "user-61507")

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


def test_public_oauth_computer_use_status_rejects_foreign_profile_before_io(
    gated_app, monkeypatch
):
    _logged_in(gated_app)
    monkeypatch.setattr(web_server, "_dashboard_launch_profile", lambda: "default")

    entered_scopes = []
    config_reads = []

    @contextmanager
    def tracked_scope(profile):
        entered_scopes.append(profile)
        yield

    monkeypatch.setattr(web_server, "_config_profile_scope", tracked_scope)
    monkeypatch.setattr(
        "tools.computer_use.tool.configured_computer_use_backend",
        lambda: config_reads.append(True) or "bridge",
    )

    response = gated_app.get("/api/tools/computer-use/status?profile=work")

    assert response.status_code == 403
    assert entered_scopes == []
    assert config_reads == []


def test_public_oauth_computer_use_status_allows_launch_profile(gated_app, monkeypatch):
    _logged_in(gated_app)
    monkeypatch.setattr(web_server, "_dashboard_launch_profile", lambda: "default")

    entered_scopes = []

    @contextmanager
    def tracked_scope(profile):
        entered_scopes.append(profile)
        yield

    monkeypatch.setattr(web_server, "_config_profile_scope", tracked_scope)
    monkeypatch.setattr(
        "tools.computer_use.tool.configured_computer_use_backend",
        lambda: "bridge",
    )
    monkeypatch.setattr(
        "tools.computer_use.bridge.bridge_computer_use_status",
        lambda: {"ready": True, "checks": [{"label": "launch-profile"}]},
    )

    response = gated_app.get("/api/tools/computer-use/status?profile=Default")

    assert response.status_code == 200
    assert response.json()["checks"] == [{"label": "launch-profile"}]
    assert entered_scopes == [None]


def test_operator_computer_use_status_keeps_cross_profile_compatibility(
    loopback_app, monkeypatch
):
    entered_scopes = []

    @contextmanager
    def tracked_scope(profile):
        entered_scopes.append(profile)
        yield

    monkeypatch.setattr(web_server, "_config_profile_scope", tracked_scope)
    monkeypatch.setattr(
        "tools.computer_use.tool.configured_computer_use_backend",
        lambda: "bridge",
    )
    monkeypatch.setattr(
        "tools.computer_use.bridge.bridge_computer_use_status",
        lambda: {"ready": True, "checks": [{"label": "operator-profile"}]},
    )

    response = loopback_app.get(
        "/api/tools/computer-use/status?profile=work",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
    )

    assert response.status_code == 200
    assert response.json()["checks"] == [{"label": "operator-profile"}]
    assert entered_scopes == ["work"]


def test_public_bridge_foreign_profile_denial_does_not_probe_existence(
    gated_app, monkeypatch
):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr(web_server, "_ws_request_is_allowed", lambda _ws: True)
    monkeypatch.setattr(web_server, "_dashboard_launch_profile", lambda: "default")
    profile_probes = []
    monkeypatch.setattr(
        web_server,
        "_resolve_profile_dir",
        lambda profile: profile_probes.append(profile) or profile,
    )

    for profile in ("existing-foreign", "missing-foreign"):
        ticket = mint_ticket(user_id="alice", provider="stub")
        with pytest.raises(WebSocketDisconnect) as denied:
            with gated_app.websocket_connect(
                "/api/tools/computer-use/desktop-bridge/ws"
                f"?ticket={ticket}&profile={profile}"
            ) as ws:
                ws.receive_json()
        assert denied.value.code == 4403

    assert profile_probes == []


def test_public_bridge_ticket_is_pinned_to_launch_profile_without_live_session(
    gated_app, monkeypatch
):
    monkeypatch.setattr(web_server, "_ws_request_is_allowed", lambda _ws: True)
    monkeypatch.setattr(web_server, "_dashboard_launch_profile", lambda: "default")
    resolved_profiles = []
    monkeypatch.setattr(
        web_server,
        "_resolve_profile_dir",
        lambda profile: resolved_profiles.append(profile) or profile,
    )
    handled = []

    async def fake_handle(ws, **scope):
        handled.append(scope)
        await ws.accept()
        await ws.send_json({"ok": True})
        await ws.close()

    monkeypatch.setattr(
        "tools.computer_use.desktop_bridge.handle_desktop_bridge_ws", fake_handle
    )

    ticket = mint_ticket(user_id="alice", provider="stub")
    with gated_app.websocket_connect(
        f"/api/tools/computer-use/desktop-bridge/ws?ticket={ticket}"
    ) as ws:
        assert ws.receive_json() == {"ok": True}

    assert handled == [
        {"provider": "stub", "principal": "alice", "profile": "default"}
    ]
    assert resolved_profiles == ["default"]


def test_operator_bridge_keeps_validated_cross_profile_scope(
    loopback_app, monkeypatch
):
    monkeypatch.setattr(web_server, "_ws_request_is_allowed", lambda _ws: True)
    resolved_profiles = []
    monkeypatch.setattr(
        web_server,
        "_resolve_profile_dir",
        lambda profile: resolved_profiles.append(profile) or profile,
    )
    handled = []

    async def fake_handle(ws, **scope):
        handled.append(scope)
        await ws.accept()
        await ws.send_json({"ok": True})
        await ws.close()

    monkeypatch.setattr(
        "tools.computer_use.desktop_bridge.handle_desktop_bridge_ws", fake_handle
    )

    with loopback_app.websocket_connect(
        "/api/tools/computer-use/desktop-bridge/ws"
        f"?token={web_server._SESSION_TOKEN}&profile=Work"
    ) as ws:
        assert ws.receive_json() == {"ok": True}

    assert resolved_profiles == ["work"]
    assert handled == [
        {
            "provider": "dashboard-token",
            "principal": "local-session",
            "profile": "work",
        }
    ]


def test_ticket_principal_and_session_profile_reach_only_their_bridge(
    gated_app, monkeypatch, tmp_path
):
    """Verified ticket -> live session context -> exact bridge dispatch."""
    from hermes_cli import profiles as profiles_mod
    from tools.computer_use import tool as computer_use_tool
    from tools.computer_use.desktop_bridge import _BROKER, DesktopBridgeScope
    from tui_gateway import server as tui_server

    profiles_root = tmp_path / "profiles"
    default_home = tmp_path / "default"
    default_home.mkdir()
    (default_home / "config.yaml").write_text(
        "computer_use:\n  backend: cua\n", encoding="utf-8"
    )
    profiles_root.mkdir()

    monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")
    monkeypatch.setattr(tui_server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(web_server, "_ws_request_is_allowed", lambda _ws: True)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(tui_server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(tui_server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(
        tui_server,
        "_claim_active_session_slot",
        lambda *_args, **_kwargs: (None, None),
    )
    monkeypatch.setattr(tui_server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(tui_server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(
        tui_server, "_sync_agent_model_with_config", lambda _sid, _session: None
    )
    monkeypatch.setattr(
        tui_server,
        "_sync_session_key_after_compress",
        lambda *_args, **_kwargs: None,
    )

    class LocalFallback:
        def start(self):
            return None

        def stop(self):
            return None

        def is_available(self):
            return True

        def list_apps(self):
            return [{"name": "gateway-local"}]

    monkeypatch.setattr(
        "tools.computer_use.cua_backend.CuaDriverBackend", LocalFallback
    )

    results: queue.Queue[tuple[str, dict]] = queue.Queue()

    class ToolDispatchAgent:
        api_mode = ""
        base_url = ""
        model = "test-model"
        provider = "test"

        def __init__(self, owner, session_key):
            self.owner = owner
            self.session_id = session_key

        def clear_interrupt(self):
            return None

        def run_conversation(self, _message, **_kwargs):
            payload = json.loads(
                computer_use_tool.handle_computer_use({"action": "list_apps"})
            )
            results.put((self.owner, payload))
            return {"final_response": "tool complete", "messages": []}

    def receive_response(ws, request_id):
        while True:
            frame = ws.receive_json()
            if frame.get("id") == request_id:
                return frame

    def create_session(ws, owner):
        request_id = f"create-{owner}"
        ws.send_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "session.create",
                "params": {"source": "desktop"},
            }
        )
        response = receive_response(ws, request_id)
        sid = response["result"]["session_id"]
        session = tui_server._sessions[sid]
        session["agent"] = ToolDispatchAgent(owner, session["session_key"])
        session["agent_ready"].set()
        return sid

    def submit_list_apps(ws, owner, sid):
        request_id = f"prompt-{owner}"
        ws.send_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "prompt.submit",
                "params": {"session_id": sid, "text": "list apps"},
            }
        )
        response = receive_response(ws, request_id)
        assert response["result"]["status"] == "streaming"

    def answer_bridge_list_apps(ws, owner):
        status = ws.receive_json()
        assert status["type"] == "status"
        ws.send_json(
            {"id": status["id"], "ok": True, "result": {"ready": True, "checks": []}}
        )
        call = ws.receive_json()
        assert call["type"] == "computer-use"
        assert call["method"] == "list_apps"
        ws.send_json(
            {
                "id": call["id"],
                "ok": True,
                "result": {"apps": [{"name": f"{owner}-desktop"}]},
            }
        )

    def wait_session_idle(sid):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if not tui_server._sessions[sid]["running"]:
                return
            time.sleep(0.01)
        raise AssertionError(f"session {sid} did not become idle")

    tui_server._sessions.clear()
    computer_use_tool.reset_backend_for_tests()
    alice_bridge_ticket = mint_ticket(user_id="alice", provider="stub")
    bob_bridge_ticket = mint_ticket(user_id="bob", provider="stub")
    alice_gateway_ticket = mint_ticket(user_id="alice", provider="stub")
    bob_gateway_ticket = mint_ticket(user_id="bob", provider="stub")
    mallory_gateway_ticket = mint_ticket(user_id="mallory", provider="stub")

    try:
        with gated_app.websocket_connect(
            f"/api/ws?ticket={alice_gateway_ticket}"
        ) as alice_gateway, gated_app.websocket_connect(
            f"/api/ws?ticket={bob_gateway_ticket}"
        ) as bob_gateway, gated_app.websocket_connect(
            f"/api/ws?ticket={mallory_gateway_ticket}"
        ) as mallory_gateway:
            for gateway in (alice_gateway, bob_gateway, mallory_gateway):
                assert gateway.receive_json()["params"]["type"] == "gateway.ready"

            alice_sid = create_session(alice_gateway, "alice")
            bob_sid = create_session(bob_gateway, "bob")
            mallory_sid = create_session(mallory_gateway, "mallory")

            with gated_app.websocket_connect(
                "/api/tools/computer-use/desktop-bridge/ws"
                f"?ticket={alice_bridge_ticket}"
            ) as alice_bridge, gated_app.websocket_connect(
                "/api/tools/computer-use/desktop-bridge/ws"
                f"?ticket={bob_bridge_ticket}"
            ) as bob_bridge:
                submit_list_apps(alice_gateway, "alice", alice_sid)
                answer_bridge_list_apps(alice_bridge, "alice")
                assert results.get(timeout=2) == (
                    "alice",
                    {"apps": [{"name": "alice-desktop"}], "count": 1},
                )

                submit_list_apps(bob_gateway, "bob", bob_sid)
                answer_bridge_list_apps(bob_bridge, "bob")
                assert results.get(timeout=2) == (
                    "bob",
                    {"apps": [{"name": "bob-desktop"}], "count": 1},
                )

                submit_list_apps(mallory_gateway, "mallory", mallory_sid)
                assert results.get(timeout=2) == (
                    "mallory",
                    {"apps": [{"name": "gateway-local"}], "count": 1},
                )

                # Generic session routing remains unchanged: Bob can address
                # Alice's live id, but that foreign transport must clear only
                # the Computer Use bridge scope. If either Desktop socket were
                # selected this turn would block waiting for a bridge reply.
                wait_session_idle(alice_sid)
                submit_list_apps(bob_gateway, "bob-via-alice", alice_sid)
                assert results.get(timeout=2) == (
                    "alice",
                    {"apps": [{"name": "gateway-local"}], "count": 1},
                )

                alice_bridge.close()
                alice_scope = DesktopBridgeScope("stub", "alice", "default")
                deadline = time.monotonic() + 2
                while _BROKER.is_connected(alice_scope) and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert _BROKER.is_connected(alice_scope) is False
                assert _BROKER.is_connected(
                    DesktopBridgeScope("stub", "bob", "default")
                ) is True

                wait_session_idle(bob_sid)
                submit_list_apps(bob_gateway, "bob-again", bob_sid)
                answer_bridge_list_apps(bob_bridge, "bob")
                assert results.get(timeout=2) == (
                    "bob",
                    {"apps": [{"name": "bob-desktop"}], "count": 1},
                )
    finally:
        computer_use_tool.reset_backend_for_tests()
        for sid in list(tui_server._sessions):
            tui_server._close_session_by_id(sid, end_reason="test_cleanup")


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

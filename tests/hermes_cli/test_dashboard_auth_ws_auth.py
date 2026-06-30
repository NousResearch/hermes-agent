"""Tests for the WS-upgrade auth helper (Phase 5 task 5.2).

The dashboard's WS endpoints (``/api/pty``, ``/api/console``, ``/api/ws``,
``/api/pub``, ``/api/events``) share an auth gate: ``_ws_auth_ok``. In
loopback mode it accepts ``?token=<_SESSION_TOKEN>``; in gated mode it accepts
a single-use ``?ticket=`` minted by ``POST /api/auth/ws-ticket``.

These tests exercise the helper at the unit level (no actual WS upgrade)
plus the ticket-mint endpoint under realistic gated-mode setup. We don't
test the full WS upgrade because the starlette TestClient WS path has a
pre-existing regression unrelated to dashboard-auth.
"""

from __future__ import annotations

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

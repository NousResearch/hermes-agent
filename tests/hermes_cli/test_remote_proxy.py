"""Tests for ``hermes dashboard proxy`` (hermes_cli/remote_proxy.py).

The proxy's contract: only /api/* crosses the tunnel, lifecycle routes are
denied by default with an audit trail, hop-by-hop headers are stripped, and
everything else streams through with the upstream's auth untouched.
"""

from __future__ import annotations



import httpx
import pytest
from fastapi.testclient import TestClient

from hermes_cli.remote_proxy import (
    DEFAULT_DENY_ROUTES,
    classify_request,
    create_proxy_app,
    filtered_headers,
    resolve_deny_routes,
)


# ---------------------------------------------------------------------------
# Pure classification / policy
# ---------------------------------------------------------------------------

class TestClassification:
    @pytest.mark.parametrize("path", [
        "/api/sessions",
        "/api/config",
        "/api/ws",
        "/api/actions/doctor/status",
    ])
    def test_api_routes_forward(self, path):
        assert classify_request(path, DEFAULT_DENY_ROUTES) == "forward"

    @pytest.mark.parametrize("path", sorted(DEFAULT_DENY_ROUTES))
    def test_lifecycle_routes_denied_by_default(self, path):
        assert classify_request(path, DEFAULT_DENY_ROUTES) == "deny"

    @pytest.mark.parametrize("path", [
        "/",
        "/index.html",
        "/assets/index.js",
        "/health",
        "/login",
        "/auth/callback",
        "/apiary",  # prefix trick must not match /api
    ])
    def test_non_api_is_not_found(self, path):
        assert classify_request(path, DEFAULT_DENY_ROUTES) == "not_found"

    def test_trailing_slash_cannot_bypass_deny(self):
        assert classify_request("/api/hermes/update/", DEFAULT_DENY_ROUTES) == "deny"

    def test_update_check_stays_reachable(self):
        # The read-only update CHECK endpoint is not on the deny-list; only
        # the mutating spawn is.
        assert classify_request("/api/hermes/update/check", DEFAULT_DENY_ROUTES) == "forward"


class TestOverrides:
    def test_allow_route_removes_from_default_deny(self):
        routes = resolve_deny_routes(allow=["/api/gateway/restart"])
        assert classify_request("/api/gateway/restart", routes) == "forward"
        assert classify_request("/api/hermes/update", routes) == "deny"

    def test_deny_route_adds(self):
        routes = resolve_deny_routes(deny=["/api/sessions/import"])
        assert classify_request("/api/sessions/import", routes) == "deny"

    def test_normalization_tolerates_slashes(self):
        routes = resolve_deny_routes(allow=["api/gateway/restart/"])
        assert classify_request("/api/gateway/restart", routes) == "forward"


class TestHeaderFiltering:
    def test_hop_by_hop_and_host_are_dropped(self):
        kept = filtered_headers([
            ("Host", "evil.example"),
            ("Connection", "keep-alive"),
            ("Transfer-Encoding", "chunked"),
            ("Upgrade", "h2c"),
            ("Authorization", "Bearer token"),
            ("X-Hermes-Session", "abc"),
        ])
        names = [k.lower() for k, _ in kept]
        assert "host" not in names
        assert "connection" not in names
        assert "transfer-encoding" not in names
        assert "upgrade" not in names
        # End-to-end headers (auth included — the upstream enforces it) pass.
        assert ("Authorization", "Bearer token") in kept
        assert ("X-Hermes-Session", "abc") in kept


# ---------------------------------------------------------------------------
# ASGI behavior (mocked upstream)
# ---------------------------------------------------------------------------

def test_forwarded_request_streams_upstream_response(tmp_path, monkeypatch):
    seen: list[httpx.Request] = []

    def _upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"ok": True},
            headers=[
                ("X-Upstream", "yes"),
                ("Set-Cookie", "access=one; Path=/"),
                ("Set-Cookie", "refresh=two; Path=/"),
                ("Connection", "keep-alive"),
            ],
        )

    real_async_client = httpx.AsyncClient

    def _mock_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(_upstream)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client)

    app = create_proxy_app(
        upstream="http://127.0.0.1:9119",
        deny_log=tmp_path / "denied.log",
    )
    with TestClient(app) as client:
        response = client.get("/api/sessions?limit=5", headers={"X-Hermes-Session": "abc"})

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["X-Upstream"] == "yes"
    assert response.headers.get_list("set-cookie") == [
        "access=one; Path=/",
        "refresh=two; Path=/",
    ]
    # Hop-by-hop from upstream must not be forwarded back.
    assert "connection" not in {k.lower() for k in response.headers}
    # The upstream saw the pass-through auth header and the query string.
    assert seen[0].headers["X-Hermes-Session"] == "abc"
    assert str(seen[0].url).endswith("/api/sessions?limit=5")


def test_proxy_preserves_real_loopback_backend_token_auth(tmp_path, monkeypatch):
    import hermes_cli.web_server as web_server

    real_async_client = httpx.AsyncClient

    def _in_process_backend(*args, **kwargs):
        kwargs["transport"] = httpx.ASGITransport(app=web_server.app)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _in_process_backend)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)

    app = create_proxy_app(
        upstream="http://127.0.0.1:9119",
        deny_log=tmp_path / "denied.log",
    )
    with TestClient(app) as client:
        denied = client.get("/api/config/raw")
        allowed = client.get(
            "/api/config/raw",
            headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
        )

    assert denied.status_code == 401
    assert allowed.status_code == 200


def test_denied_route_is_403_and_audited(tmp_path, monkeypatch):


    called: list[str] = []

    real_async_client = httpx.AsyncClient

    def _mock_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda req: called.append(str(req.url)) or httpx.Response(200)
        )
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client)

    deny_log = tmp_path / "denied.log"
    app = create_proxy_app(upstream="http://127.0.0.1:9119", deny_log=deny_log)
    with TestClient(app) as client:
        response = client.post("/api/hermes/update")

    assert response.status_code == 403
    assert called == [], "denied requests must never reach the upstream"
    logged = deny_log.read_text(encoding="utf-8")
    assert "DENIED POST /api/hermes/update" in logged


def test_non_api_is_404_and_never_reaches_upstream(tmp_path, monkeypatch):
    called: list[str] = []

    real_async_client = httpx.AsyncClient

    def _mock_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda req: called.append(str(req.url)) or httpx.Response(200)
        )
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client)

    app = create_proxy_app(
        upstream="http://127.0.0.1:9119", deny_log=tmp_path / "denied.log"
    )
    with TestClient(app) as client:
        assert client.get("/").status_code == 404
        assert client.get("/index.html").status_code == 404
        assert client.get("/assets/index-abc.js").status_code == 404
        assert client.get("/login").status_code == 404
        assert client.get("/auth/callback").status_code == 404
    assert called == [], "the SPA and static assets must never cross the proxy"


def test_denied_websocket_is_policy_closed(tmp_path):
    from starlette.websockets import WebSocketDisconnect as StarletteWSDisconnect

    app = create_proxy_app(
        upstream="http://127.0.0.1:9119",
        deny_routes=resolve_deny_routes(deny=["/api/console"]),
        deny_log=tmp_path / "denied.log",
    )
    with TestClient(app) as client:
        with client.websocket_connect("/api/console") as ws:
            with pytest.raises(StarletteWSDisconnect) as excinfo:
                ws.receive_text()
            assert excinfo.value.code == 4403

    logged = (tmp_path / "denied.log").read_text(encoding="utf-8")
    assert "DENIED WS /api/console" in logged

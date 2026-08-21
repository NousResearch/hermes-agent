"""Regression tests for signed short-lived download tickets.

Covers the mint/verify round trip, expiry rejection, tampered signature and
tampered path rejection, missing-param rejection, non-``/api/files/download``
path scoping, the middleware bypass (a valid ticket passes the cookie gate,
an invalid one is 401), the file-identity binding (a file replaced at the
same path after minting no longer satisfies the ticket), and the mint
endpoint's public-URL handling behind a reverse proxy.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from hermes_cli.dashboard_auth.download_ticket import (
    build_download_url,
    verify_download_ticket,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _request_for(url: str, *, app=None) -> Request:
    """Build a starlette Request whose URL matches ``url``."""
    parts = urlsplit(url)
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": parts.scheme or "http",
        "path": parts.path,
        "raw_path": parts.path.encode(),
        "query_string": parts.query.encode(),
        "headers": [],
        "server": (parts.hostname or "127.0.0.1", parts.port or 80),
        "client": ("127.0.0.1", 12345),
    }
    if app is not None:
        scope["app"] = app
    return Request(scope)


def _swap_query(url: str, **changes) -> str:
    """Return ``url`` with the given query params replaced."""
    parts = urlsplit(url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q.update(changes)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment)
    )


def _drop_query(url: str, key: str) -> str:
    """Return ``url`` with the given query param removed."""
    parts = urlsplit(url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q.pop(key, None)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment)
    )


@pytest.fixture
def ticket_file(tmp_path):
    f = tmp_path / "doc.docx"
    f.write_bytes(b"x" * 100)
    return f


# ---------------------------------------------------------------------------
# mint / verify round trip
# ---------------------------------------------------------------------------


def test_mint_verify_round_trip(ticket_file):
    url = build_download_url("http://127.0.0.1:9119", str(ticket_file))
    assert url.startswith("http://127.0.0.1:9119/api/files/download?")
    assert verify_download_ticket(_request_for(url)) is True


def test_base_url_trailing_slash_normalised(ticket_file):
    url = build_download_url("http://127.0.0.1:9119/", str(ticket_file))
    assert url.startswith("http://127.0.0.1:9119/api/files/download?")


def test_url_carries_path_exp_sig_params(ticket_file):
    url = build_download_url("http://h", str(ticket_file))
    q = dict(parse_qsl(urlsplit(url).query))
    assert q["path"] == str(ticket_file)
    assert int(q["exp"]) > 0
    assert q["sig"]


def test_expired_ticket_rejected(ticket_file):
    url = build_download_url("http://h", str(ticket_file), ttl_seconds=-10)
    assert verify_download_ticket(_request_for(url)) is False


def test_tampered_signature_rejected(ticket_file):
    url = build_download_url("http://h", str(ticket_file))
    tampered = _swap_query(url, sig="A" * 40)
    assert verify_download_ticket(_request_for(tampered)) is False


def test_tampered_path_rejected(ticket_file):
    url = build_download_url("http://h", str(ticket_file))
    tampered = _swap_query(url, path="/etc/passwd")
    assert verify_download_ticket(_request_for(tampered)) is False


@pytest.mark.parametrize("dropped", ["path", "exp", "sig"])
def test_missing_param_rejected(ticket_file, dropped):
    url = build_download_url("http://h", str(ticket_file))
    assert verify_download_ticket(_request_for(_drop_query(url, dropped))) is False


def test_bad_exp_value_rejected(ticket_file):
    url = build_download_url("http://h", str(ticket_file))
    assert verify_download_ticket(_request_for(_swap_query(url, exp="soon"))) is False


def test_non_download_path_never_verifies(ticket_file):
    """The same ticket query on a different route must not authenticate."""
    url = build_download_url("http://h", str(ticket_file))
    parts = urlsplit(url)
    req = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/files/read",
            "raw_path": b"/api/files/read",
            "query_string": parts.query.encode(),
            "headers": [],
            "server": ("127.0.0.1", 80),
            "client": ("127.0.0.1", 1),
        }
    )
    assert verify_download_ticket(req) is False


def test_mint_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_download_url("http://h", str(tmp_path / "nope.docx"))


# ---------------------------------------------------------------------------
# file-identity binding (TOCTOU)
# ---------------------------------------------------------------------------


def test_file_replaced_at_same_path_invalidates_ticket(ticket_file):
    url = build_download_url("http://h", str(ticket_file))
    assert verify_download_ticket(_request_for(url)) is True
    # Same path, different content/size → different identity → ticket dead.
    ticket_file.write_bytes(b"y" * 999)
    assert verify_download_ticket(_request_for(url)) is False


def test_deleted_file_invalidates_ticket(ticket_file):
    url = build_download_url("http://h", str(ticket_file))
    ticket_file.unlink()
    assert verify_download_ticket(_request_for(url)) is False


# ---------------------------------------------------------------------------
# middleware integration: the gate bypass is exactly as narrow as designed
# ---------------------------------------------------------------------------


class _FakeApp:
    def __init__(self, auth_required: bool = True):
        self.state = SimpleNamespace(auth_required=auth_required)


async def _passthrough(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True}, status_code=200)


def test_middleware_bypasses_gate_with_valid_ticket(ticket_file):
    from hermes_cli.dashboard_auth.middleware import gated_auth_middleware

    url = build_download_url("http://127.0.0.1:9119", str(ticket_file))
    req = _request_for(url, app=_FakeApp())
    resp = asyncio.run(gated_auth_middleware(req, _passthrough))
    assert resp.status_code == 200


def test_middleware_401s_with_invalid_ticket(ticket_file):
    from hermes_cli.dashboard_auth.middleware import gated_auth_middleware

    url = build_download_url("http://127.0.0.1:9119", str(ticket_file))
    tampered = _swap_query(url, sig="A" * 40)
    req = _request_for(tampered, app=_FakeApp())
    resp = asyncio.run(gated_auth_middleware(req, _passthrough))
    assert resp.status_code == 401


def test_middleware_ignores_ticket_on_other_routes(ticket_file):
    from hermes_cli.dashboard_auth.middleware import gated_auth_middleware

    url = build_download_url("http://127.0.0.1:9119", str(ticket_file))
    parts = urlsplit(url)
    req = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/files/read",
            "raw_path": b"/api/files/read",
            "query_string": parts.query.encode(),
            "headers": [],
            "server": ("127.0.0.1", 9119),
            "client": ("127.0.0.1", 1),
            "app": _FakeApp(),
        }
    )
    resp = asyncio.run(gated_auth_middleware(req, _passthrough))
    # Ticket params on a non-download route must NOT authenticate: 401.
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# endpoint-level: mint honours the declared public URL (reverse proxy)
# ---------------------------------------------------------------------------


@pytest.fixture
def loopback_client():
    from fastapi.testclient import TestClient

    from hermes_cli import web_server

    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client, web_server
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _mint(client, web_server, path):
    return client.get(
        "/api/files/download-ticket",
        params={"path": str(path)},
        headers={
            web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN,
        },
    )


def test_mint_endpoint_uses_public_url_when_declared(
    loopback_client, ticket_file, monkeypatch, tmp_path
):
    client, web_server = loopback_client
    monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_URL", "https://media.example.com/hermes")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    r = _mint(client, web_server, ticket_file)
    assert r.status_code == 200, r.text
    assert r.json()["url"].startswith(
        "https://media.example.com/hermes/api/files/download?"
    )


def test_mint_endpoint_falls_back_to_request_base(
    loopback_client, ticket_file, monkeypatch, tmp_path
):
    client, web_server = loopback_client
    monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    r = _mint(client, web_server, ticket_file)
    assert r.status_code == 200, r.text
    assert r.json()["url"].startswith("http://127.0.0.1:9119/api/files/download?")


# ---------------------------------------------------------------------------
# end-to-end: full gate + route with a real ticket (no cookie, no token)
# ---------------------------------------------------------------------------


def test_gated_download_serves_file_with_valid_ticket_only(
    ticket_file, monkeypatch, tmp_path
):
    from fastapi.testclient import TestClient

    from hermes_cli import web_server

    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = True
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    try:
        client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
        url = build_download_url("http://127.0.0.1:9119", str(ticket_file))
        parts = urlsplit(url)

        # No cookie, no token, no ticket → the gate 401s.
        r = client.get("/api/files/download", params={"path": str(ticket_file)})
        assert r.status_code == 401, r.text

        # Valid ticket → bypasses the gate and the file is streamed.
        r = client.get("/api/files/download", params=dict(parse_qsl(parts.query)))
        assert r.status_code == 200, r.text
        assert r.content == b"x" * 100

        # Tampered ticket → back to 401.
        r = client.get(
            "/api/files/download",
            params=dict(parse_qsl(urlsplit(_swap_query(url, sig="B" * 40)).query)),
        )
        assert r.status_code == 401, r.text
    finally:
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_required

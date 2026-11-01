"""Real ASGI checks of the Webapp's same-origin gate for cookie-authenticated writes.

Behind the sign-in gate, the gateway's Webapp (``ui_surface == "webapp"``) refuses a write that
carries the session cookie unless it comes from the Webapp's own origin; the standalone dashboard
keeps main's rule (``test_dashboard_auth_middleware``). The pre-parser upload limit on the same
routes is covered in ``test_webapp_upload_limit.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from starlette import formparsers

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE, SESSION_RT_COOKIE
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def upload_app(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", "0.0.0.0", raising=False)
    monkeypatch.setattr(web_server.app.state, "trusted_public_hosts", frozenset(), raising=False)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(web_server.app.state, "listener_public_url", "", raising=False)
    if hasattr(web_server.app.state, "webapp_home"):
        monkeypatch.delattr(web_server.app.state, "webapp_home")
    clear_providers()
    register_provider(StubAuthProvider())
    # Observe the actual parser's open files, retaining references so GC cannot
    # disguise a missing close. No parser or upload implementation is mocked.
    spools = []
    real_spool = formparsers.SpooledTemporaryFile

    def track_spool(*args, **kwargs):
        spool = real_spool(*args, dir=tmp_path, **kwargs)
        spools.append(spool)
        return spool

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", track_spool)
    yield home, spools
    for spool in spools:
        spool.close()
    clear_providers()


async def _login(client):
    response = await client.get("/auth/login?provider=stub", follow_redirects=True)
    assert response.status_code != 401
    return next(v for k, v in client.cookies.items() if k.endswith(SESSION_AT_COOKIE))


@pytest.mark.parametrize(
    "headers",
    [
        # No Origin, but Fetch Metadata names an initiating page: not a native client.
        {"Sec-Fetch-Site": "same-site"},
        {"Sec-Fetch-Site": "cross-site"},
        *(
            {"Origin": origin}
            for origin in (
                "null",
                "https://evil.example.com",
                "http://dashboard.example.com",
                "https://dashboard.example.com:8443",
                "file://",
                "https://dashboard.example.com/path",
                "https://dashboard.example.com:bad",
                "https://dashboard.example.com, https://evil.example.com",
            )
        ),
    ],
)
def test_cookie_upload_requires_exact_http_origin(upload_app, headers):
    home, spools = upload_app

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="https://dashboard.example.com"
        ) as client:
            await _login(client)
            response = await client.post(
                "/api/chat/file-upload",
                files={"file": ("csrf.txt", b"unwanted")},
                headers=headers,
            )
            assert response.status_code == 403, response.text
            assert not (home / "uploads").exists()
            assert not spools  # Refused before FastAPI opens a spool.

    asyncio.run(run())


def test_cookie_same_origin_and_refresh_uploads_work(upload_app):
    """The Webapp's own page uploads with its cookie session, before and after a refresh. (A bearer
    token skips the origin check; on the Webapp only a native app's own sign-in has one that works,
    covered in ``tests/gateway/test_webapp_cookie_writes.py``.)"""
    home, spools = upload_app

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="https://dashboard.example.com"
        ) as client:
            await _login(client)
            for headers in (
                {"Origin": "https://dashboard.example.com:443"},
                {"Origin": "https://dashboard.example.com", "Sec-Fetch-Site": "same-origin"},
            ):
                response = await client.post(
                    "/api/chat/file-upload", files={"file": ("ok.txt", b"allowed")}, headers=headers
                )
                assert response.status_code == 200, response.text
                assert Path(response.json()["path"]).read_bytes() == b"allowed"
            # Refresh-only cookies must face the same Origin check before rotating.
            for cookie in list(client.cookies.jar):
                if cookie.name.endswith(SESSION_AT_COOKIE):
                    client.cookies.delete(cookie.name, domain=cookie.domain, path=cookie.path)
            assert any(k.endswith(SESSION_RT_COOKIE) for k in client.cookies)
            response = await client.post(
                "/api/chat/file-upload",
                files={"file": ("bad.txt", b"bad")},
                headers={"Origin": "https://dashboard.example.com:8443", "Sec-Fetch-Site": "same-site"},
            )
            assert response.status_code == 403
            response = await client.post(
                "/api/chat/file-upload",
                files={"file": ("refresh.txt", b"allowed")},
                headers={"Origin": "https://dashboard.example.com"},
            )
            assert response.status_code == 200, response.text
        assert len(list((home / "uploads").iterdir())) == 3
        assert all(s.closed for s in spools)

    asyncio.run(run())


def test_cookie_unsafe_methods_and_logout_but_not_auth_bootstrap(upload_app):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="https://dashboard.example.com"
        ) as client:
            await _login(client)
            for method in ("POST", "PUT", "PATCH", "DELETE"):
                response = await client.request(
                    method, "/api/auth/me", headers={"Origin": "https://evil.example.com"}
                )
                assert response.status_code == 403
                response = await client.request(
                    method, "/api/auth/me", headers={"Origin": "https://dashboard.example.com"}
                )
                assert response.status_code == 405  # Authorized; method isn't routed.
            response = await client.get("/api/auth/me")
            assert response.status_code == 200
            for path, expected in (("/auth/callback", 405), ("/auth/password-login", 422)):
                response = await client.post(path, headers={"Origin": "null"})
                assert response.status_code == expected  # Bootstrap, not CSRF.
            response = await client.post(
                "/api/auth/ws-ticket",
                headers=[("Origin", "https://dashboard.example.com"), ("Origin", "https://evil.example.com")],
            )
            assert response.status_code == 403
            response = await client.post(
                "/auth/logout", headers={"Origin": "null", "Sec-Fetch-Site": "same-site"}
            )
            assert response.status_code == 403
            response = await client.post("/auth/logout", headers={"Origin": "https://dashboard.example.com"})
            assert response.status_code == 302
            assert (await client.get("/api/auth/me")).status_code == 401
            # Logout stays public, including for a stale native bearer.
            response = await client.post("/auth/logout", headers={"Authorization": "Bearer expired"})
            assert response.status_code == 302

    asyncio.run(run())


@pytest.mark.parametrize("public_url", [False, True])
def test_cookie_upload_accepts_trusted_proxy_origin_only(upload_app, monkeypatch, public_url):
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1")
    monkeypatch.setattr(web_server.app.state, "trusted_public_hosts", frozenset({"dashboard.example.com"}))

    async def run():
        app = ProxyHeadersMiddleware(web_server.app, trusted_hosts=["127.0.0.1"])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://dashboard.example.com:8443"
        ) as client:
            await _login(client)
            if public_url:
                monkeypatch.setattr(web_server.app.state, "listener_public_url",
                                    "https://dashboard.example.com:8443/hermes")
            host = "127.0.0.1:9119" if public_url else "dashboard.example.com:8443"
            headers = {"X-Forwarded-Prefix": "/hermes", "Host": host, "X-Forwarded-Proto": "https"}
            for origin, expected in (
                ("https://dashboard.example.com:8443", 200),
                ("https://dashboard.example.com", 403),
                ("http://dashboard.example.com:8443", 403),
                ("https://evil.example.com:8443", 403),
            ):
                response = await client.post(
                    "http://dashboard.example.com:8443/api/chat/file-upload",
                    files={"file": ("proxy.txt", b"proxy")},
                    headers={
                        **headers,
                        "Origin": origin,
                        "Cookie": "; ".join(f"{k}={v}" for k, v in client.cookies.items()),
                    },
                )
                assert response.status_code == expected, response.text
            # Raw forwarded headers are not an authority declaration. Untrusted
            # peers cannot upgrade the ASGI scheme or invent an accepted host.
            if not public_url:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app, client=("192.0.2.10", 1234)),
                    base_url="http://dashboard.example.com:8443",
                ) as untrusted:
                    response = await untrusted.post(
                        "/api/chat/file-upload",
                        files={"file": ("bad.txt", b"bad")},
                        headers={
                            **headers,
                            "Origin": "https://dashboard.example.com:8443",
                            "Cookie": "; ".join(f"{k}={v}" for k, v in client.cookies.items()),
                        },
                    )
                    assert response.status_code == 403

    asyncio.run(run())

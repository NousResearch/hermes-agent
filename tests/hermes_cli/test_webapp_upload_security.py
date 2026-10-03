"""Real ASGI regressions for cookie CSRF and pre-parser upload limits (PR #93508)."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from starlette import formparsers

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE, SESSION_RT_COOKIE
from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES
from hermes_cli.web_upload_limit import CHAT_FILE_BODY_MAX_BYTES
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
    "origin",
    [
        None,
        "null",
        "https://evil.example.com",
        "http://dashboard.example.com",
        "https://dashboard.example.com:8443",
        "file://",
        "https://dashboard.example.com/path",
        "https://dashboard.example.com:bad",
        "https://dashboard.example.com, https://evil.example.com",
    ],
)
def test_cookie_upload_requires_exact_http_origin(upload_app, origin):
    home, spools = upload_app

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="https://dashboard.example.com"
        ) as client:
            await _login(client)
            response = await client.post(
                "/api/chat/file-upload",
                files={"file": ("csrf.txt", b"unwanted")},
                headers={} if origin is None else {"Origin": origin},
            )
            assert response.status_code == 403, response.text
            assert not (home / "uploads").exists()
            assert not spools  # Refused before FastAPI opens a spool.

    asyncio.run(run())


def test_cookie_same_origin_refresh_and_bearer_uploads_work(upload_app):
    home, spools = upload_app

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="https://dashboard.example.com"
        ) as client:
            token = await _login(client)
            for headers in (
                {"Origin": "https://dashboard.example.com:443"},
                {"Authorization": f"Bearer {token}"},
                {"Authorization": f"Bearer {token}", "Origin": "null"},
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
            response = await client.post("/api/chat/file-upload", files={"file": ("bad.txt", b"bad")})
            assert response.status_code == 403
            response = await client.post(
                "/api/chat/file-upload",
                files={"file": ("refresh.txt", b"allowed")},
                headers={"Origin": "https://dashboard.example.com"},
            )
            assert response.status_code == 200, response.text
        assert len(list((home / "uploads").iterdir())) == 4
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
            response = await client.post("/auth/logout")
            assert response.status_code == 403
            response = await client.post("/auth/logout", headers={"Origin": "https://dashboard.example.com"})
            assert response.status_code == 302
            assert (await client.get("/api/auth/me")).status_code == 401
            # Logout has always been public (including stale native bearers).
            response = await client.post("/auth/logout", headers={"Authorization": "Bearer expired"})
            assert response.status_code == 302

    asyncio.run(run())


@pytest.mark.parametrize(
    "declared_length,root_path", [(None, ""), ("1", ""), ("oversize", ""), (None, "/hermes")]
)
def test_upload_bound_stops_parser_and_closes_spools(upload_app, monkeypatch, declared_length, root_path):
    home, spools = upload_app
    # Include a small complete part before the large unfinished file: both open
    # handles must be closed when receive refuses a later chunk.
    boundary = b"upload-security-boundary"
    prefix = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="extra"; filename="first.txt"\r\n\r\nfirst\r\n'
    )
    prefix += (
        b"--" + boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="large.bin"\r\n\r\n'
    )
    body_cap = CHAT_FILE_BODY_MAX_BYTES
    received = 0

    async def body():
        nonlocal received
        received += len(prefix)
        yield prefix
        while received < body_cap:
            chunk = b"x" * min(64 * 1024, body_cap - received)
            received += len(chunk)
            yield chunk
        received += 1
        yield b"x"
        pytest.fail("body was consumed after the first over-limit byte")
        yield b"\r\n--" + boundary + b"--\r\n"

    from hermes_cli.web_routers import uploads

    def handler_must_not_run(*args, **kwargs):
        pytest.fail("upload handler ran after oversized multipart parsing")

    monkeypatch.setattr(uploads, "_resolve_upload_generation", handler_must_not_run)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="https://dashboard.example.com"
        ) as client:
            token = await _login(client)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "multipart/form-data; boundary=" + boundary.decode(),
            }
            if declared_length is not None:
                headers["Content-Length"] = (
                    str(body_cap + 1) if declared_length == "oversize" else declared_length
                )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=web_server.app, root_path=root_path),
                base_url="https://dashboard.example.com",
            ) as uploader:
                response = await uploader.post(
                    root_path + "/api/chat/file-upload", content=body(), headers=headers
                )
            assert response.status_code == 413, response.text
            assert not (home / "uploads").exists()
            if declared_length == "oversize":
                assert received == 0
                assert not spools
            else:
                assert received == body_cap + 1
                assert len(spools) == 2
                assert all(s.closed for s in spools)
                assert any(s._rolled for s in spools)

    asyncio.run(run())


@pytest.mark.parametrize("chunked", [False, True])
def test_exact_cap_file_survives_multipart_framing(upload_app, chunked):
    _, spools = upload_app

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="https://dashboard.example.com"
        ) as client:
            await _login(client)
            payload = b"x" * WEBAPP_ATTACHMENT_MAX_BYTES
            request = client.build_request(
                "POST",
                "/api/chat/file-upload",
                files={"file": ("exact.bin", payload)},
                headers={"Origin": "https://dashboard.example.com"},
            )
            if chunked:
                encoded = request.read()

                async def chunks():
                    for start in range(0, len(encoded), 64 * 1024):
                        yield encoded[start : start + 64 * 1024]

                del request.headers["Content-Length"]
                response = await client.post(
                    "/api/chat/file-upload", content=chunks(), headers=request.headers
                )
            else:
                response = await client.send(request)
            assert response.status_code == 200, response.text
            assert response.json()["size"] == len(payload)
            staged = Path(response.json()["path"])
            assert staged.read_bytes() == payload
            assert spools and all(s.closed for s in spools)

    asyncio.run(run())


@pytest.mark.parametrize("public_url", [False, True])
def test_cookie_upload_accepts_trusted_proxy_origin_only(upload_app, monkeypatch, public_url):
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    home, _ = upload_app
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1")
    monkeypatch.setattr(web_server.app.state, "trusted_public_hosts", frozenset({"dashboard.example.com"}))

    async def run():
        app = ProxyHeadersMiddleware(web_server.app, trusted_hosts=["127.0.0.1"])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://dashboard.example.com:8443"
        ) as client:
            await _login(client)
            if public_url:
                (home / "config.yaml").write_text(
                    "dashboard:\n  public_url: https://dashboard.example.com:8443/hermes\n"
                )
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


def test_other_upload_routes_keep_their_larger_caps(upload_app, monkeypatch):
    from hermes_cli.web_routers.files import _CHAT_IMAGE_UPLOAD_MAX_BYTES

    home, spools = upload_app
    root = home / "managed"
    root.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="https://dashboard.example.com"
        ) as client:
            token = await _login(client)
            headers = {"Authorization": f"Bearer {token}"}
            payload = b"x" * web_server._MANAGED_FILE_MAX_BYTES
            response = await client.post(
                "/api/files/upload-stream",
                files={"file": ("large.bin", BytesIO(payload))},
                data={"path": str(root / "large.bin")},
                headers=headers,
            )
            assert response.status_code == 200, response.text
            assert (root / "large.bin").read_bytes() == payload
            payload = b"\x89PNG\r\n\x1a\n" + b"x" * (_CHAT_IMAGE_UPLOAD_MAX_BYTES - 8)
            response = await client.post(
                "/api/chat/image-upload",
                json={
                    "filename": "large.png",
                    "data_url": "data:image/png;base64," + base64.b64encode(payload).decode(),
                },
                headers=headers,
            )
            assert response.status_code == 200, response.text
            assert Path(response.json()["path"]).read_bytes() == payload
            assert all(s.closed for s in spools)

    asyncio.run(run())

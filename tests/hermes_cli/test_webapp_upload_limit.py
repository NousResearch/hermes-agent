"""Real ASGI regressions for the Webapp's pre-parser upload limit (``hermes_cli/web_upload_limit.py``).

The cookie Origin gate on the same uploads is covered in ``test_webapp_upload_security.py``.
"""

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
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE
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
    # The Webapp behind sign-in: the only surface that caps upload bodies before parsing them.
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(web_server.app.state, "listener_public_url", "", raising=False)
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
    """Sign the browser in on the Webapp; returns what its own page's requests carry (its session
    cookies and its Origin), so another client can send the same request. On the Webapp a browser
    login's token never works alone as a bearer token."""
    response = await client.get("/auth/login?provider=stub", follow_redirects=True)
    assert response.status_code != 401
    assert any(k.endswith(SESSION_AT_COOKIE) for k in client.cookies)
    return {"Cookie": "; ".join(f"{k}={v}" for k, v in client.cookies.items()),
            "Origin": "https://dashboard.example.com"}


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

    monkeypatch.setattr(uploads, "resolve_upload_home", handler_must_not_run)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="https://dashboard.example.com"
        ) as client:
            page = await _login(client)
            headers = {
                **page,
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


@pytest.mark.parametrize("declared_length", [None, "1", "oversize"])
def test_an_oversized_image_paste_is_refused_before_its_body_is_read(upload_app, monkeypatch, declared_length):
    """``/api/chat/image-upload`` carries the image as a base64 data URL in a JSON body. A body
    longer than the largest image it may carry is refused with 413 at once when Content-Length
    declares it, and otherwise near the cap while it streams in: never read whole into memory,
    and the handler never runs."""
    from hermes_cli.web_routers import files

    home, _spools = upload_app
    # The base64 length of the largest image the route accepts, plus room for the JSON around it.
    allowed = 4 * -(-files._CHAT_IMAGE_UPLOAD_MAX_BYTES // 3) + 1024 * 1024
    prefix = b'{"filename": "huge.png", "data_url": "data:image/png;base64,'
    received = 0

    async def body():
        nonlocal received
        received += len(prefix)
        yield prefix
        while received <= allowed:
            chunk = b"A" * (256 * 1024)
            received += len(chunk)
            yield chunk
        pytest.fail("image upload body was read past its cap")
        yield b'"}'

    def handler_must_not_run(*args, **kwargs):
        pytest.fail("image upload handler ran on an oversized body")

    monkeypatch.setattr(files, "resolve_upload_home", handler_must_not_run)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="https://dashboard.example.com"
        ) as client:
            page = await _login(client)
            headers = {**page, "Content-Type": "application/json"}
            if declared_length is not None:
                headers["Content-Length"] = str(2 * allowed) if declared_length == "oversize" else declared_length
            response = await client.post("/api/chat/image-upload", content=body(), headers=headers)
            assert response.status_code == 413, response.text
            assert not (home / "cache").exists()
            if declared_length == "oversize":
                assert received == 0

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
            headers = await _login(client)
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

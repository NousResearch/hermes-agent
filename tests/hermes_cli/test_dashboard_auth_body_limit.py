"""Body budgets apply before parsing public auth requests, not to uploads."""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from starlette.types import Message, Scope

from hermes_cli import web_server


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    monkeypatch.setattr(web_server.app.state, "bound_host", None, raising=False)
    application = FastAPI(middleware=web_server.app.user_middleware)
    application.state.auth_required = True
    application.state.calls = 0

    async def endpoint(payload: dict[str, str]) -> dict[str, str]:
        application.state.calls += 1
        return payload

    for path in ("/auth/password-login", "/auth/native/token", "/auth/native/refresh",
                 "/api/upload"):
        application.post(path)(endpoint)
    yield application


async def _request(
    app: FastAPI, path: str, chunks: list[bytes], headers: list[tuple[bytes, bytes]],
) -> tuple[list[Message], int]:
    scope: Scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": [(b"host", b"testserver"),
        (b"content-type", b"application/json"), *headers],
        "client": ("127.0.0.1", 1234), "server": ("testserver", 80),
    }
    reads = 0
    messages: list[Message] = []

    async def receive() -> Message:
        nonlocal reads
        if reads == len(chunks):
            return {"type": "http.disconnect"}
        body = chunks[reads]
        reads += 1
        return {"type": "http.request", "body": body, "more_body": reads < len(chunks)}

    async def send(message: Message) -> None:
        messages.append(message)

    await app(scope, receive, send)
    return messages, reads


@pytest.mark.asyncio
async def test_oversized_content_length_rejected_before_handler(app: FastAPI) -> None:
    messages, reads = await _request(
        app, "/auth/password-login", [b'{}'], [(b"content-length", b"65537")])
    assert messages[0]["status"] == 413
    assert app.state.calls == 0
    assert reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/auth/password-login", "/auth/native/token", "/auth/native/refresh"])
@pytest.mark.parametrize("headers", [[], [(b"content-length", b"2")]], ids=["chunked", "understated"])
async def test_stream_cut_off_before_remaining_body(
    app: FastAPI, path: str, headers: list[tuple[bytes, bytes]],
) -> None:
    messages, reads = await _request(
        app, path, [b' ', b' ' * 65536, b'{}'], headers)
    assert messages[0]["status"] == 413
    assert app.state.calls == 0
    assert reads == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [2, 65536], ids=["small", "at-limit"])
async def test_normal_login_reaches_handler(app: FastAPI, size: int) -> None:
    messages, _ = await _request(app, "/auth/password-login", [b' ' * (size - 2) + b'{}'], [])
    assert messages[0]["status"] == 200
    assert app.state.calls == 1


@pytest.mark.asyncio
async def test_non_auth_upload_is_not_limited(app: FastAPI) -> None:
    app.state.auth_required = False
    messages, _ = await _request(
        app, "/api/upload", [b' ' * 65536 + b'{}'],
        [(web_server._SESSION_HEADER_NAME.lower().encode(), web_server._SESSION_TOKEN.encode()),
         (b"content-length", b"65538")])
    assert messages[0]["status"] == 200
    assert app.state.calls == 1


@pytest.mark.asyncio
async def test_non_public_path_rejected_without_reading_body(app: FastAPI) -> None:
    messages, reads = await _request(app, "/api/upload", [b' ' * 65537], [])
    assert messages[0]["status"] == 401
    assert reads == 0
    assert app.state.calls == 0

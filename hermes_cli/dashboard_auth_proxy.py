"""Authenticated HTTPS reverse proxy for a local Hermes dashboard.

This module is intentionally small and operational: it lets an operator keep
``hermes dashboard`` bound to loopback while exposing a separate, fail-closed
HTTPS listener protected by Basic auth.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Mapping
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask


DEFAULT_UPSTREAM = "http://127.0.0.1:9119"
DEFAULT_LISTEN_HOST = "0.0.0.0"
DEFAULT_LISTEN_PORT = 9443

AUTH_REALM = "Hermes Dashboard"
USERNAME_ENV = "HERMES_DASHBOARD_PROXY_USERNAME"
PASSWORD_FILE_ENV = "HERMES_DASHBOARD_PROXY_PASSWORD_FILE"
UPSTREAM_ENV = "HERMES_DASHBOARD_PROXY_UPSTREAM"
HOST_ENV = "HERMES_DASHBOARD_PROXY_HOST"
PORT_ENV = "HERMES_DASHBOARD_PROXY_PORT"
CERT_FILE_ENV = "HERMES_DASHBOARD_PROXY_CERT_FILE"
KEY_FILE_ENV = "HERMES_DASHBOARD_PROXY_KEY_FILE"

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


@asynccontextmanager
async def _lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    fastapi_app.state.client = httpx.AsyncClient(follow_redirects=False, timeout=None)
    try:
        yield
    finally:
        client = getattr(fastapi_app.state, "client", None)
        if client is not None:
            await client.aclose()


app = FastAPI(title="Hermes Dashboard Auth Proxy", lifespan=_lifespan)


def _password_from_file() -> str | None:
    path = os.environ.get(PASSWORD_FILE_ENV, "").strip()
    if not path:
        return None
    try:
        first_line = Path(path).read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    password = first_line.strip()
    return password or None


def _configured_credentials() -> tuple[str, str] | None:
    username = os.environ.get(USERNAME_ENV, "").strip()
    password = _password_from_file()
    if not username or not password:
        return None
    return username, password


def _basic_auth_valid(header: str, credentials: tuple[str, str]) -> bool:
    prefix = "Basic "
    if not header.startswith(prefix):
        return False
    try:
        decoded = base64.b64decode(header[len(prefix):], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    if ":" not in decoded:
        return False
    username, password = decoded.split(":", 1)
    expected_username, expected_password = credentials
    return hmac.compare_digest(username, expected_username) and hmac.compare_digest(
        password,
        expected_password,
    )


def _auth_challenge() -> Response:
    return Response(
        status_code=401,
        headers={
            "WWW-Authenticate": f'Basic realm="{AUTH_REALM}", charset="UTF-8"',
            **SECURITY_HEADERS,
        },
    )


def _with_security_headers(response: Response) -> Response:
    for key, value in SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    return response


def _upstream_base() -> str:
    return os.environ.get(UPSTREAM_ENV, DEFAULT_UPSTREAM).rstrip("/")


def _upstream_url(path: str, query: str) -> str:
    url = f"{_upstream_base()}/{path}"
    if query:
        url = f"{url}?{query}"
    return url


def _proxy_headers(request_headers: Mapping[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    upstream_host = urlsplit(_upstream_base()).netloc
    for key, value in request_headers.items():
        lower = key.lower()
        if lower in HOP_BY_HOP_HEADERS or lower in {"authorization", "host"}:
            continue
        headers[key] = value
    if upstream_host:
        headers["Host"] = upstream_host
    return headers


def _response_headers(upstream_headers: Mapping[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in upstream_headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        headers[key] = value
    headers.update(SECURITY_HEADERS)
    return headers


async def _client() -> httpx.AsyncClient:
    client = getattr(app.state, "client", None)
    if client is None:
        client = httpx.AsyncClient(follow_redirects=False, timeout=None)
        app.state.client = client
    return client


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    credentials = _configured_credentials()
    if credentials is None:
        return _with_security_headers(
            JSONResponse(
                status_code=503,
                content={"detail": "Dashboard proxy credentials are not configured."},
            ),
        )
    if not _basic_auth_valid(request.headers.get("authorization", ""), credentials):
        return _auth_challenge()
    response = await call_next(request)
    return _with_security_headers(response)


async def _proxy_to_upstream(request: Request, path: str) -> Response:
    client = await _client()
    body = await request.body()
    upstream_request = client.build_request(
        request.method,
        _upstream_url(path, request.url.query),
        headers=_proxy_headers(request.headers),
        content=body,
    )
    try:
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={"detail": f"Dashboard upstream unavailable: {exc.__class__.__name__}"},
            headers=SECURITY_HEADERS,
        )

    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=_response_headers(upstream_response.headers),
        background=BackgroundTask(upstream_response.aclose),
    )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy(request: Request, path: str = "") -> Response:
    return await _proxy_to_upstream(request, path)


def main() -> None:
    cert_file = os.environ.get(CERT_FILE_ENV)
    key_file = os.environ.get(KEY_FILE_ENV)
    if not cert_file or not key_file:
        raise SystemExit(
            f"{CERT_FILE_ENV} and {KEY_FILE_ENV} are required for HTTPS.",
        )
    uvicorn.run(
        app,
        host=os.environ.get(HOST_ENV, DEFAULT_LISTEN_HOST),
        port=int(os.environ.get(PORT_ENV, str(DEFAULT_LISTEN_PORT))),
        ssl_certfile=cert_file,
        ssl_keyfile=key_file,
        log_level=os.environ.get("HERMES_DASHBOARD_PROXY_LOG_LEVEL", "warning"),
    )


if __name__ == "__main__":
    main()

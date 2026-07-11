from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import urllib.parse

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from plugins.platforms.telegram.mini_app.app import RequestBodyLimitMiddleware
from plugins.platforms.telegram.mini_app.app import app as mini_app
from plugins.platforms.telegram.mini_app.app import auth
from plugins.platforms.telegram.mini_app.auth import SESSION_COOKIE

BOT_TOKEN = "mini-app-test-token"
OWNER_ID = "123456789"


def _signed_init_data(
    *, user_id: str = OWNER_ID, auth_date: int | None = None, query_id: str = "query-1"
) -> str:
    values = {
        "auth_date": str(int(time.time()) if auth_date is None else auth_date),
        "query_id": query_id,
        "user": json.dumps(
            {"id": int(user_id), "first_name": "Owner"}, separators=(",", ":")
        ),
    }
    data_check = "\n".join(f"{key}={value}" for key, value in sorted(values.items()))
    secret = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
    values["hash"] = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    return urllib.parse.urlencode(values)


@pytest.fixture(autouse=True)
def _reset_auth_state(monkeypatch):
    monkeypatch.setattr(auth, "bot_token", BOT_TOKEN)
    monkeypatch.setattr(auth, "allowed_users_raw", OWNER_ID)
    monkeypatch.setattr(auth, "public_url", "https://mini.hermes.test")
    auth.sessions.clear()
    auth.init_uses.clear()
    auth.exchange_events.clear()


@pytest.fixture
def client():
    with TestClient(
        mini_app,
        base_url="https://mini.hermes.test",
        headers={"User-Agent": "mini-app-security-tests/1"},
    ) as value:
        yield value


def _exchange(client: TestClient, init_data: str):
    return client.post("/api/auth/session", headers={"X-Telegram-Init-Data": init_data})


def test_private_surface_is_exactly_read_only():
    methods = {
        (route.path, method)
        for route in mini_app.routes
        if isinstance(route, APIRoute)
        for method in route.methods or set()
    }
    private_gets = {
        path for path, method in methods if path.startswith("/api/") and method == "GET"
    }
    assert private_gets == {
        "/api/me",
        "/api/status",
        "/api/live-usage",
        "/api/swarm/board",
        "/api/memory",
        "/api/sessions",
        "/api/tools/toolsets",
        "/api/skills",
    }
    writes = {
        (path, method)
        for path, method in methods
        if path.startswith("/api/") and method in {"POST", "PUT", "PATCH", "DELETE"}
    }
    assert writes == {
        ("/api/auth/session", "POST"),
        ("/api/auth/session", "DELETE"),
    }
    assert not any(
        "pty" in path.lower() or "dashboard" in path.lower() for path, _ in methods
    )


def test_exchange_issues_opaque_secure_cookie_and_init_data_is_one_use(client):
    init_data = _signed_init_data()
    response = _exchange(client, init_data)
    assert response.status_code == 200
    assert len(response.json()["csrf_token"]) >= 32
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=strict" in cookie
    raw = client.cookies.get(SESSION_COOKIE)
    assert raw and OWNER_ID not in raw and "hash=" not in raw

    other = TestClient(
        mini_app,
        base_url="https://mini.hermes.test",
        headers={"User-Agent": "mini-app-security-tests/1"},
    )
    assert _exchange(other, init_data).status_code == 401


def test_exchange_rejects_expired_invalid_and_non_owner_launches(client):
    expired = _signed_init_data(auth_date=int(time.time()) - 301, query_id="expired")
    assert _exchange(client, expired).status_code == 401
    assert _exchange(client, "auth_date=1&hash=forged").status_code == 401
    assert (
        _exchange(
            client, _signed_init_data(user_id="999", query_id="stranger")
        ).status_code
        == 403
    )


def test_init_data_is_not_a_bearer_token_and_logout_requires_csrf(client):
    init_data = _signed_init_data(query_id="csrf")
    assert (
        client.get("/api/me", headers={"X-Telegram-Init-Data": init_data}).status_code
        == 401
    )
    exchange = _exchange(client, init_data)
    assert exchange.status_code == 200
    assert client.get("/api/me").status_code == 200
    assert client.delete("/api/auth/session").status_code == 403
    assert (
        client.delete(
            "/api/auth/session",
            headers={
                "X-CSRF-Token": exchange.json()["csrf_token"],
                "Origin": "https://mini.hermes.test",
                "Sec-Fetch-Site": "same-origin",
            },
        ).status_code
        == 200
    )
    assert client.get("/api/me").status_code == 401


def test_errors_and_private_responses_are_not_cached(client):
    response = client.get("/api/me")
    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["strict-transport-security"].startswith("max-age=")


def test_logout_origin_uses_stable_config_not_spoofable_request_host(client):
    exchange = _exchange(client, _signed_init_data(query_id="stable-origin"))
    csrf = exchange.json()["csrf_token"]
    response = client.delete(
        "/api/auth/session",
        headers={
            "Host": "attacker.example",
            "Origin": "https://attacker.example",
            "X-CSRF-Token": csrf,
            "Sec-Fetch-Site": "same-origin",
        },
    )
    assert response.status_code == 403


def test_logout_fails_closed_without_stable_public_origin(client, monkeypatch):
    exchange = _exchange(client, _signed_init_data(query_id="missing-origin"))
    monkeypatch.setattr(auth, "public_url", "")
    response = client.delete(
        "/api/auth/session",
        headers={
            "Origin": "https://mini.hermes.test",
            "X-CSRF-Token": exchange.json()["csrf_token"],
            "Sec-Fetch-Site": "same-origin",
        },
    )
    assert response.status_code == 503


def test_streamed_oversized_body_is_rejected_with_security_headers(client):
    response = client.post(
        "/api/auth/session",
        content=(b"x" * (64 * 1024) for _ in range(17)),
        headers={"X-Telegram-Init-Data": _signed_init_data(query_id="oversized")},
    )
    assert response.status_code == 413
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_slow_request_body_hits_secured_deadline():
    async def exercise():
        async def downstream(scope, receive, send):
            raise AssertionError("timed-out body reached downstream app")

        async def receive():
            await asyncio.sleep(1)
            return {"type": "http.request", "body": b"", "more_body": False}

        messages = []

        async def send(message):
            messages.append(message)

        middleware = RequestBodyLimitMiddleware(
            downstream, max_bytes=1024, timeout_seconds=0.01, max_concurrent=1
        )
        await middleware(
            {"type": "http", "method": "POST", "path": "/", "headers": []},
            receive,
            send,
        )
        return messages

    messages = asyncio.run(exercise())
    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    headers = {key.decode().lower(): value.decode() for key, value in start["headers"]}
    assert start["status"] == 408
    assert headers["cache-control"] == "no-store"
    assert headers["strict-transport-security"].startswith("max-age=")


def test_concurrency_is_bounded_before_slow_body_buffering():
    async def exercise():
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_receive_called = False

        async def downstream(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = RequestBodyLimitMiddleware(
            downstream,
            max_bytes=1024,
            timeout_seconds=1,
            max_concurrent=1,
            admission_timeout_seconds=0.01,
        )
        scope = {"type": "http", "method": "POST", "path": "/", "headers": []}

        async def first_receive():
            first_started.set()
            await release_first.wait()
            return {"type": "http.request", "body": b"", "more_body": False}

        async def second_receive():
            nonlocal second_receive_called
            second_receive_called = True
            return {"type": "http.request", "body": b"", "more_body": False}

        first_messages: list[dict] = []
        second_messages: list[dict] = []

        async def send_first(message):
            first_messages.append(message)

        async def send_second(message):
            second_messages.append(message)

        first = asyncio.create_task(middleware(scope, first_receive, send_first))
        await first_started.wait()
        await middleware(scope, second_receive, send_second)
        assert second_receive_called is False
        assert (
            next(
                message
                for message in second_messages
                if message["type"] == "http.response.start"
            )["status"]
            == 503
        )

        release_first.set()
        await first
        assert (
            next(
                message
                for message in first_messages
                if message["type"] == "http.response.start"
            )["status"]
            == 204
        )

    asyncio.run(exercise())


def test_exchange_and_session_requests_are_rate_limited(client, monkeypatch):
    for index in range(6):
        current = TestClient(
            mini_app,
            base_url="https://mini.hermes.test",
            headers={"User-Agent": f"rate-test/{index}"},
        )
        assert (
            _exchange(current, _signed_init_data(query_id=f"rate-{index}")).status_code
            == 200
        )
    extra = TestClient(mini_app, base_url="https://mini.hermes.test")
    assert _exchange(extra, _signed_init_data(query_id="rate-extra")).status_code == 429

    auth.exchange_events.clear()
    exchange = _exchange(client, _signed_init_data(query_id="private-rate"))
    assert exchange.status_code == 200
    monkeypatch.setattr(
        "plugins.platforms.telegram.mini_app.app.PRIVATE_SESSION_RATE_LIMIT", 2
    )
    assert [client.get("/api/me").status_code for _ in range(3)] == [200, 200, 429]

import hashlib
import hmac
import json
import time
from urllib.parse import quote

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from hermes_cli.telegram_miniapp.server import MiniAppSettings, create_app


BOT_TOKEN = "123456:test-token"
USER = {"id": 777, "first_name": "Andrey", "username": "daiver"}


def build_init_data(*, bot_token=BOT_TOKEN, user=USER, auth_date=None):
    auth_date = int(time.time()) if auth_date is None else auth_date
    fields = {
        "auth_date": str(auth_date),
        "query_id": "AAHdF6IQAAAAAN0XohDhrOrc",
        "user": json.dumps(user, separators=(",", ":")),
    }
    data_check = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    return "&".join(f"{quote(key)}={quote(value)}" for key, value in fields.items())


def make_client(settings=None, status_snapshot=None, now=1_700_000_100):
    settings = settings or MiniAppSettings(
        bot_token=BOT_TOKEN,
        allowed_users={"777"},
        now=lambda: now,
    )
    app = create_app(settings=settings, status_provider=lambda: status_snapshot or {"ok": True, "gateway": {}})
    return TestClient(app, base_url="http://127.0.0.1:9120")


def auth_client(client):
    response = client.post(
        "/api/auth/telegram",
        json={"initData": build_init_data(auth_date=1_700_000_000)},
        headers={"origin": "http://127.0.0.1:5175"},
    )
    assert response.status_code == 200
    return response


def test_public_health_and_ready_do_not_leak_internals():
    client = make_client()

    for path in ("/healthz", "/readyz"):
        response = client.get(path)
        assert response.status_code == 200
        body = response.text
        assert "telegram-miniapp" in body
        assert "/Volumes/Diver Pro/hermes" not in body
        assert BOT_TOKEN not in body
        assert "pid" not in body.lower()


def test_route_inventory_and_forbidden_routes_are_absent():
    app = create_app(
        settings=MiniAppSettings(bot_token=BOT_TOKEN, allowed_users={"777"}, now=lambda: 1_700_000_100),
        status_provider=lambda: {"ok": True, "gateway": {}},
    )
    client = TestClient(app, base_url="http://127.0.0.1:9120")
    routes = {
        (next(iter(route.methods - {"HEAD"})), route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for _ in [0]
    }
    assert routes == {
        ("GET", "/healthz"),
        ("GET", "/readyz"),
        ("POST", "/api/auth/telegram"),
        ("POST", "/api/logout"),
        ("GET", "/api/me"),
        ("GET", "/api/status"),
        ("GET", "/api/approvals"),
    }

    assert client.get("/does-not-exist").status_code == 404
    for path in ["/api/actions/restart", "/api/restart", "/api/config/model", "/api/model/switch"]:
        assert client.post(path).status_code == 404


def test_auth_sets_httponly_api_scoped_cookie_without_raw_init_data():
    client = make_client()
    response = auth_client(client)

    cookies = response.headers.get_list("set-cookie")
    assert any("hermes_tma_session=" in cookie for cookie in cookies)
    cookie = next(cookie for cookie in cookies if "hermes_tma_session=" in cookie)
    assert "HttpOnly" in cookie
    assert "Path=/api" in cookie
    assert "Secure" not in cookie
    assert build_init_data(auth_date=1_700_000_000) not in response.text
    assert BOT_TOKEN not in response.text


def test_auth_fails_closed_for_empty_allowlist():
    settings = MiniAppSettings(bot_token=BOT_TOKEN, allowed_users=set(), now=lambda: 1_700_000_100)
    client = make_client(settings=settings)

    response = client.post(
        "/api/auth/telegram",
        json={"initData": build_init_data(auth_date=1_700_000_000)},
        headers={"origin": "http://127.0.0.1:5175"},
    )

    assert response.status_code == 403
    assert BOT_TOKEN not in response.text


def test_me_and_status_require_auth_then_return_safe_schema():
    snapshot = {
        "ok": True,
        "updated_at": "2026-07-01T10:00:00+00:00",
        "hermes_home": "configured",
        "gateway": {
            "running": True,
            "state": "running",
            "busy": False,
            "drainable": True,
            "active_agents": 0,
            "restart_requested": False,
        },
        "miniapp": {"mode": "local-read-only", "actions_enabled": False, "public_exposure": False},
    }
    client = make_client(status_snapshot=snapshot)

    assert client.get("/api/me").status_code == 401
    assert client.get("/api/status").status_code == 401

    auth_client(client)
    me = client.get("/api/me")
    status = client.get("/api/status")

    assert me.status_code == 200
    assert me.json()["user"]["id"] == "777"
    assert status.status_code == 200
    assert status.json() == snapshot
    serialized = status.text
    assert "/Volumes/Diver Pro/hermes" not in serialized
    assert BOT_TOKEN not in serialized


def test_approvals_require_auth_then_return_safe_read_only_queue():
    client = make_client()

    assert client.get("/api/approvals").status_code == 401

    auth_client(client)
    response = client.get("/api/approvals")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert len(body["items"]) == 2
    first = body["items"][0]
    assert set(first) == {"id", "title", "source", "risk", "summary", "requested_at", "status", "checks"}
    assert first["risk"] in {"read_only", "critical"}
    assert first["status"] in {"waiting", "blocked"}
    serialized = response.text
    assert "/Volumes/Diver Pro/hermes" not in serialized
    assert BOT_TOKEN not in serialized
    assert "TELEGRAM_BOT_TOKEN" not in serialized
    assert "pid" not in serialized.lower()


def test_logout_clears_only_miniapp_session():
    client = make_client()
    auth_client(client)

    response = client.post("/api/logout", headers={"origin": "http://127.0.0.1:5175"})

    assert response.status_code == 200
    assert "Max-Age=0" in response.headers.get("set-cookie", "")
    assert client.get("/api/me").status_code == 401


def test_origin_host_and_cors_fail_closed():
    client = make_client()
    init_data = build_init_data(auth_date=1_700_000_000)

    bad_origin = client.post(
        "/api/auth/telegram",
        json={"initData": init_data},
        headers={"origin": "https://evil.example"},
    )
    assert bad_origin.status_code == 403

    bad_host = client.post(
        "/api/auth/telegram",
        json={"initData": init_data},
        headers={"origin": "http://127.0.0.1:5175", "host": "attacker.com"},
    )
    assert bad_host.status_code == 400

    cors = client.options(
        "/api/status",
        headers={
            "origin": "https://evil.example",
            "access-control-request-method": "GET",
        },
    )
    assert cors.headers.get("access-control-allow-origin") is None

    bad_get_origin = client.get("/api/status", headers={"origin": "https://evil.example"})
    assert bad_get_origin.status_code == 403


def test_non_loopback_bind_fails_closed():
    settings = MiniAppSettings(bot_token=BOT_TOKEN, allowed_users={"777"}, host="0.0.0.0")
    client = make_client(settings=settings)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert "/Volumes/Diver Pro/hermes" not in response.text
    assert BOT_TOKEN not in response.text

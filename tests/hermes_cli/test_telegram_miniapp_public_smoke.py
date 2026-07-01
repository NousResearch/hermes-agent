import hashlib
import hmac
import json
import sys
import time
import types
from urllib.parse import quote

from fastapi.testclient import TestClient

from hermes_cli.telegram_miniapp.config import settings_from_config
from hermes_cli.telegram_miniapp.server import MiniAppSettings, create_app


BOT_TOKEN = "123456:test-token"
USER = {"id": 777, "first_name": "Andrey", "username": "daiver"}
SMOKE_ORIGIN = "https://hermes-smoke.example"


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


def public_settings(**overrides):
    values = {
        "bot_token": BOT_TOKEN,
        "allowed_users": {"777"},
        "public_smoke": True,
        "public_base_url": SMOKE_ORIGIN,
        "cors_allowed_origins": {SMOKE_ORIGIN},
        "now": lambda: 1_700_000_100,
    }
    values.update(overrides)
    return MiniAppSettings(**values)


def make_client(settings=None):
    app = create_app(
        settings=settings or public_settings(),
        status_provider=lambda: {
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
            "miniapp": {"mode": "https-smoke", "actions_enabled": False, "public_exposure": True},
        },
    )
    return TestClient(app, base_url=SMOKE_ORIGIN)


def auth_client(client, *, origin=SMOKE_ORIGIN):
    response = client.post(
        "/api/auth/telegram",
        json={"initData": build_init_data(auth_date=1_700_000_000)},
        headers={"origin": origin, "host": "hermes-smoke.example"},
    )
    assert response.status_code == 200
    return response


def test_public_smoke_requires_explicit_https_origin_and_safe_inputs():
    for settings in [
        public_settings(public_base_url="http://hermes-smoke.example"),
        public_settings(public_base_url="https://user:pass@hermes-smoke.example"),
        public_settings(public_base_url="https://hermes-smoke.example/path"),
        public_settings(public_base_url="https://hermes-smoke.example?x=1"),
        public_settings(public_base_url="https://hermes-smoke.example#frag"),
        public_settings(allowed_users=set()),
        public_settings(bot_token=""),
        public_settings(enable_actions=True),
    ]:
        response = make_client(settings).get("/readyz", headers={"host": "hermes-smoke.example"})
        assert response.status_code == 503
        assert BOT_TOKEN not in response.text
        assert "/Volumes/Diver Pro/hermes" not in response.text


def test_public_smoke_cannot_be_activated_by_durable_config():
    settings = settings_from_config({
        "telegram_miniapp": {
            "public_smoke": True,
            "https_smoke": True,
            "public_base_url": SMOKE_ORIGIN,
            "allowed_users": ["777"],
            "cors_allowed_origins": [SMOKE_ORIGIN],
        }
    })

    assert settings.public_smoke is False
    assert settings.public_base_url is None
    assert settings.enable_actions is False


def test_public_smoke_cli_forces_loopback_bind(monkeypatch):
    from hermes_cli.telegram_miniapp import cli

    captured = {}
    settings = public_settings(host="0.0.0.0")
    monkeypatch.setattr(cli, "settings_from_config", lambda: settings)
    monkeypatch.setattr(cli, "create_app", lambda *, settings: object())
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **kwargs: captured.update(kwargs)))

    cli.run_foreground(https_smoke=True, public_base_url=SMOKE_ORIGIN)

    assert captured["host"] == "127.0.0.1"
    assert settings.host == "127.0.0.1"


def test_public_smoke_host_port_must_match_public_base_url():
    settings = public_settings(
        public_base_url="https://hermes-smoke.example:8443",
        cors_allowed_origins={"https://hermes-smoke.example:8443"},
    )
    client = make_client(settings)

    assert client.get("/healthz", headers={"host": "hermes-smoke.example"}).status_code == 400
    assert client.get("/healthz", headers={"host": "hermes-smoke.example:8443"}).status_code == 200


def test_public_smoke_rejects_host_origin_null_and_local_dev_origins():
    client = make_client()

    assert client.get("/healthz", headers={"host": "attacker.example"}).status_code == 400
    bad_host_api = client.get("/api/status", headers={"host": "attacker.example"})
    assert bad_host_api.status_code == 400
    assert bad_host_api.headers["cache-control"] == "no-store"
    bad_origin = client.get("/api/status", headers={"host": "hermes-smoke.example", "origin": "https://evil.example"})
    assert bad_origin.status_code == 403
    assert bad_origin.headers["cache-control"] == "no-store"
    assert client.get("/api/status", headers={"host": "hermes-smoke.example", "origin": "null"}).status_code == 403
    assert client.get("/api/status", headers={"host": "hermes-smoke.example", "origin": "http://127.0.0.1:5175"}).status_code == 403

    preflight = client.options(
        "/api/status",
        headers={
            "host": "hermes-smoke.example",
            "origin": "https://evil.example",
            "access-control-request-method": "GET",
        },
    )
    assert preflight.status_code == 204
    assert preflight.headers.get("access-control-allow-origin") is None


def test_public_smoke_missing_origin_fails_closed_except_tested_fetch_metadata_get():
    client = make_client()
    init_data = build_init_data(auth_date=1_700_000_000)

    missing_origin_post = client.post(
        "/api/auth/telegram",
        json={"initData": init_data},
        headers={"host": "hermes-smoke.example"},
    )
    assert missing_origin_post.status_code == 403

    missing_origin_get = client.get("/api/status", headers={"host": "hermes-smoke.example"})
    assert missing_origin_get.status_code == 403

    fetch_metadata_get = client.get(
        "/api/status",
        headers={"host": "hermes-smoke.example", "sec-fetch-site": "same-origin"},
    )
    assert fetch_metadata_get.status_code == 401
    assert fetch_metadata_get.headers["cache-control"] == "no-store"


def test_public_smoke_cookie_security_headers_and_status_shape():
    client = make_client()
    auth = auth_client(client)
    cookie = auth.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "samesite=none" in cookie.lower()
    assert "Path=/api" in cookie

    status = client.get("/api/status", headers={"origin": SMOKE_ORIGIN, "host": "hermes-smoke.example"})
    assert status.status_code == 200
    body = status.json()
    assert body["miniapp"] == {"mode": "https-smoke", "actions_enabled": False, "public_exposure": True}
    assert SMOKE_ORIGIN not in status.text
    assert BOT_TOKEN not in status.text
    assert status.headers["cache-control"] == "no-store"
    assert status.headers["x-content-type-options"] == "nosniff"
    assert status.headers["referrer-policy"] == "no-referrer"
    assert "camera=()" in status.headers["permissions-policy"]


def test_public_smoke_approvals_are_authenticated_no_store_and_rate_limited():
    client = make_client(public_settings(status_rate_limit_per_minute=1))
    headers = {"origin": SMOKE_ORIGIN, "host": "hermes-smoke.example"}

    unauthenticated = client.get("/api/approvals", headers=headers)
    assert unauthenticated.status_code == 401
    assert unauthenticated.headers["cache-control"] == "no-store"

    auth_client(client)
    first = client.get("/api/approvals", headers=headers)
    limited = client.get("/api/approvals", headers=headers)

    assert first.status_code == 200
    assert first.headers["cache-control"] == "no-store"
    assert first.headers["x-content-type-options"] == "nosniff"
    assert first.json()["ok"] is True
    assert len(first.json()["items"]) == 2
    assert SMOKE_ORIGIN not in first.text
    assert BOT_TOKEN not in first.text
    assert "/Volumes/Diver Pro/hermes" not in first.text
    assert limited.status_code == 429
    assert BOT_TOKEN not in limited.text


def test_public_smoke_default_status_provider_overlays_authoritative_public_mode():
    app = create_app(settings=public_settings())
    client = TestClient(app, base_url=SMOKE_ORIGIN)
    auth_client(client)

    status = client.get("/api/status", headers={"origin": SMOKE_ORIGIN, "host": "hermes-smoke.example"})

    assert status.status_code == 200
    assert status.json()["miniapp"] == {"mode": "https-smoke", "actions_enabled": False, "public_exposure": True}
    assert SMOKE_ORIGIN not in status.text


def test_public_smoke_auth_failure_does_not_reveal_allowlist_membership():
    client = make_client(public_settings(allowed_users={"999"}))

    response = client.post(
        "/api/auth/telegram",
        json={"initData": build_init_data(auth_date=1_700_000_000)},
        headers={"origin": SMOKE_ORIGIN, "host": "hermes-smoke.example"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}
    assert "not allowed" not in response.text.lower()
    assert "777" not in response.text


def test_public_smoke_forbidden_action_routes_remain_absent_and_logout_invalidates_session():
    client = make_client()
    auth_client(client)

    assert client.post("/api/restart", headers={"origin": SMOKE_ORIGIN, "host": "hermes-smoke.example"}).status_code == 404
    assert client.post("/api/actions/restart", headers={"origin": SMOKE_ORIGIN, "host": "hermes-smoke.example"}).status_code == 404

    logout = client.post("/api/logout", headers={"origin": SMOKE_ORIGIN, "host": "hermes-smoke.example"})
    assert logout.status_code == 200
    replay = client.get("/api/status", headers={"origin": SMOKE_ORIGIN, "host": "hermes-smoke.example"})
    assert replay.status_code == 401


def test_public_smoke_rate_limits_auth_and_ignores_spoofed_forwarded_headers():
    current_time = [1_700_000_100]
    client = make_client(public_settings(auth_rate_limit_per_minute=99, auth_global_limit=2, now=lambda: current_time[0]))
    headers = {
        "origin": SMOKE_ORIGIN,
        "host": "hermes-smoke.example",
        "x-forwarded-for": "198.51.100.1",
        "forwarded": "for=198.51.100.1",
    }

    first = client.post("/api/auth/telegram", json={"initData": "bad"}, headers=headers)
    current_time[0] += 120
    second = client.post(
        "/api/auth/telegram",
        json={"initData": "bad"},
        headers={**headers, "x-forwarded-for": "198.51.100.2", "forwarded": "for=198.51.100.2"},
    )
    current_time[0] += 120
    third = client.post(
        "/api/auth/telegram",
        json={"initData": "bad"},
        headers={**headers, "x-forwarded-for": "198.51.100.3", "forwarded": "for=198.51.100.3"},
    )

    assert first.status_code != 429
    assert second.status_code != 429
    assert third.status_code == 429
    assert third.headers["cache-control"] == "no-store"
    assert BOT_TOKEN not in third.text


def test_public_smoke_status_rate_limit():
    client = make_client(public_settings(status_rate_limit_per_minute=1))
    auth_client(client)
    headers = {"origin": SMOKE_ORIGIN, "host": "hermes-smoke.example"}

    assert client.get("/api/status", headers=headers).status_code == 200
    limited = client.get("/api/status", headers=headers)
    assert limited.status_code == 429
    assert BOT_TOKEN not in limited.text

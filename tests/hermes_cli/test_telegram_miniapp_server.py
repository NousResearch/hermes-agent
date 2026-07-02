import hashlib
import hmac
import json
import threading
import time
from typing import get_args
from urllib.parse import quote

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from hermes_cli.telegram_miniapp.models import ActionDecisionValue
from hermes_cli.telegram_miniapp.previews import build_logs_snapshot, build_sessions_snapshot
from hermes_cli.telegram_miniapp.server import MiniAppSettings, RateLimiter, create_app


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


def assert_safe_preview_meta(meta):
    assert meta == {
        "source": meta["source"],
        "source_label": meta["source_label"],
        "redaction": "safe-preview",
        "contains_live_actions": False,
    }
    assert meta["source"] in {"preview", "live-safe"}
    assert meta["source_label"]
    serialized = json.dumps(meta)
    assert "/Volumes/Diver Pro/hermes" not in serialized
    assert BOT_TOKEN not in serialized
    assert "TELEGRAM_BOT_TOKEN" not in serialized
    assert "pid" not in serialized.lower()
    assert "command" not in serialized.lower()


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
        ("GET", "/api/capabilities"),
        ("GET", "/api/approvals"),
        ("GET", "/api/sessions"),
        ("GET", "/api/logs"),
    }

    assert client.get("/does-not-exist").status_code == 404
    forbidden_posts = [
        "/api/actions",
        "/api/actions/restart",
        "/api/actions/approve",
        "/api/actions/reject",
        "/api/actions/decision",
        "/api/execute",
        "/api/tool",
        "/api/command",
        "/api/process",
        "/api/process/kill",
        "/api/restart",
        "/api/approvals/system-mode-change-preview/approve",
        "/api/approvals/system-mode-change-preview/reject",
        "/api/approvals/system-mode-change-preview/decision",
        "/api/config",
        "/api/config/model",
        "/api/model/switch",
    ]
    forbidden_gets = [
        "/api/actions",
        "/api/restart",
        "/api/execute",
        "/api/tool",
        "/api/command",
        "/api/process",
    ]
    for path in forbidden_posts:
        assert client.post(path).status_code == 404
    for path in forbidden_gets:
        assert client.get(path).status_code == 404


def test_action_decision_contract_is_phase_one_only_and_dormant():
    assert set(get_args(ActionDecisionValue)) == {"approve_once", "reject_once"}
    assert "restart" not in get_args(ActionDecisionValue)
    assert "approve_all" not in get_args(ActionDecisionValue)
    assert "approve_session" not in get_args(ActionDecisionValue)
    assert "approve_always" not in get_args(ActionDecisionValue)


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


def test_capabilities_require_auth_then_return_safe_read_only_matrix():
    client = make_client()

    assert client.get("/api/capabilities").status_code == 401

    auth_client(client)
    response = client.get("/api/capabilities")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["ok"] is True
    assert_safe_preview_meta(body["meta"])
    assert len(body["items"]) == 5
    ids = {item["id"] for item in body["items"]}
    assert {"status-read", "approvals-read", "sessions-read", "approve-action", "restart-action"} == ids
    for item in body["items"]:
        assert set(item) == {"id", "label", "enabled", "mode", "reason"}
        assert isinstance(item["enabled"], bool)
        assert item["mode"] in {"read-only", "blocked"}
    blocked = [item for item in body["items"] if not item["enabled"]]
    assert {item["id"] for item in blocked} == {"approve-action", "restart-action"}
    serialized = response.text
    assert "/Volumes/Diver Pro/hermes" not in serialized
    assert BOT_TOKEN not in serialized
    assert "TELEGRAM_BOT_TOKEN" not in serialized
    assert "pid" not in serialized.lower()
    assert "command" not in serialized.lower()


def test_approvals_require_auth_then_return_safe_read_only_queue():
    client = make_client()

    assert client.get("/api/approvals").status_code == 401

    auth_client(client)
    response = client.get("/api/approvals")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert_safe_preview_meta(body["meta"])
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


def test_sessions_and_logs_require_auth_then_return_safe_read_only_previews():
    client = make_client()

    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/logs").status_code == 401

    auth_client(client)
    sessions = client.get("/api/sessions")
    logs = client.get("/api/logs")

    assert sessions.status_code == 200
    sessions_body = sessions.json()
    assert sessions_body["ok"] is True
    assert_safe_preview_meta(sessions_body["meta"])
    assert isinstance(sessions_body["items"], list)
    for first_session in sessions_body["items"][:1]:
        assert set(first_session) == {"id", "agent", "state", "meta", "time", "tone"}
        assert first_session["state"] in {"observing", "waiting", "completed"}
        assert first_session["tone"] in {"ok", "warn", "muted"}

    assert logs.status_code == 200
    logs_body = logs.json()
    assert logs_body["ok"] is True
    assert_safe_preview_meta(logs_body["meta"])
    assert len(logs_body["items"]) == 4
    first_log = logs_body["items"][0]
    assert set(first_log) == {"level", "message", "time"}
    assert first_log["level"] in {"info", "warn", "error"}

    for serialized in (sessions.text, logs.text):
        assert "/Volumes/Diver Pro/hermes" not in serialized
        assert BOT_TOKEN not in serialized
        assert "TELEGRAM_BOT_TOKEN" not in serialized
        assert "pid" not in serialized.lower()
        assert "command" not in serialized.lower()


def test_sessions_snapshot_uses_safe_read_only_projection_without_raw_row_leaks():
    raw_session_id = "session-secret-raw-id"
    leaked_values = [
        raw_session_id,
        "/Volumes/Diver Pro/projects/private-app",
        "anthropic/secret-model",
        "SYSTEM PROMPT SECRET",
        "raw user preview with /private/path and token",
        "TELEGRAM_BOT_TOKEN",
        "pid",
        "command",
    ]

    class FakeSessionSource:
        def list_sessions_rich(self, **kwargs):
            assert kwargs == {
                "limit": 5,
                "include_children": False,
                "order_by_last_active": True,
                "include_archived": False,
            }
            return [
                {
                    "id": raw_session_id,
                    "source": "telegram",
                    "model": "anthropic/secret-model",
                    "model_config": {"provider": "secret-provider"},
                    "system_prompt": "SYSTEM PROMPT SECRET",
                    "cwd": "/Volumes/Diver Pro/projects/private-app",
                    "git_repo_root": "/Volumes/Diver Pro/projects/private-app",
                    "git_branch": "private-branch",
                    "preview": "raw user preview with /private/path and token",
                    "started_at": 1_700_000_000,
                    "last_active": 1_700_000_040,
                    "ended_at": None,
                    "message_count": 7,
                }
            ]

    snapshot = build_sessions_snapshot(session_db_factory=lambda: FakeSessionSource(), now=lambda: 1_700_000_100)

    assert snapshot["ok"] is True
    assert snapshot["meta"]["source"] == "live-safe"
    assert len(snapshot["items"]) == 1
    item = snapshot["items"][0]
    assert item == {
        "id": item["id"],
        "agent": "Telegram",
        "state": "observing",
        "meta": "7 сообщений · Telegram",
        "time": "сейчас",
        "tone": "warn",
    }
    assert item["id"].startswith("session-")
    assert item["id"] != raw_session_id
    serialized = json.dumps(snapshot, ensure_ascii=False)
    for leaked in leaked_values:
        assert leaked not in serialized


def test_sessions_snapshot_returns_safe_empty_when_source_unreadable():
    def broken_factory():
        raise RuntimeError("/Volumes/Diver Pro/private/state.db exploded with token")

    snapshot = build_sessions_snapshot(session_db_factory=broken_factory)

    assert snapshot == {
        "ok": True,
        "meta": {
            "source": "live-safe",
            "source_label": "Safe session index",
            "redaction": "safe-preview",
            "contains_live_actions": False,
        },
        "items": [],
    }


def test_logs_snapshot_is_derived_from_safe_facts_not_raw_log_lines():
    sessions_snapshot = {
        "ok": True,
        "meta": {"source": "live-safe", "source_label": "Safe session index", "redaction": "safe-preview", "contains_live_actions": False},
        "items": [{"id": "session-safe", "agent": "Telegram", "state": "observing", "meta": "1 сообщение · Telegram", "time": "сейчас", "tone": "warn"}],
    }

    snapshot = build_logs_snapshot(sessions_provider=lambda: sessions_snapshot)

    assert snapshot["ok"] is True
    assert snapshot["meta"]["source"] == "live-safe"
    assert len(snapshot["items"]) == 4
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert "Есть активные сессии наблюдения" in serialized
    assert "/Volumes/Diver Pro" not in serialized
    assert "Traceback" not in serialized
    assert "TELEGRAM_BOT_TOKEN" not in serialized
    assert "pid" not in serialized.lower()
    assert "command" not in serialized.lower()


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


def test_loopback_auth_is_rate_limited_per_minute_without_lifetime_cap():
    current_time = [1_700_000_100]
    settings = MiniAppSettings(
        bot_token=BOT_TOKEN,
        allowed_users={"777"},
        auth_rate_limit_per_minute=2,
        auth_global_limit=1,
        now=lambda: current_time[0],
    )
    client = make_client(settings=settings)
    payload = {"initData": build_init_data(auth_date=1_700_000_000)}
    headers = {"origin": "http://127.0.0.1:5175"}

    first = client.post("/api/auth/telegram", json=payload, headers=headers)
    second = client.post("/api/auth/telegram", json=payload, headers=headers)
    limited = client.post("/api/auth/telegram", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert limited.status_code == 429
    assert "Too many requests" in limited.text
    assert BOT_TOKEN not in limited.text

    # Next minute window: the per-minute limiter resets, and the public-smoke
    # lifetime cap (auth_global_limit=1 already exceeded above) must not lock
    # out auth on a long-lived loopback sidecar.
    current_time[0] += 61
    recovered = client.post("/api/auth/telegram", json=payload, headers=headers)
    assert recovered.status_code == 200


def test_rate_limiter_evicts_stale_minute_buckets():
    limiter = RateLimiter()

    assert limiter.check("auth:a", limit=5, now=0)
    assert limiter.check("auth:b", limit=5, now=30)
    assert limiter.check("auth:a", limit=5, now=61)

    assert set(limiter._counts) == {("auth:a", 1)}


def test_session_expires_after_ttl_and_requires_reauth():
    current_time = [1_700_000_100]
    settings = MiniAppSettings(
        bot_token=BOT_TOKEN,
        allowed_users={"777"},
        session_ttl_seconds=60,
        now=lambda: current_time[0],
    )
    client = make_client(settings=settings)
    auth = client.post(
        "/api/auth/telegram",
        json={"initData": build_init_data(auth_date=1_700_000_000)},
        headers={"origin": "http://127.0.0.1:5175"},
    )
    assert auth.status_code == 200

    assert client.get("/api/me").status_code == 200

    current_time[0] += 61
    expired = client.get("/api/me")
    assert expired.status_code == 401


def test_local_mode_csrf_posture_missing_origin_is_allowed_by_design():
    """Lock the loopback CSRF posture as an explicit, reviewed decision.

    A cross-site HTML form POST may omit the Origin header and local mode lets
    it through the origin guard. That is intentional and safe today: auth
    requires unforgeable Telegram initData in a JSON body, logout only drops
    the caller's own session, and no state-changing action endpoints exist.
    Revisit before any action endpoint ships (see M18 action-gate spec).
    """
    client = make_client()

    auth = client.post(
        "/api/auth/telegram",
        json={"initData": build_init_data(auth_date=1_700_000_000)},
    )
    assert auth.status_code == 200

    # Browser-level CSRF backstop: the local-mode session cookie must stay
    # HttpOnly + SameSite=Lax so cross-site form POSTs do not carry it.
    set_cookie = auth.headers.get("set-cookie", "")
    assert "httponly" in set_cookie.lower()
    assert "samesite=lax" in set_cookie.lower()

    # HTML forms cannot send application/json; a real cross-site form POST is
    # rejected by body validation even though it passes the origin guard.
    form_attempt = client.post(
        "/api/auth/telegram",
        data={"initData": "forged"},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert form_attempt.status_code == 422, "body validation must reject form bodies, not a 5xx"

    # Without the session cookie, a missing-Origin logout is unauthorized.
    anonymous = make_client()
    assert anonymous.post("/api/logout").status_code == 401

    logout = client.post("/api/logout")
    assert logout.status_code == 200


def test_rate_limiter_is_thread_safe_across_bucket_rollover():
    limiter = RateLimiter()
    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            for step in range(500):
                limiter.check(f"auth:{idx % 3}", limit=1_000_000, now=step * 0.5)
        except Exception as exc:  # noqa: BLE001 - the test asserts no exception at all
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []

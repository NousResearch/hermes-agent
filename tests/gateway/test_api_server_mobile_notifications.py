"""Tests for the Hermes mobile notification inbox API."""

import os
import stat
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/mobile/notifications", adapter._handle_mobile_notifications)
    app.router.add_post(
        "/api/mobile/notifications/{notification_id}/read",
        adapter._handle_mark_mobile_notification_read,
    )
    app.router.add_post(
        "/api/mobile/notifications/{notification_id}/actions",
        adapter._handle_mobile_notification_action,
    )
    return app


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    return _make_adapter()


@pytest.fixture
def auth_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    return _make_adapter(api_key="sk-secret")


def _seed_approval(adapter: APIServerAdapter, expires_at=None):
    return adapter._get_mobile_notifications().upsert_notification(
        {
            "id": "remember-1",
            "kind": "memory_approval",
            "title": "保存候補があります",
            "body": "2件の候補を確認してください。",
            "detail_ref": "memory-review:2026-06-11",
            "expires_at": expires_at,
        },
        actions=[
            {
                "id": "yes",
                "group_key": "remember-1",
                "label": "はい",
                "value": "approve",
            },
            {
                "id": "no",
                "group_key": "remember-1",
                "label": "いいえ",
                "value": "reject",
            },
        ],
    )


@pytest.mark.asyncio
async def test_mobile_notifications_require_auth(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/mobile/notifications")
        assert resp.status == 401

        resp = await cli.get(
            "/api/mobile/notifications",
            headers={"Authorization": "Bearer sk-secret"},
        )
        assert resp.status == 200


@pytest.mark.asyncio
async def test_list_mobile_notifications_returns_open_items(adapter):
    _seed_approval(adapter)
    adapter._get_mobile_notifications().upsert_notification(
        {
            "id": "old-closed",
            "kind": "memory_approval",
            "title": "完了済み",
            "body": "これは表示しません。",
            "status": "closed",
        },
        actions=[],
    )
    adapter._get_mobile_notifications().upsert_notification(
        {
            "id": "old-expired",
            "kind": "memory_approval",
            "title": "期限切れ",
            "body": "これも表示しません。",
            "expires_at": time.time() - 1,
        },
        actions=[],
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/mobile/notifications")
        assert resp.status == 200
        data = await resp.json()

    assert [item["id"] for item in data["notifications"]] == ["remember-1"]
    assert {action["id"] for action in data["notifications"][0]["actions"]} == {"yes", "no"}


@pytest.mark.asyncio
async def test_mark_mobile_notification_read(adapter):
    _seed_approval(adapter)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/mobile/notifications/remember-1/read")
        assert resp.status == 200
        data = await resp.json()

    assert data["notification"]["status"] == "read"


@pytest.mark.asyncio
async def test_resolve_mobile_notification_action_closes_group(adapter):
    _seed_approval(adapter)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/mobile/notifications/remember-1/actions",
            json={"action_id": "yes"},
        )
        assert resp.status == 200
        data = await resp.json()

        retry = await cli.post(
            "/api/mobile/notifications/remember-1/actions",
            json={"action_id": "yes"},
        )
        assert retry.status == 200

        conflict = await cli.post(
            "/api/mobile/notifications/remember-1/actions",
            json={"action_id": "no"},
        )
        assert conflict.status == 409

    assert data["notification"]["status"] == "closed"
    statuses = {action["id"]: action["status"] for action in data["notification"]["actions"]}
    assert statuses == {"yes": "done", "no": "rejected"}


@pytest.mark.asyncio
async def test_resolve_expired_mobile_notification_returns_gone(adapter):
    _seed_approval(adapter, expires_at=time.time() - 1)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/mobile/notifications/remember-1/actions",
            json={"action_id": "yes"},
        )
        data = await resp.json()

    assert resp.status == 410
    assert data["notification"]["status"] == "expired"


@pytest.mark.asyncio
async def test_mobile_notification_validation_errors(adapter):
    _seed_approval(adapter)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        bad_status = await cli.get("/api/mobile/notifications?status=closed")
        bad_limit = await cli.get("/api/mobile/notifications?limit=0")
        missing_action = await cli.post(
            "/api/mobile/notifications/remember-1/actions",
            json={},
        )

    assert bad_status.status == 400
    assert bad_limit.status == 400
    assert missing_action.status == 400


@pytest.mark.asyncio
async def test_mobile_capabilities_advertise_inbox(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    assert data["features"]["mobile_notifications"] is True
    assert data["endpoints"]["mobile_notifications"]["path"] == "/api/mobile/notifications"


def test_mobile_notification_db_is_owner_only(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX file permissions are not available on Windows")
    db_path = tmp_path / "mobile_notifications.db"
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    adapter = _make_adapter()
    _seed_approval(adapter)

    mode = stat.S_IMODE(db_path.stat().st_mode)
    assert mode == 0o600

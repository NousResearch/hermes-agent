from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.mobile_notifications import MobilePairingStore
from gateway.platforms.api_server import APIServerAdapter


def _app(adapter):
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/mobile/pairing/exchange", adapter._handle_mobile_pairing_exchange)
    app.router.add_get("/v1/mobile/pairing/devices", adapter._handle_mobile_paired_devices)
    app.router.add_delete("/v1/mobile/pairing/devices/{device_id}", adapter._handle_mobile_paired_device_delete)
    app.router.add_post("/v1/mobile/devices/{installation_id}/test", adapter._handle_mobile_notification_test)
    return app


@pytest.mark.asyncio
async def test_pairing_exchange_then_device_token_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "master-secret"}))
    grant = MobilePairingStore().create_grant()
    app = _app(adapter)

    with patch("gateway.platforms.api_server.mobile_extension_enabled", return_value=True):
        async with TestClient(TestServer(app)) as cli:
            exchanged = await cli.post("/v1/mobile/pairing/exchange", json={
                "grant": grant.secret,
                "installation_id": "pixel-1",
                "device_name": "Pixel",
            })
            assert exchanged.status == 200
            token = (await exchanged.json())["token"]

            devices = await cli.get(
                "/v1/mobile/pairing/devices",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert devices.status == 200
            body = await devices.json()
            assert body["data"][0]["installation_id"] == "pixel-1"
            assert "token" not in body["data"][0]


@pytest.mark.asyncio
async def test_pairing_is_dormant_when_extension_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "master-secret"}))
    store = MobilePairingStore()
    grant = store.create_grant()
    device = store.exchange(grant.secret, installation_id="pixel-1")
    app = _app(adapter)
    with patch("gateway.platforms.api_server.mobile_extension_enabled", return_value=False):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/mobile/pairing/exchange", json={})
            assert resp.status == 404

            caps = await cli.get(
                "/v1/capabilities",
                headers={"Authorization": "Bearer master-secret"},
            )
            assert "hermes.mobile" not in (await caps.json())["extensions"]
            rejected_device = await cli.get(
                "/v1/capabilities",
                headers={"Authorization": f"Bearer {device.token}"},
            )
            assert rejected_device.status == 401


@pytest.mark.asyncio
async def test_pairing_exchange_is_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "master-secret"}))
    app = _app(adapter)
    with patch("gateway.platforms.api_server.mobile_extension_enabled", return_value=True):
        async with TestClient(TestServer(app)) as cli:
            for _ in range(10):
                resp = await cli.post("/v1/mobile/pairing/exchange", json={
                    "grant": "wrong",
                    "installation_id": "pixel-1",
                })
                assert resp.status == 401
            limited = await cli.post("/v1/mobile/pairing/exchange", json={
                "grant": "wrong",
                "installation_id": "pixel-1",
            })
            assert limited.status == 429
            assert limited.headers["Retry-After"] == "60"


@pytest.mark.asyncio
async def test_capability_v12_and_targeted_self_test(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "master-secret"}))
    app = _app(adapter)
    notifier = MagicMock()
    notifier.send.return_value = 1
    with patch("gateway.platforms.api_server.mobile_extension_enabled", return_value=True), patch(
        "gateway.mobile_notifications.FCMNotifier", return_value=notifier
    ):
        async with TestClient(TestServer(app)) as cli:
            headers = {"Authorization": "Bearer master-secret"}
            caps = await cli.get("/v1/capabilities", headers=headers)
            data = await caps.json()
            mobile = data["extensions"]["hermes.mobile"]
            assert mobile["version"] == "1.2"
            assert all(mobile["features"].values())
            assert data["features"]["jobs_admin"] is True

            sent = await cli.post("/v1/mobile/devices/pixel-1/test", headers=headers)
            assert sent.status == 200
            assert await sent.json() == {"sent": True, "installation_id": "pixel-1"}
    notifier.send.assert_called_once_with(
        {"event": "notification.test", "state": "test", "title": "Hermes notification test"},
        installation_id="pixel-1",
    )

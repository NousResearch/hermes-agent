import uuid

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


ISO_TIME = "2026-04-21T07:00:00Z"


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    middlewares = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=middlewares)
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/iphone/register", adapter._handle_iphone_register)
    app.router.add_post("/api/iphone/events", adapter._handle_iphone_events)
    app.router.add_get("/api/iphone/recommendations/latest", adapter._handle_iphone_recommendation_latest)
    app.router.add_post("/api/iphone/location-requests", adapter._handle_iphone_location_request_create)
    app.router.add_get("/api/iphone/location-requests/next", adapter._handle_iphone_location_request_next)
    app.router.add_post(
        "/api/iphone/location-requests/{request_id}/decision",
        adapter._handle_iphone_location_request_decision,
    )
    return app


def _registration_payload(client_id: str | None = None) -> dict:
    return {
        "clientID": client_id or "11111111-2222-3333-4444-555555555555",
        "deviceName": "屙屎屙唔出",
        "bundleIdentifier": "com.glasserdraco.CompanionNodeApp",
        "platform": "ios-companion",
        "osVersion": "iOS 26.4",
        "capabilities": {
            "supportsLocation": True,
            "supportsMessageRelay": True,
            "supportsShortcutsBridge": True,
        },
        "submittedAt": ISO_TIME,
    }


def _events_payload(client_id: str = "11111111-2222-3333-4444-555555555555") -> dict:
    return {
        "clientID": client_id,
        "deviceName": "屙屎屙唔出",
        "submittedAt": ISO_TIME,
        "events": [
            {
                "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "createdAt": ISO_TIME,
                "payload": {
                    "kind": "message_trigger",
                    "messageTrigger": {
                        "ruleID": "from-alice-contains-arrive",
                        "senderLabel": "Alice",
                        "matchedText": "到了",
                        "occurredAt": ISO_TIME,
                    },
                },
            }
        ],
    }


def _personal_context_events_payload(
    client_id: str = "11111111-2222-3333-4444-555555555555",
    *,
    notification_authorization: str = "denied",
    battery_level: int = 18,
    charging_state: str = "unplugged",
    due_today_count: int = 3,
    due_soon_count: int = 1,
) -> dict:
    return {
        "clientID": client_id,
        "deviceName": "屙屎屙唔出",
        "submittedAt": ISO_TIME,
        "events": [
            {
                "id": "bbbbbbbb-cccc-dddd-eeee-ffffffffffff",
                "createdAt": ISO_TIME,
                "payload": {
                    "kind": "personal_context_snapshot",
                    "personalContextSnapshot": {
                        "capturedAt": ISO_TIME,
                        "notificationAuthorization": notification_authorization,
                        "battery": {
                            "levelPercent": battery_level,
                            "chargingState": charging_state,
                        },
                        "calendar": {
                            "access": "fullAccess",
                            "upcomingEventCount": 2,
                            "nextEventStartAt": "2026-04-21T08:00:00Z",
                            "nextEventEndAt": "2026-04-21T09:00:00Z",
                        },
                        "reminders": {
                            "access": "fullAccess",
                            "dueTodayCount": due_today_count,
                            "dueSoonCount": due_soon_count,
                        },
                    },
                },
            }
        ],
    }


def _location_request_payload(client_id: str = "11111111-2222-3333-4444-555555555555") -> dict:
    return {
        "clientID": client_id,
        "reason": "Hermes wants your current location. Open Companion Node and tap Allow once if you want to share it.",
        "requestedAt": ISO_TIME,
    }


@pytest.fixture
def auth_adapter() -> APIServerAdapter:
    return _make_adapter(api_key="dragon-api-key")


@pytest.mark.asyncio
async def test_iphone_register_returns_device_receipt(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/iphone/register",
            json=_registration_payload(),
            headers={"Authorization": "Bearer dragon-api-key"},
        )

        assert response.status == 200
        body = await response.json()
        assert body["clientID"] == "11111111-2222-3333-4444-555555555555"
        assert uuid.UUID(body["registeredDeviceID"])
        assert body["deviceToken"].startswith("iphone_")
        assert body["registeredAt"].endswith("Z")


@pytest.mark.asyncio
async def test_iphone_register_is_idempotent_for_same_client_id(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        first = await cli.post(
            "/api/iphone/register",
            json=_registration_payload(),
            headers={"Authorization": "Bearer dragon-api-key"},
        )
        second = await cli.post(
            "/api/iphone/register",
            json=_registration_payload(),
            headers={"Authorization": "Bearer dragon-api-key"},
        )

        assert first.status == 200
        assert second.status == 200
        first_body = await first.json()
        second_body = await second.json()
        assert second_body == first_body


@pytest.mark.asyncio
async def test_iphone_events_accept_registered_device_token(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        register = await cli.post(
            "/api/iphone/register",
            json=_registration_payload(),
            headers={"Authorization": "Bearer dragon-api-key"},
        )
        receipt = await register.json()

        response = await cli.post(
            "/api/iphone/events",
            json=_events_payload(),
            headers={"Authorization": f"Bearer {receipt['deviceToken']}"},
        )

        assert response.status == 202
        body = await response.json()
        assert body == {
            "status": "accepted",
            "acceptedCount": 1,
            "clientID": "11111111-2222-3333-4444-555555555555",
        }


@pytest.mark.asyncio
async def test_iphone_events_reject_unknown_device_token(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/iphone/events",
            json=_events_payload(),
            headers={"Authorization": "Bearer not-a-real-device-token"},
        )

        assert response.status == 401
        body = await response.json()
        assert body["error"] == "Invalid device token"


@pytest.mark.asyncio
async def test_iphone_events_reject_client_id_mismatch(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        register = await cli.post(
            "/api/iphone/register",
            json=_registration_payload(),
            headers={"Authorization": "Bearer dragon-api-key"},
        )
        receipt = await register.json()

        response = await cli.post(
            "/api/iphone/events",
            json=_events_payload(client_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            headers={"Authorization": f"Bearer {receipt['deviceToken']}"},
        )

        assert response.status == 403
        body = await response.json()
        assert body["error"] == "Device token does not match clientID"


@pytest.mark.asyncio
async def test_iphone_recommendation_returns_card_for_latest_snapshot(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        register = await cli.post(
            "/api/iphone/register",
            json=_registration_payload(),
            headers={"Authorization": "Bearer dragon-api-key"},
        )
        receipt = await register.json()

        uploaded = await cli.post(
            "/api/iphone/events",
            json=_personal_context_events_payload(),
            headers={"Authorization": f"Bearer {receipt['deviceToken']}"},
        )
        assert uploaded.status == 202

        response = await cli.get(
            "/api/iphone/recommendations/latest",
            headers={"Authorization": f"Bearer {receipt['deviceToken']}"},
        )

        assert response.status == 200
        body = await response.json()
        assert body["kind"] == "enable_notifications"
        assert body["title"] == "Enable notifications for nudges"
        assert "notification" in body["body"].lower()


@pytest.mark.asyncio
async def test_iphone_recommendation_returns_204_without_snapshot(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        register = await cli.post(
            "/api/iphone/register",
            json=_registration_payload(),
            headers={"Authorization": "Bearer dragon-api-key"},
        )
        receipt = await register.json()

        response = await cli.get(
            "/api/iphone/recommendations/latest",
            headers={"Authorization": f"Bearer {receipt['deviceToken']}"},
        )

        assert response.status == 204


@pytest.mark.asyncio
async def test_location_request_round_trip_create_poll_and_decide(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        register = await cli.post(
            "/api/iphone/register",
            json=_registration_payload(),
            headers={"Authorization": "Bearer dragon-api-key"},
        )
        receipt = await register.json()

        created = await cli.post(
            "/api/iphone/location-requests",
            json=_location_request_payload(),
            headers={"Authorization": "Bearer dragon-api-key"},
        )

        assert created.status == 202
        created_body = await created.json()
        assert uuid.UUID(created_body["requestID"])
        assert created_body["status"] == "pending"
        assert created_body["clientID"] == "11111111-2222-3333-4444-555555555555"

        pending = await cli.get(
            "/api/iphone/location-requests/next",
            headers={"Authorization": f"Bearer {receipt['deviceToken']}"},
        )

        assert pending.status == 200
        pending_body = await pending.json()
        assert pending_body == created_body

        decided = await cli.post(
            f"/api/iphone/location-requests/{created_body['requestID']}/decision",
            json={"decision": "fulfilled", "decidedAt": ISO_TIME},
            headers={"Authorization": f"Bearer {receipt['deviceToken']}"},
        )

        assert decided.status == 200
        decided_body = await decided.json()
        assert decided_body["status"] == "fulfilled"
        assert decided_body["requestID"] == created_body["requestID"]

        empty = await cli.get(
            "/api/iphone/location-requests/next",
            headers={"Authorization": f"Bearer {receipt['deviceToken']}"},
        )
        assert empty.status == 204


@pytest.mark.asyncio
async def test_location_request_create_rejects_unknown_client(auth_adapter):
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/iphone/location-requests",
            json=_location_request_payload(client_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            headers={"Authorization": "Bearer dragon-api-key"},
        )

        assert response.status == 404
        body = await response.json()
        assert body["error"] == "Unknown iPhone clientID"

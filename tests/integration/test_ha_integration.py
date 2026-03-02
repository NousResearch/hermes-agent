"""Integration tests for Home Assistant (tool + gateway).

Spins up a real in-process fake HA server (HTTP + WebSocket) and exercises
the full adapter and tool handler paths over real TCP connections.
No mocks -- only real async I/O against a fake server.

Run with:  uv run pytest tests/integration/test_ha_integration.py -v
"""

import asyncio

import pytest

pytestmark = pytest.mark.integration

from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.homeassistant import HomeAssistantAdapter
from tests.fakes.fake_ha_server import FakeHAServer, ENTITY_STATES
from tools.homeassistant_tool import (
    _async_call_service,
    _async_get_state,
    _async_list_entities,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapter_for(server: FakeHAServer, **extra) -> HomeAssistantAdapter:
    """Create an adapter pointed at the fake server."""
    config = PlatformConfig(
        enabled=True,
        token=server.token,
        extra={"url": server.url, **extra},
    )
    return HomeAssistantAdapter(config)


# ---------------------------------------------------------------------------
# 1. Gateway -- WebSocket lifecycle
# ---------------------------------------------------------------------------


class TestGatewayWebSocket:
    @pytest.mark.asyncio
    async def test_connect_auth_subscribe(self):
        """Full WS handshake succeeds: auth_required -> auth -> auth_ok -> subscribe -> ACK."""
        async with FakeHAServer() as server:
            adapter = _adapter_for(server)
            connected = await adapter.connect()
            assert connected is True
            assert adapter._running is True
            assert adapter._ws is not None
            assert not adapter._ws.closed
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_connect_auth_rejected(self):
        """connect() returns False when the server rejects auth."""
        async with FakeHAServer() as server:
            server.reject_auth = True
            adapter = _adapter_for(server)
            connected = await adapter.connect()
            assert connected is False

    @pytest.mark.asyncio
    async def test_event_received_and_forwarded(self):
        """Server pushes event -> adapter calls handle_message with correct MessageEvent."""
        async with FakeHAServer() as server:
            adapter = _adapter_for(server)
            adapter.handle_message = AsyncMock()

            await adapter.connect()

            # Push a state_changed event
            await server.push_event({
                "data": {
                    "entity_id": "light.bedroom",
                    "old_state": {"state": "off", "attributes": {}},
                    "new_state": {
                        "state": "on",
                        "attributes": {"friendly_name": "Bedroom Light"},
                    },
                }
            })

            # Wait for the adapter to process it
            for _ in range(50):
                if adapter.handle_message.call_count > 0:
                    break
                await asyncio.sleep(0.05)

            assert adapter.handle_message.call_count == 1
            msg_event = adapter.handle_message.call_args[0][0]
            assert "Bedroom Light" in msg_event.text
            assert "turned on" in msg_event.text
            assert msg_event.source.platform == Platform.HOMEASSISTANT

            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_event_filtering_ignores_unwatched(self):
        """Events outside watch_domains are silently dropped."""
        async with FakeHAServer() as server:
            adapter = _adapter_for(server, watch_domains=["climate"])
            adapter.handle_message = AsyncMock()

            await adapter.connect()

            # Push a light event (not in watch_domains)
            await server.push_event({
                "data": {
                    "entity_id": "light.bedroom",
                    "old_state": {"state": "off", "attributes": {}},
                    "new_state": {
                        "state": "on",
                        "attributes": {"friendly_name": "Bedroom Light"},
                    },
                }
            })

            await asyncio.sleep(0.5)
            assert adapter.handle_message.call_count == 0

            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_closes_cleanly(self):
        """disconnect() cancels listener and closes WebSocket."""
        async with FakeHAServer() as server:
            adapter = _adapter_for(server)
            await adapter.connect()
            ws_ref = adapter._ws

            await adapter.disconnect()

            assert adapter._running is False
            assert adapter._listen_task is None
            assert adapter._ws is None
            # The original WS reference should be closed
            assert ws_ref.closed


# ---------------------------------------------------------------------------
# 2. REST tool handlers (real HTTP against fake server)
# ---------------------------------------------------------------------------


class TestToolRest:
    """Call the async tool functions directly against the fake server.

    Note: we call ``_async_*`` instead of the sync ``_handle_*`` wrappers
    because the sync wrappers use ``_run_async`` which blocks the event
    loop, deadlocking with the in-process fake server.  The async functions
    are the real logic; the sync wrappers are trivial bridge code already
    covered by unit tests.
    """

    @pytest.mark.asyncio
    async def test_list_entities_returns_all(self, monkeypatch):
        """_async_list_entities returns all entities from the fake server."""
        async with FakeHAServer() as server:
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_URL", server.url,
            )
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_TOKEN", server.token,
            )

            result = await _async_list_entities()

            assert result["count"] == len(ENTITY_STATES)
            ids = {e["entity_id"] for e in result["entities"]}
            assert "light.bedroom" in ids
            assert "climate.thermostat" in ids

    @pytest.mark.asyncio
    async def test_list_entities_domain_filter(self, monkeypatch):
        """Domain filter is applied after fetching from server."""
        async with FakeHAServer() as server:
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_URL", server.url,
            )
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_TOKEN", server.token,
            )

            result = await _async_list_entities(domain="light")

            assert result["count"] == 2
            for e in result["entities"]:
                assert e["entity_id"].startswith("light.")

    @pytest.mark.asyncio
    async def test_get_state_single_entity(self, monkeypatch):
        """_async_get_state returns full entity details."""
        async with FakeHAServer() as server:
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_URL", server.url,
            )
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_TOKEN", server.token,
            )

            result = await _async_get_state("light.bedroom")

            assert result["entity_id"] == "light.bedroom"
            assert result["state"] == "on"
            assert result["attributes"]["brightness"] == 200
            assert result["last_changed"] is not None

    @pytest.mark.asyncio
    async def test_get_state_not_found(self, monkeypatch):
        """Non-existent entity raises an aiohttp error (404)."""
        import aiohttp as _aiohttp

        async with FakeHAServer() as server:
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_URL", server.url,
            )
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_TOKEN", server.token,
            )

            with pytest.raises(_aiohttp.ClientResponseError) as exc_info:
                await _async_get_state("light.nonexistent")
            assert exc_info.value.status == 404

    @pytest.mark.asyncio
    async def test_call_service_turn_on(self, monkeypatch):
        """_async_call_service sends correct payload and server records it."""
        async with FakeHAServer() as server:
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_URL", server.url,
            )
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_TOKEN", server.token,
            )

            result = await _async_call_service(
                domain="light",
                service="turn_on",
                entity_id="light.bedroom",
                data={"brightness": 255},
            )

            assert result["success"] is True
            assert result["service"] == "light.turn_on"
            assert len(result["affected_entities"]) == 1
            assert result["affected_entities"][0]["state"] == "on"

            # Verify fake server recorded the call
            assert len(server.received_service_calls) == 1
            call = server.received_service_calls[0]
            assert call["domain"] == "light"
            assert call["service"] == "turn_on"
            assert call["data"]["entity_id"] == "light.bedroom"
            assert call["data"]["brightness"] == 255


# ---------------------------------------------------------------------------
# 3. send() -- REST notification
# ---------------------------------------------------------------------------


class TestSendNotification:
    @pytest.mark.asyncio
    async def test_send_notification_delivered(self):
        """Adapter send() delivers notification to fake server REST endpoint."""
        async with FakeHAServer() as server:
            adapter = _adapter_for(server)

            result = await adapter.send("ha_events", "Test notification from agent")

            assert result.success is True
            assert len(server.received_notifications) == 1
            notif = server.received_notifications[0]
            assert notif["title"] == "Hermes Agent"
            assert notif["message"] == "Test notification from agent"

    @pytest.mark.asyncio
    async def test_send_auth_failure(self):
        """send() returns failure when token is wrong."""
        async with FakeHAServer() as server:
            config = PlatformConfig(
                enabled=True,
                token="wrong-token",
                extra={"url": server.url},
            )
            adapter = HomeAssistantAdapter(config)

            result = await adapter.send("ha_events", "Should fail")

            assert result.success is False
            assert "401" in result.error


# ---------------------------------------------------------------------------
# 4. Auth and error cases
# ---------------------------------------------------------------------------


class TestAuthAndErrors:
    @pytest.mark.asyncio
    async def test_rest_unauthorized(self, monkeypatch):
        """Async function raises on 401 when token is wrong."""
        import aiohttp as _aiohttp

        async with FakeHAServer() as server:
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_URL", server.url,
            )
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_TOKEN", "bad-token",
            )

            with pytest.raises(_aiohttp.ClientResponseError) as exc_info:
                await _async_list_entities()
            assert exc_info.value.status == 401

    @pytest.mark.asyncio
    async def test_rest_server_error(self, monkeypatch):
        """Async function raises on 500 response."""
        import aiohttp as _aiohttp

        async with FakeHAServer() as server:
            server.force_500 = True
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_URL", server.url,
            )
            monkeypatch.setattr(
                "tools.homeassistant_tool._HASS_TOKEN", server.token,
            )

            with pytest.raises(_aiohttp.ClientResponseError) as exc_info:
                await _async_list_entities()
            assert exc_info.value.status == 500

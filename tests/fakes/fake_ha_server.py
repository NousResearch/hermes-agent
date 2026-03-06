"""Fake Home Assistant server for integration testing.

Provides a real HTTP + WebSocket server (via aiohttp.web) that mimics the
Home Assistant API surface used by hermes-agent:

- ``/api/websocket``  -- WebSocket auth handshake + event push
- ``/api/states``     -- GET all entity states
- ``/api/states/{entity_id}`` -- GET single entity state
- ``/api/services/{domain}/{service}`` -- POST service call
- ``/api/services/persistent_notification/create`` -- POST notification

Usage::

    async with FakeHAServer(token="test-token") as server:
        url = server.url            # e.g. "http://127.0.0.1:54321"
        await server.push_event(event_data)
        assert server.received_notifications  # verify what arrived
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer


# -- Sample entity data -------------------------------------------------------

ENTITY_STATES: List[Dict[str, Any]] = [
    {
        "entity_id": "light.bedroom",
        "state": "on",
        "attributes": {"friendly_name": "Bedroom Light", "brightness": 200},
        "last_changed": "2025-01-15T10:30:00+00:00",
        "last_updated": "2025-01-15T10:30:00+00:00",
    },
    {
        "entity_id": "light.kitchen",
        "state": "off",
        "attributes": {"friendly_name": "Kitchen Light"},
        "last_changed": "2025-01-15T09:00:00+00:00",
        "last_updated": "2025-01-15T09:00:00+00:00",
    },
    {
        "entity_id": "sensor.temperature",
        "state": "22.5",
        "attributes": {
            "friendly_name": "Kitchen Temperature",
            "unit_of_measurement": "C",
        },
        "last_changed": "2025-01-15T10:00:00+00:00",
        "last_updated": "2025-01-15T10:00:00+00:00",
    },
    {
        "entity_id": "switch.fan",
        "state": "on",
        "attributes": {"friendly_name": "Living Room Fan"},
        "last_changed": "2025-01-15T08:00:00+00:00",
        "last_updated": "2025-01-15T08:00:00+00:00",
    },
    {
        "entity_id": "climate.thermostat",
        "state": "heat",
        "attributes": {
            "friendly_name": "Main Thermostat",
            "current_temperature": 21,
            "temperature": 23,
        },
        "last_changed": "2025-01-15T07:00:00+00:00",
        "last_updated": "2025-01-15T07:00:00+00:00",
    },
]


class FakeHAServer:
    """In-process fake Home Assistant for integration tests.

    Parameters
    ----------
    token : str
        The expected Bearer token for authentication.
    """

    def __init__(self, token: str = "test-token-123"):
        self.token = token

        # Observability -- tests inspect these after exercising the adapter.
        self.received_service_calls: List[Dict[str, Any]] = []
        self.received_notifications: List[Dict[str, Any]] = []

        # Control -- tests push events, server forwards them over WS.
        self._event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        # Flag to simulate auth rejection.
        self.reject_auth = False

        # Flag to simulate server errors.
        self.force_500 = False

        # Internal bookkeeping.
        self._app: Optional[web.Application] = None
        self._server: Optional[TestServer] = None
        self._ws_connections: List[web.WebSocketResponse] = []

    # -- Public helpers --------------------------------------------------------

    @property
    def url(self) -> str:
        """Base URL of the running server, e.g. ``http://127.0.0.1:12345``."""
        assert self._server is not None, "Server not started"
        host = self._server.host
        port = self._server.port
        return f"http://{host}:{port}"

    async def push_event(self, event_data: Dict[str, Any]) -> None:
        """Enqueue a state_changed event for delivery over WebSocket."""
        await self._event_queue.put(event_data)

    # -- Lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        self._app = self._build_app()
        self._server = TestServer(self._app)
        await self._server.start_server()

    async def stop(self) -> None:
        # Close any remaining WS connections.
        for ws in self._ws_connections:
            if not ws.closed:
                await ws.close()
        self._ws_connections.clear()
        if self._server is not None:
            await self._server.close()

    async def __aenter__(self) -> "FakeHAServer":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # -- Application construction ----------------------------------------------

    def _build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/api/websocket", self._handle_ws)
        app.router.add_get("/api/states", self._handle_get_states)
        app.router.add_get("/api/states/{entity_id}", self._handle_get_state)
        # Notification endpoint must be registered before the generic service
        # route so that it takes priority.
        app.router.add_post(
            "/api/services/persistent_notification/create",
            self._handle_notification,
        )
        app.router.add_post(
            "/api/services/{domain}/{service}",
            self._handle_call_service,
        )
        return app

    # -- Auth helper -----------------------------------------------------------

    def _check_rest_auth(self, request: web.Request) -> Optional[web.Response]:
        """Return a 401 response if the Bearer token is wrong, else None."""
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {self.token}":
            return web.Response(status=401, text="Unauthorized")
        if self.force_500:
            return web.Response(status=500, text="Internal Server Error")
        return None

    # -- WebSocket handler -----------------------------------------------------

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_connections.append(ws)

        # Step 1: auth_required
        await ws.send_json({"type": "auth_required", "ha_version": "2025.1.0"})

        # Step 2: receive auth
        msg = await ws.receive()
        if msg.type != aiohttp.WSMsgType.TEXT:
            await ws.close()
            return ws
        auth_msg = json.loads(msg.data)

        # Step 3: validate
        if self.reject_auth or auth_msg.get("access_token") != self.token:
            await ws.send_json({"type": "auth_invalid", "message": "Invalid token"})
            await ws.close()
            return ws

        await ws.send_json({"type": "auth_ok", "ha_version": "2025.1.0"})

        # Step 4: subscribe_events
        msg = await ws.receive()
        if msg.type != aiohttp.WSMsgType.TEXT:
            await ws.close()
            return ws
        sub_msg = json.loads(msg.data)
        sub_id = sub_msg.get("id", 1)

        # Step 5: ACK
        await ws.send_json({
            "id": sub_id,
            "type": "result",
            "success": True,
            "result": None,
        })

        # Step 6: push events from queue until closed
        try:
            while not ws.closed:
                try:
                    event_data = await asyncio.wait_for(
                        self._event_queue.get(), timeout=0.1,
                    )
                    await ws.send_json({
                        "id": sub_id,
                        "type": "event",
                        "event": event_data,
                    })
                except asyncio.TimeoutError:
                    continue
        except (ConnectionResetError, asyncio.CancelledError):
            pass

        return ws

    # -- REST handlers ---------------------------------------------------------

    async def _handle_get_states(self, request: web.Request) -> web.Response:
        err = self._check_rest_auth(request)
        if err:
            return err
        return web.json_response(ENTITY_STATES)

    async def _handle_get_state(self, request: web.Request) -> web.Response:
        err = self._check_rest_auth(request)
        if err:
            return err
        entity_id = request.match_info["entity_id"]
        for s in ENTITY_STATES:
            if s["entity_id"] == entity_id:
                return web.json_response(s)
        return web.Response(status=404, text=f"Entity {entity_id} not found")

    async def _handle_notification(self, request: web.Request) -> web.Response:
        err = self._check_rest_auth(request)
        if err:
            return err
        body = await request.json()
        self.received_notifications.append(body)
        return web.json_response([])

    async def _handle_call_service(self, request: web.Request) -> web.Response:
        err = self._check_rest_auth(request)
        if err:
            return err
        domain = request.match_info["domain"]
        service = request.match_info["service"]
        body = await request.json()

        self.received_service_calls.append({
            "domain": domain,
            "service": service,
            "data": body,
        })

        # Return affected entities (mimics real HA behaviour for light/switch).
        affected = []
        entity_id = body.get("entity_id")
        if entity_id:
            new_state = "on" if service == "turn_on" else "off"
            for s in ENTITY_STATES:
                if s["entity_id"] == entity_id:
                    affected.append({
                        "entity_id": entity_id,
                        "state": new_state,
                        "attributes": s.get("attributes", {}),
                    })
                    break

        return web.json_response(affected)

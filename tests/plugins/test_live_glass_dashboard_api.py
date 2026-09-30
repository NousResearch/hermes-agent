"""Tests for the live-glass dashboard WebSocket API (AVA-19)."""
from __future__ import annotations

import pytest


def _fresh_ws_app():
    """Build a standalone FastAPI app with the live-glass dashboard router."""
    from fastapi import FastAPI
    from plugins.observability.live_glass.dashboard.plugin_api import router
    from plugins.observability.live_glass import reset_event_bus_for_tests
    reset_event_bus_for_tests()
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def ws_client():
    from fastapi.testclient import TestClient
    app = _fresh_ws_app()
    with TestClient(app) as client:
        yield client


class TestWebSocketEndpoint:
    def test_connect_and_receive_events(self, ws_client):
        with ws_client.websocket_connect("/events") as ws:
            from plugins.observability.live_glass import publish
            publish("frame", {
                "image_url": "data:image/png;base64,test",
                "mime_type": "image/png",
                "source": "test",
            }, session_id="s1")

            data = ws.receive_json()
            assert data["type"] == "frame"

    def test_heartbeat(self, ws_client):
        with ws_client.websocket_connect("/events") as ws:
            # Should receive a heartbeat (at 2s, set via env var)
            for _ in range(10):
                data = ws.receive_json()
                if data["type"] == "heartbeat":
                    break

    def test_replays_last_frame_on_connect(self, ws_client):
        from plugins.observability.live_glass import publish
        publish("frame", {
            "image_url": "data:image/png;base64,replay_me",
            "source": "test",
        }, session_id="pre_connect")

        with ws_client.websocket_connect("/events") as ws:
            data = ws.receive_json()
            assert data["type"] == "frame"
            assert data["payload"]["image_url"] == "data:image/png;base64,replay_me"

    def test_multiple_concurrent_clients(self, ws_client):
        from plugins.observability.live_glass import publish

        with (
            ws_client.websocket_connect("/events") as ws1,
            ws_client.websocket_connect("/events") as ws2,
        ):
            publish("log", {"tool_name": "multi", "status": "ok",
                            "duration_ms": 42, "source": "test"},
                    session_id="s1")

            # Both clients should receive the log event.
            found1 = _receive_until_type(ws1, "log")
            found2 = _receive_until_type(ws2, "log")
            assert found1["payload"]["tool_name"] == "multi"
            assert found2["payload"]["tool_name"] == "multi"

    def test_disconnect_cleanup(self, ws_client):
        from plugins.observability.live_glass import publish

        with ws_client.websocket_connect("/events") as ws:
            _ = ws.receive_json()  # heartbeat or replay

        # After disconnect, publish — should not raise.
        publish("log", {"tool_name": "after_disconnect", "status": "ok",
                        "duration_ms": 1, "source": "test"},
                session_id="s1")

        # Reconnect — bus is still healthy.
        with ws_client.websocket_connect("/events") as ws:
            publish("log", {"tool_name": "reconnect_test", "status": "ok",
                           "duration_ms": 1, "source": "test"},
                    session_id="s1")
            data = _receive_until_type(ws, "log")
            assert data["payload"]["tool_name"] == "reconnect_test"


def _receive_until_type(ws, event_type: str, max_attempts: int = 20):
    for _ in range(max_attempts):
        data = ws.receive_json()
        if data.get("type") == event_type:
            return data
    raise AssertionError(f"never received event type {event_type!r}")

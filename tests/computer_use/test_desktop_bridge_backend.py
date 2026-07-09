from __future__ import annotations

import asyncio
import json

import pytest

from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use.bridge import capture_to_payload
from tools.computer_use.desktop_bridge import (
    DesktopBridgeBroker,
    DesktopComputerUseBridgeBackend,
    desktop_bridge_computer_use_status,
)


class _DoneFuture:
    def result(self, timeout=None):
        return None


class _FakeWs:
    async def send_text(self, text):
        self.sent = text


def test_desktop_bridge_broker_sends_request_and_returns_reply(monkeypatch):
    broker = DesktopBridgeBroker()
    ws = _FakeWs()

    with broker._lock:  # broker internals deliberately covered by this unit test
        broker._ws = ws
        broker._loop = object()

    def fake_run_coroutine_threadsafe(coro, _loop):
        asyncio.run(coro)
        frame = json.loads(ws.sent)
        assert frame["type"] == "status"
        broker._handle_message({"id": frame["id"], "ok": True, "result": {"ready": True}})
        return _DoneFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)

    assert broker.request({"type": "status"}, timeout=1) == {"ready": True}


def test_desktop_bridge_backend_round_trips_capture(monkeypatch):
    import tools.computer_use.desktop_bridge as desktop_bridge

    class FakeBroker:
        def is_connected(self):
            return True

        def connection_info(self):
            return {"connected": True, "client_id": "desktop-test", "pending": 0}

        def request(self, payload, timeout=None):
            if payload["type"] == "status":
                return {"ready": True, "checks": []}
            assert payload == {"type": "computer-use", "method": "capture", "args": {"mode": "ax", "app": "Finder"}}
            return capture_to_payload(
                CaptureResult(
                    mode="ax",
                    width=800,
                    height=600,
                    elements=[UIElement(index=1, role="AXButton", label="OK")],
                    app="Finder",
                )
            )

    monkeypatch.setattr(desktop_bridge, "_BROKER", FakeBroker())

    backend = DesktopComputerUseBridgeBackend()
    backend.start()
    capture = backend.capture(mode="ax", app="Finder")

    assert capture.app == "Finder"
    assert capture.elements[0].label == "OK"


def test_desktop_bridge_status_reports_offline_without_secret(monkeypatch):
    import tools.computer_use.desktop_bridge as desktop_bridge

    class OfflineBroker:
        def is_connected(self):
            return False

    monkeypatch.setattr(desktop_bridge, "_BROKER", OfflineBroker())

    status = desktop_bridge_computer_use_status()

    assert status["platform"] == "desktop-bridge"
    assert status["ready"] is False
    assert status["bridge"] == {"kind": "desktop", "connected": False}
    assert "token" not in json.dumps(status).lower()

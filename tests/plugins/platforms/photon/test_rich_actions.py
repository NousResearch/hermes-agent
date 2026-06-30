"""Rich native iMessage action tests for PhotonAdapter."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture_sidecar(adapter: PhotonAdapter) -> List[Tuple[str, Dict[str, Any]]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        return {"ok": True, "messageId": "msg-123"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


@pytest.mark.asyncio
async def test_send_with_effect_posts_send_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_with_effect("+15551234567", "boom", "confetti")

    assert result.success is True
    assert result.message_id == "msg-123"
    assert calls == [
        (
            "/send-effect",
            {
                "spaceId": "+15551234567",
                "text": "boom",
                "effect": "confetti",
                "format": "markdown",
            },
        )
    ]


@pytest.mark.asyncio
async def test_reply_to_message_posts_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.reply_to_message("+15551234567", "inbound-1", "threaded")

    assert result.success is True
    assert calls == [
        (
            "/reply",
            {
                "spaceId": "+15551234567",
                "messageId": "inbound-1",
                "text": "threaded",
                "format": "markdown",
            },
        )
    ]


@pytest.mark.asyncio
async def test_edit_native_message_posts_edit(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.edit_native_message("+15551234567", "outbound-1", "edited")

    assert result.success is True
    assert calls == [
        (
            "/edit",
            {
                "spaceId": "+15551234567",
                "messageId": "outbound-1",
                "text": "edited",
                "format": "markdown",
            },
        )
    ]


@pytest.mark.asyncio
async def test_unsend_message_posts_unsend(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.unsend_message("+15551234567", "outbound-1")

    assert result.success is True
    assert calls == [
        (
            "/unsend",
            {"spaceId": "+15551234567", "messageId": "outbound-1"},
        )
    ]


@pytest.mark.asyncio
async def test_send_poll_posts_send_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_poll("+15551234567", "Lunch?", ["Sushi", "Pizza"])

    assert result.success is True
    assert calls == [
        (
            "/send-poll",
            {
                "spaceId": "+15551234567",
                "question": "Lunch?",
                "options": ["Sushi", "Pizza"],
            },
        )
    ]


@pytest.mark.asyncio
async def test_send_mini_app_posts_send_mini_app(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_mini_app(
        "+15551234567",
        app_name="Hermes",
        extension_bundle_id="net.example.MessagesExtension",
        team_id="TEAM12345",
        url="hermes://run/abc",
        layout={"caption": "Done", "summary": "Hermes finished"},
    )

    assert result.success is True
    assert calls == [
        (
            "/send-mini-app",
            {
                "spaceId": "+15551234567",
                "appName": "Hermes",
                "extensionBundleId": "net.example.MessagesExtension",
                "teamId": "TEAM12345",
                "url": "hermes://run/abc",
                "layout": {"caption": "Done", "summary": "Hermes finished"},
            },
        )
    ]


@pytest.mark.asyncio
async def test_send_contact_posts_send_contact(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_contact(
        "+15551234567",
        name="Daniel Pinto",
        phone=["+15551234567"],
    )

    assert result.success is True
    assert calls == [
        (
            "/send-contact",
            {
                "spaceId": "+15551234567",
                "name": "Daniel Pinto",
                "phone": ["+15551234567"],
            },
        )
    ]


@pytest.mark.asyncio
async def test_set_background_posts_set_background(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.set_background("+15551234567", "/tmp/bg.png")

    assert result.success is True
    assert calls == [
        (
            "/set-background",
            {"spaceId": "+15551234567", "clear": False, "background": "/tmp/bg.png"},
        )
    ]

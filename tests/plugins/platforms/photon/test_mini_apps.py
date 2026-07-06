"""Feature-gated mini-app launch tests for PhotonAdapter."""
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
        return {"ok": True, "messageId": "mini-app-msg-123"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


@pytest.mark.asyncio
async def test_send_mini_app_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MINI_APPS", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_mini_app(
        "+155****4567",
        "https://hermes.example.test/control-room",
    )

    assert result.success is False
    assert "disabled" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_send_mini_app_rejects_non_https_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_MINI_APPS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_mini_app("+155****4567", "http://localhost/control-room")

    assert result.success is False
    assert "https" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_send_mini_app_posts_minimal_app_url_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_MINI_APPS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_mini_app(
        "+155****4567",
        " https://hermes.example.test/control-room?token=redacted ",
    )

    assert result.success is True
    assert result.message_id == "mini-app-msg-123"
    assert calls == [
        (
            "/send-mini-app",
            {
                "spaceId": "+155****4567",
                "url": "https://hermes.example.test/control-room?token=redacted",
            },
        )
    ]


@pytest.mark.asyncio
async def test_send_mini_app_accepts_explicit_metadata_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_MINI_APPS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_mini_app(
        "+155****4567",
        "https://hermes.example.test/control-room",
        app_name="Hermes Control Room",
        extension_bundle_id="codes.nous.hermes.ControlRoom.MessagesExtension",
        team_id="TEAM12345",
    )

    assert result.success is True
    assert calls == [
        (
            "/send-mini-app",
            {
                "spaceId": "+155****4567",
                "url": "https://hermes.example.test/control-room",
                "appName": "Hermes Control Room",
                "extensionBundleId": "codes.nous.hermes.ControlRoom.MessagesExtension",
                "teamId": "TEAM12345",
            },
        )
    ]

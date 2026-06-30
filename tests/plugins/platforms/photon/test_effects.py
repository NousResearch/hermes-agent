"""Native iMessage effect tests for PhotonAdapter.

The effect path is intentionally narrow and feature-gated. These tests stub
``_sidecar_call`` and do not spawn the Node sidecar or bind ports.
"""
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
        return {"ok": True, "messageId": "effect-msg-123"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


@pytest.mark.asyncio
async def test_send_with_effect_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_NATIVE_EFFECTS", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_with_effect("+155****4567", "hello", "confetti")

    assert result.success is False
    assert "disabled" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_send_with_effect_posts_minimal_effect_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_EFFECTS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_with_effect("+155****4567", "**hello**", "confetti")

    assert result.success is True
    assert result.message_id == "effect-msg-123"
    assert calls == [
        (
            "/send-effect",
            {
                "spaceId": "+155****4567",
                "text": "**hello**",
                "effect": "confetti",
                "format": "markdown",
            },
        )
    ]


@pytest.mark.asyncio
async def test_send_with_effect_respects_markdown_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_EFFECTS", "true")
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_with_effect("+155****4567", "hello", "slam")

    assert result.success is True
    assert calls == [
        (
            "/send-effect",
            {"spaceId": "+155****4567", "text": "hello", "effect": "slam"},
        )
    ]

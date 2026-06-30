"""Native iMessage reply tests for PhotonAdapter.

The reply path is deliberately narrow and feature-gated. These tests stub
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
        return {"ok": True, "messageId": "reply-msg-123"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


@pytest.mark.asyncio
async def test_reply_to_message_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_NATIVE_REPLIES", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.reply_to_message(
        "+155****4567", "target-msg-1", "native reply"
    )

    assert result.success is False
    assert "disabled" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_reply_to_message_posts_minimal_reply_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_REPLIES", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.reply_to_message(
        "+155****4567", "target-msg-1", "**native reply**"
    )

    assert result.success is True
    assert result.message_id == "reply-msg-123"
    assert result.raw_response == {"ok": True, "messageId": "reply-msg-123"}
    assert calls == [
        (
            "/reply",
            {
                "spaceId": "+155****4567",
                "messageId": "target-msg-1",
                "text": "**native reply**",
                "format": "markdown",
            },
        )
    ]


@pytest.mark.asyncio
async def test_reply_to_message_respects_markdown_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_REPLIES", "true")
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.reply_to_message(
        "+155****4567", "target-msg-1", "native reply"
    )

    assert result.success is True
    assert calls == [
        (
            "/reply",
            {
                "spaceId": "+155****4567",
                "messageId": "target-msg-1",
                "text": "native reply",
            },
        )
    ]


@pytest.mark.asyncio
async def test_send_uses_native_reply_when_reply_anchor_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_REPLIES", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send(
        "+155****4567", "normal gateway reply", reply_to="target-msg-1"
    )

    assert result.success is True
    assert result.message_id == "reply-msg-123"
    assert calls == [
        (
            "/reply",
            {
                "spaceId": "+155****4567",
                "messageId": "target-msg-1",
                "text": "normal gateway reply",
                "format": "markdown",
            },
        )
    ]


@pytest.mark.asyncio
async def test_send_reply_anchor_falls_back_when_native_replies_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_NATIVE_REPLIES", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send(
        "+155****4567", "normal gateway reply", reply_to="target-msg-1"
    )

    assert result.success is True
    assert result.message_id == "reply-msg-123"
    assert calls == [
        (
            "/send",
            {
                "spaceId": "+155****4567",
                "text": "normal gateway reply",
                "format": "markdown",
            },
        )
    ]

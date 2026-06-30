"""Native iMessage poll tests for PhotonAdapter.

Polls are feature-gated. Hermes exposes creation plus the remaining explicit
poll mutation helpers from OpenClaw: add option, vote, and unvote.
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
        return {
            "ok": True,
            "messageId": "poll-msg-123",
            "pollMessageId": body.get("pollMessageId", "poll-msg-123"),
        }

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


@pytest.mark.asyncio
async def test_send_poll_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHOTON_NATIVE_POLLS", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_poll("+155****4567", "Tea?", ["Yes", "No"])

    assert result.success is False
    assert "disabled" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_send_poll_requires_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_POLLS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_poll("+155****4567", "  ", ["Yes", "No"])

    assert result.success is False
    assert "question" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_send_poll_requires_two_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_POLLS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_poll("+155****4567", "Tea?", ["Yes", " "])

    assert result.success is False
    assert "two options" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_send_poll_posts_trimmed_creation_payload_and_tracks_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_POLLS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_poll(
        "+155****4567", "  Tea?  ", [" Yes ", "No", " "]
    )

    assert result.success is True
    assert result.message_id == "poll-msg-123"
    assert "poll-msg-123" in adapter._sent_message_ids
    assert calls == [
        (
            "/send-poll",
            {
                "spaceId": "+155****4567",
                "question": "Tea?",
                "options": ["Yes", "No"],
            },
        )
    ]


@pytest.mark.asyncio
async def test_add_poll_option_posts_trimmed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_POLLS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.add_poll_option("+155****4567", "poll-msg-123", " Maybe ")

    assert result.success is True
    assert result.message_id == "poll-msg-123"
    assert calls == [
        (
            "/poll-add-option",
            {
                "spaceId": "+155****4567",
                "pollMessageId": "poll-msg-123",
                "option": "Maybe",
            },
        )
    ]


@pytest.mark.asyncio
async def test_add_poll_option_requires_option(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_POLLS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.add_poll_option("+155****4567", "poll-msg-123", " ")

    assert result.success is False
    assert "option" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_vote_poll_posts_option_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_POLLS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.vote_poll("+155****4567", "poll-msg-123", " option-1 ")

    assert result.success is True
    assert result.message_id == "poll-msg-123"
    assert calls == [
        (
            "/poll-vote",
            {
                "spaceId": "+155****4567",
                "pollMessageId": "poll-msg-123",
                "optionId": "option-1",
            },
        )
    ]


@pytest.mark.asyncio
async def test_vote_poll_requires_option_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_POLLS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.vote_poll("+155****4567", "poll-msg-123", " ")

    assert result.success is False
    assert "optionId" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_unvote_poll_posts_poll_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_POLLS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.unvote_poll("+155****4567", "poll-msg-123")

    assert result.success is True
    assert result.message_id == "poll-msg-123"
    assert calls == [
        (
            "/poll-unvote",
            {"spaceId": "+155****4567", "pollMessageId": "poll-msg-123"},
        )
    ]

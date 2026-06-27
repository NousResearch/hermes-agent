"""Group-chat mention-gating tests for PhotonAdapter.

Parity with the BlueBubbles iMessage channel: when ``require_mention`` is
enabled, group messages are dropped unless they hit a wake-word pattern,
and the leading wake word is stripped from the ones that pass. DMs are
never gated.

These call ``_dispatch_inbound`` directly (no aiohttp / ports) and assert
on what reaches ``handle_message``.
"""
from __future__ import annotations

from typing import List

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch, extra: dict | None = None) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    monkeypatch.delenv("PHOTON_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("PHOTON_MENTION_PATTERNS", raising=False)
    cfg = PlatformConfig(enabled=True, token="", extra=extra or {})
    return PhotonAdapter(cfg)


def _group_payload(text: str) -> dict:
    return {
        "messageId": f"grp-{abs(hash(text))}",
        "space": {"id": "group-guid-xyz", "type": "group", "phone": None},
        "sender": {"id": "+15551234567"},
        "content": {"type": "text", "text": text},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


def _dm_payload(text: str) -> dict:
    return {
        "messageId": f"dm-{abs(hash(text))}",
        "space": {"id": "+155****4567", "type": "dm", "phone": "+155****4567"},
        "sender": {"id": "+155****4567"},
        "content": {"type": "text", "text": text},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


def _richlink_payload(url: str, *, title: str | None = None) -> dict:
    content = {"type": "richlink", "url": url}
    if title is not None:
        content["title"] = title
    return {
        "messageId": f"link-{abs(hash((url, title)))}",
        "space": {"id": "+155****4567", "type": "dm", "phone": "+155****4567"},
        "sender": {"id": "+155****4567"},
        "content": content,
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


def _group_with_richlink_payload(text: str, url: str) -> dict:
    return {
        "messageId": f"grp-link-{abs(hash((text, url)))}",
        "space": {"id": "group-guid-xyz", "type": "group", "phone": None},
        "sender": {"id": "+155****4567"},
        "content": {
            "type": "group",
            "items": [
                {"content": {"type": "text", "text": text}},
                {"content": {"type": "richlink", "url": url}},
            ],
        },
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


def _capture(adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def test_require_mention_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    assert adapter.require_mention is False
    # Defaults compile to the two Hermes wake-word patterns.
    assert len(adapter._mention_patterns) == 2


@pytest.mark.asyncio
async def test_group_message_dropped_without_mention(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch, extra={"require_mention": True})
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_group_payload("just chatting, no wake word"))
    assert captured == []


@pytest.mark.asyncio
async def test_group_message_passes_and_strips_wake_word(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch, extra={"require_mention": True})
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_group_payload("Hermes what's the weather"))
    assert len(captured) == 1
    # Leading wake word stripped before dispatch.
    assert captured[0].text == "what's the weather"


@pytest.mark.asyncio
async def test_dm_never_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch, extra={"require_mention": True})
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_dm_payload("no wake word here"))
    assert len(captured) == 1
    assert captured[0].text == "no wake word here"


@pytest.mark.asyncio
async def test_dm_richlink_extracts_url(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(
        _richlink_payload("https://example.com/article", title="Example Article")
    )
    assert len(captured) == 1
    assert captured[0].text == "Example Article\nhttps://example.com/article"


@pytest.mark.asyncio
async def test_group_richlink_preserves_text_and_url(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(
        _group_with_richlink_payload("Evaluate this", "https://example.com/article")
    )
    assert len(captured) == 1
    assert captured[0].text == "Evaluate this\nhttps://example.com/article"


@pytest.mark.asyncio
async def test_require_mention_off_passes_group_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)  # require_mention defaults off
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_group_payload("plain group chatter"))
    assert len(captured) == 1
    assert captured[0].text == "plain group chatter"


def test_custom_mention_patterns_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(
        monkeypatch,
        extra={"require_mention": True, "mention_patterns": [r"(?<![\w@])@?amos\b[,:\-]?"]},
    )
    assert adapter.require_mention is True
    assert len(adapter._mention_patterns) == 1
    assert adapter._message_matches_mention_patterns("amos help me") is True
    assert adapter._message_matches_mention_patterns("hermes help me") is False


def test_mention_patterns_env_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    monkeypatch.setenv("PHOTON_REQUIRE_MENTION", "true")
    monkeypatch.setenv("PHOTON_MENTION_PATTERNS", r"bot\b, assistant\b")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    adapter = PhotonAdapter(cfg)
    assert adapter.require_mention is True
    assert len(adapter._mention_patterns) == 2
    assert adapter._message_matches_mention_patterns("hey bot") is True


def test_invalid_pattern_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(
        monkeypatch,
        extra={"require_mention": True, "mention_patterns": ["(unclosed", r"good\b"]},
    )
    # Bad regex dropped, good one kept.
    assert len(adapter._mention_patterns) == 1
    assert adapter._message_matches_mention_patterns("a good thing") is True

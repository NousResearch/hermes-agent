"""Tests for WhatsApp owner-message metadata and source-level text tagging.

The Node bridge sets ``fromOwner: true`` on inbound `fromMe` messages that
look owner-typed (linked-device send, not echoed from /send) when the
operator opts into ``WHATSAPP_FORWARD_OWNER_MESSAGES``.  These tests pin
the adapter's responsibility: lift that flag onto
``MessageEvent.metadata["whatsapp_from_owner"]``, prefix ``MessageEvent.text``
with ``[owner reply] ``, and otherwise leave metadata absent and text
unchanged.  The env-var gate itself lives in the bridge — the adapter just
trusts the payload.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.fixture(autouse=True)
def _whatsapp_open_optin(monkeypatch):
    """Opt into WhatsApp allow-all so ``dm_policy: open`` dispatch tests run.

    The adapter fails closed on ``open`` without an allow-all opt-in
    (SECURITY.md 2.6); these owner-DM tests set ``_dm_policy = "open"``.
    """
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")


def _make_adapter():
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._mention_patterns = []
    adapter._free_response_chats = set()
    adapter._whatsapp_free_response_chats = lambda: set()
    return adapter


def _dm_payload(**overrides):
    payload = {
        "messageId": "M1",
        "chatId": "6281234567890@s.whatsapp.net",
        "senderId": "6281234567890@s.whatsapp.net",
        "senderName": "Customer",
        "chatName": "Customer",
        "isGroup": False,
        "body": "hi from the linked phone",
        "hasMedia": False,
        "mediaType": "",
        "mediaUrls": [],
        "mentionedIds": [],
        "quotedParticipant": "",
        "botIds": [],
        "timestamp": 0,
    }
    payload.update(overrides)
    return payload


def test_metadata_flag_set_when_payload_has_from_owner():
    adapter = _make_adapter()
    payload = _dm_payload(fromOwner=True)

    event = asyncio.run(adapter._build_message_event(payload))

    assert event is not None
    assert event.metadata.get("whatsapp_from_owner") is True
    assert event.text.startswith("[owner reply] ")
    assert event.text == "[owner reply] hi from the linked phone"


def test_from_owner_does_not_double_prefix_when_already_tagged():
    adapter = _make_adapter()
    payload = _dm_payload(
        fromOwner=True,
        body="[owner reply] already tagged",
    )

    event = asyncio.run(adapter._build_message_event(payload))

    assert event is not None
    assert event.metadata.get("whatsapp_from_owner") is True
    assert event.text == "[owner reply] already tagged"


def test_from_owner_prefixes_empty_body_for_uniform_media_placeholders():
    """Owner media with empty caption still gets the marker (bridge may
    substitute placeholders like ``[image received]`` upstream; empty stays
    tagged for consistency)."""
    adapter = _make_adapter()
    payload = _dm_payload(fromOwner=True, body="")

    event = asyncio.run(adapter._build_message_event(payload))

    assert event is not None
    assert event.metadata.get("whatsapp_from_owner") is True
    assert event.text == "[owner reply] "


def test_metadata_flag_absent_by_default():
    """Default bridge payload (env flag off → field never present) must not
    leak the metadata key.  Plugins use ``.get(...)`` and rely on absence."""
    adapter = _make_adapter()
    payload = _dm_payload()

    event = asyncio.run(adapter._build_message_event(payload))

    assert event is not None
    assert "whatsapp_from_owner" not in event.metadata


def test_metadata_flag_absent_when_explicitly_false():
    """Explicit fromOwner=false must not set the metadata key — plugins
    test for truthiness, but absence is the canonical "not owner" state."""
    adapter = _make_adapter()
    payload = _dm_payload(fromOwner=False)

    event = asyncio.run(adapter._build_message_event(payload))

    assert event is not None
    assert "whatsapp_from_owner" not in event.metadata


def test_plaintext_skill_is_markdown_and_injected(tmp_path, monkeypatch):
    skill_path = tmp_path / "demo.skill"
    skill_path.write_text("---\nname: whatsapp-demo\n---\n# Instructions")
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path",
        lambda _path: True,
    )
    adapter = _make_adapter()
    payload = _dm_payload(
        body="",
        hasMedia=True,
        mediaType="document",
        mediaUrls=[str(skill_path)],
        mime="application/octet-stream",
    )

    event = asyncio.run(adapter._build_message_event(payload))

    assert event is not None
    assert event.media_types == ["text/markdown"]
    assert "[Content of demo.skill]:" in event.text
    assert "name: whatsapp-demo" in event.text


def test_zip_skill_is_archive_and_not_injected(tmp_path, monkeypatch):
    skill_path = tmp_path / "bundle.skill"
    skill_path.write_bytes(b"PK\x03\x04skill-bundle")
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path",
        lambda _path: True,
    )
    adapter = _make_adapter()
    payload = _dm_payload(
        body="",
        hasMedia=True,
        mediaType="document",
        mediaUrls=[str(skill_path)],
        mime="application/octet-stream",
    )

    event = asyncio.run(adapter._build_message_event(payload))

    assert event is not None
    assert event.media_types == ["application/zip"]
    assert "Content of bundle.skill" not in (event.text or "")

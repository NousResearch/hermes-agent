"""Tests for WhatsApp owner-message metadata and source-level text tagging.

The Node bridge sets ``fromOwner: true`` on inbound `fromMe` messages that
look owner-typed (linked-device send, not echoed from /send) when the
operator opts into ``forward_owner_messages`` (projected internally to the
bridge environment). These tests pin
the adapter's responsibility: lift that flag onto
``MessageEvent.metadata["whatsapp_from_owner"]``, prefix ``MessageEvent.text``
with ``[owner reply] ``, and otherwise leave metadata absent and text
unchanged.  The gate itself lives in the bridge; the adapter owns its resolved config.
"""

from __future__ import annotations

import asyncio
import stat
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.whatsapp import adapter as whatsapp_adapter
from plugins.platforms.whatsapp.adapter import (
    WhatsAppAdapter,
    _load_or_create_owner_message_secret,
)


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


def test_yaml_owner_forwarding_setting_overrides_internal_environment(monkeypatch):
    monkeypatch.setenv("WHATSAPP_FORWARD_OWNER_MESSAGES", "false")
    config = PlatformConfig(enabled=True, extra={"forward_owner_messages": True})

    adapter = WhatsAppAdapter(config)

    assert adapter._forward_owner_messages is True


def test_owner_message_secret_is_strong_private_and_stable_per_session(tmp_path):
    secret_path = tmp_path / "consumer.secret"

    first = _load_or_create_owner_message_secret(secret_path)
    second = _load_or_create_owner_message_secret(secret_path)

    assert first == second
    assert len(first) >= 43
    assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600


def test_owner_message_secret_uses_path_chmod_when_fchmod_is_unavailable(
    tmp_path, monkeypatch
):
    secret_path = tmp_path / "consumer.secret"
    monkeypatch.delattr(whatsapp_adapter.os, "fchmod")

    first = _load_or_create_owner_message_secret(secret_path)
    second = _load_or_create_owner_message_secret(secret_path)

    assert first == second
    assert len(first) >= 43
    assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600


def test_owner_message_consumer_secret_file_is_configurable(tmp_path):
    secret_path = tmp_path / "integrations" / "owner-consumer.secret"
    config = PlatformConfig(
        enabled=True,
        extra={
            "forward_owner_messages": True,
            "session_path": str(secret_path.parent),
            "owner_message_secret_file": str(secret_path),
        },
    )

    adapter = WhatsAppAdapter(config)
    adapter._ensure_owner_message_secret()

    assert adapter._owner_message_secret_file == secret_path
    assert adapter._owner_message_secret == secret_path.read_text(encoding="utf-8").strip()


def test_owner_message_secret_rejects_symlinked_parent_outside_session(tmp_path):
    session_path = tmp_path / "session"
    outside = tmp_path / "outside"
    session_path.mkdir()
    outside.mkdir()
    linked_parent = session_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)
    config = PlatformConfig(
        enabled=True,
        extra={
            "forward_owner_messages": True,
            "session_path": str(session_path),
            "owner_message_secret_file": str(linked_parent / "owner.secret"),
        },
    )

    with pytest.raises(ValueError, match="inside the WhatsApp session directory"):
        WhatsAppAdapter(config)._ensure_owner_message_secret()


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



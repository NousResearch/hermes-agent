"""Tests for the Facebook Messenger platform plugin."""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_messenger = load_plugin_adapter("messenger")

MessengerAdapter = _messenger.MessengerAdapter
MessageType = _messenger.MessageType
_MessageDeduplicator = _messenger._MessageDeduplicator
_apply_yaml_config = _messenger._apply_yaml_config
_env_enablement = _messenger._env_enablement
parse_accounts = _messenger.parse_accounts
register = _messenger.register
split_for_messenger = _messenger.split_for_messenger
strip_markdown_for_messenger = _messenger.strip_markdown_for_messenger
validate_config = _messenger.validate_config
verify_messenger_signature = _messenger.verify_messenger_signature


def _signature(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_verify_messenger_signature_accepts_valid_header():
    body = b'{"object":"page"}'
    assert verify_messenger_signature(body, _signature(body, "secret"), "secret")


def test_verify_messenger_signature_rejects_tampered_body():
    body = b'{"object":"page"}'
    assert not verify_messenger_signature(body + b" ", _signature(body, "secret"), "secret")


def test_verify_messenger_signature_rejects_missing_sha256_prefix():
    body = b"{}"
    assert not verify_messenger_signature(body, _signature(body, "secret").removeprefix("sha256="), "secret")


def test_split_for_messenger_respects_2000_char_cap():
    chunks = split_for_messenger("a" * 1999 + " " + "b" * 1999)
    assert len(chunks) == 2
    assert all(len(chunk) <= 2000 for chunk in chunks)


def test_strip_markdown_preserves_links_as_plain_urls():
    text = strip_markdown_for_messenger("**bold** [OpenAI](https://openai.com) `code`")
    assert "**" not in text
    assert "`" not in text
    assert "OpenAI (https://openai.com)" in text


def test_parse_accounts_reads_default_env(monkeypatch):
    monkeypatch.setenv("MESSENGER_PAGE_ACCESS_TOKEN", "page-token")
    monkeypatch.setenv("MESSENGER_APP_SECRET", "app-secret")
    monkeypatch.setenv("MESSENGER_VERIFY_TOKEN", "verify-token")
    accounts = parse_accounts({})
    assert accounts["default"].page_access_token == "page-token"
    assert accounts["default"].complete


def test_validate_config_accepts_yaml_only_credentials(monkeypatch):
    monkeypatch.delenv("MESSENGER_PAGE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("MESSENGER_APP_SECRET", raising=False)
    monkeypatch.delenv("MESSENGER_VERIFY_TOKEN", raising=False)
    config = PlatformConfig(
        enabled=True,
        extra={
            "page_access_token": "page-token",
            "app_secret": "app-secret",
            "verify_token": "verify-token",
        },
    )
    assert validate_config(config)


def test_env_enablement_seeds_runtime_options(monkeypatch):
    monkeypatch.setenv("MESSENGER_PAGE_ACCESS_TOKEN", "page-token")
    monkeypatch.setenv("MESSENGER_APP_SECRET", "app-secret")
    monkeypatch.setenv("MESSENGER_VERIFY_TOKEN", "verify-token")
    monkeypatch.setenv("MESSENGER_PORT", "9999")
    monkeypatch.setenv("MESSENGER_WEBHOOK_PATH", "messenger/custom")
    seeded = _env_enablement()
    assert seeded["port"] == 9999
    assert seeded["webhook_path"] == "/messenger/custom"
    assert seeded["dm_policy"] == "pairing"


def test_apply_yaml_config_translates_direct_and_extra_keys():
    seeded = _apply_yaml_config(
        {},
        {
            "host": "127.0.0.1",
            "extra": {
                "pageAccessToken": "page-token",
                "appSecret": "app-secret",
                "verifyToken": "verify-token",
            },
        },
    )
    assert seeded["host"] == "127.0.0.1"
    assert seeded["pageAccessToken"] == "page-token"
    assert seeded["appSecret"] == "app-secret"
    assert seeded["verifyToken"] == "verify-token"


def test_deduplicator_tracks_repeated_mid():
    dedup = _MessageDeduplicator(max_size=2)
    assert not dedup.is_duplicate("m1")
    assert dedup.is_duplicate("m1")
    assert not dedup.is_duplicate("")
    assert not dedup.is_duplicate("")


def test_register_metadata():
    class FakeCtx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kwargs):
            self.kwargs = kwargs

    ctx = FakeCtx()
    register(ctx)
    assert ctx.kwargs["name"] == "messenger"
    assert ctx.kwargs["allowed_users_env"] == "MESSENGER_ALLOWED_USERS"
    assert ctx.kwargs["allow_all_env"] == "MESSENGER_ALLOW_ALL_USERS"
    assert ctx.kwargs["cron_deliver_env_var"] == "MESSENGER_HOME_CHANNEL"


def test_adapter_exposes_gateway_dm_policy_name(monkeypatch):
    monkeypatch.delenv("MESSENGER_DM_POLICY", raising=False)
    config = PlatformConfig(
        enabled=True,
        extra={
            "dm_policy": "disabled",
            "page_access_token": "page-token",
            "app_secret": "app-secret",
            "verify_token": "verify-token",
        },
    )
    adapter = MessengerAdapter(config)
    assert adapter._dm_policy == "disabled"
    assert adapter.dm_policy == "disabled"


def test_package_exports_register():
    module_name = "plugin_package_messenger_test"
    init_path = Path(__file__).resolve().parents[2] / "plugins" / "platforms" / "messenger" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        assert callable(module.register)
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_to_message_event_builds_scoped_source_and_attachment_note(monkeypatch):
    config = PlatformConfig(
        enabled=True,
        extra={
            "accounts": {
                "work": {
                    "page_access_token": "page-token",
                    "app_secret": "app-secret",
                    "verify_token": "verify-token",
                }
            }
        },
    )
    adapter = MessengerAdapter(config)
    account = adapter._complete_accounts["work"]

    cached = type(
        "Cached",
        (),
        {
            "path": "/tmp/messenger-photo.jpg",
            "media_type": "image/jpeg",
            "context_note": lambda self: "[image saved at: /tmp/messenger-photo.jpg]",
        },
    )()
    monkeypatch.setattr(adapter, "_download_attachment", AsyncMock(return_value=cached))
    event = await adapter._to_message_event(
        account,
        {
            "sender": {"id": "PSID123"},
            "recipient": {"id": "PAGE"},
            "message": {
                "mid": "mid.1",
                "text": "hello",
                "attachments": [{"type": "image", "payload": {"url": "https://example.com/a.jpg"}}],
            },
        },
    )

    assert event.source.chat_id == "work:PSID123"
    assert event.source.user_id == "PSID123"
    assert event.message_type is MessageType.PHOTO
    assert event.media_urls == ["/tmp/messenger-photo.jpg"]
    assert "hello" in event.text


@pytest.mark.asyncio
async def test_disabled_dm_policy_drops_inbound_before_dispatch(monkeypatch):
    config = PlatformConfig(
        enabled=True,
        extra={
            "dm_policy": "disabled",
            "page_access_token": "page-token",
            "app_secret": "app-secret",
            "verify_token": "verify-token",
        },
    )
    adapter = MessengerAdapter(config)
    account = adapter._complete_accounts["default"]
    handle_message = AsyncMock()
    monkeypatch.setattr(adapter, "handle_message", handle_message)
    await adapter._process_messaging_event(
        account,
        {
            "sender": {"id": "PSID123"},
            "recipient": {"id": "PAGE"},
            "message": {"mid": "mid.1", "text": "hello"},
        },
    )

    handle_message.assert_not_called()

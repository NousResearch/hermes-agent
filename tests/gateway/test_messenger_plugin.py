"""Tests for the Facebook Messenger platform plugin."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_messenger = load_plugin_adapter("messenger")

MessengerAdapter = _messenger.MessengerAdapter
MessengerGraphError = _messenger.MessengerGraphError
MessageType = _messenger.MessageType
_MessageDeduplicator = _messenger._MessageDeduplicator
_YAML_BRIDGED_ENV_KEYS = _messenger._YAML_BRIDGED_ENV_KEYS
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


def test_adapter_disables_streaming_message_editing():
    assert MessengerAdapter.SUPPORTS_MESSAGE_EDITING is False


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


def test_validate_config_rejects_duplicate_or_reserved_webhook_paths(monkeypatch):
    monkeypatch.delenv("MESSENGER_PAGE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("MESSENGER_APP_SECRET", raising=False)
    monkeypatch.delenv("MESSENGER_VERIFY_TOKEN", raising=False)
    monkeypatch.delenv("MESSENGER_WEBHOOK_PATH", raising=False)
    duplicate = PlatformConfig(
        enabled=True,
        extra={
            "accounts": {
                "a": {
                    "page_access_token": "page-a",
                    "app_secret": "secret-a",
                    "verify_token": "verify-a",
                    "webhook_path": "/messenger/page",
                },
                "b": {
                    "page_access_token": "page-b",
                    "app_secret": "secret-b",
                    "verify_token": "verify-b",
                    "webhook_path": "/messenger/page",
                },
            },
        },
    )
    reserved = PlatformConfig(
        enabled=True,
        extra={
            "accounts": {
                "a": {
                    "page_access_token": "page",
                    "app_secret": "secret",
                    "verify_token": "verify",
                    "webhook_path": "/messenger/health",
                }
            },
        },
    )

    assert not validate_config(duplicate)
    assert not validate_config(reserved)


def test_env_enablement_seeds_runtime_options(monkeypatch):
    monkeypatch.setenv("MESSENGER_PAGE_ACCESS_TOKEN", "page-token")
    monkeypatch.setenv("MESSENGER_APP_SECRET", "app-secret")
    monkeypatch.setenv("MESSENGER_VERIFY_TOKEN", "verify-token")
    monkeypatch.delenv("MESSENGER_DM_POLICY", raising=False)
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


def test_apply_yaml_config_bridges_nested_auth_and_direct_overrides_extra(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("MESSENGER_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("MESSENGER_ALLOW_ALL_USERS", raising=False)
    _YAML_BRIDGED_ENV_KEYS.clear()

    seeded = _apply_yaml_config(
        {},
        {
            "pageAccessToken": "direct-page-token",
            "allow_from": ["outer-user"],
            "allow_all_users": False,
            "extra": {
                "pageAccessToken": "extra-page-token",
                "allow_from": ["inner-user"],
                "allow_all_users": True,
            },
        },
    )

    assert seeded["pageAccessToken"] == "direct-page-token"
    assert seeded["allow_from"] == ["outer-user"]
    assert seeded["allow_all_users"] is False
    assert os.environ["MESSENGER_ALLOWED_USERS"] == "outer-user"
    assert os.environ["MESSENGER_ALLOW_ALL_USERS"] == "false"

    _apply_yaml_config({}, {"extra": {"pageAccessToken": "new-token"}})
    assert "MESSENGER_ALLOWED_USERS" not in os.environ
    assert "MESSENGER_ALLOW_ALL_USERS" not in os.environ


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
    assert event.source.user_id == "work:PSID123"
    assert event.message_type is MessageType.PHOTO
    assert event.media_urls == ["/tmp/messenger-photo.jpg"]
    assert "hello" in event.text


@pytest.mark.asyncio
async def test_process_webhook_payload_skips_malformed_entries(monkeypatch):
    config = PlatformConfig(
        enabled=True,
        extra={
            "page_access_token": "page-token",
            "app_secret": "app-secret",
            "verify_token": "verify-token",
        },
    )
    adapter = MessengerAdapter(config)
    account = adapter._complete_accounts["default"]
    process = AsyncMock()
    monkeypatch.setattr(adapter, "_process_messaging_event", process)

    await adapter._process_webhook_payload(
        account,
        {
            "object": "page",
            "entry": [
                "bad-entry",
                {"messaging": "bad-events"},
                {"messaging": ["bad-event", {"message": {"text": "hello"}}]},
            ],
        },
    )

    process.assert_awaited_once_with(account, {"message": {"text": "hello"}})


@pytest.mark.asyncio
async def test_webhook_handler_tracks_post_processing_task(monkeypatch):
    if _messenger.web is None:
        pytest.skip("aiohttp is not installed")

    config = PlatformConfig(
        enabled=True,
        extra={
            "page_access_token": "page-token",
            "app_secret": "app-secret",
            "verify_token": "verify-token",
        },
    )
    adapter = MessengerAdapter(config)
    account = adapter._complete_accounts["default"]
    started = asyncio.Event()
    release = asyncio.Event()

    async def wait_payload(_account, _payload):
        started.set()
        await release.wait()

    monkeypatch.setattr(adapter, "_process_webhook_payload", wait_payload)
    body = json.dumps({"object": "page"}).encode("utf-8")

    class Request:
        headers = {"X-Hub-Signature-256": _signature(body, "app-secret")}

        async def read(self):
            return body

    response = await adapter._build_webhook_handler(account)(Request())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert response.status == 200
    assert [task for task in adapter._background_tasks if not task.done()]

    release.set()
    await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_send_uses_metadata_messaging_type(monkeypatch):
    config = PlatformConfig(
        enabled=True,
        extra={
            "page_access_token": "page-token",
            "app_secret": "app-secret",
            "verify_token": "verify-token",
        },
    )
    adapter = MessengerAdapter(config)
    post_graph = AsyncMock(return_value={"message_id": "mid.1"})
    monkeypatch.setattr(adapter, "_post_graph", post_graph)

    result = await adapter.send(
        "PSID123",
        "hello",
        metadata={"messenger_messaging_type": "UPDATE"},
    )

    assert result.success is True
    assert post_graph.await_args.args[2]["messaging_type"] == "UPDATE"

    utility_result = await adapter.send(
        "PSID123",
        "utility",
        metadata={"messenger_messaging_type": "UTILITY"},
    )

    assert utility_result.success is True
    assert post_graph.await_args.args[2]["messaging_type"] == "UTILITY"


@pytest.mark.asyncio
async def test_chunked_send_partial_failure_is_not_retryable(monkeypatch):
    config = PlatformConfig(
        enabled=True,
        extra={
            "page_access_token": "page-token",
            "app_secret": "app-secret",
            "verify_token": "verify-token",
        },
    )
    adapter = MessengerAdapter(config)
    post_graph = AsyncMock(
        side_effect=[
            {"message_id": "mid.1"},
            MessengerGraphError(500, {"error": "temporary"}),
        ]
    )
    monkeypatch.setattr(adapter, "_post_graph", post_graph)
    account = adapter._complete_accounts["default"]

    result = await adapter._send_text_chunks(account, "PSID123", "a" * 2001)

    assert result.success is False
    assert result.retryable is False
    assert result.raw_response["delivered_message_ids"] == ["mid.1"]


@pytest.mark.asyncio
async def test_disabled_dm_policy_drops_inbound_before_dispatch(monkeypatch):
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

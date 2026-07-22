"""Unit tests for the X Chat platform plugin.

All tests run offline: the X API layer is replaced with fakes and the Chat
XDK crypto core is replaced with a stub — no chatxdk native module, no
network, no gateway process.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.xchat import adapter as xchat_adapter
from plugins.platforms.xchat.adapter import (
    XChatAdapter,
    _compile_mention_patterns,
    _env_enablement,
    check_requirements,
)
from plugins.platforms.xchat.crypto import XChatCrypto, message_text


# ---------------------------------------------------------------------------
# Helpers / fakes


class FakeCrypto:
    """Stands in for XChatCrypto — records calls, no native SDK."""

    def __init__(self) -> None:
        self.encrypted: List[tuple] = []
        self.batch_calls: List[List[str]] = []
        self.signing_keys: List[Dict[str, str]] = []
        self.decrypt_map: Dict[str, Dict[str, Any]] = {}
        self.fail_encrypt: Optional[Exception] = None

    def decrypt_one(self, event_b64, conversation_keys=None):
        return self.decrypt_map[event_b64]

    def decrypt_batch(self, events_b64):
        self.batch_calls.append(list(events_b64))
        return {"messages": [], "conversation_keys": {"keys": {"1": b"k"}}, "errors": {}}

    def encrypt_text(self, conversation_id, text):
        if self.fail_encrypt is not None:
            raise self.fail_encrypt
        self.encrypted.append((conversation_id, text))
        return {
            "message_id": "mid-1",
            "encoded_message_create_event": "ZW5j",
            "encoded_message_event_signature": "c2ln",
        }

    def set_signing_keys(self, keys):
        self.signing_keys = list(keys)


class FakeApi:
    """Stands in for XChatApi — canned responses, records sends."""

    def __init__(self) -> None:
        self.sent: List[tuple] = []
        self.typing: List[str] = []
        self.public_keys: Dict[str, List[Dict[str, Any]]] = {}
        self.events_pages: Dict[str, Dict[str, Any]] = {}
        self.conversations: List[str] = []

    async def get_my_user(self):
        return {"id": "999"}

    async def get_public_keys(self, user_id):
        return self.public_keys.get(user_id, [])

    async def get_conversations(self, *, max_results=100, pagination_token=None):
        return {
            "data": [{"conversation_id": c} for c in self.conversations],
            "meta": {},
        }

    async def get_events(self, conversation_id, *, max_results=50, pagination_token=None):
        return self.events_pages.get(conversation_id, {"data": []})

    async def send_message(self, conversation_id, body):
        self.sent.append((conversation_id, body))
        return {"data": {"message_id": body.get("message_id", ""), "event_id": "evt-echo"}}

    async def send_typing(self, conversation_id):
        self.typing.append(conversation_id)

    async def aclose(self):
        pass


def _make_adapter(monkeypatch: pytest.MonkeyPatch, **extra) -> XChatAdapter:
    monkeypatch.setenv("XCHAT_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("XCHAT_USER_ID", "999")
    cfg = PlatformConfig(enabled=True, token="", extra=dict(extra))
    return XChatAdapter(cfg)


def _wire(adapter: XChatAdapter) -> tuple[FakeApi, FakeCrypto]:
    api, crypto = FakeApi(), FakeCrypto()
    adapter._api = api
    adapter._crypto = crypto
    adapter._bot_user_id = "999"
    return api, crypto


def _capture(adapter: XChatAdapter, monkeypatch: pytest.MonkeyPatch) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


# ---------------------------------------------------------------------------
# check_fn / config


def test_check_requirements_needs_token_and_blob(monkeypatch, tmp_path):
    monkeypatch.delenv("XCHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("XCHAT_PRIVATE_KEYS_B64", raising=False)
    assert check_requirements() is False

    monkeypatch.setenv("XCHAT_ACCESS_TOKEN", "tok")
    assert check_requirements() is False  # no key blob

    monkeypatch.setenv("XCHAT_PRIVATE_KEYS_B64", "YmxvYg==")
    assert check_requirements() is True


def test_env_enablement_seeds_extra(monkeypatch):
    monkeypatch.delenv("XCHAT_ACCESS_TOKEN", raising=False)
    assert _env_enablement() is None

    monkeypatch.setenv("XCHAT_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("XCHAT_USER_ID", "42")
    monkeypatch.setenv("XCHAT_HOME_CHANNEL", "123-456")
    seed = _env_enablement()
    assert seed is not None
    assert seed["access_token"] == "tok"
    assert seed["user_id"] == "42"
    assert seed["home_channel"]["chat_id"] == "123-456"


def test_adapter_reads_config_extra_over_defaults(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        poll_interval="30",
        conversation_ids="111-222, g333",
    )
    assert adapter._poll_interval == 30.0
    assert adapter._pinned_conversations == ["111-222", "g333"]
    assert adapter.platform == Platform("xchat")


def test_poll_interval_floor(monkeypatch):
    adapter = _make_adapter(monkeypatch, poll_interval="0.1")
    assert adapter._poll_interval == 2.0


# ---------------------------------------------------------------------------
# Mention gating


def test_mention_patterns_json_and_csv():
    pats = _compile_mention_patterns('["^bot\\\\b"]')
    assert pats[0].pattern == "^bot\\b"
    pats = _compile_mention_patterns("alpha, beta")
    assert len(pats) == 2
    # None → defaults
    assert _compile_mention_patterns(None)


@pytest.mark.asyncio
async def test_group_mention_gate(monkeypatch):
    monkeypatch.setenv("XCHAT_REQUIRE_MENTION", "true")
    adapter = _make_adapter(monkeypatch)
    _wire(adapter)
    captured = _capture(adapter, monkeypatch)

    # Group message without wake word → dropped
    await adapter._dispatch_inbound(
        conv_id="g123", sender_id="5", text="just chatting", message_id="e1", raw={}
    )
    assert captured == []

    # Group message with wake word → dispatched, wake word stripped
    await adapter._dispatch_inbound(
        conv_id="g123", sender_id="5", text="hermes what time is it", message_id="e2", raw={}
    )
    assert len(captured) == 1
    assert captured[0].text == "what time is it"

    # DMs are never gated
    await adapter._dispatch_inbound(
        conv_id="111-222", sender_id="5", text="no wake word", message_id="e3", raw={}
    )
    assert len(captured) == 2


@pytest.mark.asyncio
async def test_dispatch_sets_chat_type(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    _wire(adapter)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(
        conv_id="g42", sender_id="7", text="hi", message_id="e1", raw={}
    )
    await adapter._dispatch_inbound(
        conv_id="111-999", sender_id="7", text="hi", message_id="e2", raw={}
    )
    assert captured[0].source.chat_type == "group"
    assert captured[1].source.chat_type == "dm"
    assert captured[0].message_type == MessageType.TEXT
    assert captured[1].source.user_id == "7"


# ---------------------------------------------------------------------------
# Poll loop mechanics


@pytest.mark.asyncio
async def test_backlog_seeds_keys_without_replying(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    api, crypto = _wire(adapter)
    captured = _capture(adapter, monkeypatch)

    api.events_pages["111-999"] = {
        "data": [
            {"id": "e1", "encoded_event": "AAA", "sender_id": "111"},
            {"id": "e2", "encoded_event": "BBB", "sender_id": "111"},
        ]
    }
    await adapter._poll_conversation("111-999")

    # Backlog: batch-decrypted for keys, nothing dispatched, ids marked seen.
    assert crypto.batch_calls == [["BBB", "AAA"]]  # newest-first reversed
    assert captured == []
    assert "e1" in adapter._seen_event_ids and "e2" in adapter._seen_event_ids
    assert "111-999" in adapter._backlog_loaded


@pytest.mark.asyncio
async def test_new_message_dispatched_after_backlog(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    api, crypto = _wire(adapter)
    captured = _capture(adapter, monkeypatch)
    adapter._backlog_loaded.add("111-999")

    crypto.decrypt_map["CCC"] = {
        "type": "Message",
        "id": "e3",
        "sender_id": "111",
        "conversation_id": "111:999",
        "content": {"text": "hello agent"},
    }
    api.events_pages["111-999"] = {
        "data": [{"id": "e3", "encoded_event": "CCC", "sender_id": "111"}]
    }
    await adapter._poll_conversation("111-999")

    assert len(captured) == 1
    ev = captured[0]
    assert ev.text == "hello agent"
    # Reply target uses the canonical id embedded in the signed event.
    assert ev.source.chat_id == "111:999"

    # Second poll with the same event id → dedup, no double dispatch.
    await adapter._poll_conversation("111-999")
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_own_messages_filtered(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    api, crypto = _wire(adapter)
    captured = _capture(adapter, monkeypatch)
    adapter._backlog_loaded.add("111-999")

    crypto.decrypt_map["DDD"] = {
        "type": "Message",
        "id": "e4",
        "sender_id": "999",  # the bot itself
        "content": {"text": "echo of our own reply"},
    }
    api.events_pages["111-999"] = {
        "data": [{"id": "e4", "encoded_event": "DDD", "sender_id": "999"}]
    }
    await adapter._poll_conversation("111-999")
    assert captured == []


@pytest.mark.asyncio
async def test_keychange_routes_through_batch(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    api, crypto = _wire(adapter)
    captured = _capture(adapter, monkeypatch)
    adapter._backlog_loaded.add("111-999")

    crypto.decrypt_map["KEY"] = {"type": "KeyChange", "id": "e5"}
    api.events_pages["111-999"] = {
        "data": [{"id": "e5", "encoded_event": "KEY", "sender_id": "111"}]
    }
    await adapter._poll_conversation("111-999")

    assert crypto.batch_calls == [["KEY"]]
    assert captured == []
    assert adapter._conversation_keys["111-999"] == {"1": b"k"}


@pytest.mark.asyncio
async def test_signing_keys_fetched_once_per_sender(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    api, crypto = _wire(adapter)
    api.public_keys["111"] = [
        {
            "public_key_version": "3",
            "signing_public_key": "SPK",
            "public_key": "IPK",
            "identity_public_key_signature": "SIG",
        }
    ]
    events = [{"id": "e1", "sender_id": "111"}]
    await adapter._register_signing_keys(events)
    await adapter._register_signing_keys(events)  # second call — cached

    assert len(crypto.signing_keys) == 1
    entry = crypto.signing_keys[0]
    assert entry["user_id"] == "111"
    assert entry["public_key"] == "SPK"
    assert entry["identity_public_key"] == "IPK"


@pytest.mark.asyncio
async def test_discovery_adds_conversations(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    api, _ = _wire(adapter)
    api.conversations = ["111-222", "g333"]
    await adapter._discover_conversations()
    assert adapter._conversations == {"111-222", "g333"}


def test_dedup_prune_bounds_memory(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    now = time.time()
    for i in range(xchat_adapter.DEDUP_MAX_SIZE + 100):
        adapter._seen_event_ids[f"e{i}"] = now + i
    adapter._prune_dedup()
    assert len(adapter._seen_event_ids) <= xchat_adapter.DEDUP_MAX_SIZE
    # Newest entries survive the prune.
    assert f"e{xchat_adapter.DEDUP_MAX_SIZE + 99}" in adapter._seen_event_ids


# ---------------------------------------------------------------------------
# Outbound


@pytest.mark.asyncio
async def test_send_encrypts_and_posts(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    api, crypto = _wire(adapter)

    result = await adapter.send("111:999", "hi there")
    assert result.success
    assert result.message_id == "mid-1"
    assert crypto.encrypted == [("111:999", "hi there")]
    conv, body = api.sent[0]
    assert conv == "111:999"
    assert body["encoded_message_create_event"] == "ZW5j"
    # Echo suppression: returned event id marked as seen.
    assert "evt-echo" in adapter._seen_event_ids


@pytest.mark.asyncio
async def test_send_without_conversation_key(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    _, crypto = _wire(adapter)
    crypto.fail_encrypt = ValueError("no key")

    result = await adapter.send("111:999", "hi")
    assert not result.success
    assert "conversation key" in (result.error or "")


@pytest.mark.asyncio
async def test_send_disconnected(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    result = await adapter.send("111:999", "hi")
    assert not result.success


@pytest.mark.asyncio
async def test_get_chat_info_types(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    assert (await adapter.get_chat_info("g123"))["type"] == "group"
    assert (await adapter.get_chat_info("111-222"))["type"] == "dm"


# ---------------------------------------------------------------------------
# Crypto wrapper (fake Chat object — no native SDK)


class _FakePayload:
    message_id = "m1"
    encrypted_content = "ENC"
    encoded_event_signature = "SIG"


class _FakeChat:
    def __init__(self) -> None:
        self.identity = None
        self.cache = None
        self.imported = None

    def import_keys(self, blob, version=None):
        self.imported = (blob, version)

    def set_identity(self, user_id, version):
        self.identity = (user_id, version)

    def set_cache_keys(self, enabled):
        self.cache = enabled

    def set_signing_keys(self, keys):
        self.signing = keys

    def encrypt_message(self, conversation_id, text):
        return _FakePayload()

    def decrypt_event(self, event_b64, conversation_keys, signing_keys):
        return {"type": "Message", "content": {"text": "plain"}}

    def decrypt_events(self, events, signing_keys):
        return {"messages": [{"event": {"type": "Message"}}], "conversation_keys": {}, "errors": {}}


def test_crypto_load_keys_and_identity():
    crypto = XChatCrypto(chat=_FakeChat())
    crypto.load_keys("YmxvYg==", "7")  # b64("blob")
    assert crypto.chat.imported == (b"blob", "7")
    assert crypto.signing_key_version == "7"
    crypto.set_identity("42")
    assert crypto.chat.identity == ("42", "7")


def test_crypto_encrypt_shapes_send_body():
    crypto = XChatCrypto(chat=_FakeChat())
    body = crypto.encrypt_text("1:2", "hello")
    assert body == {
        "message_id": "m1",
        "encoded_message_create_event": "ENC",
        "encoded_message_event_signature": "SIG",
    }


def test_message_text_extraction():
    assert message_text({"type": "Message", "content": {"text": "x"}}) == "x"
    assert message_text({"type": "KeyChange"}) is None
    assert message_text({"type": "Message", "content": {}}) is None


# ---------------------------------------------------------------------------
# Registry integration


def test_platform_registry_entry_parity():
    """Every parity knob must be populated on the registered entry."""
    from gateway.platform_registry import PlatformEntry

    captured: Dict[str, Any] = {}

    class Ctx:
        class manifest:
            name = "xchat-platform"

        def register_platform(self, **kwargs):
            captured.update(kwargs)

        def register_cli_command(self, **kwargs):
            captured["cli"] = kwargs

    xchat_adapter.register(Ctx())

    assert captured["name"] == "xchat"
    assert captured["allowed_users_env"] == "XCHAT_ALLOWED_USERS"
    assert captured["allow_all_env"] == "XCHAT_ALLOW_ALL_USERS"
    assert captured["cron_deliver_env_var"] == "XCHAT_HOME_CHANNEL"
    assert callable(captured["standalone_sender_fn"])
    assert callable(captured["setup_fn"])
    assert callable(captured["env_enablement_fn"])
    assert captured["platform_hint"]
    assert captured["max_message_length"] > 0
    assert captured["cli"]["name"] == "xchat"
    # The kwargs must construct a valid PlatformEntry.
    entry_kwargs = {k: v for k, v in captured.items() if k != "cli"}
    entry = PlatformEntry(**entry_kwargs)
    assert entry.name == "xchat"


# ---------------------------------------------------------------------------
# Standalone send (config-error paths — no network)


@pytest.mark.asyncio
async def test_standalone_send_requires_token(monkeypatch):
    monkeypatch.delenv("XCHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("XCHAT_PRIVATE_KEYS_B64", raising=False)
    cfg = PlatformConfig(enabled=True, extra={})
    out = await xchat_adapter._standalone_send(cfg, "111", "msg")
    assert "XCHAT_ACCESS_TOKEN" in out["error"]


@pytest.mark.asyncio
async def test_standalone_send_requires_key_blob(monkeypatch):
    monkeypatch.setenv("XCHAT_ACCESS_TOKEN", "tok")
    monkeypatch.delenv("XCHAT_PRIVATE_KEYS_B64", raising=False)
    cfg = PlatformConfig(enabled=True, extra={})
    out = await xchat_adapter._standalone_send(cfg, "111", "msg")
    assert "private-key blob" in out["error"]

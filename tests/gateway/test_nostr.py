"""Tests for the Nostr platform adapter (plugin)."""

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.anyio

from gateway.config import PlatformConfig


def _make_fake_nostr_sdk():
    """Create fake nostr_sdk modules for testing."""
    nostr_sdk = types.ModuleType("nostr_sdk")
    nostr_sdk.Kind = MagicMock()
    nostr_sdk.Keys = MagicMock()
    nostr_sdk.Message = MagicMock()
    nostr_sdk.Filter = MagicMock()
    nostr_sdk.Tag = MagicMock()
    nostr_sdk.EventBuilder = MagicMock()
    nostr_sdk.Client = MagicMock()
    nostr_sdk.NostrSigner = MagicMock()
    return {"nostr_sdk": nostr_sdk}


def _import_nostr_module():
    with patch.dict("sys.modules", _make_fake_nostr_sdk()):
        import plugins.platforms.nostr.adapter as _mod
        return _mod


_mod = _import_nostr_module()

NostrAdapter = _mod.NostrAdapter
check_nostr_requirements = _mod.check_nostr_requirements


def _config(relays=None, nsec=None):
    extra = {}
    if relays:
        extra["relays"] = relays
    if nsec:
        extra["nsec"] = nsec
    return PlatformConfig(enabled=True, extra=extra)


class TestNostrRequirements:
    def test_returns_true_when_nostr_sdk_installed(self):
        with patch.dict("sys.modules", {"nostr_sdk": MagicMock()}):
            assert check_nostr_requirements() is True

    def test_returns_false_when_nostr_sdk_missing(self):
        saved = __import__("sys").modules.pop("nostr_sdk", None)
        try:
            assert check_nostr_requirements() is False
        finally:
            if saved:
                __import__("sys").modules["nostr_sdk"] = saved


class TestNostrConnect:
    async def test_connect_success(self):
        mock_keys = MagicMock()
        mock_keys.public_key().to_hex.return_value = "abc123"
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()

        with (
            patch.object(_mod, "Keys") as mock_keys_cls,
            patch.object(_mod, "NostrSigner") as _,
            patch.object(_mod, "Client", return_value=mock_client),
            patch.object(_mod, "asyncio") as mock_asyncio,
        ):
            mock_keys_cls.from_nsec.return_value = mock_keys
            mock_asyncio.create_task = MagicMock()

            adapter = NostrAdapter(_config(nsec="nsec1test"))
            result = await adapter.connect()

        assert result is True
        assert adapter.nsec == "nsec1test"
        assert adapter.pubkey == "abc123"
        assert adapter.relays == [
            "wss://relay.damus.io",
            "wss://relay.primal.net",
            "wss://relay.snort.social",
        ]
        mock_keys_cls.from_nsec.assert_called_once_with("nsec1test")
        mock_client.connect.assert_awaited_once()

    async def test_connect_success_with_is_reconnect(self):
        mock_keys = MagicMock()
        mock_keys.public_key().to_hex.return_value = "abc123"
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()

        with (
            patch.object(_mod, "Keys") as mock_keys_cls,
            patch.object(_mod, "NostrSigner") as _,
            patch.object(_mod, "Client", return_value=mock_client),
            patch.object(_mod, "asyncio") as mock_asyncio,
        ):
            mock_keys_cls.from_nsec.return_value = mock_keys
            mock_asyncio.create_task = MagicMock()

            adapter = NostrAdapter(_config(nsec="nsec1test"))
            result = await adapter.connect(is_reconnect=True)

        assert result is True
        mock_client.connect.assert_awaited_once()

    async def test_connect_uses_config_relays(self):
        mock_keys = MagicMock()
        mock_keys.public_key().to_hex.return_value = "abc123"
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()

        with (
            patch.object(_mod, "Keys") as mock_keys_cls,
            patch.object(_mod, "NostrSigner") as _,
            patch.object(_mod, "Client", return_value=mock_client),
            patch.object(_mod, "asyncio") as mock_asyncio,
        ):
            mock_keys_cls.from_nsec.return_value = mock_keys
            mock_asyncio.create_task = MagicMock()

            adapter = NostrAdapter(
                _config(nsec="nsec1test", relays=["wss://custom.relay"])
            )
            result = await adapter.connect()

        assert result is True
        assert adapter.relays == ["wss://custom.relay"]

    async def test_connect_fails_when_nsec_missing(self):
        adapter = NostrAdapter(_config())
        result = await adapter.connect()
        assert result is False
        assert not adapter.nsec

    async def test_connect_fails_on_exception(self):
        with patch.object(_mod, "Keys") as mock_keys_cls:
            mock_keys_cls.from_nsec.side_effect = Exception("bad key")
            adapter = NostrAdapter(_config(nsec="nsec1bad"))
            result = await adapter.connect()
        assert result is False


class TestNostrDisconnect:
    async def test_disconnect_cleans_up(self):
        mock_client = AsyncMock()
        adapter = NostrAdapter(_config())
        adapter.client = mock_client
        adapter.keys = MagicMock()
        adapter.nsec = "nsec1test"
        adapter.pubkey = "abc123"

        await adapter.disconnect()

        assert adapter._listening is False
        assert adapter.client is None
        assert adapter.keys is None
        assert adapter.nsec is None
        assert adapter.pubkey is None
        mock_client.disconnect.assert_awaited_once()

    async def test_disconnect_when_no_client(self):
        adapter = NostrAdapter(_config())
        await adapter.disconnect()
        assert adapter._listening is False


class TestNostrSend:
    async def test_send_success(self):
        mock_keys = MagicMock()
        mock_keys.encrypt.return_value = "ciphertext"
        mock_client = AsyncMock()

        mock_signed = MagicMock()
        mock_signed.id().to_hex.return_value = "event123"

        with patch.object(_mod, "EventBuilder") as mock_eb:
            mock_eb.return_value.sign_with_keys.return_value = mock_signed
            adapter = NostrAdapter(_config())
            adapter.client = mock_client
            adapter.keys = mock_keys

            result = await adapter.send("recipient_pubkey", "hello")

        assert result.success is True
        assert result.message_id == "event123"
        mock_keys.encrypt.assert_called_once_with("recipient_pubkey", "hello")

    async def test_send_fails_when_not_connected(self):
        adapter = NostrAdapter(_config())
        adapter.client = None
        adapter.keys = None

        result = await adapter.send("recipient", "hello")

        assert result.success is False
        assert "Not connected" in result.error

    async def test_send_fails_on_exception(self):
        mock_keys = MagicMock()
        mock_keys.encrypt.side_effect = Exception("encrypt error")

        adapter = NostrAdapter(_config())
        adapter.client = MagicMock()
        adapter.keys = mock_keys

        result = await adapter.send("recipient", "hello")

        assert result.success is False
        assert "encrypt error" in result.error


class TestNostrSendTyping:
    async def test_send_typing_is_noop(self):
        adapter = NostrAdapter(_config())
        result = await adapter.send_typing("chat_id")
        assert result is None


class TestNostrSendImage:
    async def test_send_image_delegates_to_send(self):
        adapter = NostrAdapter(_config())
        with patch.object(adapter, "send", AsyncMock()) as mock_send:
            await adapter.send_image("chat_id", "https://example.com/img.png", "caption")
        mock_send.assert_awaited_once_with(
            "chat_id", "caption\nhttps://example.com/img.png",
            reply_to=None, metadata=None,
        )

    async def test_send_image_no_caption(self):
        adapter = NostrAdapter(_config())
        with patch.object(adapter, "send", AsyncMock()) as mock_send:
            await adapter.send_image("chat_id", "https://example.com/img.png")
        mock_send.assert_awaited_once_with(
            "chat_id", "https://example.com/img.png",
            reply_to=None, metadata=None,
        )


class TestNostrGetChatInfo:
    async def test_get_chat_info_no_client(self):
        adapter = NostrAdapter(_config())
        adapter.client = None
        info = await adapter.get_chat_info("pubkey123")
        assert info["chat_id"] == "pubkey123"
        assert info["type"] == "user"

    async def test_get_chat_info_with_profile(self):
        mock_client = AsyncMock()
        mock_event = MagicMock()
        mock_event.content.return_value = '{"display_name": "Alice", "name": "alice"}'
        mock_client.query.return_value = [mock_event]

        adapter = NostrAdapter(_config())
        adapter.client = mock_client

        info = await adapter.get_chat_info("pubkey123")
        assert info["name"] == "Alice"
        assert info["profile"]["display_name"] == "Alice"

    async def test_get_chat_info_no_profile_found(self):
        mock_client = AsyncMock()
        mock_client.query.return_value = []

        adapter = NostrAdapter(_config())
        adapter.client = mock_client

        info = await adapter.get_chat_info("pubkey123")
        assert info["name"] == "pubkey123"

    async def test_get_chat_info_fetch_error(self):
        mock_client = AsyncMock()
        mock_client.query.side_effect = Exception("timeout")

        adapter = NostrAdapter(_config())
        adapter.client = mock_client

        info = await adapter.get_chat_info("pubkey123")
        assert info["chat_id"] == "pubkey123"


class TestNostrHandleIncomingMessage:
    async def test_creates_correct_message_event(self, monkeypatch):
        adapter = NostrAdapter(_config())
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        await adapter._handle_incoming_message(
            "sender_pubkey", "hello world", "evt001", 1710000000,
        )

        assert len(handled) == 1
        event = handled[0]
        assert event.text == "hello world"
        assert event.source.platform.value == "nostr"
        assert event.source.chat_id == "sender_pubkey"
        assert event.source.user_id == "sender_pubkey"
        assert event.source.user_name == "sender_p..."

    async def test_logs_exception_on_failure(self, caplog):
        adapter = NostrAdapter(_config())
        import logging
        caplog.set_level(logging.ERROR)

        with patch.object(adapter, "handle_message", side_effect=Exception("boom")):
            await adapter._handle_incoming_message(
                "pk", "hello", "evt001", 1710000000,
            )

        assert "Error handling incoming Nostr message" in caplog.text


class TestNostrProcessEvent:
    async def test_process_kind4_decrypts_and_dispatches(self):
        mock_keys = MagicMock()
        mock_keys.decrypt.return_value = "decrypted hello"

        adapter = NostrAdapter(_config())
        adapter.keys = mock_keys
        adapter.pubkey = "our_pubkey"

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            await adapter._process_event({
                "id": "evt001",
                "pubkey": "sender_pk",
                "kind": 4,
                "content": "encrypted_data",
                "tags": [],
                "created_at": 1710000000,
            })

        assert len(handled) == 1
        assert handled[0] == ("sender_pk", "decrypted hello", "evt001", 1710000000)

    async def test_process_kind4_decrypt_failure_skipped(self):
        mock_keys = MagicMock()
        mock_keys.decrypt.side_effect = Exception("decrypt failed")

        adapter = NostrAdapter(_config())
        adapter.keys = mock_keys

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            await adapter._process_event({
                "id": "evt001",
                "pubkey": "sender_pk",
                "kind": 4,
                "content": "encrypted_data",
                "tags": [],
                "created_at": 1710000000,
            })

        assert len(handled) == 0

    async def test_process_kind1_with_mention_dispatches(self):
        adapter = NostrAdapter(_config())
        adapter.pubkey = "our_pubkey"

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            await adapter._process_event({
                "id": "evt002",
                "pubkey": "sender_pk",
                "kind": 1,
                "content": "hello @bot",
                "tags": [["p", "our_pubkey"]],
                "created_at": 1710000001,
            })

        assert len(handled) == 1
        assert handled[0] == ("sender_pk", "hello @bot", "evt002", 1710000001)

    async def test_process_kind1_without_mention_skipped(self):
        adapter = NostrAdapter(_config())
        adapter.pubkey = "our_pubkey"

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            await adapter._process_event({
                "id": "evt003",
                "pubkey": "sender_pk",
                "kind": 1,
                "content": "hello",
                "tags": [["p", "other_pubkey"]],
                "created_at": 1710000002,
            })

        assert len(handled) == 0

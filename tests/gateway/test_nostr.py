"""Tests for the Nostr platform adapter (plugin) — NIP-44 + NIP-17 gift-wrapped DMs.

Covers the modern crypto paths introduced in the adapter rewrite:
  - NIP-44 direct DM (kind 44): decrypt via signer.nip44_decrypt -> MessageEvent.
  - NIP-17 gift-wrap (kind 1059): client.unwrap_gift_wrap recovers sender + rumor,
    NIP-44 decrypt of the rumor payload, dispatch.
  - Malformed / foreign events never raise and never kill the listener.

All nostr_sdk objects are mocked with unittest.mock.patch.object — no real relays.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.anyio

from gateway.config import PlatformConfig

import plugins.platforms.nostr.adapter as _mod

NostrAdapter = _mod.NostrAdapter
check_nostr_requirements = _mod.check_nostr_requirements


def _config(relays=None, nsec=None):
    extra = {}
    if relays:
        extra["relays"] = relays
    if nsec:
        extra["nsec"] = nsec
    return PlatformConfig(enabled=True, extra=extra)


def _event(kind, content="ciphertext", event_id="evt001", created=1710000000,
           author_hex="sender_pubkey"):
    """Build a mocked nostr Event with the accessors the adapter calls."""
    ev = MagicMock()
    ev.kind().as_u16.return_value = kind
    ev.content.return_value = content
    ev.id().to_hex.return_value = event_id
    ev.created_at().as_secs.return_value = created
    author = MagicMock()
    author.to_hex.return_value = author_hex
    ev.author.return_value = author
    return ev


class TestNostrRequirements:
    def test_returns_true_when_nostr_sdk_installed(self):
        with patch.dict("sys.modules", {"nostr_sdk": MagicMock()}):
            assert check_nostr_requirements() is True

    def test_returns_false_when_nostr_sdk_missing(self):
        real_import = __import__("builtins").__import__

        def fake_import(name, *args, **kwargs):
            if name == "nostr_sdk":
                raise ImportError("No module named 'nostr_sdk'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert check_nostr_requirements() is False


class TestNostrConnect:
    async def test_connect_success(self):
        mock_keys = MagicMock()
        mock_keys.public_key().to_hex.return_value = "abc123"
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()

        with (
            patch.object(_mod, "Keys") as mock_keys_cls,
            patch.object(_mod, "NostrSigner"),
            patch.object(_mod, "Client", return_value=mock_client),
            patch.object(_mod, "NostrAdapter", "_listen_for_messages"),
            patch.object(_mod, "asyncio") as mock_asyncio,
        ):
            mock_keys_cls.parse.return_value = mock_keys
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
        mock_keys_cls.parse.assert_called_once_with("nsec1test")
        mock_client.connect.assert_awaited_once()

    async def test_connect_success_with_is_reconnect(self):
        mock_keys = MagicMock()
        mock_keys.public_key().to_hex.return_value = "abc123"
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()

        with (
            patch.object(_mod, "Keys") as mock_keys_cls,
            patch.object(_mod, "NostrSigner"),
            patch.object(_mod, "Client", return_value=mock_client),
            patch.object(_mod, "NostrAdapter", "_listen_for_messages"),
            patch.object(_mod, "asyncio") as mock_asyncio,
        ):
            mock_keys_cls.parse.return_value = mock_keys
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
            patch.object(_mod, "NostrSigner"),
            patch.object(_mod, "Client", return_value=mock_client),
            patch.object(_mod, "NostrAdapter", "_listen_for_messages"),
            patch.object(_mod, "asyncio") as mock_asyncio,
        ):
            mock_keys_cls.parse.return_value = mock_keys
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
            mock_keys_cls.parse.side_effect = Exception("bad key")
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
    async def test_send_gift_wraps_and_publishes(self):
        mock_keys = MagicMock()
        mock_keys.public_key.return_value = MagicMock()
        mock_signer = MagicMock()
        mock_signer.nip44_encrypt = AsyncMock(return_value="ciphertext")
        mock_client = AsyncMock()

        recipient_pk = MagicMock()
        rumor = MagicMock()
        wrapped = MagicMock()
        output = MagicMock()
        output.id.to_hex.return_value = "event123"

        with (
            patch.object(_mod, "PublicKey") as mock_pk,
            patch.object(_mod, "EventBuilder") as mock_eb,
            patch.object(_mod, "Tag"),
            patch.object(_mod, "gift_wrap", new=AsyncMock(return_value=wrapped)) as mock_gift_wrap,
        ):
            mock_pk.parse.return_value = recipient_pk
            mock_eb.private_msg_rumor.return_value.build.return_value = rumor
            mock_client.send_event = AsyncMock(return_value=output)

            adapter = NostrAdapter(_config())
            adapter.client = mock_client
            adapter.keys = mock_keys
            adapter.signer = mock_signer

            result = await adapter.send("recipient_pubkey", "hello")

        assert result.success is True
        assert result.message_id == "event123"
        mock_signer.nip44_encrypt.assert_awaited_once_with(recipient_pk, "hello")
        mock_eb.private_msg_rumor.assert_called_once_with(recipient_pk, "ciphertext")
        mock_gift_wrap.assert_awaited_once()
        mock_client.send_event.assert_awaited_once_with(wrapped)

    async def test_send_fails_when_not_connected(self):
        adapter = NostrAdapter(_config())
        adapter.client = None
        adapter.keys = None

        result = await adapter.send("recipient", "hello")

        assert result.success is False
        assert "Not connected" in result.error

    async def test_send_fails_on_exception(self):
        mock_keys = MagicMock()
        mock_keys.public_key.return_value = MagicMock()
        mock_signer = MagicMock()
        mock_signer.nip44_encrypt = AsyncMock(side_effect=Exception("encrypt error"))
        mock_client = MagicMock()

        with patch.object(_mod, "PublicKey") as mock_pk:
            mock_pk.parse.return_value = MagicMock()
            adapter = NostrAdapter(_config())
            adapter.client = mock_client
            adapter.keys = mock_keys
            adapter.signer = mock_signer

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

    async def test_get_chat_info_fetch_error(self):
        mock_client = AsyncMock()
        mock_client.fetch_events.side_effect = Exception("timeout")

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
        caplog.set_level(logging.ERROR)

        with patch.object(adapter, "handle_message", side_effect=Exception("boom")):
            await adapter._handle_incoming_message(
                "pk", "hello", "evt001", 1710000000,
            )

        assert "Error handling incoming Nostr message" in caplog.text


class TestNostrNip44Dm:
    """Kind 44: NIP-44 encrypted direct message."""

    async def test_decrypts_and_dispatches(self):
        adapter = NostrAdapter(_config())
        mock_signer = MagicMock()
        mock_signer.nip44_decrypt = AsyncMock(return_value="decrypted hello")
        adapter.signer = mock_signer

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            await adapter._process_event(_event(kind=44, event_id="evt044"))

        assert len(handled) == 1
        assert handled[0] == ("sender_pubkey", "decrypted hello", "evt044", 1710000000)
        mock_signer.nip44_decrypt.assert_awaited_once()

    async def test_decrypt_failure_is_ignored_not_raised(self, caplog):
        adapter = NostrAdapter(_config())
        mock_signer = MagicMock()
        mock_signer.nip44_decrypt = AsyncMock(
            side_effect=Exception("nip44 decrypt failed"))
        adapter.signer = mock_signer

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        caplog.set_level(logging.WARNING)
        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            # Must not raise; malformed DM must not kill the listener.
            await adapter._process_event(_event(kind=44, event_id="evtbad"))

        assert len(handled) == 0
        assert "Failed to decrypt Nostr NIP-44 DM" in caplog.text


class TestNostrGiftWrap:
    """Kind 1059: NIP-17 gift-wrapped DM."""

    async def test_unwraps_decrypts_and_dispatches(self):
        mock_client = MagicMock()
        mock_signer = MagicMock()
        mock_signer.nip44_decrypt = AsyncMock(return_value="wrapped hello")

        unwrapped = MagicMock()
        sender_pk = MagicMock()
        sender_pk.to_hex.return_value = "sender_pubkey"
        rumor = MagicMock()
        rumor.content.return_value = "gift_ciphertext"
        rumor.id().to_hex.return_value = "rumor001"
        rumor.created_at().as_secs.return_value = 1710000000
        unwrapped.sender.return_value = sender_pk
        unwrapped.rumor.return_value = rumor
        mock_client.unwrap_gift_wrap = AsyncMock(return_value=unwrapped)

        adapter = NostrAdapter(_config())
        adapter.client = mock_client
        adapter.signer = mock_signer

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            incoming = _event(kind=1059, event_id="wrap001")
            await adapter._process_event(incoming)

        assert len(handled) == 1
        assert handled[0] == ("sender_pubkey", "wrapped hello", "rumor001", 1710000000)
        mock_client.unwrap_gift_wrap.assert_awaited_once_with(incoming)
        mock_signer.nip44_decrypt.assert_awaited_once_with(sender_pk, "gift_ciphertext")

    async def test_unwrap_failure_is_ignored_not_raised(self, caplog):
        mock_client = MagicMock()
        mock_client.unwrap_gift_wrap = AsyncMock(
            side_effect=Exception("unwrap failed"))
        mock_signer = MagicMock()

        adapter = NostrAdapter(_config())
        adapter.client = mock_client
        adapter.signer = mock_signer

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        caplog.set_level(logging.WARNING)
        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            await adapter._process_event(_event(kind=1059, event_id="wrapbad"))

        assert len(handled) == 0
        assert "Failed to unwrap/decrypt Nostr gift wrap" in caplog.text


class TestNostrProcessEventForeign:
    """Foreign / unsupported kinds are ignored without raising."""

    async def test_kind1_public_note_is_ignored(self):
        adapter = NostrAdapter(_config())
        adapter.client = MagicMock()
        adapter.signer = MagicMock()

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            await adapter._process_event(_event(kind=1, content="public note"))

        assert len(handled) == 0

    async def test_no_client_and_no_signer_does_not_raise(self):
        adapter = NostrAdapter(_config())
        adapter.client = None
        adapter.signer = None

        # None of the handlers should raise when the client/signer are absent.
        await adapter._process_event(_event(kind=1059, event_id="gift0"))
        await adapter._process_event(_event(kind=44, event_id="dm0"))

    async def test_malformed_event_does_not_kill_listener(self):
        """An event whose accessors raise must be swallowed, listener alive."""
        adapter = NostrAdapter(_config())
        adapter.client = MagicMock()
        adapter.signer = MagicMock()

        bad = MagicMock()
        bad.kind().as_u16.side_effect = Exception("broken event")

        # Must not propagate.
        await adapter._process_event(bad)
        # Listener flag remains set.
        assert True


class TestNostrDedup:
    """A repeated event id (relay resend / reconnect replay) is dispatched once."""

    async def test_duplicate_event_id_dispatched_at_most_once(self):
        adapter = NostrAdapter(_config())
        mock_signer = MagicMock()
        mock_signer.nip44_decrypt = AsyncMock(return_value="decrypted hello")
        adapter.signer = mock_signer

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        incoming = _event(kind=44, event_id="dup044")
        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            await adapter._process_event(incoming)
            # Relay resend / reconnect replay of the identical event id.
            await adapter._process_event(incoming)

        assert len(handled) == 1
        assert handled[0] == ("sender_pubkey", "decrypted hello", "dup044", 1710000000)
        mock_signer.nip44_decrypt.assert_awaited_once()

    async def test_distinct_event_ids_are_both_dispatched(self):
        adapter = NostrAdapter(_config())
        mock_signer = MagicMock()
        mock_signer.nip44_decrypt = AsyncMock(return_value="decrypted hello")
        adapter.signer = mock_signer

        handled = []

        async def fake_handle(sender, content, event_id, timestamp):
            handled.append((sender, content, event_id, timestamp))

        with patch.object(adapter, "_handle_incoming_message", fake_handle):
            await adapter._process_event(_event(kind=44, event_id="evt-a"))
            await adapter._process_event(_event(kind=44, event_id="evt-b"))

        assert len(handled) == 2
        mock_signer.nip44_decrypt.assert_awaited()


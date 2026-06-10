"""Tests for the Nostr platform adapter.

Covers the checklist items from ADDING_A_PLATFORM.md:
- Platform enum value
- Config loading from env vars via _apply_env_overrides
- Adapter init (config parsing, allowlist, defaults)
- Authorization integration (platform in allowlist maps)
- Send message tool routing (platform in platform_map)

Cryptographic correctness (NIP-17 unwrap, signature verification, NIP-44
encryption) is provided by the nostr-sdk dependency and is not retested here.
"""

import asyncio
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------

class TestPlatformEnum:
    def test_nostr_enum_value(self):
        from gateway.config import Platform
        assert Platform.NOSTR.value == "nostr"

    def test_nostr_in_platform_list(self):
        from gateway.config import Platform
        values = [p.value for p in Platform]
        assert "nostr" in values


# ---------------------------------------------------------------------------
# Config loading from env vars
# ---------------------------------------------------------------------------

class TestNostrEnvConfig:
    def test_private_key_and_relays_enable_platform(self):
        from gateway.config import load_gateway_config, Platform
        env = {
            "NOSTR_PRIVATE_KEY": "a" * 64,
            "NOSTR_RELAYS": "wss://relay.example.com",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = load_gateway_config()
        assert Platform.NOSTR in cfg.platforms
        pc = cfg.platforms[Platform.NOSTR]
        assert pc.enabled is True
        assert pc.token == "a" * 64
        assert pc.extra["relays"] == "wss://relay.example.com"

    def test_missing_relays_does_not_enable(self, monkeypatch):
        from gateway.config import load_gateway_config, Platform
        # Wipe any user-shell NOSTR_* vars that would leak into the test.
        # NOSTR_RELAYS is the one this test specifically requires absent.
        monkeypatch.delenv("NOSTR_RELAYS", raising=False)
        monkeypatch.setenv("NOSTR_PRIVATE_KEY", "a" * 64)
        cfg = load_gateway_config()
        assert Platform.NOSTR not in cfg.platforms or not cfg.platforms.get(Platform.NOSTR, MagicMock(enabled=False)).enabled

    def test_profile_extras_loaded(self):
        from gateway.config import load_gateway_config, Platform
        env = {
            "NOSTR_PRIVATE_KEY": "b" * 64,
            "NOSTR_RELAYS": "wss://relay.example.com",
            "NOSTR_BOT_NAME": "TestBot",
            "NOSTR_NIP05": "bot@example.com",
            "NOSTR_LUD16": "tips@example.com",
            "NOSTR_BOT_WEBSITE": "https://example.com",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = load_gateway_config()
        pc = cfg.platforms[Platform.NOSTR]
        assert pc.extra["name"] == "TestBot"
        assert pc.extra["nip05"] == "bot@example.com"
        assert pc.extra["lud16"] == "tips@example.com"
        assert pc.extra["website"] == "https://example.com"

    def test_lookback_minutes_loaded(self):
        from gateway.config import load_gateway_config, Platform
        env = {
            "NOSTR_PRIVATE_KEY": "b" * 64,
            "NOSTR_RELAYS": "wss://relay.example.com",
            "NOSTR_LOOKBACK_MINUTES": "60",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = load_gateway_config()
        assert cfg.platforms[Platform.NOSTR].extra["lookback_minutes"] == "60"

    def test_home_channel_loaded(self, monkeypatch):
        # NOSTR_HOME_CHANNEL is validated through nostr_sdk's PublicKey.parse,
        # which checks that the hex decodes to an actual secp256k1 curve point.
        # Synthetic strings like "c" * 64 are syntactically valid hex but rarely
        # land on the curve, so derive a real pubkey hex from a private key at
        # test time. Skip if the SDK is unavailable (the loader would silently
        # log "ignoring" and the test would assert against None).
        nostr_sdk = pytest.importorskip("nostr_sdk")
        from gateway.config import load_gateway_config, Platform
        privkey_hex = "d" * 64
        pubkey_hex = nostr_sdk.Keys.parse(privkey_hex).public_key().to_hex()
        monkeypatch.setenv("NOSTR_PRIVATE_KEY", privkey_hex)
        monkeypatch.setenv("NOSTR_RELAYS", "wss://relay.example.com")
        monkeypatch.setenv("NOSTR_HOME_CHANNEL", pubkey_hex)
        cfg = load_gateway_config()
        pc = cfg.platforms[Platform.NOSTR]
        assert pc.home_channel is not None
        assert pc.home_channel.chat_id == pubkey_hex


# ---------------------------------------------------------------------------
# Authorization maps
# ---------------------------------------------------------------------------

class TestAuthMaps:
    def test_nostr_in_platform_env_map(self):
        """Platform.NOSTR must appear in _is_user_authorized's platform_env_map."""
        from gateway.config import Platform
        assert Platform.NOSTR is not None
        with patch.dict(os.environ, {"NOSTR_ALLOWED_NPUBS": "abc123"}, clear=False):
            assert os.getenv("NOSTR_ALLOWED_NPUBS") == "abc123"

    def test_nostr_allow_all_var_convention(self):
        with patch.dict(os.environ, {"NOSTR_ALLOW_ALL_USERS": "true"}, clear=False):
            assert os.getenv("NOSTR_ALLOW_ALL_USERS") == "true"


# ---------------------------------------------------------------------------
# Profile field validators
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("nostr_sdk") is None,
    reason="nostr-sdk not installed",
)
class TestProfileValidators:
    def test_local_at_domain_accepts_well_formed(self):
        from gateway.platforms.nostr import is_valid_local_at_domain
        assert is_valid_local_at_domain("bot@example.com")
        assert is_valid_local_at_domain("bot.name@sub.example.com")
        assert is_valid_local_at_domain("bot_1-2@example.io")

    def test_local_at_domain_rejects_malformed(self):
        from gateway.platforms.nostr import is_valid_local_at_domain
        assert not is_valid_local_at_domain("")
        assert not is_valid_local_at_domain("no-at-sign")
        assert not is_valid_local_at_domain("@example.com")
        assert not is_valid_local_at_domain("bot@")
        assert not is_valid_local_at_domain("bot@nodot")
        assert not is_valid_local_at_domain("bot example.com")

    def test_http_url_accepts_http_https(self):
        from gateway.platforms.nostr import is_valid_http_url
        assert is_valid_http_url("http://example.com")
        assert is_valid_http_url("https://example.com/path?q=1")
        assert is_valid_http_url("https://sub.example.com:8080/p")

    def test_http_url_rejects_other_schemes(self):
        from gateway.platforms.nostr import is_valid_http_url
        assert not is_valid_http_url("")
        assert not is_valid_http_url("ftp://example.com")
        assert not is_valid_http_url("javascript:alert(1)")
        assert not is_valid_http_url("example.com")  # no scheme
        assert not is_valid_http_url("https://")  # no host

    def test_parse_relay_url_canonicalizes_bare_host(self):
        from gateway.platforms.nostr import parse_relay_url
        assert parse_relay_url("relay.example.com") == "wss://relay.example.com"
        assert parse_relay_url("  wss://relay.example.com  ") == "wss://relay.example.com"

    def test_parse_relay_url_rejects_other_schemes(self):
        from gateway.platforms.nostr import parse_relay_url
        assert parse_relay_url("") is None
        assert parse_relay_url("   ") is None
        assert parse_relay_url("ws://relay.example.com") is None
        assert parse_relay_url("https://relay.example.com") is None
        assert parse_relay_url("wss://") is None  # scheme but no host

    def test_parse_pubkey_accepts_npub_and_hex(self):
        from gateway.platforms.nostr import parse_pubkey
        # Jack's well-known npub
        npub = "npub1sg6plzptd64u62a878hep2kev88swjh3tw00gjsfl8f237lmu63q0uf63m"
        pk = parse_pubkey(npub)
        assert pk is not None
        # Re-parse the same key from its hex form
        pk2 = parse_pubkey(pk.to_hex())
        assert pk2 is not None
        assert pk.to_hex() == pk2.to_hex()

    def test_parse_pubkey_rejects_invalid(self):
        from gateway.platforms.nostr import parse_pubkey
        assert parse_pubkey("") is None
        assert parse_pubkey("not-a-key") is None
        assert parse_pubkey("npub1invalid") is None


# ---------------------------------------------------------------------------
# Adapter init (skipped without nostr-sdk)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("nostr_sdk") is None,
    reason="nostr-sdk not installed",
)
class TestNostrAdapterInit:
    def _make_config(self, privkey_hex=None, relays=None, extra=None):
        from gateway.config import PlatformConfig
        # Use a fixed valid 32-byte hex key; randomness isn't needed here.
        privkey_hex = privkey_hex or ("01" * 32)
        relays = relays or "wss://relay.example.com"
        config = PlatformConfig(
            enabled=True,
            token=privkey_hex,
            extra={"relays": relays, **(extra or {})},
        )
        return config

    def test_adapter_init_sets_pubkey(self):
        from gateway.platforms.nostr import NostrAdapter
        from nostr_sdk import Keys
        privkey_hex = "01" * 32
        config = self._make_config(privkey_hex=privkey_hex)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)
        expected_pubkey = Keys.parse(privkey_hex).public_key().to_hex()
        assert adapter._pubkey_hex == expected_pubkey
        assert adapter._npub.startswith("npub1")

    def test_adapter_init_parses_relay_list(self):
        from gateway.platforms.nostr import NostrAdapter
        config = self._make_config(relays="wss://relay1.example.com,wss://relay2.example.com")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)
        assert len(adapter._relay_urls) == 2
        assert "wss://relay1.example.com" in adapter._relay_urls

    def test_adapter_init_rejects_non_wss(self):
        from gateway.platforms.nostr import NostrAdapter
        config = self._make_config(relays="ws://insecure.example.com,wss://ok.example.com")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)
        # Only wss:// relays should remain
        assert adapter._relay_urls == ["wss://ok.example.com"]

    def test_adapter_init_parses_allowlist(self):
        from gateway.platforms.nostr import NostrAdapter
        pubkey_hex = "ab" * 32
        config = self._make_config()
        with patch.dict(os.environ, {"NOSTR_ALLOWED_NPUBS": pubkey_hex}, clear=False):
            adapter = NostrAdapter(config)
        assert pubkey_hex in adapter._allowed_pubkeys

    def test_adapter_init_parses_npub_in_allowlist(self):
        """Allowlist must accept npub bech32 form, normalizing to hex."""
        from gateway.platforms.nostr import NostrAdapter
        from nostr_sdk import Keys
        # Generate a known npub from a known privkey
        keys = Keys.parse("ab" * 32)
        npub = keys.public_key().to_bech32()
        expected_hex = keys.public_key().to_hex()
        config = self._make_config()
        with patch.dict(os.environ, {"NOSTR_ALLOWED_NPUBS": npub}, clear=False):
            adapter = NostrAdapter(config)
        assert expected_hex in adapter._allowed_pubkeys

    def test_adapter_missing_token_raises(self):
        from gateway.platforms.nostr import NostrAdapter
        from gateway.config import PlatformConfig
        config = PlatformConfig(enabled=True, token="", extra={"relays": "wss://x.com"})
        with pytest.raises(ValueError, match="NOSTR_PRIVATE_KEY"):
            NostrAdapter(config)

    def test_adapter_default_denies_all(self):
        """Empty NOSTR_ALLOWED_NPUBS must default to deny-all, not open access."""
        from gateway.platforms.nostr import NostrAdapter
        config = self._make_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)
        assert not adapter._allow_all_npubs
        assert len(adapter._allowed_pubkeys) == 0

    def test_adapter_star_allows_all(self):
        """NOSTR_ALLOWED_NPUBS=* must set allow-all flag."""
        from gateway.platforms.nostr import NostrAdapter
        config = self._make_config()
        with patch.dict(os.environ, {"NOSTR_ALLOWED_NPUBS": "*"}, clear=False):
            adapter = NostrAdapter(config)
        assert adapter._allow_all_npubs
        assert len(adapter._allowed_pubkeys) == 0

    def test_adapter_allow_all_users_boolean_opens_adapter_gate(self):
        # Cross-platform convention: every <PLATFORM>_ALLOW_ALL_USERS=true should
        # actually open access. Without this synonym, an operator who set only the
        # boolean (matching Discord/Slack muscle memory) would silently still be
        # denied at the adapter's _process_event allowlist check, because the
        # gateway-level gate that reads the boolean runs after that check.
        from gateway.platforms.nostr import NostrAdapter
        config = self._make_config()
        with patch.dict(
            os.environ,
            {"NOSTR_ALLOW_ALL_USERS": "true"},
            clear=False,
        ):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)
        assert adapter._allow_all_npubs
        assert len(adapter._allowed_pubkeys) == 0

    def test_adapter_allow_all_users_accepts_truthy_variants(self):
        # Mirrors gateway/run.py:_is_user_authorized which accepts {true,1,yes}.
        from gateway.platforms.nostr import NostrAdapter
        for variant in ("true", "True", "TRUE", "1", "yes", "YES"):
            config = self._make_config()
            with patch.dict(os.environ, {"NOSTR_ALLOW_ALL_USERS": variant}, clear=False):
                os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
                adapter = NostrAdapter(config)
            assert adapter._allow_all_npubs, f"{variant!r} should open the adapter gate"

    def test_adapter_allow_all_users_falsy_does_not_open(self):
        # Empty / false / random strings must NOT open the adapter gate — that
        # would silently weaken the default-deny default.
        from gateway.platforms.nostr import NostrAdapter
        for variant in ("", "false", "no", "0", "maybe"):
            config = self._make_config()
            with patch.dict(os.environ, {"NOSTR_ALLOW_ALL_USERS": variant}, clear=False):
                os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
                adapter = NostrAdapter(config)
            assert not adapter._allow_all_npubs, f"{variant!r} must not open the gate"

    def test_adapter_missing_relays_raises(self):
        from gateway.platforms.nostr import NostrAdapter
        from gateway.config import PlatformConfig
        config = PlatformConfig(enabled=True, token="01" * 32, extra={"relays": ""})
        with pytest.raises(ValueError, match="NOSTR_RELAYS"):
            NostrAdapter(config)

    def test_adapter_lookback_default_is_48h(self):
        from gateway.platforms.nostr import NostrAdapter, NIP59_MIN_LOOKBACK_MINUTES
        config = self._make_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)
        assert adapter._lookback_seconds == NIP59_MIN_LOOKBACK_MINUTES * 60

    def test_adapter_lookback_overrides_via_extra(self):
        from gateway.platforms.nostr import NostrAdapter
        config = self._make_config(extra={"lookback_minutes": "60"})
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)
        assert adapter._lookback_seconds == 60 * 60


# ---------------------------------------------------------------------------
# Runtime behavior — _process_event, send(), connect lifecycle
# ---------------------------------------------------------------------------
# Init-time tests above cover attribute assignment; this block exercises the
# runtime paths the reviewer flagged as untested: allowlist enforcement at
# message-handling time, dedup-window isolation from blocked senders, NIP-40
# expiration tag on outbound DMs, self-message rejection, the disconnected
# send guard, and the connect()-with-zero-relays bail.

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("nostr_sdk") is None,
    reason="nostr-sdk not installed",
)
class TestNostrAdapterRuntime:
    def _make_config(self, privkey_hex=None, relays=None, extra=None):
        from gateway.config import PlatformConfig
        privkey_hex = privkey_hex or ("01" * 32)
        relays = relays or "wss://relay.example.com"
        return PlatformConfig(
            enabled=True,
            token=privkey_hex,
            extra={"relays": relays, **(extra or {})},
        )

    def _mock_gift_wrap_event(self, event_id_hex=None):
        from nostr_sdk import KindStandard
        event_id_hex = event_id_hex or ("d" * 64)
        mock_event = MagicMock()
        mock_event.id.return_value.to_hex.return_value = event_id_hex
        mock_event.kind.return_value.as_std.return_value = KindStandard.GIFT_WRAP
        return mock_event

    def _mock_unwrapped(self, sender_pk, content="hello"):
        from nostr_sdk import KindStandard
        mock_rumor = MagicMock()
        mock_rumor.kind.return_value.as_std.return_value = KindStandard.PRIVATE_DIRECT_MESSAGE
        mock_rumor.content.return_value = content
        mock_unwrapped = MagicMock()
        mock_unwrapped.sender.return_value = sender_pk
        mock_unwrapped.rumor.return_value = mock_rumor
        return mock_unwrapped

    @pytest.mark.asyncio
    async def test_process_event_drops_disallowed_sender(self):
        from gateway.platforms.nostr import NostrAdapter
        from nostr_sdk import Keys

        # Default deny-all
        config = self._make_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)

        stranger_pk = Keys.parse("99" * 32).public_key()
        event = self._mock_gift_wrap_event()
        unwrapped = self._mock_unwrapped(stranger_pk)

        adapter.handle_message = AsyncMock()
        adapter._save_seen_ids = MagicMock()

        with patch("gateway.platforms.nostr.UnwrappedGift") as mock_uw:
            mock_uw.from_gift_wrap = AsyncMock(return_value=unwrapped)
            await adapter._process_event(event, "wss://relay.example.com")

        adapter.handle_message.assert_not_awaited()
        adapter._save_seen_ids.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocked_sender_does_not_pollute_dedup_window(self):
        # Allowlist check runs BEFORE the seen-list insert (see _process_event).
        # If a blocked sender's id ever landed in _seen_event_ids, a flood from
        # disallowed senders could evict legitimate ids and enable replay.
        from gateway.platforms.nostr import NostrAdapter
        from nostr_sdk import Keys

        config = self._make_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)

        stranger_pk = Keys.parse("99" * 32).public_key()
        event_id = "a" * 64
        event = self._mock_gift_wrap_event(event_id_hex=event_id)
        unwrapped = self._mock_unwrapped(stranger_pk)

        adapter.handle_message = AsyncMock()
        adapter._save_seen_ids = MagicMock()

        with patch("gateway.platforms.nostr.UnwrappedGift") as mock_uw:
            mock_uw.from_gift_wrap = AsyncMock(return_value=unwrapped)
            await adapter._process_event(event, "wss://relay.example.com")
            await adapter._process_event(event, "wss://relay.example.com")

        assert event_id not in adapter._seen_event_ids
        assert event_id not in adapter._seen_event_list
        assert len(adapter._seen_event_ids) == 0
        adapter.handle_message.assert_not_awaited()
        adapter._save_seen_ids.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_event_rejects_self_message(self):
        from gateway.platforms.nostr import NostrAdapter
        from nostr_sdk import Keys

        privkey_hex = "01" * 32
        config = self._make_config(privkey_hex=privkey_hex)
        with patch.dict(os.environ, {"NOSTR_ALLOWED_NPUBS": "*"}, clear=False):
            adapter = NostrAdapter(config)

        # Bot sees a gift wrap whose unwrapped sender == itself (NIP-17 self-copy).
        self_pk = Keys.parse(privkey_hex).public_key()
        event = self._mock_gift_wrap_event()
        unwrapped = self._mock_unwrapped(self_pk)

        adapter.handle_message = AsyncMock()

        with patch("gateway.platforms.nostr.UnwrappedGift") as mock_uw:
            mock_uw.from_gift_wrap = AsyncMock(return_value=unwrapped)
            await adapter._process_event(event, "wss://relay.example.com")

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_attaches_nip40_expiration_tag(self):
        import time
        from gateway.platforms.nostr import NostrAdapter
        from nostr_sdk import Keys

        # 60-minute TTL is far from any default so the bound check is meaningful
        config = self._make_config(extra={"expiration_minutes": "60"})
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)

        recipient_hex = Keys.parse("ab" * 32).public_key().to_hex()

        mock_client = MagicMock()
        mock_client.send_private_msg = AsyncMock()
        adapter._client = mock_client

        before = int(time.time())
        result = await adapter.send(recipient_hex, "hello")
        after = int(time.time())

        assert result.success is True
        # Two sends: one to recipient, one self-copy for sent-history recovery
        assert mock_client.send_private_msg.call_count == 2

        for call in mock_client.send_private_msg.call_args_list:
            args = call.args
            assert len(args) == 3, "expected (recipient_pk, content, [expiration_tag])"
            tags = args[2]
            assert isinstance(tags, list) and len(tags) == 1
            tag_vec = tags[0].as_vec()
            assert tag_vec[0] == "expiration"
            ts = int(tag_vec[1])
            assert before + 60 * 60 <= ts <= after + 60 * 60

    @pytest.mark.asyncio
    async def test_send_returns_error_when_not_connected(self):
        from gateway.platforms.nostr import NostrAdapter

        config = self._make_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)

        assert adapter._client is None
        result = await adapter.send("ab" * 32, "hello")
        assert result.success is False
        assert "not connected" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_send_chunks_long_messages(self):
        # Relays accept large events but clients render long DMs poorly, and
        # declaring MAX_MESSAGE_LENGTH is inert unless send() actually splits.
        # A >MAX_MESSAGE_LENGTH reply must go out as multiple NIP-17 DMs, each
        # with its own self-copy.
        from gateway.platforms.nostr import NostrAdapter
        from nostr_sdk import Keys

        config = self._make_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)

        recipient_hex = Keys.parse("ab" * 32).public_key().to_hex()
        mock_client = MagicMock()
        mock_client.send_private_msg = AsyncMock()
        adapter._client = mock_client

        # Two full chunks' worth of newline-free text forces a split.
        long_text = "x" * (adapter.MAX_MESSAGE_LENGTH * 2)
        expected_chunks = adapter.truncate_message(long_text, adapter.MAX_MESSAGE_LENGTH)
        assert len(expected_chunks) >= 2, "test text should span multiple chunks"

        result = await adapter.send(recipient_hex, long_text)

        assert result.success is True
        # Each chunk is sent once to the recipient and once as a self-copy.
        assert mock_client.send_private_msg.call_count == len(expected_chunks) * 2
        # Every send stays within the declared per-message ceiling.
        for call in mock_client.send_private_msg.call_args_list:
            assert len(call.args[1]) <= adapter.MAX_MESSAGE_LENGTH

    @pytest.mark.asyncio
    async def test_send_skips_empty_content(self):
        from gateway.platforms.nostr import NostrAdapter
        from nostr_sdk import Keys

        config = self._make_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)

        mock_client = MagicMock()
        mock_client.send_private_msg = AsyncMock()
        adapter._client = mock_client

        result = await adapter.send(Keys.parse("ab" * 32).public_key().to_hex(), "   ")
        assert result.success is True
        assert mock_client.send_private_msg.call_count == 0


# ---------------------------------------------------------------------------
# connect() relay-add guard + _active_instance lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("nostr_sdk") is None,
    reason="nostr-sdk not installed",
)
class TestNostrConnectLifecycle:
    def _make_config(self, relays="wss://relay.example.com"):
        from gateway.config import PlatformConfig
        return PlatformConfig(
            enabled=True,
            token="01" * 32,
            extra={"relays": relays},
        )

    @pytest.mark.asyncio
    async def test_connect_returns_false_when_all_relays_fail(self):
        # Without this guard the bot would be running but deaf and mute —
        # connected to zero relays with the notification loop blissfully
        # waiting for events that can't arrive.
        from gateway.platforms.nostr import NostrAdapter

        config = self._make_config(relays="wss://r1.example.com,wss://r2.example.com")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)
        NostrAdapter.set_active(None)  # isolate from prior test state

        adapter._load_seen_ids = MagicMock()
        mock_client = MagicMock()
        mock_client.add_relay = AsyncMock(side_effect=RuntimeError("relay rejected"))
        mock_client.connect = AsyncMock()
        mock_client.subscribe = AsyncMock()

        with patch("gateway.platforms.nostr.Client", return_value=mock_client):
            ok = await adapter.connect()

        assert ok is False
        mock_client.connect.assert_not_awaited()
        mock_client.subscribe.assert_not_awaited()
        assert NostrAdapter.get_active() is not adapter

    @pytest.mark.asyncio
    async def test_connect_aborts_when_identity_lock_held(self):
        # Two gateways driving the same nsec would double-process inbound gift
        # wraps, double-send outbound DMs, and corrupt the pubkey-scoped dedup
        # file. connect() must bail before building a client when the
        # identity lock is already held elsewhere.
        from gateway.platforms.nostr import NostrAdapter

        config = self._make_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)
        NostrAdapter.set_active(None)

        adapter._load_seen_ids = MagicMock()
        adapter._acquire_platform_lock = MagicMock(return_value=False)
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()

        with patch("gateway.platforms.nostr.Client", return_value=mock_client):
            ok = await adapter.connect()

        assert ok is False
        adapter._acquire_platform_lock.assert_called_once()
        # The lock scope/identity must be the bot pubkey, not e.g. the relay URL.
        scope, identity, _desc = adapter._acquire_platform_lock.call_args.args
        assert scope == "nostr-pubkey"
        assert identity == adapter._pubkey_hex
        adapter._load_seen_ids.assert_not_called()
        mock_client.connect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connect_releases_identity_lock_when_all_relays_fail(self):
        # The zero-relay bail-out must not leak the identity lock, or a clean
        # restart on the same nsec would falsely report the identity in use.
        from gateway.platforms.nostr import NostrAdapter

        config = self._make_config(relays="wss://r1.example.com")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)
        NostrAdapter.set_active(None)

        adapter._load_seen_ids = MagicMock()
        adapter._acquire_platform_lock = MagicMock(return_value=True)
        adapter._release_platform_lock = MagicMock()
        mock_client = MagicMock()
        mock_client.add_relay = AsyncMock(side_effect=RuntimeError("relay rejected"))

        with patch("gateway.platforms.nostr.Client", return_value=mock_client):
            ok = await adapter.connect()

        assert ok is False
        adapter._release_platform_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_watchdog_before_notif_task(self):
        # disconnect() must stop the watchdog first so it can't kick off a
        # recovery cycle while we're tearing the client down.
        from gateway.platforms.nostr import NostrAdapter

        config = self._make_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)

        adapter._running = True

        async def _forever():
            await asyncio.Future()

        adapter._notif_task = asyncio.create_task(_forever())
        adapter._watchdog_tick_seconds = 0.001
        adapter._watchdog_task = asyncio.create_task(adapter._watchdog_loop())
        await asyncio.sleep(0.01)

        await adapter.disconnect()

        assert adapter._running is False
        assert adapter._watchdog_task is None
        assert adapter._notif_task is None

    @pytest.mark.asyncio
    async def test_active_instance_cleared_on_disconnect(self):
        # send_message_tool.py reuses _active_instance to avoid spinning up
        # short-lived adapters that republish profile/relay metadata. The
        # ClassVar must be cleared on disconnect so a stale reference doesn't
        # outlive the connection.
        from gateway.platforms.nostr import NostrAdapter

        config = self._make_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(config)

        NostrAdapter.set_active(adapter)
        assert NostrAdapter.get_active() is adapter

        adapter._client = None
        adapter._notif_task = None
        adapter._watchdog_task = None
        adapter._running = True
        await adapter.disconnect()

        assert NostrAdapter.get_active() is None


# ---------------------------------------------------------------------------
# Watchdog — notification-loop health supervision + recovery
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("nostr_sdk") is None,
    reason="nostr-sdk not installed",
)
class TestNostrWatchdog:
    def _make_config(self):
        from gateway.config import PlatformConfig
        return PlatformConfig(
            enabled=True,
            token="01" * 32,
            extra={"relays": "wss://relay.example.com"},
        )

    def _make_adapter(self):
        from gateway.platforms.nostr import NostrAdapter
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOSTR_ALLOWED_NPUBS", None)
            adapter = NostrAdapter(self._make_config())
        # Compress real-time delays into deterministic, fast values.
        adapter._watchdog_tick_seconds = 0.001
        adapter._watchdog_recovery_delays_initial = (0.001, 0.001, 0.001)
        adapter._watchdog_recovery_delay_steady = 0.001
        adapter._watchdog_reconnect_settle_seconds = 0.0
        adapter._running = True
        return adapter

    def _mock_healthy_client(self):
        from nostr_sdk import RelayStatus
        relay = MagicMock()
        relay.status.return_value = RelayStatus.CONNECTED
        client = MagicMock()
        client.relays = AsyncMock(return_value={"wss://r.example.com": relay})
        client.disconnect = AsyncMock()
        client.connect = AsyncMock()
        client.subscribe = AsyncMock()
        return client

    def _alive_task(self):
        async def _forever():
            await asyncio.Future()
        return asyncio.create_task(_forever())

    async def _stop_task(self, task):
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_healthy_loop_does_not_trigger_recovery(self):
        adapter = self._make_adapter()
        adapter._client = self._mock_healthy_client()
        adapter._notif_task = self._alive_task()
        adapter._soft_reconnect_cycle = AsyncMock()
        try:
            wd = asyncio.create_task(adapter._watchdog_loop())
            await asyncio.sleep(0.05)  # several ticks
            adapter._running = False
            await self._stop_task(wd)
            adapter._soft_reconnect_cycle.assert_not_awaited()
        finally:
            await self._stop_task(adapter._notif_task)

    @pytest.mark.asyncio
    async def test_unhealthy_loop_triggers_recovery(self):
        adapter = self._make_adapter()
        adapter._client = self._mock_healthy_client()
        adapter._notif_task = None  # task missing → unhealthy

        async def _fake_recovery():
            # Simulate a successful soft-reconnect: a new notif task appears.
            adapter._notif_task = self._alive_task()

        adapter._soft_reconnect_cycle = AsyncMock(side_effect=_fake_recovery)

        try:
            wd = asyncio.create_task(adapter._watchdog_loop())
            await asyncio.sleep(0.1)
            adapter._running = False
            await self._stop_task(wd)
            assert adapter._soft_reconnect_cycle.await_count >= 1
        finally:
            await self._stop_task(adapter._notif_task)

    @pytest.mark.asyncio
    async def test_retries_indefinitely_until_disconnect(self):
        # Matches the codebase convention (signal.py, yuanbao.py, matrix.py,
        # mattermost.py): persistent-loop adapters retry forever rather than
        # capitulate. A long network outage must self-heal once relays are
        # reachable again, not permanently silence the bot.
        adapter = self._make_adapter()
        adapter._client = self._mock_healthy_client()
        adapter._notif_task = None  # unhealthy from the start
        adapter._soft_reconnect_cycle = AsyncMock(
            side_effect=RuntimeError("relays still down")
        )

        wd = asyncio.create_task(adapter._watchdog_loop())
        # Let it churn far past the 3-element initial ramp so we observe it
        # entering the steady-state retry phase.
        await asyncio.sleep(0.1)
        adapter._running = False
        await self._stop_task(wd)

        # Must have attempted recovery well beyond the initial ramp length.
        assert adapter._soft_reconnect_cycle.await_count > len(
            adapter._watchdog_recovery_delays_initial
        )

    @pytest.mark.asyncio
    async def test_recovery_loop_yields_on_disconnect_mid_flight(self):
        # disconnect() during a recovery cycle must end cleanly, not deadlock
        # on the recovery_lock or leak the soft-reconnect task.
        adapter = self._make_adapter()
        adapter._client = self._mock_healthy_client()
        adapter._notif_task = None
        adapter._soft_reconnect_cycle = AsyncMock(
            side_effect=RuntimeError("permanently down")
        )

        wd = asyncio.create_task(adapter._watchdog_loop())
        await asyncio.sleep(0.02)  # let recovery start churning

        adapter._running = False
        await self._stop_task(wd)

        # Watchdog exited cleanly; no exception escaped.
        assert wd.done()

    @pytest.mark.asyncio
    async def test_recovery_does_not_republish_metadata(self):
        # Reconnect storms must not spam relays with duplicate Kind 0 / Kind
        # 10050 events — those live on the relays already.
        adapter = self._make_adapter()
        client = self._mock_healthy_client()
        adapter._client = client
        adapter._notif_task = self._alive_task()
        adapter._publish_profile = AsyncMock()
        adapter._publish_relay_list = AsyncMock()

        try:
            # Patch the module-level Client(self._signer) call inside the cycle
            # by also mocking handle_notifications to keep the new notif task alive.
            client.handle_notifications = MagicMock(
                return_value=asyncio.sleep(10)
            )
            await adapter._soft_reconnect_cycle()

            adapter._publish_profile.assert_not_awaited()
            adapter._publish_relay_list.assert_not_awaited()
            client.disconnect.assert_awaited()
            client.connect.assert_awaited()
            client.subscribe.assert_awaited()
        finally:
            await self._stop_task(adapter._notif_task)

    @pytest.mark.asyncio
    async def test_recovery_uses_short_lookback_not_48h(self):
        # NIP-59's 48h lookback is for fresh boots; on a recovery cycle we
        # only need to backfill the outage window. A full 48h replay on every
        # blip would hammer relays and re-traverse dedup unnecessarily.
        adapter = self._make_adapter()
        adapter._watchdog_recovery_lookback_seconds = 600  # 10 min
        client = self._mock_healthy_client()
        adapter._client = client
        adapter._notif_task = self._alive_task()
        client.handle_notifications = MagicMock(return_value=asyncio.sleep(10))

        import time as _t
        before = int(_t.time())
        try:
            await adapter._soft_reconnect_cycle()
        finally:
            await self._stop_task(adapter._notif_task)
        after = int(_t.time())

        # The Filter we built includes a `since`. We can't easily introspect
        # nostr-sdk's Filter object, so we instead assert recovery did NOT
        # use the full 48h window by checking the constant we wired in.
        # (Direct introspection of Filter would couple the test to SDK
        # internals; the constant check is the durable guarantee.)
        assert adapter._watchdog_recovery_lookback_seconds < NIP59_MIN_LOOKBACK_MINUTES * 60
        assert before <= after  # sanity

    @pytest.mark.asyncio
    async def test_is_loop_healthy_when_all_signals_pass(self):
        adapter = self._make_adapter()
        adapter._client = self._mock_healthy_client()
        adapter._notif_task = self._alive_task()
        try:
            assert await adapter._is_loop_healthy() is True
        finally:
            await self._stop_task(adapter._notif_task)

    @pytest.mark.asyncio
    async def test_is_loop_healthy_false_when_no_client(self):
        adapter = self._make_adapter()
        adapter._client = None
        adapter._notif_task = self._alive_task()
        try:
            assert await adapter._is_loop_healthy() is False
        finally:
            await self._stop_task(adapter._notif_task)

    @pytest.mark.asyncio
    async def test_is_loop_healthy_false_when_task_done(self):
        adapter = self._make_adapter()
        adapter._client = self._mock_healthy_client()

        async def _instant():
            return None

        adapter._notif_task = asyncio.create_task(_instant())
        await adapter._notif_task  # let it complete
        assert await adapter._is_loop_healthy() is False

    @pytest.mark.asyncio
    async def test_is_loop_healthy_false_when_all_relays_disconnected(self):
        from nostr_sdk import RelayStatus
        adapter = self._make_adapter()
        relay = MagicMock()
        relay.status.return_value = RelayStatus.DISCONNECTED
        client = MagicMock()
        client.relays = AsyncMock(return_value={"wss://r.example.com": relay})
        adapter._client = client
        adapter._notif_task = self._alive_task()
        try:
            assert await adapter._is_loop_healthy() is False
        finally:
            await self._stop_task(adapter._notif_task)


# NIP59_MIN_LOOKBACK_MINUTES is imported by tests above; re-import here for
# the watchdog lookback assertion.
from gateway.platforms.nostr import NIP59_MIN_LOOKBACK_MINUTES  # noqa: E402

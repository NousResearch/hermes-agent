"""Tests for the Nostr NIP-17 gateway adapter."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.session import SessionSource


def _make_nostr_adapter(**extra):
    from gateway.platforms.nostr import NostrAdapter

    config = PlatformConfig(enabled=True, token=extra.pop("token", "nsec1test"))
    config.extra = {"relays": ["wss://relay.example"], **extra}
    return NostrAdapter(config)


class _FakeEvent:
    def __init__(self, payload):
        self._payload = payload

    def as_json(self):
        return json.dumps(self._payload)


class _BadEvent:
    pass


class _FakeFilter:
    def kind(self, _value):
        return self

    def pubkey(self, _value):
        return self

    def pubkeys(self, _value):
        return self

    def author(self, _value):
        return self

    def authors(self, _value):
        return self

    def limit(self, _value):
        return self


class _FakePublicKey:
    def __init__(self, value):
        self.value = value

    def to_hex(self):
        return self.value

    def to_bech32(self):
        return f"npub_{self.value}"


class _FakeKeys:
    @staticmethod
    def parse(_secret):
        return _FakeKeys()

    def public_key(self):
        return _FakePublicKey("bothex")


class _FakeClient:
    def __init__(self, _signer, fetch_result=None):
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.sent = []
        self.relays = []

    async def add_relay(self, relay):
        self.relays.append(relay)

    async def connect(self):
        return None

    async def disconnect(self):
        return None

    async def fetch_events(self, _filter, _timeout):
        return self.fetch_result

    async def send_private_msg(self, recipient, content, rumor_extra_tags=None):
        self.sent.append((recipient, content, rumor_extra_tags))


class TestNostrPlatformEnum:
    def test_nostr_enum_exists(self):
        assert Platform.NOSTR.value == "nostr"

    def test_nostr_in_platform_list(self):
        platforms = [p.value for p in Platform]
        assert "nostr" in platforms


class TestNostrConfigLoading:
    def test_apply_env_overrides_nostr(self, monkeypatch):
        monkeypatch.setenv("NOSTR_SECRET_KEY", "nsec1example")
        monkeypatch.setenv("NOSTR_RELAYS", "wss://relay.one,wss://relay.two")
        monkeypatch.setenv("NOSTR_HOME_CHANNEL", "npub1home")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.NOSTR in config.platforms
        nostr_cfg = config.platforms[Platform.NOSTR]
        assert nostr_cfg.enabled is True
        assert nostr_cfg.token == "nsec1example"
        assert nostr_cfg.extra["relays"] == ["wss://relay.one", "wss://relay.two"]
        assert nostr_cfg.home_channel.chat_id == "npub1home"

    def test_connected_platforms_requires_relays(self):
        config = GatewayConfig(
            platforms={
                Platform.NOSTR: PlatformConfig(enabled=True, token="nsec1example", extra={})
            }
        )
        assert Platform.NOSTR not in config.get_connected_platforms()

    def test_connected_platforms_includes_nostr_with_secret_and_relays(self):
        config = GatewayConfig(
            platforms={
                Platform.NOSTR: PlatformConfig(
                    enabled=True,
                    token="nsec1example",
                    extra={"relays": ["wss://relay.example"]},
                )
            }
        )
        assert Platform.NOSTR in config.get_connected_platforms()


class TestNostrHelpers:
    def test_check_requirements(self, monkeypatch):
        monkeypatch.setenv("NOSTR_SECRET_KEY", "nsec1example")
        monkeypatch.setenv("NOSTR_RELAYS", "wss://relay.example")
        with patch("gateway.platforms.nostr.importlib.util.find_spec", return_value=object()):
            from gateway.platforms.nostr import check_nostr_requirements

            assert check_nostr_requirements() is True

    def test_check_requirements_missing(self, monkeypatch):
        monkeypatch.delenv("NOSTR_SECRET_KEY", raising=False)
        monkeypatch.delenv("NOSTR_NSEC", raising=False)
        monkeypatch.delenv("NOSTR_RELAYS", raising=False)
        with patch("gateway.platforms.nostr.importlib.util.find_spec", return_value=None):
            from gateway.platforms.nostr import check_nostr_requirements

            assert check_nostr_requirements() is False

    def test_check_requirements_accepts_config_without_env(self, monkeypatch):
        monkeypatch.delenv("NOSTR_SECRET_KEY", raising=False)
        monkeypatch.delenv("NOSTR_NSEC", raising=False)
        monkeypatch.delenv("NOSTR_RELAYS", raising=False)
        with patch("gateway.platforms.nostr.importlib.util.find_spec", return_value=object()):
            from gateway.platforms.nostr import check_nostr_requirements

            config = PlatformConfig(enabled=True, token="nsec1example", extra={"relays": ["wss://relay.example"]})
            assert check_nostr_requirements(config) is True

    def test_extract_dm_relays_from_events(self):
        from gateway.platforms.nostr import _extract_dm_relays_from_events

        events = [
            _FakeEvent({"created_at": 1, "tags": [["relay", "wss://relay.one"]]}),
            _FakeEvent({"created_at": 2, "tags": [["relay", "wss://relay.one"], ["relay", "wss://relay.two"]]}),
        ]

        assert _extract_dm_relays_from_events(events) == ["wss://relay.one", "wss://relay.two"]

    def test_extract_dm_relays_from_events_prefers_latest_replaceable_event(self):
        from gateway.platforms.nostr import _extract_dm_relays_from_events

        events = [
            _FakeEvent({"created_at": 10, "tags": [["relay", "wss://relay.old"], ["relay", "wss://relay.shared"]]}),
            _FakeEvent({"created_at": 20, "tags": [["relay", "wss://relay.new"], ["relay", "wss://relay.shared"]]}),
        ]

        assert _extract_dm_relays_from_events(events) == ["wss://relay.new", "wss://relay.shared"]

    def test_message_event_from_rumor_maps_sender_to_dm_session(self):
        from gateway.platforms.nostr import _message_event_from_rumor

        event = _message_event_from_rumor(
            sender_hex="abcd1234",
            sender_display="npub1sender",
            rumor_data={"id": "rumor1", "created_at": 1_700_000_000, "content": "hello"},
            raw_message={"id": "gift1"},
        )

        assert event.text == "hello"
        assert event.source.platform == Platform.NOSTR
        assert event.source.chat_id == "abcd1234"
        assert event.source.user_id == "abcd1234"
        assert event.source.chat_type == "dm"

    def test_build_dm_relay_list_builder_uses_kind_10050_and_relay_tags(self):
        from gateway.platforms.nostr import _build_dm_relay_list_builder

        parsed_tags = []

        class _FakeBuilder:
            def __init__(self, kind, content):
                self.kind = kind
                self.content = content
                self.tags_value = None

            def tags(self, tags):
                self.tags_value = tags
                return self

        fake_sdk = SimpleNamespace(
            EventBuilder=_FakeBuilder,
            Tag=SimpleNamespace(parse=lambda tag: parsed_tags.append(tag) or tag),
            Kind=lambda value: value,
        )

        builder = _build_dm_relay_list_builder(fake_sdk, ["wss://relay.one", "wss://relay.two"])

        assert builder.kind == 10050
        assert builder.content == ""
        assert parsed_tags == [["relay", "wss://relay.one"], ["relay", "wss://relay.two"]]
        assert builder.tags_value == parsed_tags


class TestNostrAuthorization:
    def _make_runner(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.NOSTR: PlatformConfig(
                    enabled=True,
                    token="nsec1test",
                    extra={"relays": ["wss://relay.example"]},
                )
            }
        )
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False
        return runner

    def test_allowlist_npub_authorizes_hex_sender(self, monkeypatch):
        runner = self._make_runner()
        monkeypatch.setenv("NOSTR_ALLOWED_USERS", "npub1allowed")

        with patch(
            "gateway.platforms.nostr.normalize_nostr_identifier",
            side_effect=lambda value: {
                "npub1allowed": "abcdef1234",
                "abcdef1234": "abcdef1234",
            }.get(value, str(value).lower()),
        ):
            source = SessionSource(
                platform=Platform.NOSTR,
                user_id="abcdef1234",
                chat_id="abcdef1234",
                user_name="npub1sender",
                chat_type="dm",
            )
            assert runner._is_user_authorized(source) is True

    def test_allowlist_hex_authorizes_npub_sender(self, monkeypatch):
        runner = self._make_runner()
        monkeypatch.setenv("NOSTR_ALLOWED_USERS", "abcdef1234")

        with patch(
            "gateway.platforms.nostr.normalize_nostr_identifier",
            side_effect=lambda value: {
                "abcdef1234": "abcdef1234",
                "npub1sender": "abcdef1234",
            }.get(value, str(value).lower()),
        ):
            source = SessionSource(
                platform=Platform.NOSTR,
                user_id="npub1sender",
                chat_id="npub1sender",
                user_name="npub1sender",
                chat_type="dm",
            )
            assert runner._is_user_authorized(source) is True


class TestNostrAdapter:
    def test_init_parses_config(self):
        adapter = _make_nostr_adapter(token="nsec1abc", relays=["wss://relay.one", "wss://relay.two"])
        assert adapter.secret_key == "nsec1abc"
        assert adapter.relays == ["wss://relay.one", "wss://relay.two"]

    @pytest.mark.asyncio
    async def test_handle_gift_wrap_ignores_self_echo(self):
        adapter = _make_nostr_adapter()
        adapter._sdk = SimpleNamespace()
        adapter._signer = object()
        adapter._public_key_hex = "bothex"
        adapter.handle_message = AsyncMock()

        class _FakeUnwrapped:
            def rumor(self):
                return _FakeEvent({"kind": 14, "content": "hi"})

            def sender(self):
                return _FakePublicKey("bothex")

        class _FakeUnwrappedGift:
            @staticmethod
            async def from_gift_wrap(_signer, _event):
                return _FakeUnwrapped()

        adapter._sdk.UnwrappedGift = _FakeUnwrappedGift
        await adapter._handle_gift_wrap(_FakeEvent({"id": "gift1"}), {"id": "gift1"})
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_gift_wrap_swallow_unwrap_failure(self):
        adapter = _make_nostr_adapter()
        adapter._sdk = SimpleNamespace()
        adapter._signer = object()

        class _FakeUnwrappedGift:
            @staticmethod
            async def from_gift_wrap(_signer, _event):
                raise RuntimeError("boom")

        adapter._sdk.UnwrappedGift = _FakeUnwrappedGift
        await adapter._handle_gift_wrap(_FakeEvent({"id": "gift1"}), {"id": "gift1"})

    @pytest.mark.asyncio
    async def test_publish_dm_relays_sends_kind_10050_to_configured_relays(self):
        adapter = _make_nostr_adapter(relays=["wss://relay.one", "wss://relay.two"])
        sent = {}

        class _FakeBuilder:
            def __init__(self, kind, content):
                self.kind = kind
                self.content = content
                self.tags_value = None

            def tags(self, tags):
                self.tags_value = tags
                return self

        class _FakeClientWithPublish(_FakeClient):
            async def send_event_builder_to(self, urls, builder):
                sent["urls"] = urls
                sent["builder"] = builder

        adapter._sdk = SimpleNamespace(
            EventBuilder=_FakeBuilder,
            Tag=SimpleNamespace(parse=lambda tag: tag),
            Kind=lambda value: value,
            RelayUrl=SimpleNamespace(parse=lambda relay: relay),
        )
        adapter._client = _FakeClientWithPublish(object())

        await adapter._publish_dm_relays()

        assert sent["urls"] == ["wss://relay.one", "wss://relay.two"]
        assert sent["builder"].kind == 10050
        assert sent["builder"].tags_value == [["relay", "wss://relay.one"], ["relay", "wss://relay.two"]]

    @pytest.mark.asyncio
    async def test_poll_inbox_once_skips_malformed_events_and_keeps_valid_order(self):
        adapter = _make_nostr_adapter()
        adapter._sdk = SimpleNamespace(Filter=_FakeFilter, Kind=lambda value: value)
        adapter._public_key = _FakePublicKey("bothex")
        adapter._client = _FakeClient(
            object(),
            fetch_result=[
                _BadEvent(),
                _FakeEvent({"id": "gift-late", "created_at": 20}),
                _FakeEvent({"id": "gift-early", "created_at": 10}),
            ],
        )
        adapter._handle_gift_wrap = AsyncMock()

        await adapter._poll_inbox_once()

        handled_ids = [call.args[1]["id"] for call in adapter._handle_gift_wrap.await_args_list]
        assert handled_ids == ["gift-early", "gift-late"]

    @pytest.mark.asyncio
    async def test_poll_inbox_once_ignores_replayed_seen_event_ids(self):
        adapter = _make_nostr_adapter()
        adapter._sdk = SimpleNamespace(Filter=_FakeFilter, Kind=lambda value: value)
        adapter._public_key = _FakePublicKey("bothex")
        adapter._client = _FakeClient(
            object(),
            fetch_result=[
                _FakeEvent({"id": "gift-2", "created_at": 20}),
                _FakeEvent({"id": "gift-1", "created_at": 10}),
                _FakeEvent({"id": "gift-2", "created_at": 20}),
            ],
        )
        adapter._handle_gift_wrap = AsyncMock()

        await adapter._poll_inbox_once()

        handled_ids = [call.args[1]["id"] for call in adapter._handle_gift_wrap.await_args_list]
        assert handled_ids == ["gift-1", "gift-2"]


class TestNostrSend:
    @pytest.mark.asyncio
    async def test_send_nostr_dm_once_fails_without_kind_10050(self):
        from gateway.platforms import nostr as nostr_mod

        fake_client = _FakeClient(object(), fetch_result=[])
        fake_sdk = SimpleNamespace(
            Keys=_FakeKeys,
            NostrSigner=SimpleNamespace(keys=lambda keys: object()),
            Client=lambda signer: fake_client,
            RelayUrl=SimpleNamespace(parse=lambda relay: relay),
            PublicKey=SimpleNamespace(parse=lambda value: value),
            Filter=_FakeFilter,
            Kind=lambda value: value,
            uniffi_set_event_loop=lambda _loop: None,
        )

        with patch.object(nostr_mod, "_load_nostr_sdk", return_value=fake_sdk):
            result = await nostr_mod.send_nostr_dm_once(
                "nsec1secret",
                ["wss://relay.example"],
                "npub1recipient",
                "hello",
            )

        assert result.success is False
        assert "kind 10050" in result.error

    @pytest.mark.asyncio
    async def test_send_nostr_dm_once_sends_without_extra_tag_relays(self):
        from gateway.platforms import nostr as nostr_mod

        relay_list_event = _FakeEvent({"tags": [["relay", "wss://dm.example"]]})
        fake_client = _FakeClient(object(), fetch_result=[relay_list_event])
        fake_sdk = SimpleNamespace(
            Keys=_FakeKeys,
            NostrSigner=SimpleNamespace(keys=lambda keys: object()),
            Client=lambda signer: fake_client,
            RelayUrl=SimpleNamespace(parse=lambda relay: relay),
            PublicKey=SimpleNamespace(parse=lambda value: value),
            Filter=_FakeFilter,
            Kind=lambda value: value,
            uniffi_set_event_loop=lambda _loop: None,
        )

        with patch.object(nostr_mod, "_load_nostr_sdk", return_value=fake_sdk):
            result = await nostr_mod.send_nostr_dm_once(
                "nsec1secret",
                ["wss://relay.example"],
                "npub1recipient",
                "hello",
            )

        assert result.success is True
        assert fake_client.sent == [("npub1recipient", "hello", None)]

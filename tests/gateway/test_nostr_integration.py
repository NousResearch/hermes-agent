"""Deterministic NIP-17 transport harness tests."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms import nostr as nostr_mod


def _pubhex_for_secret(secret: str) -> str:
    if secret == "nsec-bot":
        return "bothex1234"
    if secret == "nsec-sender":
        return "senderhex5678"
    return f"hex-{secret}"


def _npub_for_hex(hex_value: str) -> str:
    return f"npub_{hex_value}"


class _FakeRelayEvent:
    def __init__(self, payload, *, sender_hex=None, rumor_data=None):
        self.payload = payload
        self.sender_hex = sender_hex
        self.rumor_data = rumor_data

    def as_json(self):
        return json.dumps(self.payload)


class _FakeFilter:
    def __init__(self):
        self.kind_value = None
        self.pubkey_values = []
        self.author_values = []
        self.limit_value = None

    def kind(self, value):
        self.kind_value = value
        return self

    def pubkey(self, value):
        self.pubkey_values = [value]
        return self

    def pubkeys(self, values):
        self.pubkey_values = list(values)
        return self

    def author(self, value):
        self.author_values = [value]
        return self

    def authors(self, values):
        self.author_values = list(values)
        return self

    def limit(self, value):
        self.limit_value = value
        return self


class _FakePublicKey:
    def __init__(self, value):
        self.value = value

    def to_hex(self):
        return self.value

    def to_bech32(self):
        return _npub_for_hex(self.value)


class _FakeKeys:
    def __init__(self, secret):
        self.secret = secret
        self._public_key = _FakePublicKey(_pubhex_for_secret(secret))

    @staticmethod
    def parse(secret):
        return _FakeKeys(secret)

    def public_key(self):
        return self._public_key


class _FakeSigner:
    def __init__(self, keys):
        self.keys = keys


class _FakeTag:
    @staticmethod
    def parse(tag):
        return list(tag)


class _FakeEventBuilder:
    def __init__(self, kind, content):
        self.kind = kind
        self.content = content
        self.tags_value = []

    def tags(self, tags):
        self.tags_value = list(tags)
        return self


class _FakeRelayNetwork:
    def __init__(self):
        self.events = []
        self._counter = 0

    def _next_counter(self):
        self._counter += 1
        return self._counter

    def publish_dm_relays(self, *, author_hex, builder):
        counter = self._next_counter()
        self.events.append(
            _FakeRelayEvent(
                {
                    "id": f"relay-list-{counter}",
                    "kind": int(builder.kind),
                    "created_at": counter,
                    "author": author_hex,
                    "pubkey": author_hex,
                    "tags": list(builder.tags_value),
                    "content": builder.content,
                }
            )
        )

    def publish_private_dm(self, *, sender_hex, recipient_hex, content):
        counter = self._next_counter()
        self.events.append(
            _FakeRelayEvent(
                {
                    "id": f"gift-{counter}",
                    "kind": nostr_mod.GIFT_WRAP_KIND,
                    "created_at": counter,
                    "pubkey": recipient_hex,
                    "tags": [["p", recipient_hex]],
                },
                sender_hex=sender_hex,
                rumor_data={
                    "id": f"rumor-{counter}",
                    "kind": nostr_mod.NIP17_DM_KIND,
                    "created_at": counter,
                    "content": content,
                },
            )
        )

    def fetch(self, flt):
        kind_value = int(flt.kind_value) if flt.kind_value is not None else None
        pubkeys = {value.to_hex() if hasattr(value, "to_hex") else str(value) for value in flt.pubkey_values}
        authors = {value.to_hex() if hasattr(value, "to_hex") else str(value) for value in flt.author_values}

        matches = []
        for event in self.events:
            payload = event.payload
            if kind_value is not None and int(payload.get("kind", -1)) != kind_value:
                continue
            if pubkeys and payload.get("pubkey") not in pubkeys:
                continue
            if authors and payload.get("author") not in authors:
                continue
            matches.append(event)

        if flt.limit_value:
            matches = matches[-flt.limit_value:]
        return list(matches)


class _FakeClient:
    def __init__(self, network, signer):
        self.network = network
        self.signer = signer
        self.public_key = signer.keys.public_key()
        self.relays = []

    async def add_relay(self, relay):
        self.relays.append(relay)

    async def add_discovery_relay(self, relay):
        self.relays.append(relay)

    async def add_write_relay(self, relay):
        self.relays.append(relay)

    async def connect(self):
        return None

    async def disconnect(self):
        return None

    async def fetch_events(self, flt, _timeout):
        return self.network.fetch(flt)

    async def send_event_builder_to(self, _relay_urls, builder):
        self.network.publish_dm_relays(
            author_hex=self.public_key.to_hex(),
            builder=builder,
        )

    async def send_private_msg(self, recipient, content, rumor_extra_tags=None):
        self.network.publish_private_dm(
            sender_hex=self.public_key.to_hex(),
            recipient_hex=recipient.to_hex() if hasattr(recipient, "to_hex") else str(recipient),
            content=content,
        )


class _FakeUnwrappedGift:
    def __init__(self, event):
        self._event = event

    def rumor(self):
        return _FakeRelayEvent(self._event.rumor_data)

    def sender(self):
        return _FakePublicKey(self._event.sender_hex)

    @staticmethod
    async def from_gift_wrap(_signer, event):
        return _FakeUnwrappedGift(event)


class _DummyTask:
    def __init__(self, coro):
        coro.close()

    def cancel(self):
        return None

    def __await__(self):
        async def _done():
            return None

        return _done().__await__()


def _build_fake_sdk(network):
    def _parse_public_key(value):
        if isinstance(value, _FakePublicKey):
            return value
        text = str(value)
        if text.startswith("npub_"):
            return _FakePublicKey(text.removeprefix("npub_"))
        return _FakePublicKey(text)

    return SimpleNamespace(
        Keys=_FakeKeys,
        NostrSigner=SimpleNamespace(keys=lambda keys: _FakeSigner(keys)),
        Client=lambda signer: _FakeClient(network, signer),
        PublicKey=SimpleNamespace(parse=_parse_public_key),
        RelayUrl=SimpleNamespace(parse=lambda relay: relay),
        Filter=_FakeFilter,
        Kind=lambda value: value,
        Tag=_FakeTag,
        EventBuilder=_FakeEventBuilder,
        UnwrappedGift=_FakeUnwrappedGift,
        uniffi_set_event_loop=lambda _loop: None,
    )


@pytest.mark.asyncio
async def test_nostr_transport_harness_round_trip_and_replay_suppression(monkeypatch):
    network = _FakeRelayNetwork()
    fake_sdk = _build_fake_sdk(network)

    monkeypatch.setattr(nostr_mod, "_load_nostr_sdk", lambda: fake_sdk)
    monkeypatch.setattr(nostr_mod.asyncio, "create_task", lambda coro: _DummyTask(coro))
    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda *args, **kwargs: None)

    adapter = nostr_mod.NostrAdapter(
        nostr_mod.PlatformConfig(
            enabled=True,
            token="nsec-bot",
            extra={"relays": ["wss://relay.test"]},
        )
    )

    async def _reply_once(event):
        await adapter.send(event.source.chat_id, f"reply:{event.text}")

    adapter.handle_message = AsyncMock(side_effect=_reply_once)

    connected = await adapter.connect()
    assert connected is True
    assert any(
        event.payload["kind"] == nostr_mod.DM_RELAY_LIST_KIND
        and event.payload["author"] == "bothex1234"
        for event in network.events
    )

    sender_client = fake_sdk.Client(fake_sdk.NostrSigner.keys(fake_sdk.Keys.parse("nsec-sender")))
    sender_builder = nostr_mod._build_dm_relay_list_builder(fake_sdk, ["wss://relay.test"])
    await sender_client.send_event_builder_to(["wss://relay.test"], sender_builder)

    send_result = await nostr_mod.send_nostr_dm_once(
        "nsec-sender",
        ["wss://relay.test"],
        _npub_for_hex("bothex1234"),
        "ping",
    )
    assert send_result.success is True

    await adapter._poll_inbox_once()
    await adapter._poll_inbox_once()

    assert adapter.handle_message.await_count == 1
    inbound_event = adapter.handle_message.await_args.args[0]
    assert inbound_event.text == "ping"
    assert inbound_event.source.chat_id == "senderhex5678"

    sender_filter = nostr_mod._filter_for_recipient_gift_wraps(
        fake_sdk,
        fake_sdk.PublicKey.parse(_npub_for_hex("senderhex5678")),
    )
    sender_events = await sender_client.fetch_events(sender_filter, nostr_mod.FETCH_TIMEOUT)
    assert len(sender_events) == 1

    unwrapped = await fake_sdk.UnwrappedGift.from_gift_wrap(None, sender_events[0])
    rumor = json.loads(unwrapped.rumor().as_json())
    assert rumor["content"] == "reply:ping"

    await adapter.disconnect()

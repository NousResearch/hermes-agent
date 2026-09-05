"""Inbound replay dedupe on the relay adapter (transplanted from the
live-cards branch for the rc.4 relay-fixes train).

Live-canary finding #3 (Alice, staging): the relay inbound leg is
at-least-once. On WS re-handshake the connector replays its durable
per-instance buffer; a long multi-tool turn straddling a quiet socket drop
got its ORIGINAL inbound replayed after the turn finished, re-running the
entire turn — the user saw the final answer posted 2-5x. Platform message
identity (chat_id + message_id/ts) is stable across replays, so a bounded
seen-set drops them. Fail-open: events without a message_id never dedupe
(dropping a real message is strictly worse than rerunning one).
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SessionSource
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from tests.gateway.relay.stub_connector import StubConnector


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="slack",
        label="Slack",
        max_message_length=39000,
        supports_draft_streaming=True,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="slack",
        len_unit="chars",
        emoji="\U0001f4ac",
        platform_hint="",
        pii_safe=False,
        supported_ops=("send", "edit", "typing"),
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


def _connected_adapter(**desc_kw):
    desc = make_desc(**desc_kw)
    stub = StubConnector(desc)
    adapter = RelayAdapter(PlatformConfig(), desc, transport=stub)
    return adapter, stub


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


def _record(bucket, event):
    async def _coro():
        bucket.append(event)
    return _coro()


async def _false_coro():
    return False


async def _none_coro():
    return None


class TestInboundReplayDedupe:
    """Finding #3 (live canary): connector replay of the original inbound
    after a WS re-handshake must not re-run the turn."""

    def _event(self, message_id="1700.100", chat_id="C1", text="hi"):
        # A REAL MessageEvent, shaped exactly as _event_from_wire produces it:
        # chat identity lives on event.source, NOT as a top-level attribute.
        # (The first version of these tests used a SimpleNamespace with a
        # top-level chat_id — a shape no production code path produces — and
        # green-lit a dedupe key that read the wrong field.)
        source = SessionSource(
            platform=Platform.SLACK,
            chat_id=chat_id,
            chat_type="channel",
            user_id="U1",
            message_id=message_id,
        )
        return MessageEvent(text=text, source=source, message_id=message_id)

    def _tap(self, adapter, handled):
        adapter.handle_message = lambda e: _record(handled, e)
        adapter._consume_prompt_response = lambda e: _false_coro()
        adapter._localize_inbound_media = lambda e: _none_coro()

    def test_replayed_inbound_dropped(self, loop):
        adapter, _ = _connected_adapter()
        handled = []
        self._tap(adapter, handled)
        e = self._event()
        loop.run_until_complete(adapter._on_inbound(e))
        loop.run_until_complete(adapter._on_inbound(e))  # replay
        assert len(handled) == 1

    def test_distinct_messages_both_handled(self, loop):
        adapter, _ = _connected_adapter()
        handled = []
        self._tap(adapter, handled)
        loop.run_until_complete(adapter._on_inbound(self._event("1700.100")))
        loop.run_until_complete(adapter._on_inbound(self._event("1700.200")))
        assert len(handled) == 2

    def test_missing_message_id_fails_open(self, loop):
        adapter, _ = _connected_adapter()
        handled = []
        self._tap(adapter, handled)
        e = self._event(message_id=None)
        loop.run_until_complete(adapter._on_inbound(e))
        loop.run_until_complete(adapter._on_inbound(e))
        assert len(handled) == 2  # never dedupe without identity

    def test_seen_set_bounded(self, loop):
        adapter, _ = _connected_adapter()
        adapter.handle_message = lambda e: _none_coro()
        adapter._consume_prompt_response = lambda e: _false_coro()
        adapter._localize_inbound_media = lambda e: _none_coro()
        for i in range(600):
            loop.run_until_complete(adapter._on_inbound(self._event(f"ts.{i}")))
        assert len(adapter._seen_inbound) <= adapter._SEEN_INBOUND_MAX


class TestWireLevelReplayDedupe:
    """The full production inbound path: a connector wire frame decoded by
    _event_from_wire, then dispatched through RelayAdapter._on_inbound.

    This is the layer the hand-built-event tests above cannot vouch for: the
    dedupe key must work on the exact object shape the wire decoder emits.
    The original dedupe commit shipped green on hand-built events while being
    a no-op on decoded ones — this class exists so that can't recur.
    """

    WIRE = {
        "text": "hi",
        "message_type": "text",
        "message_id": "1700.100",
        "source": {
            "platform": "slack",
            "chat_id": "C1",
            "chat_type": "channel",
            "user_id": "U1",
            "message_id": "1700.100",
        },
    }

    def _tap(self, adapter, handled):
        adapter.handle_message = lambda e: _record(handled, e)
        adapter._consume_prompt_response = lambda e: _false_coro()
        adapter._localize_inbound_media = lambda e: _none_coro()

    def _decode(self, **overrides):
        from gateway.relay.ws_transport import _event_from_wire

        raw = {**self.WIRE, **overrides}
        if "source" in overrides:
            raw["source"] = {**self.WIRE["source"], **overrides["source"]}
        return _event_from_wire(raw)

    def test_decoded_event_yields_a_dedupe_key(self):
        adapter, _ = _connected_adapter()
        key = adapter._inbound_dedupe_key(self._decode())
        assert key is not None, (
            "the wire decoder's event shape must produce a dedupe key — "
            "None here means the dedupe is fail-open for ALL production "
            "traffic (the original ship-broken state)"
        )

    def test_replayed_wire_frame_dropped(self, loop):
        adapter, _ = _connected_adapter()
        handled = []
        self._tap(adapter, handled)
        loop.run_until_complete(adapter._on_inbound(self._decode()))
        # The connector re-delivers the SAME frame on re-handshake; the
        # decoder builds a fresh object each time, so identity must come
        # from the key, not object identity.
        loop.run_until_complete(adapter._on_inbound(self._decode()))
        assert len(handled) == 1

    def test_same_ids_on_different_platforms_not_conflated(self, loop):
        # Phase 1.5 multiplex: one adapter fronts several platforms. Numeric
        # chat/message ids can collide across platforms; both must dispatch.
        adapter, _ = _connected_adapter()
        handled = []
        self._tap(adapter, handled)
        loop.run_until_complete(adapter._on_inbound(self._decode()))
        loop.run_until_complete(
            adapter._on_inbound(self._decode(source={"platform": "discord"}))
        )
        assert len(handled) == 2


class TestDedupeKeyPlatformNormalization:
    """The platform component of the key must be spelling-invariant: a
    Platform enum and its plain-string form are ONE platform (one key), and
    two different string platforms must never collapse into a shared empty
    component. Production wire decoding always yields the enum; alternate
    event constructors may carry the string."""

    def _key(self, platform):
        adapter, _ = _connected_adapter()
        source = SessionSource(
            platform=platform, chat_id="C1", chat_type="channel", message_id="m1"
        )
        event = MessageEvent(text="hi", source=source, message_id="m1")
        return adapter._inbound_dedupe_key(event)

    def test_enum_and_string_spellings_produce_one_key(self):
        assert self._key(Platform.SLACK) == self._key("slack")

    def test_distinct_string_platforms_stay_distinct(self):
        assert self._key("slack") != self._key("discord")

    def test_missing_platform_still_yields_a_key(self):
        # Fail-open on identity is reserved for missing message/chat ids;
        # a missing platform alone must not disable dedupe.
        key = self._key(None)
        assert key is not None
        assert key.startswith(":")

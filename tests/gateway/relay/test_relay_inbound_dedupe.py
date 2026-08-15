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
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
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
        return SimpleNamespace(
            message_id=message_id, chat_id=chat_id, text=text, media=None
        )

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

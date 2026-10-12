"""Regression tests for Buzz forum-kind dispatch (#90309).

The inbound path hardcoded Nostr kind 9 (chat) at both the WebSocket
subscription filter and the dispatch gate, so forum channels (kind 45001
thread roots and 45003 comment replies) were silently never dispatched.
The DM-reclassification check deliberately stays kind-9-only: widening it
would let a p-tagged forum post be reclassified as a DM and bypass
mention gating.
"""

import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.buzz.adapter import (
    _CHAT_KIND,
    _DISPATCH_KINDS,
    BuzzAdapter,
)

CHANNEL = "7c83e8f7bb1d4db2bb4074d5c14f2a7f6a9e1c21d3b5a90f8e7d6c5b4a392801"
SELF_PUBKEY = "aa" * 32
OTHER_PUBKEY = "bb" * 32


def _dummy_nsec() -> str:
    # Test-only placeholder, assembled to avoid a credential-shaped literal.
    return "-".join(("nsec", "1", "test"))


def _event(event_id, content="hey @Chip", kind=9, p_tag_self=False):
    tags = [["h", CHANNEL]]
    if p_tag_self:
        tags.append(["p", SELF_PUBKEY])
    return {
        "id": event_id,
        "pubkey": OTHER_PUBKEY,
        "content": content,
        "created_at": 100,
        "kind": kind,
        "tags": tags,
    }


def _make_group_adapter():
    adapter = BuzzAdapter(
        PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay"})
    )
    adapter._self_pubkey = SELF_PUBKEY
    adapter._self_npub = "npub1" + "a" * 50
    adapter._display_name = "Chip"
    adapter._private_key = _dummy_nsec()
    adapter._dispatched = []

    async def capture(**kwargs):
        adapter._dispatched.append(kwargs)

    adapter._dispatch_message = capture
    adapter._message_handler = AsyncMock()
    adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
    return adapter


class _ScriptedCli:
    def __init__(self, events):
        import copy

        self._events = copy.deepcopy(events)

    async def __call__(self, args, *, input_text=None):
        return 0, json.dumps(self._events), ""


class TestForumKindDispatch:
    @pytest.mark.asyncio
    async def test_forum_post_root_dispatched(self):
        adapter = _make_group_adapter()
        adapter._run_cli = _ScriptedCli([_event("f1", kind=45001)])
        await adapter._poll_channel(CHANNEL)
        assert [d["message_id"] for d in adapter._dispatched] == ["f1"]

    @pytest.mark.asyncio
    async def test_forum_comment_reply_dispatched(self):
        adapter = _make_group_adapter()
        adapter._run_cli = _ScriptedCli([_event("f2", kind=45003)])
        await adapter._poll_channel(CHANNEL)
        assert [d["message_id"] for d in adapter._dispatched] == ["f2"]

    @pytest.mark.asyncio
    async def test_undocumented_stream_kind_still_ignored(self):
        """Stream kinds stay out of scope until their dispatch semantics are
        confirmed — the dispatch set must not widen beyond chat + forum."""
        adapter = _make_group_adapter()
        adapter._run_cli = _ScriptedCli(
            [_event("s1", kind=46010), _event("s2", kind=40007)]
        )
        await adapter._poll_channel(CHANNEL)
        assert adapter._dispatched == []


class TestSubscriptionFilter:
    @pytest.mark.asyncio
    async def test_websocket_subscription_includes_forum_kinds(self):
        """The REQ filter must subscribe to every dispatchable kind, or the
        live path never even receives forum events to drop."""
        adapter = _make_group_adapter()
        sent = []

        class _FakeWS:
            async def send(self, payload):
                sent.append(json.loads(payload))

        adapter._channel_state[CHANNEL]["last_ts"] = 100
        await adapter._send_channel_subscription(_FakeWS(), "sub-1", CHANNEL)
        assert sent, "subscription request must be sent"
        assert sent[0][2]["kinds"] == sorted(_DISPATCH_KINDS)
        assert 45001 in sent[0][2]["kinds"] and 45003 in sent[0][2]["kinds"]


class TestDmReclassificationStaysChatOnly:
    def test_ptagged_forum_post_is_not_a_dm(self):
        """Security pin: a p-tagged kind-45001 post must NOT be reclassified
        as a DM — that would bypass mention gating for forum content
        (#90309 caveat). The DM check keeps the kind-9-only comparison."""
        adapter = _make_group_adapter()
        adapter._may_reclassify_as_dm = lambda channel_id: True
        assert adapter._is_direct_message_event(
            CHANNEL, _event("f3", kind=45001, p_tag_self=True)
        ) is False

    def test_ptagged_chat_message_can_still_be_a_dm(self):
        adapter = _make_group_adapter()
        adapter._may_reclassify_as_dm = lambda channel_id: True
        event = _event("d1", kind=_CHAT_KIND, p_tag_self=True, content="no mention")
        assert adapter._is_direct_message_event(CHANNEL, event) is True

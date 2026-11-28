"""NIP-25 reactions must reach the agent when they answer the agent.

A reaction is how a human resolves a question with one tap — an approval gate, a "yes, continue",
a thumbs-up on a proposed plan. The adapter subscribed only to chat and forum kinds, so a reaction
landed on the relay and never existed as far as the agent was concerned.

The scope is deliberately narrow: a reaction is dispatched **only** when it targets a message this
agent wrote. Reactions between humans are social noise, and a reaction to a message we never cached
cannot be quoted back to the model, so both stay ignored.
"""

import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.buzz.adapter import (
    _DISPATCH_KINDS,
    _REACTION_KIND,
    BuzzAdapter,
    _reaction_target,
    _reaction_text,
)

CHANNEL = "7c83e8f7bb1d4db2bb4074d5c14f2a7f6a9e1c21d3b5a90f8e7d6c5b4a392801"
SELF_PUBKEY = "aa" * 32
HUMAN_PUBKEY = "bb" * 32
STRANGER_PUBKEY = "cc" * 32
OUR_MESSAGE = "d1" * 32
THEIR_MESSAGE = "e2" * 32


def _dummy_nsec() -> str:
    return "-".join(("nsec", "1", "test"))


def _reaction(event_id, target, emoji="👍", pubkey=HUMAN_PUBKEY):
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": emoji,
        "created_at": 200,
        "kind": _REACTION_KIND,
        "tags": [["h", CHANNEL], ["e", target], ["p", SELF_PUBKEY]],
    }


def _make_adapter(allowed=None):
    adapter = BuzzAdapter(PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay"}))
    adapter._self_pubkey = SELF_PUBKEY
    adapter._self_npub = "npub1" + "a" * 50
    adapter._display_name = "Chip"
    adapter._private_key = _dummy_nsec()
    adapter._allowed_pubkeys = set(allowed or [])
    adapter._dispatched = []

    async def capture(**kwargs):
        adapter._dispatched.append(kwargs)

    adapter._dispatch_message = capture
    adapter._message_handler = AsyncMock()
    adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}, "event_meta": {}}
    # Both messages are known to the adapter; only one of them is ours.
    adapter._remember_event_meta(CHANNEL, OUR_MESSAGE, SELF_PUBKEY, "Approve the brief?")
    adapter._remember_event_meta(CHANNEL, THEIR_MESSAGE, HUMAN_PUBKEY, "lunch?")
    return adapter


class _ScriptedCli:
    def __init__(self, events):
        import copy

        self._events = copy.deepcopy(events)

    async def __call__(self, args, *, input_text=None):
        return 0, json.dumps(self._events), ""


class TestReactionDispatch:
    @pytest.mark.asyncio
    async def test_reaction_on_our_message_is_dispatched_with_context(self):
        adapter = _make_adapter()
        adapter._run_cli = _ScriptedCli([_reaction("r1", OUR_MESSAGE)])
        await adapter._poll_channel(CHANNEL)
        assert len(adapter._dispatched) == 1
        sent = adapter._dispatched[0]
        assert "👍" in sent["text"]
        assert "Approve the brief?" in sent["text"]
        assert sent["reply_to_message_id"] == OUR_MESSAGE
        assert sent["reply_to_is_own_message"] is True

    @pytest.mark.asyncio
    async def test_reaction_on_someone_elses_message_is_ignored(self):
        adapter = _make_adapter()
        adapter._run_cli = _ScriptedCli([_reaction("r2", THEIR_MESSAGE)])
        await adapter._poll_channel(CHANNEL)
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_reaction_on_an_uncached_message_is_ignored(self):
        adapter = _make_adapter()
        adapter._run_cli = _ScriptedCli([_reaction("r3", "ff" * 32)])
        await adapter._poll_channel(CHANNEL)
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_reaction_from_an_unlisted_pubkey_is_ignored(self):
        adapter = _make_adapter(allowed=[HUMAN_PUBKEY])
        adapter._run_cli = _ScriptedCli([_reaction("r4", OUR_MESSAGE, pubkey=STRANGER_PUBKEY)])
        await adapter._poll_channel(CHANNEL)
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_a_plain_like_becomes_a_readable_mark(self):
        adapter = _make_adapter()
        adapter._run_cli = _ScriptedCli([_reaction("r5", OUR_MESSAGE, emoji="+")])
        await adapter._poll_channel(CHANNEL)
        assert "👍" in adapter._dispatched[0]["text"]


class TestSubscriptionAndHelpers:
    @pytest.mark.asyncio
    async def test_websocket_subscription_includes_reactions(self):
        adapter = _make_adapter()
        sent = []

        class _FakeWS:
            async def send(self, payload):
                sent.append(json.loads(payload))

        await adapter._send_channel_subscription(_FakeWS(), "sub-1", CHANNEL)
        assert _REACTION_KIND in sent[0][2]["kinds"]
        assert sent[0][2]["kinds"] == sorted(_DISPATCH_KINDS)

    def test_target_is_the_last_e_tag(self):
        event = {"tags": [["e", "a" * 64], ["e", "b" * 64]]}
        assert _reaction_target(event) == "b" * 64
        assert _reaction_target({"tags": []}) is None

    def test_reaction_text_quotes_the_answered_message(self):
        assert _reaction_text("👎", "ship it?") == '[reaction 👎 to "ship it?"]'
        assert _reaction_text("+", "") == "[reaction 👍]"

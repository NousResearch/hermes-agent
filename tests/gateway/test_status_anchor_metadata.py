"""The generic ``STATUS_ANCHOR_MESSAGE_ID`` metadata key is inert for adapters that do not read it.

Every platform's reply metadata now carries the id of the inbound message the turn answers (the
Slack adapter anchors a flat turn's status indicator on it). Routing never reads the key, so a
non-Slack adapter must route, send and type exactly as it does without it.
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import _thread_metadata_for_source
from gateway.platforms.base_status_anchor import STATUS_ANCHOR_MESSAGE_ID
from tests.gateway.test_telegram_thread_fallback import _inject_fake_telegram, _make_adapter

ANCHOR = "4242"


def _source(**overrides):
    return SimpleNamespace(**{
        "platform": Platform.TELEGRAM, "chat_id": "775566675", "chat_type": "dm", "thread_id": None,
        "message_id": ANCHOR, "profile": None, **overrides})


def _without_anchor(metadata):
    return {k: v for k, v in (metadata or {}).items() if k != STATUS_ANCHOR_MESSAGE_ID} or None


@pytest.mark.parametrize("overrides", [
    pytest.param({}, id="telegram-plain-dm"),
    pytest.param({"thread_id": "270453"}, id="telegram-dm-topic"),
    pytest.param({"platform": Platform.DISCORD, "chat_type": "group"}, id="discord-channel"),
])
def test_anchor_adds_no_routing_keys(overrides):
    """A plain Telegram DM still gets no DM-topic fallback; a topic lane keeps exactly its keys."""
    anchored = _thread_metadata_for_source(_source(**overrides), ANCHOR)
    assert anchored[STATUS_ANCHOR_MESSAGE_ID] == ANCHOR
    assert _without_anchor(anchored) == _thread_metadata_for_source(_source(**overrides), None)


def _telegram_adapter(calls):
    adapter = _make_adapter()
    adapter._telegram_typing_cooldown_until = {}

    async def send_message(**kwargs):
        calls.append(("send_message", kwargs))
        return SimpleNamespace(message_id=1)

    async def send_chat_action(**kwargs):
        calls.append(("send_chat_action", kwargs))

    adapter._bot = SimpleNamespace(send_message=send_message, send_chat_action=send_chat_action)
    return adapter


@pytest.mark.parametrize("routing", [
    pytest.param({}, id="plain-dm"),
    pytest.param({"thread_id": "270453"}, id="topic"),
])
@pytest.mark.asyncio
async def test_telegram_send_and_typing_ignore_the_anchor(routing):
    plain, anchored = [], []
    for calls, metadata in ((plain, routing or None), (anchored, {**routing, STATUS_ANCHOR_MESSAGE_ID: ANCHOR})):
        adapter = _telegram_adapter(calls)
        await adapter.send_typing("775566675", metadata=metadata)
        assert (await adapter.send("775566675", "done", metadata=metadata)).success
        await asyncio.gather(*adapter.__dict__.get("_telegram_typing_retrigger_tasks", {}).values())
    assert anchored == plain and plain

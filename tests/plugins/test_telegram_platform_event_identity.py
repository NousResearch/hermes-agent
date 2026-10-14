"""Native Telegram facts for deterministic approval/reply consumers.

Exercise real PTB decoding and the existing post-auth observer, without a
Telegram connection, another listener, model calls or production state.
"""

import asyncio
import json
from contextvars import ContextVar
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("telegram")
from telegram import Update

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter():
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = SimpleNamespace(extra={"allow_from": ["*"]})
    adapter.gateway_runner = None
    return adapter


def _reaction(*, anonymous=False, bot=False):
    reaction = {
        "chat": {"id": -10042, "type": "supergroup", "title": "Fixture"},
        "message_id": 7,
        "date": 1791955140,
        "old_reaction": [{"type": "emoji", "emoji": "👍"}],
        "new_reaction": [],
    }
    if anonymous:
        reaction["actor_chat"] = {"id": -10099, "type": "channel", "title": "Actor"}
    else:
        reaction["user"] = {"id": 67890, "is_bot": bot, "first_name": "Fixture"}
    return Update.de_json({"update_id": 103, "message_reaction": reaction}, None)


def _message(*, edited=False, forwarded=False, reply=True):
    chat = {"id": -10042, "type": "supergroup", "title": "Fixture"}
    message = {
        "message_id": 8,
        "date": 1791955140,
        "chat": chat,
        "from": {"id": 67890, "is_bot": False, "first_name": "Fixture"},
        "text": "post",
        "is_topic_message": True,
        "message_thread_id": 3,
        "reply_to_message": {
            "message_id": 7,
            "date": 1791954900,
            "chat": chat,
            "from": {"id": 555, "is_bot": True, "first_name": "Fixture bot"},
            "text": "Approval request",
        },
    }
    if not reply:
        del message["reply_to_message"]
    if forwarded:
        message["forward_origin"] = {
            "type": "hidden_user",
            "date": 1791954900,
            "sender_user_name": "Original",
        }
    if edited:
        message["edit_date"] = 1791955200
    return Update.de_json(
        {"update_id": 104, "edited_message" if edited else "message": message}, None
    )


@pytest.mark.parametrize(
    "anonymous,bot", [(False, False), (False, True), (True, False)]
)
def test_reaction_preserves_order_time_and_explicit_actor_facts(anonymous, bot):
    payload = _adapter()._normalize_platform_event(
        _reaction(anonymous=anonymous, bot=bot)
    )["payload"]
    assert payload["update_id"] == "103"
    assert payload["occurred_at"] == "2026-10-14T05:19:00+00:00"
    assert payload["actor_id"] == (None if anonymous else "67890")
    assert payload["actor_is_bot"] == (None if anonymous else bot)
    assert payload["actor_chat_id"] == ("-10099" if anonymous else None)
    assert payload["emojis"] == []  # A withdrawal must retain its identity.
    assert payload["chat_id"] == "-10042" and payload["message_id"] == "7"
    json.dumps(payload)


@pytest.mark.parametrize("forwarded", [False, True])
def test_new_reply_preserves_literal_text_target_and_forward_facts(forwarded):
    event = _adapter()._normalize_platform_event(_message(forwarded=forwarded))
    assert event["event_type"] == "message_reply"
    payload = event["payload"]
    assert payload["actor_id"] == "67890" and payload["actor_is_bot"] is False
    assert payload["actor_chat_id"] is None
    assert (
        payload["update_id"] == "104"
        and payload["occurred_at"] == "2026-10-14T05:19:00+00:00"
    )
    assert payload["reply_to_message_id"] == "7" and payload["message_id"] == "8"
    assert payload["thread_id"] == "3" and payload["text"] == "post"
    assert payload["forwarded"] is forwarded
    json.dumps(event)
    assert (
        _adapter()._normalize_platform_event(_message(edited=True))["event_type"]
        == "message_edited"
    )


def test_missing_metadata_is_null_and_new_message_fields_are_bounded():
    message = SimpleNamespace(
        chat=SimpleNamespace(id=-10042),
        message_id=8,
        from_user=SimpleNamespace(id=object(), is_bot=object()),
        date=datetime(2026, 10, 14),
        text="x" * 8192,
        reply_to_message=SimpleNamespace(message_id="9" * 256),
    )
    update = SimpleNamespace(message=message, update_id=object())
    payload = _adapter()._normalize_platform_event(update)["payload"]
    for field in (
        "update_id",
        "actor_id",
        "actor_is_bot",
        "actor_chat_id",
        "occurred_at",
    ):
        assert payload[field] is None
    assert len(payload["text"]) == 4096
    assert len(payload["reply_to_message_id"]) == 128
    assert payload["forwarded"] is False and payload["thread_id"] is None
    json.dumps(payload)


@pytest.mark.parametrize("authorized", [False, True])
def test_new_reply_reaches_existing_plugin_bus_only_after_native_authorization(
    authorized,
):
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="approval-fixture", source="user"), manager
    )
    observed = []
    context.register_hook(
        "gateway_platform_event", lambda **event: observed.append(event)
    )
    runner = object.__new__(GatewayRunner)
    sources = []

    def authorize(source):
        sources.append(source)
        return authorized

    runner._is_user_authorized = authorize
    adapter = _adapter()
    adapter.set_platform_event_handler(runner._handle_gateway_platform_event)
    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        asyncio.run(adapter._on_platform_update(_message(), context=None))
    assert sources[0].user_id == "67890" and sources[0].chat_id == "-10042"
    if authorized:
        assert observed[0]["event_type"] == "message_reply"
        assert observed[0]["payload"]["reply_to_message_id"] == "7"
    else:
        assert observed == []


def test_new_message_without_reply_is_not_an_event():
    assert _adapter()._normalize_platform_event(_message(reply=False)) is None


@pytest.mark.parametrize("handler_fails", [False, True])
def test_reply_observer_leaves_core_message_admission_alone(handler_fails):
    """Core handlers own a new message's handoff; only reactions and edits are settled here."""
    claims = ContextVar("claim", default=None)

    async def handler(event, source):
        if handler_fails:
            raise RuntimeError("observer failure")

    def observe(update):
        claim = SimpleNamespace(accepted=False, failed=False)
        claims.set(claim)
        adapter = _adapter()
        adapter._update_admission = claims
        adapter.set_platform_event_handler(handler)
        with patch("hermes_cli.lifecycle.has_hook", return_value=True):
            asyncio.run(adapter._on_platform_update(update, context=None))
        return claim

    reply = observe(_message())
    assert (reply.accepted, reply.failed) == (False, False)
    reaction = observe(_reaction())
    assert (reaction.accepted, reaction.failed) == (True, handler_fails)

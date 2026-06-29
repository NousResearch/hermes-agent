"""Tests for Feishu bot-to-bot @mention collaboration helpers.

We never inject (or repair) a mention, so the model's @ — or its deliberate
absence — decides whether the exchange continues. These cover the bot-sender
identity notice injected into the inbound body (so the agent knows it can
@mention a peer bot to reach it) and the markdown → post path that lifts a
model-written leading ``<at>`` into a structured element so the peer is actually
notified.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from gateway.platforms.base import MessageType, SendResult
from plugins.platforms.feishu.adapter import (
    _build_bot_sender_notice,
    _build_markdown_post_payload,
)
from tests.gateway.feishu_helpers import make_adapter_skeleton, make_message, make_sender


def _adapter():
    adapter = make_adapter_skeleton()
    adapter.config = SimpleNamespace(extra={})  # channel_prompts resolver reads config.extra
    return adapter


class TestBuildBotSenderNotice:
    def test_states_identity_and_continue_then_stop_guidance(self):
        notice = _build_bot_sender_notice(name="麦香鱼🦞", open_id="ou-bot-sender")
        # Identity + the delivery mechanism (must @ to reach a bot).
        assert '"麦香鱼🦞" (open_id: ou-bot-sender) is a bot' in notice
        assert "@mention it" in notice
        # Permission to STOP (the loop-prevention lever): reply without the @.
        assert "without the @mention" in notice
        # Explicit anti-loop guidance: don't reply just to be polite.
        assert "Don't reply out of politeness" in notice

    def test_drops_the_hard_must_mention_command(self):
        # The old "You MUST @mention it" command fought strategy C (model decides
        # to stop by *omitting* the @). It must be gone.
        notice = _build_bot_sender_notice(name="麦香鱼🦞", open_id="ou-bot-sender")
        assert "You MUST" not in notice

    def test_does_not_embed_a_copyable_at_tag_template(self):
        # Regression: an inline "<at ...>...</at>" example in the notice taught
        # the model to wrap its entire reply *inside* the tag (body where the
        # ellipsis was), yielding a malformed mention that renders as literal
        # text and never reaches the peer. The @ syntax belongs to the channel
        # tutorial, not here — the notice must carry no copyable <at> template.
        notice = _build_bot_sender_notice(name="麦香鱼🦞", open_id="ou-bot-sender")
        assert "<at" not in notice

    def test_falls_back_to_generic_label_when_name_empty(self):
        notice = _build_bot_sender_notice(name="", open_id="ou-bot-sender")
        assert notice.startswith('[System: "the sender" (open_id: ou-bot-sender) is a bot')


class TestInboundAugmentation:
    def test_group_bot_message_gets_notice(self):
        adapter = _adapter()
        text = adapter._augment_inbound_for_bot_peer(
            text="在吗",
            is_group=True,
            is_bot=True,
            open_id="ou_peer",
            name="龙虾二号",
        )
        assert text.startswith('[System: "龙虾二号" (open_id: ou_peer) is a bot')
        assert text.endswith("在吗")

    def test_group_human_message_gets_no_notice(self):
        adapter = _adapter()
        text = adapter._augment_inbound_for_bot_peer(
            text="hi",
            is_group=True,
            is_bot=False,
            open_id="ou_human",
            name="小明",
        )
        assert text == "hi"  # never annotate human senders (#71396 red line)

    def test_dm_bot_message_gets_no_notice(self):
        adapter = _adapter()
        text = adapter._augment_inbound_for_bot_peer(
            text="hi",
            is_group=False,
            is_bot=True,
            open_id="ou_peer",
            name="龙虾二号",
        )
        assert text == "hi"


class TestPostPayloadAtElement:
    def test_leading_at_becomes_structured_post_element(self):
        payload = _build_markdown_post_payload('<at user_id="ou_peer">龙虾二号</at> 收到 **重点**')
        rows = json.loads(payload)["zh_cn"]["content"]
        # First element is a real Feishu post `at` element, so the peer is
        # actually notified (an inline <at> string in an md element is not).
        assert rows[0][0] == {"tag": "at", "user_id": "ou_peer"}
        md_text = "".join(el.get("text", "") for el in rows[0] if el["tag"] == "md")
        assert "<at" not in md_text
        assert "收到" in md_text

    def test_post_payload_without_at_is_unchanged(self):
        payload = _build_markdown_post_payload("收到 **重点**")
        rows = json.loads(payload)["zh_cn"]["content"]
        assert rows == [[{"tag": "md", "text": "收到 **重点**"}]]

    def test_post_at_conversion_is_logged_with_user_id(self, caplog):
        with caplog.at_level(logging.INFO, logger="plugins.platforms.feishu.adapter"):
            _build_markdown_post_payload('<at user_id="ou_peer">龙虾二号</at> 收到 **重点**')
        assert any(
            "structured post" in r.message and "ou_peer" in r.message for r in caplog.records
        )

    def test_post_without_at_logs_no_conversion(self, caplog):
        with caplog.at_level(logging.INFO, logger="plugins.platforms.feishu.adapter"):
            _build_markdown_post_payload("收到 **重点**")
        assert not any("structured post" in r.message for r in caplog.records)


class TestProcessInboundWiring:
    @pytest.mark.asyncio
    async def test_group_bot_message_injects_notice(self):
        adapter = _adapter()

        async def _extract(message):
            return ("在吗", MessageType.TEXT, [], [], [])

        async def _chat_info(chat_id):
            return {"name": "群"}

        async def _resolve_profile(sender_id, *, is_bot):
            return {"user_id": "ou_peer", "user_name": "龙虾二号", "user_id_alt": None}

        captured = {}

        async def _dispatch(event):
            captured["event"] = event

        adapter._extract_message_content = _extract
        adapter.get_chat_info = _chat_info
        adapter._resolve_sender_profile = _resolve_profile
        adapter._resolve_source_chat_type = lambda **kwargs: "group"
        adapter.build_source = lambda **kwargs: SimpleNamespace(**kwargs)
        adapter._dispatch_inbound_event = _dispatch

        sender = make_sender(sender_type="bot", open_id="ou_peer")
        message = make_message(message_id="om_g", chat_type="group", chat_id="oc_g")

        await adapter._process_inbound_message(
            data=SimpleNamespace(),
            message=message,
            sender_id=sender.sender_id,
            chat_type="group",
            message_id="om_g",
            is_bot=True,
        )

        event = captured["event"]
        assert event.text.startswith('[System: "龙虾二号" (open_id: ou_peer) is a bot')
        assert event.text.endswith("在吗")


class TestSendWiring:
    @pytest.mark.asyncio
    async def test_send_markdown_reply_mentions_peer_via_structured_at(self):
        adapter = _adapter()
        adapter._client = object()
        captured = {}

        async def fake_retry(*, chat_id, msg_type, payload, reply_to, metadata):
            captured["msg_type"] = msg_type
            captured["payload"] = payload
            return SimpleNamespace()

        adapter._feishu_send_with_retry = fake_retry
        adapter.truncate_message = lambda text, limit: [text]
        adapter._finalize_send_result = lambda response, message: SendResult(success=True)

        # Model wrote the leading @ itself; markdown → post path must lift it
        # into a structured `at` element so the peer is actually notified.
        await adapter.send("oc_g", '<at user_id="ou_peer">龙虾二号</at> 收到 **重点**')

        assert captured["msg_type"] == "post"
        rows = json.loads(captured["payload"])["zh_cn"]["content"]
        assert rows[0][0] == {"tag": "at", "user_id": "ou_peer"}

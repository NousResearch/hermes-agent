"""QQ group replies: mention gate, always mode, and pending context.

GROUP_AT_MESSAGE_CREATE is an @. GROUP_MESSAGE_CREATE replies only in always
mode, or when the payload shows this bot was mentioned. Other lines wait in a
per-group buffer and ride channel_context on the next @.
"""

from unittest import mock

import pytest

from gateway.config import PlatformConfig


def _adapter(**extra):
    from plugins.platforms.qqbot.adapter import QQAdapter

    cfg = {"app_id": "1903885637", "client_secret": "b", "group_policy": "open"}
    cfg.update(extra)
    adapter = QQAdapter(PlatformConfig(enabled=True, extra=cfg))
    adapter.handle_message = mock.AsyncMock()
    return adapter


def _group(content="hello", *, group="g1", member="m1", mentions=None, msg_id="m"):
    payload = {
        "id": msg_id,
        "group_openid": group,
        "content": content,
        "author": {"member_openid": member},
    }
    if mentions is not None:
        payload["mentions"] = mentions
    return payload


class TestMentionDetection:
    def test_at_event_is_a_mention_and_another_members_tag_is_not(self):
        from plugins.platforms.qqbot.group_activation import detect_mentioned

        assert detect_mentioned("GROUP_AT_MESSAGE_CREATE", {}, "hi", "1") is True
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {}, "<@!999> hi", "1903885637") is False
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {}, "<@!1903885637> hi", "1903885637") is True
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {"mentions": [{"is_you": "false"}]}, "hi", "1") is False
        assert detect_mentioned(
            "GROUP_MESSAGE_CREATE", {"mentions": [{"is_you": True}]}, "hi", "1") is True

    def test_runtime_override_beats_per_group_and_global(self):
        from plugins.platforms.qqbot.group_activation import resolve_require_mention

        assert resolve_require_mention("g", global_default=True, per_group={"g": False}) is False
        assert resolve_require_mention(
            "g", global_default=True, per_group={"g": False},
            runtime_overrides={"g": True}) is True


class TestGroupReplyGate:
    @pytest.mark.asyncio
    async def test_mention_mode_buffers_a_plain_message_and_flushes_it_on_at(self):
        adapter = _adapter()
        await adapter._on_message("GROUP_MESSAGE_CREATE", _group("earlier line", msg_id="1"))
        adapter.handle_message.assert_not_awaited()

        await adapter._on_message(
            "GROUP_AT_MESSAGE_CREATE", _group("@bot what?", msg_id="2"))
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "what?"
        assert "earlier line" in event.channel_context
        assert "m1:" in event.channel_context
        assert event.source.chat_id == "g1"
        assert event.source.user_id == "m1"

    @pytest.mark.asyncio
    async def test_always_mode_replies_without_an_at(self):
        adapter = _adapter(group_require_mention=False)
        await adapter._on_message("GROUP_MESSAGE_CREATE", _group("anyone there", msg_id="1"))
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "anyone there"
        assert event.channel_context is None

    @pytest.mark.asyncio
    async def test_per_group_always_overrides_the_global_mention_gate(self):
        adapter = _adapter(groups={"g1": {"require_mention": False}})
        await adapter._on_message("GROUP_MESSAGE_CREATE", _group("in this group", msg_id="1"))
        assert adapter.handle_message.await_args.args[0].text == "in this group"

    @pytest.mark.asyncio
    async def test_runtime_override_restores_mention_mode(self):
        adapter = _adapter(group_require_mention=False)
        adapter._set_group_mode_override("g1", True)
        await adapter._on_message("GROUP_MESSAGE_CREATE", _group("quiet", msg_id="1"))
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_blocked_group_does_not_buffer(self):
        adapter = _adapter(group_policy="disabled")
        await adapter._on_message("GROUP_MESSAGE_CREATE", _group("nope", msg_id="1"))
        await adapter._on_message("GROUP_AT_MESSAGE_CREATE", _group("@bot hi", msg_id="2"))
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bare_at_still_delivers_pending_context(self):
        adapter = _adapter()
        await adapter._handle_group_message(
            _group("before"), "1", "before", {"member_openid": "m1"}, "",
            event_type="GROUP_MESSAGE_CREATE")
        await adapter._handle_group_message(
            _group("<@!1903885637>"), "2", "<@!1903885637>", {"member_openid": "m1"}, "",
            event_type="GROUP_MESSAGE_CREATE")
        event = adapter.handle_message.await_args.args[0]
        assert event.text == ""
        assert "before" in event.channel_context

    @pytest.mark.asyncio
    async def test_zero_history_limit_drops_non_mentions(self):
        adapter = _adapter(group_history_limit=0)
        await adapter._on_message("GROUP_MESSAGE_CREATE", _group("gone", msg_id="1"))
        await adapter._on_message("GROUP_AT_MESSAGE_CREATE", _group("@bot now", msg_id="2"))
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "now"
        assert event.channel_context is None

    def test_mention_tag_is_stripped(self):
        from plugins.platforms.qqbot.adapter import QQAdapter

        assert QQAdapter._strip_at_mention("<@!1903885637> hello") == "hello"
        assert QQAdapter._strip_at_mention("@BotUser hello there") == "hello there"

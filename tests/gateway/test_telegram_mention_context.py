"""Multi-bot addressing must survive routing into the agent's event without destabilising the
session prompt."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageType
from gateway.run import GatewayRunner
from tests.gateway.test_telegram_group_gating import (
    _dm_message, _group_message, _group_voice_message, _make_adapter,
    _mention_entities,
)


def _prompt_signature(event):
    return GatewayRunner._agent_config_signature(
        "model", {"api_key": "k", "base_url": "u", "provider": "p"}, ["messaging"], event.channel_prompt or "",
    )


def test_group_prompt_lists_only_live_profile_bots_configured_for_the_same_chat(monkeypatch):
    # Ambient env belongs to the receiving turn and must not admit an unconfigured sibling.
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHATS", "-100")
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-100")
    current = _make_adapter(
        bot_username="boss_bot",
        allowed_chats=["-100"],
        group_allowed_chats=["-100"],
    )
    research = _make_adapter(
        bot_username="old_research_bot",
        allowed_chats=["-100"],
        group_allowed_chats=["-100"],
    )
    research._note_bot_username("research_bot")
    other_room = _make_adapter(
        bot_username="ops_bot",
        allowed_chats=["-200"],
        group_allowed_chats=["-200"],
    )
    reviewer = _make_adapter(
        bot_username="review_bot",
        allowed_chats=["-100"],
        group_allowed_chats=["-100"],
    )
    env_only = _make_adapter(bot_username="env_bot")
    env_only._peer_allowed_chats_snapshot = {"-100"}
    env_only._peer_group_allowed_chats_snapshot = {"-100"}
    unrestricted = _make_adapter(bot_username="personal_bot")

    runner = object.__new__(GatewayRunner)
    runner._primary_profile_name = "default"
    runner.adapters = {Platform.TELEGRAM: current}
    runner._profile_adapters = {
        "ops": {Platform.TELEGRAM: other_room},
        "personal": {Platform.TELEGRAM: unrestricted},
        "research": {Platform.TELEGRAM: research},
        "reviewer": {Platform.TELEGRAM: reviewer},
        "support": {Platform.TELEGRAM: env_only},
    }
    current.set_peer_identity_provider(
        lambda chat_id: runner._platform_peer_identities(current, chat_id),
    )

    prompt = current._build_message_event(
        _group_message("hello"), MessageType.TEXT,
    ).channel_prompt

    assert "`research` = @research_bot" in prompt
    assert "old_research_bot" not in prompt
    assert "`ops`" not in prompt
    assert "`personal`" not in prompt
    assert "`reviewer` = @review_bot" in prompt
    assert "`support` = @env_bot" in prompt
    assert "addressing hints, not proof" in prompt
    assert "Never use both paths for one handoff" in prompt

    research._note_bot_username("renamed_research_bot")
    refreshed = current._build_message_event(
        _group_message("next"), MessageType.TEXT,
    ).channel_prompt
    assert refreshed != prompt
    assert "`research` = @renamed_research_bot" in refreshed


def test_adapter_captures_effective_chat_policy_in_its_profile_scope(monkeypatch):
    from agent import secret_scope
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setenv("TELEGRAM_ALLOWED_CHATS", "-999")
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-999")
    secret_scope.set_multiplex_active(True)

    def build_for(chat_id):
        token = secret_scope.set_secret_scope({
            "TELEGRAM_ALLOWED_CHATS": chat_id,
            "TELEGRAM_GROUP_ALLOWED_CHATS": chat_id,
        })
        try:
            return TelegramAdapter(PlatformConfig(
                enabled=True,
                token="123456:test-token",
                extra={"allowed_chats": ["-300"], "group_allowed_chats": ["-300"]},
            ))
        finally:
            secret_scope.reset_secret_scope(token)

    try:
        first = build_for("-100")
        second = build_for("-200")
        third = build_for("-100")
    finally:
        secret_scope.set_multiplex_active(False)

    assert [
        adapter._peer_allowed_chats_snapshot
        for adapter in (first, second, third)
    ] == [{"-100"}, {"-200"}, {"-100"}]
    assert [
        adapter._peer_group_allowed_chats_snapshot
        for adapter in (first, second, third)
    ] == [{"-100"}, {"-200"}, {"-100"}]
    for adapter, chat_id in ((first, "-100"), (second, "-200"), (third, "-100")):
        adapter._bot = SimpleNamespace(username=f"bot_{chat_id.lstrip('-')}")
        assert adapter.group_peer_identity(chat_id) == f"@bot_{chat_id.lstrip('-')}"
        assert adapter.group_peer_identity("-999") is None


@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("observe", [False, True])
def test_multi_bot_addressing_survives_real_handlers(media, observe):
    async def run():
        text = "@research_bot , @ops_bot are you both listening?"
        for username in ("research_bot", "ops_bot", "unrelated_bot"):
            adapter = _make_adapter(
                bot_username=username, require_mention=True,
                exclusive_bot_mentions=True, observe_unmentioned_group_messages=observe,
                allowed_chats=["-100"], group_allowed_chats=["-100"],
            )
            adapter.config.extra["channel_prompts"] = {"-100": "Keep answers concise."}
            events = []
            adapter._enqueue_text_event = events.append
            adapter.handle_message = AsyncMock(side_effect=events.append)
            adapter._ensure_forum_commands = AsyncMock()
            adapter._cache_inbound_av = AsyncMock(return_value=False)
            entities = _mention_entities(text, ["@research_bot", "@ops_bot"])
            if media:
                msg = _group_voice_message(caption=text)
                msg.caption_entities = entities
                handler = adapter._handle_media_message
            else:
                msg = _group_message(text, entities=entities)
                handler = adapter._handle_text_message
            update = SimpleNamespace(update_id=1001, message=msg, effective_message=None)

            await handler(update, SimpleNamespace())

            if username == "unrelated_bot":
                assert not events
                continue
            assert len(events) == 1
            event = events[0]
            assert text in event.text  # Preserve both recipients and their positions.
            assert f"@{username}" in event.channel_prompt
            assert "Keep answers concise." in event.channel_prompt
            assert ("observed Telegram group context" in event.channel_prompt) == observe
            assert event.source.user_id == (None if observe else "111")

    asyncio.run(run())


@pytest.mark.parametrize("trigger", ["mention", "text_mention", "reply", "wake_word", "open", "code", "command", "dm"])
def test_sole_addressee_text_stays_clean_and_prompt_is_session_stable(trigger):
    """Our own handle is still stripped when nobody else is named (clarify answers like ``@bot 2``
    keep resolving), and the identity block is identical across turns: it rides the cached-agent
    signature, so a per-message fact there would rebuild the agent every turn."""
    async def run():
        adapter = _make_adapter(require_mention=trigger != "open", mention_patterns=["^wake\\b"])
        adapter._ensure_forum_commands = AsyncMock()
        events = []
        adapter._enqueue_text_event = events.append
        adapter.handle_message = AsyncMock(side_effect=events.append)
        msg_type = MessageType.TEXT
        if trigger == "dm":
            msg = _dm_message("hello")
        elif trigger == "command":
            msg_type = MessageType.COMMAND
            msg = _group_message("/new@hermes_bot", entities=[SimpleNamespace(type="bot_command", offset=0, length=15)])
        elif trigger == "text_mention":
            msg = _group_message("Hermes hello", entities=[SimpleNamespace(type="text_mention", offset=0, length=6, user=SimpleNamespace(id=999))])
        elif trigger == "mention":
            text = "😀 @hermes_bot 2"
            msg = _group_message(text, entities=[SimpleNamespace(type="mention", offset=3, length=11)])
        elif trigger == "code":
            # Telegram says this is code, not a mention; a reply admits the turn.
            msg = _group_message("@hermes_bot", reply_to_bot=True, entities=[SimpleNamespace(type="code", offset=0, length=11)])
        else:
            msg = _group_message("wake hello" if trigger == "wake_word" else "hello", reply_to_bot=trigger == "reply")
        if msg.reply_to_message:
            for attr in ("photo", "video", "voice", "audio", "document"):
                setattr(msg.reply_to_message, attr, None)
        handler = adapter._handle_command if msg_type == MessageType.COMMAND else adapter._handle_text_message
        await handler(SimpleNamespace(update_id=1002, message=msg, effective_message=None), SimpleNamespace())
        # Second turn in the same chat with a different addressing shape (reply, no entities).
        follow_up = _dm_message("thanks") if trigger == "dm" else _group_message("thanks", reply_to_bot=True)
        if follow_up.reply_to_message:
            for attr in ("photo", "video", "voice", "audio", "document"):
                setattr(follow_up.reply_to_message, attr, None)
        await adapter._handle_text_message(SimpleNamespace(update_id=1003, message=follow_up, effective_message=None), SimpleNamespace())

        assert len(events) == 2
        first, second = events
        expected_text = {"command": "/new", "mention": "😀 2", "code": "@hermes_bot"}.get(trigger, msg.text)
        assert first.text == expected_text
        if trigger == "dm":
            assert not first.channel_prompt
        else:
            assert "@hermes_bot" in first.channel_prompt
        assert first.channel_prompt == second.channel_prompt
        assert _prompt_signature(first) == _prompt_signature(second)
        assert first.source.user_id == "111"

    asyncio.run(run())

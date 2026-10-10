"""Multi-bot addressing must survive routing into the agent's event without destabilising the
session prompt."""

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageType
from gateway.run import GatewayRunner
from tests.gateway.test_telegram_group_gating import (
    _bot_command_entity, _dm_message, _group_message, _group_voice_message, _make_adapter,
    _mention_entities,
)


def _prompt_signature(event):
    return GatewayRunner._agent_config_signature(
        "model", {"api_key": "k", "base_url": "u", "provider": "p"}, ["messaging"], event.channel_prompt or "",
    )


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


async def _sole_addressee_turns(msg, msg_type=MessageType.TEXT):
    """Send ``msg`` and a reply follow-up in the same chat through the real handlers."""
    adapter = _make_adapter(require_mention=True)
    adapter._ensure_forum_commands = AsyncMock()
    events = []
    adapter._enqueue_text_event = events.append
    adapter.handle_message = AsyncMock(side_effect=events.append)
    handler = adapter._handle_command if msg_type == MessageType.COMMAND else adapter._handle_text_message
    await handler(SimpleNamespace(update_id=1002, message=msg, effective_message=None), SimpleNamespace())
    follow_up = _group_message("thanks", reply_to_bot=True)
    for attr in ("photo", "video", "voice", "audio", "document"):
        setattr(follow_up.reply_to_message, attr, None)
    await adapter._handle_text_message(SimpleNamespace(update_id=1003, message=follow_up, effective_message=None), SimpleNamespace())
    return events


def _addressed_entities(text, command=None):
    """Telegram's entities for ``text``: the bot command, plus every bot mention outside it."""
    start = text.index(command) if command else -1
    entities = [_bot_command_entity(text, command)] if command else []
    for handle in ("@hermes_bot", "@ops_bot"):
        for match in re.finditer(re.escape(handle), text, re.IGNORECASE):
            if not start <= match.start() < start + len(command or ""):
                entities.append(SimpleNamespace(type="mention", offset=match.start(), length=len(handle)))
    return sorted(entities, key=lambda entity: entity.offset)


@pytest.mark.parametrize("text,command,expected_text,parsed", [
    ("@hermes_bot ok", None, "ok", None),
    ("@hermes_bot /status", None, "/status", ("status", "")),
    ("@hermes_bot /model@hermes_bot gpt-5", "/model@hermes_bot", "/model gpt-5", ("model", "gpt-5")),
    ("@hermes_bot /btw did @hermes_bot answer Alice?\nKeep it short  ", "/btw",
     "/btw did @hermes_bot answer Alice?\nKeep it short  ", ("btw", "did @hermes_bot answer Alice?\nKeep it short  ")),
    (" @HeRmEs_BoT, @hermes_bot: /btw did @hermes_bot answer? ", "/btw",
     "/btw did @hermes_bot answer? ", ("btw", "did @hermes_bot answer? ")),
    ("@hermes_bot explain /model gpt-5", None, "explain /model gpt-5", None),
    ("@hermes_bot ! /model gpt-5", None, "! /model gpt-5", None),
    ("@hermes_bot @ops_bot /model gpt-5", None, "@hermes_bot @ops_bot /model gpt-5", None),
    ("/btw what did @ops_bot tell @hermes_bot", "/btw",
     "/btw what did @ops_bot tell @hermes_bot", ("btw", "what did @ops_bot tell @hermes_bot")),
])
def test_addressed_commands_keep_their_arguments_and_a_stable_prompt(text, command, expected_text, parsed):
    """A leading own address is removed and the command's arguments keep every byte, including our
    own handle inside them. Text that only mentions a command, or that names another bot, stays
    plain text. A slash-first command that names another participant keeps its text. The identity
    block stays identical across turns, as for every other addressing shape."""
    msg_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
    first, second = asyncio.run(_sole_addressee_turns(_group_message(text, entities=_addressed_entities(text, command)), msg_type))
    assert first.text == expected_text
    if parsed:
        assert (first.get_command(), first.get_command_args()) == parsed
    else:
        assert not first.is_command()
    assert "@hermes_bot" in first.channel_prompt
    assert first.channel_prompt == second.channel_prompt
    assert _prompt_signature(first) == _prompt_signature(second)
    assert first.source.user_id == "111"


def _command_entities(text):
    return _addressed_entities(text, text.lstrip().split(maxsplit=1)[0])


@pytest.mark.parametrize("chat", ["group", "observed_group", "private"])
@pytest.mark.parametrize("text,command,args", [
    ("/model@hermes_bot gpt-5", "model", "gpt-5"),
    ("/model@HeRmEs_BoT   anthropic/claude-sonnet-4  --global  ", "model", "anthropic/claude-sonnet-4  --global  "),
    ("/queue@hermes_bot\nsummarise the thread\nthen ask @hermes_bot for a recap", "queue",
     "summarise the thread\nthen ask @hermes_bot for a recap"),
    (" \t/personality@hermes_bot\tconcise", "personality", "concise"),
    ("/btw did @hermes_bot already answer Alice?\nKeep it short", "btw", "did @hermes_bot already answer Alice?\nKeep it short"),
    ("/steer@hermes_bot ask @ops_bot what @hermes_bot missed", "steer", "ask @ops_bot what @hermes_bot missed"),
    ("/kanban@hermes_bot list", "kanban", "list"),
    ("/reasoning@hermes_bot: high --global", "reasoning", "high --global"),
    ("/model gpt-5 @hermes_bot", "model", "gpt-5"),
    ("/new@hermes_bot", "new", ""),
    ("/help\n", "help", ""),
])
def test_command_arguments_reach_real_event_parser_intact(chat, text, command, args):
    async def run():
        adapter = _make_adapter(
            require_mention=False, observe_unmentioned_group_messages=chat == "observed_group",
            allowed_chats=["-100"], group_allowed_chats=["-100"],
        )
        adapter._ensure_forum_commands = AsyncMock()
        adapter.handle_message = AsyncMock()
        msg = _dm_message(text) if chat == "private" else _group_message(text)
        msg.entities = _command_entities(text)
        await adapter._handle_command(SimpleNamespace(update_id=1004, message=msg, effective_message=None), SimpleNamespace())

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.COMMAND
        assert event.raw_message is msg
        assert event.is_command()
        assert event.get_command() == command
        assert event.get_command_args() == args
        # Handlers that re-read the text (/kanban) see the command as a DM carries it.
        assert event.text.split(maxsplit=1)[0] == f"/{command}"
        assert event.source.user_id == "111"

    asyncio.run(run())


@pytest.mark.parametrize("carrier,other_bot,require_mention", [
    ("text", False, True), ("caption", False, True),
    ("command", True, False), ("command", True, True),
])
def test_trigger_paths_preserve_commands_without_widening_admission(carrier, other_bot, require_mention):
    async def run():
        text = "/model@ops_bot gpt-5" if other_bot else "/queue@hermes_bot\nsummarise what @hermes_bot missed"
        adapter = _make_adapter(require_mention=require_mention, free_response_chats=["-100"])
        adapter._ensure_forum_commands = AsyncMock()
        adapter._cache_inbound_av = AsyncMock(return_value=False)
        events = []
        adapter._enqueue_text_event = events.append
        adapter.handle_message = AsyncMock(side_effect=events.append)
        if carrier == "caption":
            msg = _group_voice_message(caption=text)
            msg.caption_entities = _command_entities(text)
        else:
            msg = _group_message(text, reply_to_bot=other_bot, entities=_command_entities(text))
        handler = {"text": adapter._handle_text_message, "caption": adapter._handle_media_message,
                   "command": adapter._handle_command}[carrier]
        await handler(SimpleNamespace(update_id=1005, message=msg, effective_message=None), SimpleNamespace())
        if other_bot:
            assert not events
            adapter._ensure_forum_commands.assert_not_awaited()
            return
        assert len(events) == 1
        assert events[0].get_command() == "queue"
        assert events[0].get_command_args() == "summarise what @hermes_bot missed"
        assert events[0].text == "/queue\nsummarise what @hermes_bot missed"

    asyncio.run(run())

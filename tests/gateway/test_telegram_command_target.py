"""A command suffix owns its recipient even when the argument names other bots."""

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.gateway.test_telegram_group_gating import _group_message, _group_voice_message, _make_adapter


def entities(text):
    spans = []
    command = re.search(r"/\w+@\w+", text)
    if command:
        spans.append(("bot_command", command.start(), command.end()))
    for mention in re.finditer(r"@\w+", text):
        if not command or not command.start() <= mention.start() < command.end():
            spans.append(("mention", mention.start(), mention.end()))
    return [SimpleNamespace(type=kind, offset=len(text[:start].encode("utf-16-le")) // 2,
                            length=len(text[start:end].encode("utf-16-le")) // 2)
            for kind, start, end in spans]


def test_foreign_command_target_is_not_overridden_by_mentions_in_arguments():
    async def run():
        for carrier in ("command", "text", "caption"):
            for text, exclusive in (("/model@ops_bot gpt-5 @hermes_bot", True),
                                    ("@hermes_bot /queue@ops_bot 😀 ask @hermes_bot for details", True),
                                    ("/model@ops_bot gpt-5 @hermes_bot", False)):
                adapter = _make_adapter(require_mention=True, exclusive_bot_mentions=exclusive)
                adapter._ensure_forum_commands = AsyncMock()
                adapter._cache_inbound_av = AsyncMock(return_value=False)
                events = []
                adapter._enqueue_text_event = events.append
                adapter.handle_message = AsyncMock(side_effect=events.append)
                if carrier == "caption":
                    message = _group_voice_message(caption=text)
                    message.caption_entities = entities(text)
                    handler = adapter._handle_media_message
                else:
                    message = _group_message(text, entities=entities(text))
                    handler = adapter._handle_command if carrier == "command" else adapter._handle_text_message
                await handler(SimpleNamespace(update_id=1701, message=message, effective_message=None), SimpleNamespace())
                assert not events, (carrier, text, [(event.get_command(), event.get_command_args()) for event in events])
    asyncio.run(run())


def test_command_arguments_and_noncommand_text_survive_their_real_carriers():
    async def run():
        for carrier in ("command", "edited_command", "text", "edited_text", "caption"):
            texts = (("/queue@hermes_bot 😀 ask @ops_bot what @hermes_bot missed\nkeep this  ",
                      "/queue 😀 ask @ops_bot what @hermes_bot missed\nkeep this  "),)
            if carrier not in ("command", "edited_command"):
                texts += (("😀 @hermes_bot , @ops_bot are you both listening?",
                           "😀 @hermes_bot , @ops_bot are you both listening?"),)
            for text, expected in texts:
                adapter = _make_adapter(require_mention=True, exclusive_bot_mentions=True)
                adapter._ensure_forum_commands = AsyncMock()
                adapter._cache_inbound_av = AsyncMock(return_value=False)
                events = []
                adapter._enqueue_text_event = events.append
                adapter.handle_message = AsyncMock(side_effect=events.append)
                if carrier == "caption":
                    message = _group_voice_message(caption=text)
                    message.caption_entities = entities(text)
                    handler = adapter._handle_media_message
                else:
                    message = _group_message(text, entities=entities(text))
                    handler = adapter._handle_command if carrier.endswith("command") else adapter._handle_text_message
                edited = carrier.startswith("edited")
                await handler(SimpleNamespace(update_id=1702, message=None if edited else message,
                                              edited_message=message if edited else None,
                                              effective_message=message), SimpleNamespace())
                assert len(events) == 1
                event = events[0]
                assert event.text == expected
                if text.startswith("/"):
                    assert event.get_command() == "queue"
                    assert event.get_command_args() == expected.split(maxsplit=1)[1]
                else:
                    assert not event.is_command()
    asyncio.run(run())

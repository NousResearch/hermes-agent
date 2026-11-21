"""Configured draft markers must not reach any Discord context path."""
from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import pytest

from gateway.config import Platform, load_gateway_config
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from plugins.platforms.discord.adapter import DiscordAdapter
import plugins.platforms.discord.adapter as transport


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["top-level", "nested-with-root", "nested-and-root-conflict"])
async def test_profile_drafts_are_filtered_from_ingress_history_and_replies(tmp_path, monkeypatch, shape):
    class Channel:
        id = 23
        name = "dm"
    monkeypatch.setattr(transport.discord, "DMChannel", Channel)
    homes = [tmp_path / "a", tmp_path / "b"]
    for home, markers in zip(homes, [["Draft", "Drafts"], []]):
        home.mkdir()
        config = {"platforms": {"discord": {"enabled": True}}, "discord": {"draft_markers": markers}}
        if shape != "top-level":
            config["platforms"]["discord"]["extra"] = {"draft_markers": markers}
            config["discord"] = {"require_mention": False}
            if shape == "nested-and-root-conflict":
                config["discord"]["draft_markers"] = ["RootOnly"]
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    for home in [homes[0], homes[1], homes[0]]:
        token = set_hermes_home_override(home)
        try:
            config = load_gateway_config().platforms[Platform.DISCORD]
            adapter = DiscordAdapter(config)
            adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
            adapter._text_batch_delay_seconds = 0
            adapter.handle_message = AsyncMock()
            channel = Channel()
            private = SimpleNamespace(id=11, content="Draft: private topic", clean_content="Draft: private topic",
                author=SimpleNamespace(id=42, name="Example", display_name="Example", bot=False),
                type=transport.discord.MessageType.default, attachments=[SimpleNamespace(filename="private.txt", content_type="text/plain")])
            def message(text, reference=None):
                return SimpleNamespace(id=30, content=text, clean_content=text, mentions=[], attachments=[],
                    reference=reference, created_at=datetime.now(UTC), channel=channel, author=private.author)
            enabled = home == homes[0]
            if enabled:
                assert await adapter._handle_message(message(private.content)) is False
                adapter.handle_message.assert_not_awaited()
            # A near-match remains ordinary input, as does a draft in the disabled profile.
            await adapter._handle_message(message("Drafting the release plan"))
            assert adapter.handle_message.await_count == 1
            adapter.handle_message.reset_mock()
            reference = SimpleNamespace(message_id=11, resolved=private)
            adapter._collect_attachment_media = AsyncMock(return_value=([], [], [], ""))
            await adapter._handle_message(message("Review the public plan", reference))
            event = adapter.handle_message.call_args.args[0]
            assert event.reply_to_text == (None if enabled else private.content)
            assert adapter._collect_attachment_media.call_args.args[0] == ([] if enabled else private.attachments)
            async def history(**kwargs):
                yield private
            channel.history = history
            adapter._is_sender_authorized = lambda *args, **kwargs: True
            context = await adapter._fetch_channel_context(channel, before=SimpleNamespace(id=40))
            assert (private.content in context) is not enabled
            if enabled:
                assert not await adapter._should_backfill_discord_message(private)
                adapter.handle_message.reset_mock()
                forwarded = message("")
                forwarded.message_snapshots = [private]
                assert await adapter._handle_message(forwarded) is False
                adapter.handle_message.assert_not_awaited()
                forwarded.message_snapshots.append(SimpleNamespace(content="Public summary", attachments=[]))
                await adapter._handle_message(forwarded)
                assert adapter.handle_message.call_args.args[0].text == "Public summary"
                assert adapter._collect_attachment_media.call_args.args[0] == []
                forwarded.content = "My public request"
                forwarded.message_snapshots = [private]
                await adapter._handle_message(forwarded)
                assert adapter.handle_message.call_args.args[0].text == "My public request"
        finally:
            reset_hermes_home_override(token)

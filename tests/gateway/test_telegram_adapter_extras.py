"""Telegram adapter extras: stop button (can_stop), message effects, ephemeral whispers.

Regression coverage for the Bot API additions wired into the Telegram adapter:
- the stop button sends drafts via the raw API with ``can_stop=True`` when enabled,
  and a ``stopped_message_generation`` update is re-dispatched as a synthetic ``/stop``;
- completion message effects resolve emoji → id for private chats only;
- ephemeral whispers use the raw API, are group-only, and never leak to the whole
  group on failure.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter, _MESSAGE_EFFECT_IDS


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    inst = TelegramAdapter(config)
    if not hasattr(inst, "gateway_runner"):
        inst.gateway_runner = None
    return inst


class TestStopButton:
    @pytest.mark.asyncio
    async def test_draft_uses_can_stop_when_enabled(self, adapter):
        adapter._stop_button_enabled = True
        adapter._bot = MagicMock()
        adapter._bot.do_api_request = AsyncMock(return_value=True)

        result = await adapter.send_draft("123", 7, "partial text")

        assert result.success
        args, kwargs = adapter._bot.do_api_request.call_args
        assert args[0] == "sendMessageDraft"
        payload = kwargs["api_kwargs"]
        assert payload["can_stop"] is True
        assert payload["draft_id"] == 7
        assert payload["parse_mode"] == "MarkdownV2"  # enum converted for the raw body

    @pytest.mark.asyncio
    async def test_draft_default_path_without_stop_button(self, adapter):
        adapter._stop_button_enabled = False
        adapter._bot = MagicMock()
        adapter._bot.send_message_draft = AsyncMock(return_value=True)

        result = await adapter.send_draft("123", 7, "partial text")

        assert result.success
        adapter._bot.send_message_draft.assert_awaited_once()
        adapter._bot.do_api_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_stopped_update_dispatches_slash_stop(self, adapter):
        adapter._stop_button_enabled = True
        adapter.handle_message = AsyncMock()
        update = SimpleNamespace(api_kwargs={
            "stopped_message_generation": {
                "chat": {"id": 860026573, "type": "private"}, "draft_id": 9}})

        await adapter._maybe_handle_generation_stopped(update)

        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/stop"
        assert event.source.chat_id == "860026573"
        assert event.source.chat_type == "dm"
        assert event.source.user_id == "860026573"

    @pytest.mark.asyncio
    async def test_stopped_update_ignored_when_disabled(self, adapter):
        adapter._stop_button_enabled = False
        adapter.handle_message = AsyncMock()
        update = SimpleNamespace(api_kwargs={
            "stopped_message_generation": {"chat": {"id": 1, "type": "private"}, "draft_id": 1}})

        await adapter._maybe_handle_generation_stopped(update)

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_group_press_has_no_user_fail_closed(self, adapter):
        adapter._stop_button_enabled = True
        adapter.handle_message = AsyncMock()
        update = SimpleNamespace(api_kwargs={
            "stopped_message_generation": {
                "chat": {"id": -100123, "type": "supergroup"}, "draft_id": 2}})

        await adapter._maybe_handle_generation_stopped(update)

        event = adapter.handle_message.await_args.args[0]
        assert event.source.user_id is None  # the auth chain drops it (fail closed)


class TestMessageEffects:
    def test_private_chat_resolves(self, adapter):
        assert adapter._message_effect_id_for(
            "860026573", {"message_effect": "🎉"}) == _MESSAGE_EFFECT_IDS["🎉"]

    def test_group_and_channel_skip(self, adapter):
        assert adapter._message_effect_id_for("-100123", {"message_effect": "🎉"}) is None

    def test_unknown_emoji_skips(self, adapter):
        assert adapter._message_effect_id_for("123", {"message_effect": "🤖"}) is None

    def test_missing_metadata_skips(self, adapter):
        assert adapter._message_effect_id_for("123", None) is None


class TestEphemeral:
    @pytest.mark.asyncio
    async def test_group_whisper_uses_raw_api(self, adapter):
        adapter._ephemeral_messages_enabled = True
        adapter._bot = MagicMock()
        adapter._bot.do_api_request = AsyncMock(return_value=True)

        result = await adapter._maybe_send_ephemeral("-100123", "psst", {"ephemeral_for": "42"})

        assert result is not None and result.success
        payload = adapter._bot.do_api_request.call_args.kwargs["api_kwargs"]
        assert payload["ephemeral_message_parameters"] == {"receiver_user_id": 42}

    @pytest.mark.asyncio
    async def test_dm_is_not_ephemeral(self, adapter):
        adapter._ephemeral_messages_enabled = True
        result = await adapter._maybe_send_ephemeral("860026573", "hello", {"ephemeral_for": "42"})
        assert result is None

    @pytest.mark.asyncio
    async def test_failure_does_not_leak_to_group(self, adapter):
        adapter._ephemeral_messages_enabled = True
        adapter._bot = MagicMock()
        adapter._bot.do_api_request = AsyncMock(side_effect=RuntimeError("nope"))

        result = await adapter._maybe_send_ephemeral("-100123", "psst", {"ephemeral_for": "42"})

        assert result is not None and not result.success

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, adapter):
        result = await adapter._maybe_send_ephemeral("-100123", "x", {"ephemeral_for": "42"})
        assert result is None

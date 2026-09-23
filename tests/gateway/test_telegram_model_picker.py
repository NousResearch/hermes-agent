"""Tests for Telegram model picker thread fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramModelPicker:
    @pytest.mark.asyncio
    async def test_send_model_picker_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=101)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_model_picker(
            chat_id="12345",
            providers=[
                {"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}
            ],
            current_model="model_1",
            current_provider="provider_one",
            session_key="s",
            on_model_selected=AsyncMock(),
            metadata={"thread_id": "99999"},
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        assert "provider\\_one" in sent["text"]
        assert "`model_1`" in sent["text"]

    @pytest.mark.asyncio
    async def test_back_button_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        adapter._model_picker_state["12345"] = {
            "providers": [{"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}],
            "current_model": "model_1",
            "current_provider": "provider_one",
            "session_key": "s",
            "on_model_selected": AsyncMock(),
            "msg_id": 42,
        }

        query = AsyncMock()
        query.data = "mb"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.message_thread_id = None
        query.from_user = MagicMock()
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mb", "12345")

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "provider\\_one" in edit_kwargs["text"]
        assert "`model_1`" in edit_kwargs["text"]

    @pytest.mark.asyncio
    async def test_topic_picker_callbacks_keep_each_topics_selection_state(self):
        adapter = _make_adapter()
        work_callback = AsyncMock(return_value="Switched work model")
        personal_callback = AsyncMock(return_value="Switched personal model")
        chat_id = "12345"
        adapter._bot.send_message = AsyncMock(
            side_effect=[SimpleNamespace(message_id=101), SimpleNamespace(message_id=202)]
        )

        for thread_id, model_id, callback in (
            ("101", "work-model", work_callback),
            ("202", "personal-model", personal_callback),
        ):
            await adapter.send_model_picker(
                chat_id=chat_id,
                providers=[{"slug": "codex", "name": "Codex", "models": [model_id]}],
                current_model=model_id,
                current_provider="codex",
                session_key=f"topic-{thread_id}",
                on_model_selected=callback,
                metadata={"thread_id": thread_id},
            )

        assert len(adapter._model_picker_state) == 2

        async def select_in_topic(thread_id):
            query = AsyncMock()
            query.message = MagicMock(chat_id=int(chat_id), message_thread_id=int(thread_id))
            query.answer = AsyncMock()
            query.edit_message_text = AsyncMock()
            await adapter._handle_model_picker_callback(query, "mp:codex", chat_id)
            await adapter._handle_model_picker_callback(query, "mc:0", chat_id)

        await select_in_topic("101")
        await select_in_topic("202")

        work_callback.assert_awaited_once_with(chat_id, "work-model", "codex")
        personal_callback.assert_awaited_once_with(chat_id, "personal-model", "codex")

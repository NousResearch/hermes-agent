import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from tests.gateway.test_telegram_clarify_buttons import _ensure_telegram_mock

_ensure_telegram_mock()

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _make_callback_query(data, *, chat_id=12345, user_id="777", thread_id=None):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = chat_id
    query.message.chat = MagicMock()
    query.message.chat.type = "private"
    query.message.text = "Deploy where?"
    query.message.message_id = 321
    query.message.message_thread_id = thread_id
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query
    return update, query


@pytest.mark.asyncio
async def test_durable_send_clarify_uses_durable_prefix_without_in_memory_state():
    adapter = _make_adapter()
    mock_msg = MagicMock()
    mock_msg.message_id = 200
    adapter._bot.send_message = AsyncMock(return_value=mock_msg)

    result = await adapter.send_clarify(
        chat_id="12345",
        question="Which option?",
        choices=["alpha", "beta"],
        clarify_id="cld_test",
        session_key="session-key",
        metadata={"durable_clarify": True},
    )

    assert result.success is True
    assert "cld_test" not in adapter._clarify_state
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_durable_callback_resolves_choice_and_dispatches_recovery_turn(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        from tools.clarify_interaction import create_clarify_interaction, get_interaction

        adapter = _make_adapter()
        adapter._message_handler = AsyncMock(return_value="continued")
        adapter.send = AsyncMock()

        interaction = create_clarify_interaction(
            session_key="telegram:12345:777",
            platform="telegram",
            question="Deploy where?",
            choices=["staging", "production"],
            chat_id="12345",
            user_id="777",
            ttl_seconds=3600,
        )

        update, query = _make_callback_query(f"cld:{interaction.interaction_id}:1")

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())

        loaded = get_interaction(interaction.interaction_id)
        assert loaded.status == "resolved"
        assert loaded.answer == "production"
        adapter._message_handler.assert_awaited_once()
        dispatched_event = adapter._message_handler.call_args.args[0]
        assert "Deploy where?" in dispatched_event.text
        assert "production" in dispatched_event.text
        adapter.send.assert_awaited_once()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_durable_other_marks_awaiting_text_without_dispatching_turn(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        from tools.clarify_interaction import create_clarify_interaction, get_interaction

        adapter = _make_adapter()
        adapter._message_handler = AsyncMock(return_value="should not run yet")
        adapter.send = AsyncMock()

        interaction = create_clarify_interaction(
            session_key="telegram:12345:777",
            platform="telegram",
            question="Deploy where?",
            choices=["staging", "production"],
            chat_id="12345",
            user_id="777",
            ttl_seconds=3600,
        )

        update, query = _make_callback_query(f"cld:{interaction.interaction_id}:other")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())

        loaded = get_interaction(interaction.interaction_id)
        assert loaded.status == "awaiting_text"
        assert loaded.answer is None
        adapter._message_handler.assert_not_awaited()
        adapter.send.assert_not_awaited()
        query.edit_message_text.assert_awaited_once()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_durable_callback_rejects_wrong_user_without_dispatching_turn(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        from tools.clarify_interaction import create_clarify_interaction, get_interaction

        adapter = _make_adapter()
        adapter._message_handler = AsyncMock(return_value="should not run")
        adapter.send = AsyncMock()

        interaction = create_clarify_interaction(
            session_key="telegram:12345:777",
            platform="telegram",
            question="Deploy where?",
            choices=["staging", "production"],
            chat_id="12345",
            user_id="777",
            ttl_seconds=3600,
        )

        update, _query = _make_callback_query(
            f"cld:{interaction.interaction_id}:1",
            user_id="intruder",
        )
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())

        loaded = get_interaction(interaction.interaction_id)
        assert loaded.status == "pending"
        assert loaded.answer is None
        adapter._message_handler.assert_not_awaited()
        adapter.send.assert_not_awaited()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_durable_duplicate_callback_dispatches_recovery_turn_only_once(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        from tools.clarify_interaction import create_clarify_interaction, get_interaction

        adapter = _make_adapter()
        adapter._message_handler = AsyncMock(return_value="continued")
        adapter.send = AsyncMock()

        interaction = create_clarify_interaction(
            session_key="telegram:12345:777",
            platform="telegram",
            question="Deploy where?",
            choices=["staging", "production"],
            chat_id="12345",
            user_id="777",
            ttl_seconds=3600,
        )

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            first_update, _first_query = _make_callback_query(f"cld:{interaction.interaction_id}:1")
            await adapter._handle_callback_query(first_update, MagicMock())
            second_update, _second_query = _make_callback_query(f"cld:{interaction.interaction_id}:0")
            await adapter._handle_callback_query(second_update, MagicMock())

        loaded = get_interaction(interaction.interaction_id)
        assert loaded.status == "resolved"
        assert loaded.answer == "production"
        adapter._message_handler.assert_awaited_once()
        adapter.send.assert_awaited_once()
    finally:
        reset_hermes_home_override(token)

@pytest.mark.asyncio
async def test_text_intercept_resolves_awaiting_durable_answer_in_same_session(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        from tools.clarify_interaction import (
            create_clarify_interaction,
            get_interaction,
            mark_awaiting_text,
        )

        runner = _make_runner()
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="777",
        )
        event = MessageEvent(text="custom production", message_type=MessageType.TEXT, source=source)
        event.metadata = {}
        session_key = "telegram:12345:777"
        interaction = create_clarify_interaction(
            session_key=session_key,
            platform="telegram",
            question="Deploy where?",
            choices=["staging", "production"],
            chat_id="12345",
            user_id="777",
            ttl_seconds=3600,
        )
        assert mark_awaiting_text(interaction.interaction_id, user_id="777", chat_id="12345").ok

        prepared = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
            session_key=session_key,
        )

        loaded = get_interaction(interaction.interaction_id)
        assert loaded.status == "resolved"
        assert loaded.answer == "custom production"
        assert event.metadata["deferred_clarify_interaction_id"] == interaction.interaction_id
        assert "Deploy where?" in prepared
        assert "custom production" in prepared
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_text_intercept_ignores_slash_commands_for_awaiting_durable_answer(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        from tools.clarify_interaction import (
            create_clarify_interaction,
            get_interaction,
            mark_awaiting_text,
        )

        runner = _make_runner()
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="777",
        )
        event = MessageEvent(text="/help", message_type=MessageType.TEXT, source=source)
        event.metadata = {}
        session_key = "telegram:12345:777"
        interaction = create_clarify_interaction(
            session_key=session_key,
            platform="telegram",
            question="Deploy where?",
            choices=["staging", "production"],
            chat_id="12345",
            user_id="777",
            ttl_seconds=3600,
        )
        assert mark_awaiting_text(interaction.interaction_id, user_id="777", chat_id="12345").ok

        prepared = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
            session_key=session_key,
        )

        loaded = get_interaction(interaction.interaction_id)
        assert prepared == "/help"
        assert loaded.status == "awaiting_text"
        assert "deferred_clarify_interaction_id" not in event.metadata
    finally:
        reset_hermes_home_override(token)

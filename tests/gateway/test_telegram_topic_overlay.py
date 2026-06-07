"""Telegram group_topics resolves per-topic skill AND model overlay.

The model is nested under group_topics (scoped by chat_id) to avoid cross-group
thread_id collisions, then surfaced on the event as channel_model and applied in
runtime resolution.
"""

from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource

from tests.gateway.test_dm_topics import _make_adapter, _make_mock_message
from telegram.constants import ChatType as _ChatType


def test_group_topic_model_set_on_event():
    """A group_topics entry with a model surfaces channel_model on the event."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter(group_topics_config=[
        {
            "chat_id": -1001234567890,
            "topics": [
                {"name": "Engineering", "thread_id": 5, "skill": "software-development", "model": "anthropic/claude-opus-4-8"},
                {"name": "Sales", "thread_id": 12},
            ],
        }
    ])

    eng = adapter._build_message_event(
        _make_mock_message(chat_id=-1001234567890, chat_type=_ChatType.SUPERGROUP, thread_id=5, text="hi", is_topic_message=True, is_forum=True),
        MessageType.TEXT,
    )
    assert eng.channel_model == "anthropic/claude-opus-4-8"
    assert eng.auto_skill == "software-development"

    sales = adapter._build_message_event(
        _make_mock_message(chat_id=-1001234567890, chat_type=_ChatType.SUPERGROUP, thread_id=12, text="hi", is_topic_message=True, is_forum=True),
        MessageType.TEXT,
    )
    assert sales.channel_model is None


def test_group_topic_model_resolves_via_runtime(monkeypatch):
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *_: "config/default")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", dict)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig()})
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", thread_id="42", chat_type="group", user_id="u1")

    model, _ = runner._resolve_session_agent_runtime(
        source=source, user_config={}, channel_model="anthropic/claude-opus-4-8"
    )
    assert model == "anthropic/claude-opus-4-8"


def test_topic_model_loses_to_session_override(monkeypatch):
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *_: "config/default")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", dict)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig()})
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", thread_id="42", chat_type="group", user_id="u1")
    key = runner._session_key_for_source(source)
    runner._session_model_overrides[key] = {
        "model": "openai/gpt-5",
        "provider": "openai",
        "api_key": "***",
        "base_url": None,
        "api_mode": None,
    }
    model, _ = runner._resolve_session_agent_runtime(
        source=source, user_config={}, channel_model="anthropic/claude-opus-4-8"
    )
    assert model == "openai/gpt-5"

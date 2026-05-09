from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import threading

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import (
    GatewayRunner,
    _extract_auto_topic_split_decision,
)
from gateway.session import SessionEntry, SessionSource


def _history():
    return [
        {"role": "user", "content": "Hermes 的 Response truncated 问题怎么修？"},
        {
            "role": "assistant",
            "content": "已修复普通文本续写和 tool call 截断恢复逻辑。",
        },
    ]


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.WEIXIN,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )


def test_auto_topic_split_parses_llm_new_session_decision():
    should_split, reason = _extract_auto_topic_split_decision(
        '{"decision":"new_session","confidence":0.92,"reason":"different task"}'
    )

    assert should_split is True
    assert reason == "different task"


def test_auto_topic_split_parses_llm_continue_decision():
    should_split, reason = _extract_auto_topic_split_decision(
        '```json\n{"decision":"continue","confidence":0.88,"reason":"follow-up"}\n```'
    )

    assert should_split is False
    assert reason == "follow-up"


def test_auto_topic_split_low_confidence_is_uncertain():
    should_split, reason = _extract_auto_topic_split_decision(
        '{"decision":"new_session","confidence":0.2,"reason":"maybe"}',
        min_confidence=0.6,
    )

    assert should_split is None
    assert reason.startswith("low_confidence:")


def test_auto_topic_split_enabled_from_config(monkeypatch):
    import gateway.run as gateway_run

    runner = object.__new__(GatewayRunner)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"sessions": {"auto_topic_split": {"enabled": True}}},
    )

    assert runner._auto_topic_split_enabled(_source()) is True


@pytest.mark.asyncio
async def test_auto_topic_split_uses_llm_judge(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    class _FakeCompletions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"decision":"new_session","confidence":0.91,"reason":"new task"}'
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    monkeypatch.setattr(
        auxiliary_client,
        "get_text_auxiliary_client",
        lambda **_kwargs: (fake_client, "judge-model"),
    )

    runner = object.__new__(GatewayRunner)
    runner._auto_topic_split_config = MagicMock(return_value={"min_confidence": 0.6})
    runner._resolve_session_agent_runtime = MagicMock(
        return_value=("main-model", {"provider": "custom", "api_key": "k", "base_url": "http://example"})
    )

    should_split, reason = await runner._judge_auto_topic_split_with_llm(
        user_text="明天上海天气怎么样，适合出门吗？",
        history=_history(),
        source=_source(),
        session_key="agent:main:weixin:dm:chat-1",
    )

    assert should_split is True
    assert reason == "new task"


@pytest.mark.asyncio
async def test_auto_topic_split_invalid_llm_response_keeps_session(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    class _FakeCompletions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    monkeypatch.setattr(
        auxiliary_client,
        "get_text_auxiliary_client",
        lambda **_kwargs: (fake_client, "judge-model"),
    )

    runner = object.__new__(GatewayRunner)
    runner._auto_topic_split_config = MagicMock(return_value={"min_confidence": 0.6})
    runner._resolve_session_agent_runtime = MagicMock(
        return_value=("main-model", {"provider": "custom", "api_key": "k", "base_url": "http://example"})
    )

    should_split, reason = await runner._judge_auto_topic_split_with_llm(
        user_text="明天上海天气怎么样，适合出门吗？",
        history=_history(),
        source=_source(),
        session_key="agent:main:weixin:dm:chat-1",
    )

    assert should_split is False
    assert reason.startswith("invalid_json:")


@pytest.mark.asyncio
async def test_auto_reset_session_for_unrelated_input_rotates_session():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._queued_events = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._evict_cached_agent = MagicMock()
    runner._set_session_reasoning_override = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    runner._is_telegram_topic_lane = MagicMock(return_value=False)

    old_entry = SessionEntry(
        session_key="agent:main:weixin:dm:chat-1",
        session_id="old-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=_source(),
        platform=Platform.WEIXIN,
        chat_type="dm",
    )
    new_entry = SessionEntry(
        session_key=old_entry.session_key,
        session_id="new-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=_source(),
        platform=Platform.WEIXIN,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.reset_session.return_value = new_entry

    result = await runner._auto_reset_session_for_unrelated_input(
        session_key=old_entry.session_key,
        source=_source(),
        old_entry=old_entry,
    )

    assert result is new_entry
    runner.session_store.reset_session.assert_called_once_with(old_entry.session_key)
    runner._evict_cached_agent.assert_called_once_with(old_entry.session_key)
    runner._clear_session_boundary_security_state.assert_called_once_with(old_entry.session_key)
    assert runner.hooks.emit.await_count == 2

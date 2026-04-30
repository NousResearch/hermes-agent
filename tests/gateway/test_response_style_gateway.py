from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _FakeAdapter:
    async def send(self, chat_id, text, **kwargs):
        return None

    async def stop_typing(self, chat_id):
        return None


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")}
    )
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.config = MagicMock()
    runner.session_store.load_transcript.return_value = []
    runner.session_store.get_or_create_session.return_value = SimpleNamespace(
        session_id="session-1",
        session_key="telegram:dm:chat-1",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:01",
        was_auto_reset=False,
        auto_reset_reason=None,
    )
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._session_db = None
    runner._set_session_env = MagicMock()
    runner._clear_session_env = MagicMock()
    runner._should_send_voice_reply = MagicMock(return_value=False)
    return runner


@pytest.mark.asyncio
async def test_gateway_response_style_guard_loads_config_without_name_error():
    """Regression: gateway post-processing must not reference an undefined user_config."""
    runner = _make_runner()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="dm")
    event = MessageEvent(
        text="继续",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "已完成配置和代码修改。\n下一步：继续验证。",
            "messages": [],
            "api_calls": 1,
            "tools": [],
            "last_prompt_tokens": 0,
        }
    )
    cfg = {
        "response_style": {
            "enabled": True,
            "platforms": ["telegram"],
            "profile": "secretary",
            "require_labels": True,
        }
    }

    run_generation = runner._begin_session_run_generation("telegram:dm:chat-1")

    with patch("gateway.run._load_gateway_config", return_value=cfg), \
         patch("gateway.run._resolve_gateway_model", return_value="test-model"), \
         patch("gateway.run.build_session_context", return_value={}), \
         patch("gateway.run.build_session_context_prompt", return_value="context"):
        response = await runner._handle_message_with_agent(event, source, "telegram:dm:chat-1", run_generation)

    assert response.startswith("Result:\n")
    assert "Next step:" in response

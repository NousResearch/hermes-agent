import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _runner():
    runner = object.__new__(GatewayRunner)
    runner._resolve_session_agent_runtime = lambda **_: (
        "default-model",
        {
            "provider": "openai",
            "api_key": "default-key",
            "base_url": "",
        },
    )
    runner._evict_cached_agent = lambda _session_key: None
    return runner


@pytest.mark.asyncio
async def test_skill_model_override_restores_previous_session_model(monkeypatch):
    calls = {}

    def fake_switch_model(**kwargs):
        calls.update(kwargs)
        return type(
            "Result",
            (),
            {
                "success": True,
                "new_model": "claude-sonnet",
                "target_provider": "anthropic",
                "api_key": "skill-key",
                "base_url": "",
                "api_mode": "anthropic_messages",
            },
        )()

    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)

    event = MessageEvent(
        text="run it",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123"),
        metadata={
            "_hermes_skill_model": {
                "provider": "anthropic",
                "model": "claude-sonnet",
            }
        },
    )
    runner = _runner()

    assert await runner._apply_skill_model_override(event, "telegram:123") is None
    assert calls["explicit_provider"] == "anthropic"
    assert calls["raw_input"] == "claude-sonnet"
    assert runner._session_state("telegram:123").conversation.model_override == {
        "model": "claude-sonnet",
        "provider": "anthropic",
        "api_key": "skill-key",
        "base_url": "",
        "api_mode": "anthropic_messages",
    }

    runner._restore_pending_one_turn_model_override("telegram:123")
    assert runner._peek_session_state("telegram:123").conversation.model_override is None

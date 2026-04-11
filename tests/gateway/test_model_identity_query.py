from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id="179033731",
        chat_id="179033731",
        user_name="發發發",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner._is_user_authorized = lambda source: True
    runner._session_key_for_source = lambda source: "agent:main:qq_napcat:dm:179033731"
    runner._update_prompt_pending = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner.adapters = {}
    runner.pairing_store = MagicMock()
    return runner


def test_runtime_identity_query_matcher_catches_common_chinese_forms():
    assert gateway_run.GatewayRunner._is_runtime_identity_query("你什么模型现在")
    assert gateway_run.GatewayRunner._is_runtime_identity_query("你还是gpt5.4? 确定吗? 看看")
    assert not gateway_run.GatewayRunner._is_runtime_identity_query("今天天气怎么样")


@pytest.mark.asyncio
async def test_handle_message_answers_runtime_identity_without_llm(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: glm-5.1\n"
        "  provider: custom\n"
        "  base_url: https://wududu.edu.kg/v1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "glm-5.1")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "https://wududu.edu.kg/v1",
            "api_key": "test-key",
        },
    )

    runner = _make_runner()

    result = await runner._handle_message(_make_event("你什么模型现在"))

    assert "glm-5.1" in result
    assert "custom" in result
    assert "wududu.edu.kg/v1" in result


def test_runtime_identity_response_prefers_session_override(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: glm-5.1\n"
        "  provider: custom\n"
        "  base_url: https://wududu.edu.kg/v1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "glm-5.1")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "https://wududu.edu.kg/v1",
            "api_key": "test-key",
        },
    )

    runner = _make_runner()
    runner._session_model_overrides["agent:main:qq_napcat:dm:179033731"] = {
        "model": "gpt-4o",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
    }

    result = runner._format_runtime_identity_response(_make_source())

    assert "当前这个会话" in result
    assert "gpt-4o" in result
    assert "openai" in result

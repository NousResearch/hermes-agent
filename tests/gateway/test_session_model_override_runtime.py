"""Tests for gateway session-scoped model/runtime overrides."""

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


class _FreshAgent:
    """Captures the effective runtime used to build a new agent."""

    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "fresh",
            "messages": [],
            "api_calls": 1,
        }


class _CachedAgent:
    """Cached agent used to detect stale signature reuse."""

    def __init__(self):
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "cached",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._load_reasoning_config = lambda: {"enabled": True, "effort": "medium"}
    return runner


def _telegram_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


@pytest.fixture(autouse=True)
def _fake_agent_module(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _FreshAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def test_run_agent_uses_session_override_runtime_for_new_telegram_turn(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: MiniMax-M2.7\n"
        "  provider: minimax\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "minimax",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.minimax.io/anthropic",
            "api_key": "minimax-key",
            "credential_pool": None,
        },
    )

    _FreshAgent.last_init = None
    runner = _make_runner()
    runner._session_model_overrides["telegram:dm:12345"] = {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "codex-key",
    }

    result = asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=_telegram_source(),
            session_id="session-1",
            session_key="telegram:dm:12345",
        )
    )

    assert result["final_response"] == "fresh"
    assert _FreshAgent.last_init is not None
    assert _FreshAgent.last_init["model"] == "gpt-5.4"
    assert _FreshAgent.last_init["provider"] == "openai-codex"
    assert _FreshAgent.last_init["api_mode"] == "codex_responses"
    assert _FreshAgent.last_init["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert _FreshAgent.last_init["api_key"] == "codex-key"


def test_run_agent_cache_signature_uses_session_override_runtime(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: MiniMax-M2.7\n"
        "  provider: minimax\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "minimax",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.minimax.io/anthropic",
            "api_key": "minimax-key",
            "credential_pool": None,
        },
    )

    _FreshAgent.last_init = None
    runner = _make_runner()
    runner._session_model_overrides["telegram:dm:12345"] = {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "codex-key",
    }

    stale_runtime = {
        "provider": "minimax",
        "api_mode": "anthropic_messages",
        "base_url": "https://api.minimax.io/anthropic",
        "api_key": "minimax-key",
        "credential_pool": None,
    }
    stale_sig = runner._agent_config_signature("MiniMax-M2.7", stale_runtime, [], "")
    runner._agent_cache["telegram:dm:12345"] = (_CachedAgent(), stale_sig)

    result = asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=_telegram_source(),
            session_id="session-2",
            session_key="telegram:dm:12345",
        )
    )

    assert result["final_response"] == "fresh"
    assert _FreshAgent.last_init is not None
    assert _FreshAgent.last_init["provider"] == "openai-codex"

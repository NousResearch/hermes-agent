"""A gateway session's explicit reasoning effort survives a provider fallback mid-turn.

A session ``/reasoning <level>`` (messaging) or a local session's ``--reasoning`` launch level is
the session's own choice and outranks ``agent.reasoning_overrides`` and the global
``agent.reasoning_effort``. The turn hands that level to the agent; a fallback activated during the
turn must keep it, while a session without one still resolves the fallback model's config.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from run_agent import AIAgent

_FALLBACK = {"provider": "openai", "model": "fb-model"}


class _FallsBackAgent(AIAgent):
    """The real agent; its turn activates the configured fallback and reports the level it runs."""

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **_kwargs):
        client = MagicMock()
        client.base_url, client.api_key = "https://api.openai.com/v1", "fb-key"
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, _FALLBACK["model"])):
            assert self._try_activate_fallback() is True
        return {"final_response": f"{self.model}:{self.reasoning_config['effort']}", "messages": [], "api_calls": 1}


@pytest.fixture
def gateway_home(monkeypatch):
    home = get_hermes_home()
    (home / "config.yaml").write_text(
        "agent:\n  reasoning_effort: medium\n  reasoning_overrides:\n    fb-model: low\n"
        "fallback_providers:\n  - provider: openai\n    model: fb-model\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setattr(gateway_run, "_env_path", home / ".env")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "provider": "openrouter", "api_mode": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1", "api_key": "test-key"})
    monkeypatch.setattr("run_agent.AIAgent", _FallsBackAgent)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *_a, **_k: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *_a, **_k: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    return home


def _runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


def _source():
    return SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="testuser")


def _turn(runner, source):
    return asyncio.run(runner._run_agent(
        message="ping", context_prompt="", history=[], source=source, session_id="session-1",
        session_key=runner._session_key_for_source(source)))


def test_messaging_session_reasoning_survives_fallback(gateway_home):
    runner, source = _runner(), _source()
    asyncio.run(runner._handle_reasoning_command(MessageEvent(text="/reasoning high", source=source)))

    assert _turn(runner, source)["final_response"] == "fb-model:high"


def test_messaging_session_without_a_level_resolves_the_fallback_models_config(gateway_home):
    runner, source = _runner(), _source()

    assert _turn(runner, source)["final_response"] == "fb-model:low"


def test_local_launch_level_survives_fallback(gateway_home, tmp_path, monkeypatch):
    """``hermes chat --reasoning high`` runs a local session whose frozen level is its own."""
    from gateway.session_policy import build_policy
    cfg = {"agent": {"reasoning_effort": "medium", "reasoning_overrides": {"fb-model": "low"}}}
    # ignore_user_config keeps the turn off the owner's prefill store, which needs a session authority.
    policy = build_policy({"cwd": str(tmp_path), "model": "primary-model", "reasoning": "high",
                           "provider": "custom", "base_url": "http://127.0.0.1:1234/v1", "ignore_user_config": True}, cfg)
    monkeypatch.setattr("gateway.session_policy.policy_for_source", lambda _runner, _source: policy)
    runner = _runner()
    source = SessionSource(platform=Platform.LOCAL, chat_id="local-1", chat_type="dm", user_id="user-1")

    assert _turn(runner, source)["final_response"] == "fb-model:high"


def test_local_launch_level_is_the_sessions_own_until_reset(tmp_path):
    """``hermes chat --reasoning high`` freezes the level as the session's choice; ``/reasoning reset``
    gives it back to the profile, and a session launched without one never owned it."""
    from gateway.session_policy import build_policy, unpin_reasoning
    cfg = {"agent": {"reasoning_effort": "medium", "reasoning_overrides": {"fb-model": "low"}}}
    runner = object.__new__(gateway_run.GatewayRunner)

    launched = build_policy({"cwd": str(tmp_path), "model": "primary-model", "reasoning": "high"}, cfg)
    assert launched.current_reasoning_config(runner) == {"enabled": True, "effort": "high"}
    assert launched.reasoning_pinned(runner) is True

    assert build_policy({"cwd": str(tmp_path), "model": "primary-model"}, cfg).reasoning_pinned(runner) is False

    import dataclasses
    import json
    config = json.loads(launched.config_json)
    unpin_reasoning(config, cfg)
    reset = dataclasses.replace(launched, config_json=json.dumps(config))
    assert reset.reasoning_pinned(runner) is False

"""``hermes -z --reasoning X``: the run's explicit level survives a mid-run provider fallback.

``--reasoning`` outranks ``agent.reasoning_overrides`` and the global ``agent.reasoning_effort``; a
fallback activated during the run must keep it, and a run without it still resolves the fallback
model's config.
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

_CFG = {"model": {"default": "primary-model", "provider": "openrouter"},
        "agent": {"reasoning_effort": "medium", "reasoning_overrides": {"fb-model": "low"}},
        "fallback_providers": [{"provider": "openai", "model": "fb-model"}]}


class _FallsBackAgent(AIAgent):
    """The real agent; its run activates the configured fallback and reports the level it runs."""

    def run_conversation(self, user_message, conversation_history=None, **_kwargs):
        client = MagicMock()
        client.base_url, client.api_key = "https://api.openai.com/v1", "fb-key"
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "fb-model")):
            assert self._try_activate_fallback() is True
        return {"final_response": f"{self.model}:{self.reasoning_config['effort']}"}


@pytest.fixture
def oneshot(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CFG)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_kw: {
        "api_key": "k", "base_url": "https://openrouter.ai/api/v1", "provider": "openrouter",
        "api_mode": "chat_completions", "credential_pool": None})
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _p: [])
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None)
    monkeypatch.setattr("run_agent.AIAgent", _FallsBackAgent)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *_a, **_k: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *_a, **_k: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    return oneshot_mod._run_agent


def test_reasoning_flag_survives_fallback(oneshot):
    assert oneshot("ping", reasoning="high")[0] == "fb-model:high"


def test_without_the_flag_fallback_resolves_the_fallback_models_config(oneshot):
    assert oneshot("ping")[0] == "fb-model:low"


def test_an_unparsed_flag_is_not_the_runs_own_level(oneshot):
    assert oneshot("ping", reasoning="bogus")[0] == "fb-model:low"

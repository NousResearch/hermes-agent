"""Quota routing respects upstream turn vetoes and whole-turn transport dispatch."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import quota_routing as quota
from agent.turn_context import PRE_LLM_BLOCK_EXIT_REASON
from tests.agent.test_quota_preflight import fallback_client, make_agent
from tests.agent.test_quota_routing import snapshot


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    quota._cache.clear()
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **kw: 200_000)
    yield
    quota._cache.clear()


@pytest.mark.parametrize("mode", ["codex_responses", "codex_app_server"])
def test_vetoed_turn_does_not_probe_bind_compact_or_dispatch(mode):
    agent = make_agent()
    agent.api_mode = mode
    primary = agent.client
    verdict = {"action": "block", "message": "Blocked by policy"}
    with (
        patch("hermes_cli.plugins.invoke_hook", side_effect=lambda hook, **kw: [verdict] if hook == "pre_llm_call" else []),
        patch("agent.account_usage.fetch_account_usage", return_value=snapshot()) as probe,
        patch("agent.route_binding.bind_route_entry") as bind,
        patch("agent.turn_context_compaction.run_turn_start_compaction") as compact,
        patch("agent.conversation_loop._codex_app_server_turn") as dispatch,
        patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("withheld user prompt")
    assert result["final_response"] == "Blocked by policy"
    assert result["turn_exit_reason"] == PRE_LLM_BLOCK_EXIT_REASON
    assert result["api_calls"] == 0
    for boundary in (probe, bind, compact, dispatch):
        boundary.assert_not_called()
    primary.chat.completions.create.assert_not_called()
    primary.responses.stream.assert_not_called()
    assert agent.provider == "openai-codex"
    assert agent._fallback_index == 0


def test_whole_turn_transport_rechecks_quota_before_dispatch():
    from agent.conversation_loop import _turn_without_loop

    agent = make_agent()
    agent.api_mode = "codex_app_server"
    state = SimpleNamespace(active_system_prompt=agent._cached_system_prompt)
    fallback = fallback_client()
    with (
        patch("agent.account_usage.fetch_account_usage", return_value=snapshot()),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback, "fallback-model")),
        patch("agent.conversation_loop._codex_app_server_turn") as dispatch,
    ):
        assert _turn_without_loop(agent, state, "") is None
    dispatch.assert_not_called()
    assert agent.provider == "custom"
    assert agent.api_mode == "chat_completions"
    assert state.active_system_prompt == agent._cached_system_prompt


def test_context_route_synchronizes_prompt_and_publishes_bound_runtime():
    from agent.quota_preflight import prepare_quota_context

    agent = make_agent()
    old_prompt = agent._cached_system_prompt
    with (
        patch("agent.account_usage.fetch_account_usage", return_value=snapshot()),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client(), "fallback-model")),
        patch("agent.turn_context._publish_runtime_main") as publish,
    ):
        prompt = prepare_quota_context(agent, old_prompt, blocked=False)
    publish.assert_called_once_with(agent)
    assert agent.provider == "custom"
    assert prompt == agent._cached_system_prompt

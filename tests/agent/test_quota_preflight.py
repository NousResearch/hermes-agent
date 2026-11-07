"""Native main/child binding and retry integration, with synthetic provider I/O only."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from agent import quota_routing as quota
from agent.quota_preflight import candidate_quota, preflight_quota_route, retain_quota_fallback
from tests.agent.test_quota_routing import URL, snapshot

FALLBACK = {"provider": "custom", "model": "fallback-model", "base_url": "https://fallback.example/v1",
            "api_key": "fallback-synthetic-key"}


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    quota._cache.clear()
    # Context catalogs are another remote boundary, unrelated to quota evidence.
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **kw: 200_000)
    yield
    quota._cache.clear()


def make_agent(*, child=False, chain=None):
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="primary-synthetic-key", base_url="https://primary.example/v1", provider="custom",
                        model="primary-model", quiet_mode=True, skip_context_files=True, skip_memory=True,
                        parent_session_id="parent-test-session" if child else None,
                        fallback_model=chain if chain is not None else [FALLBACK])
    # Simulate the exact bound Codex runtime without touching real auth/pools.
    agent.provider = agent.requested_provider = "openai-codex"
    agent.base_url = URL
    agent.api_mode = "codex_responses"
    agent._credential_pool = None
    agent._primary_runtime.update(provider="openai-codex", requested_provider="openai-codex", base_url=URL,
                                  api_mode="codex_responses")
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client = MagicMock()
    return agent


def fallback_client():
    client = MagicMock(api_key=FALLBACK["api_key"], base_url=FALLBACK["base_url"])
    message = SimpleNamespace(content="fallback answered", tool_calls=None)
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")], model="fallback-model", usage=None)
    return client


@pytest.mark.parametrize("child", [False, True], ids=["main", "delegated-child"])
def test_first_inference_binds_entire_fallback_runtime_before_send(child, caplog):
    caplog.set_level("INFO", logger="agent.quota_preflight")
    agent = make_agent(child=child)
    primary = agent.client
    fallback = fallback_client()
    from agent.turn_context_compaction import run_turn_start_compaction
    def compaction_on_selected_route(bound_agent, **kwargs):
        assert bound_agent.provider == "custom"
        assert bound_agent.context_compressor.model == "fallback-model"
        return run_turn_start_compaction(bound_agent, **kwargs)
    # Exercise native fetch/parse + evaluator + chain binder + actual conversation request.
    payload = {"rate_limit": {"primary_window": {"used_percent": 100,
        "reset_at": (datetime.now(UTC) + timedelta(hours=1)).timestamp()}}}
    with (
        patch("agent.account_usage._get_json", return_value=payload) as usage,
        patch("agent.turn_context_compaction.run_turn_start_compaction", side_effect=compaction_on_selected_route) as compaction,
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback, "fallback-model")),
        patch.object(agent, "_create_openai_client", return_value=fallback),
        patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("synthetic quota routing probe")
    assert result["final_response"] == "fallback answered"
    compaction.assert_called_once()
    primary.chat.completions.create.assert_not_called()
    primary.responses.stream.assert_not_called()
    assert usage.call_count == 1
    assert agent.model == "fallback-model" and agent.provider == "custom"
    assert agent.api_key == FALLBACK["api_key"]
    assert agent.base_url.rstrip("/") == FALLBACK["base_url"]
    assert agent.api_mode == "chat_completions"
    assert agent.context_compressor.model == "fallback-model"
    assert agent.context_compressor.provider == "custom"
    assert agent._quota_fallback_active is True
    assert agent._credential_pool is None
    sent = fallback.chat.completions.create.call_args.kwargs
    assert sent["model"] == "fallback-model"
    assert "reason=account_window_exhausted" in caplog.text
    assert "requested=openai-codex/primary-model actual=custom/fallback-model" in caplog.text
    assert "primary-synthetic-key" not in caplog.text
    assert "fallback-synthetic-key" not in caplog.text


@pytest.mark.parametrize("state", ["unknown", "fetch_failed", "available"])
def test_unknown_failed_or_remaining_preserves_configured_route(state):
    agent = make_agent()
    with patch("agent.quota_preflight.current_quota", return_value=quota.QuotaVerdict(state, "test")), \
         patch("agent.chat_completion_helpers.try_activate_fallback") as switch:
        assert preflight_quota_route(agent) is False
    switch.assert_not_called()
    assert agent.model == "primary-model"


def test_explicit_empty_chain_disables_preflight_without_usage_request():
    agent = make_agent(chain=[])
    with patch("agent.account_usage.fetch_account_usage") as fetch:
        assert preflight_quota_route(agent) is False
    fetch.assert_not_called()


def test_same_account_sibling_is_skipped_and_no_eligible_route_keeps_primary():
    agent = make_agent(chain=[{"provider": "openai-codex", "model": "sibling-model", "api_key": "primary-synthetic-key"}])
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()), \
         patch("agent.route_binding.bind_route_entry") as bind:
        assert preflight_quota_route(agent) is False
    bind.assert_not_called()
    assert agent.model == "primary-model" and agent.provider == "openai-codex"


def test_same_group_skip_continues_to_authorized_next_route():
    agent = make_agent(chain=[{"provider": "openai-codex", "model": "sibling-model", "api_key": "primary-synthetic-key"}, FALLBACK])
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client(), "fallback-model")) as resolve:
        assert preflight_quota_route(agent) is True
    resolve.assert_called_once()
    assert agent.model == "fallback-model"


@pytest.mark.parametrize("same_account", [True, False])
def test_pool_is_not_blanket_exhausted_but_duplicate_account_tokens_share_quota(same_account, monkeypatch):
    agent = make_agent()
    monkeypatch.setattr("agent.credential_pool._codex_principal_identity",
                        lambda token: ("workspace", "member" if same_account or token == agent.api_key else "other"))
    agent._credential_pool = SimpleNamespace(entries=lambda: [
        SimpleNamespace(runtime_api_key=agent.api_key, runtime_base_url=URL),
        SimpleNamespace(runtime_api_key="second-token", runtime_base_url=URL)])
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()):
        verdict = candidate_quota(agent, {"provider": "openai-codex", "model": "sibling"})
    assert verdict.state == ("exhausted" if same_account else "unknown")


def test_failed_fallback_resolution_continues_native_chain(caplog):
    broken = {**FALLBACK, "model": "broken-model"}
    agent = make_agent(chain=[broken, FALLBACK])
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()), \
         patch("agent.auxiliary_client.resolve_provider_client", side_effect=[RuntimeError("secret-do-not-log"), (fallback_client(), "fallback-model")]):
        assert preflight_quota_route(agent) is True
    assert agent.model == "fallback-model"
    assert "secret-do-not-log" not in caplog.text


@pytest.mark.parametrize("state,retain", [("exhausted", True), ("unknown", False), ("fetch_failed", False), ("available", False)])
def test_retention_requires_reliable_exhaustion(state, retain):
    agent = SimpleNamespace(_quota_fallback_active=True)
    with patch("agent.quota_preflight.account_quota", return_value=quota.QuotaVerdict(state, "test")):
        assert retain_quota_fallback(agent, {"provider": "openai-codex", "base_url": URL, "api_key": "key"}) is retain


def test_existing_reactive_fallback_invalidates_cached_usage():
    from agent.quota_preflight import invalidate_on_quota_error
    from agent.error_classifier import FailoverReason
    agent = SimpleNamespace(provider="openai-codex", base_url=URL, api_key="key")
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot(10)) as fetch:
        assert quota.account_quota(agent.provider, URL, agent.api_key).state == "available"
        invalidate_on_quota_error(agent, FailoverReason.rate_limit)
        quota.account_quota(agent.provider, URL, agent.api_key)
    assert fetch.call_count == 2


def test_native_restore_stays_stable_then_restores_complete_primary_runtime():
    agent = make_agent()
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client(), "fallback-model")):
        assert preflight_quota_route(agent) is True
    # Reliable exhaustion retains the fallback; a failed refresh defers to native gates.
    assert agent._restore_primary_runtime() is False
    quota.invalidate_quota("openai-codex", URL, "primary-synthetic-key")
    with patch("agent.account_usage.fetch_account_usage", return_value=None), \
         patch("agent.agent_runtime_helpers._rebuild_primary_client") as rebuild:
        assert agent._restore_primary_runtime() is True
    rebuild.assert_called_once()
    assert agent.provider == "openai-codex" and agent.model == "primary-model"
    assert agent.api_key == "primary-synthetic-key"
    assert agent.api_mode == "codex_responses"
    assert agent._fallback_index == 0
    assert agent._quota_fallback_active is False


def test_iteration_limit_summary_preflights_before_building_its_transport():
    from agent.chat_completion_helpers import handle_max_iterations
    agent = make_agent()
    fallback = fallback_client()
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback, "fallback-model")), \
         patch.object(agent, "_create_openai_client", return_value=fallback):
        text = handle_max_iterations(agent, [{"role": "assistant", "content": "worked"}], 1)
    assert text == "fallback answered"
    assert fallback.chat.completions.create.call_args.kwargs["model"] == "fallback-model"
    assert agent.provider == "custom"


def test_retry_preflight_restarts_before_building_old_route_request():
    from agent.conversation_loop import _run_api_retry_loop
    from agent.turn_retry_state import TurnRetryState
    agent = SimpleNamespace(_cached_system_prompt="new prompt", ephemeral_system_prompt="")
    s = SimpleNamespace(retry_count=2, max_retries=3, compression_attempts=2, api_messages=[{"role": "system", "content": "old prompt"}],
                        active_system_prompt="old prompt", _retry=TurnRetryState())
    with patch("agent.quota_preflight.preflight_quota_route", return_value=True), \
         patch("agent.conversation_loop._run_phase") as phase:
        assert _run_api_retry_loop(agent, s) is None
    phase.assert_not_called()
    assert s._retry.restart_with_rebuilt_messages is True
    assert s.api_messages[0]["content"] == "new prompt"
    assert s.active_system_prompt == "new prompt"
    assert s.retry_count == s.compression_attempts == 0

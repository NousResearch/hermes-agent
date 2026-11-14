"""Profile opt-in, real SDK errors and real pool entries through the turn error handler."""
from copy import deepcopy
import time
from unittest.mock import patch

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from agent.route_verdicts import reset_route_verdicts
from agent.turn_api_error import handle_api_error
from agent.turn_retry_state import TurnRetryState
from providers.base import ProviderProfile
from tests.agent.test_classified_replay_recovery import BASE_URL, MODEL, _history, _item, _wire_error


class _ReplayProfile(ProviderProfile):
    def refresh_runtime_credentials(self, *, reason):
        return None


@pytest.fixture
def replay_agent(request):
    import providers
    from run_agent import AIAgent

    providers.list_providers()
    profile = _ReplayProfile(
        name="replay-test", hostnames=("replay.example",), api_mode="codex_responses",
        classify_api_error=lambda error, **kwargs: {
            "reason": "invalid_encrypted_content", "retryable": False,
            "should_rotate_credential": False, "should_fallback": False,
        },
    )
    providers.register_provider(profile)

    def cleanup():
        providers._REGISTRY.pop(profile.name, None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(cleanup)
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(api_key="offline-test-key", base_url=BASE_URL, provider="custom", model=MODEL,
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.provider, agent.api_mode = profile.name, "codex_responses"
    pool = CredentialPool(profile.name, [
        PooledCredential(provider=profile.name, id=f"entry-{i}", label=f"entry-{i}",
                         auth_type="api_key", priority=i, source="manual", access_token=f"offline-{i}")
        for i in range(2)
    ])
    agent._credential_pool = pool
    agent._credential_pool_entry_id = "entry-0"
    agent.api_key = "offline-0"
    return agent, pool


def _handle(agent, retry, history, request_history, error):
    return handle_api_error(
        agent, api_error=error, _retry=retry, thinking_spinner=None, messages=history,
        api_messages=request_history, api_kwargs={}, system_message="system", active_system_prompt="system",
        conversation_history=[], approx_tokens=10, retry_count=0, max_retries=3,
        compression_attempts=0, max_compression_attempts=1, api_call_count=1,
        api_request_id="offline-request", api_start_time=time.time(), effective_task_id="task", turn_id="turn",
    )


@pytest.mark.parametrize("kind", ["reasoning", "compaction"])
def test_route_visits_rearm_bounded_repair_without_spending_real_pool_entries(replay_agent, kind):
    agent, pool = replay_agent
    retry = TurnRetryState()
    with (patch.object(agent, "_recover_with_credential_pool", wraps=agent._recover_with_credential_pool) as recover_pool,
          patch.object(pool, "_persist", side_effect=AssertionError("unexpected pool write")),
          patch.object(agent, "_try_refresh_provider_runtime_credentials", side_effect=AssertionError("unexpected refresh")),
          patch.object(agent, "_try_activate_fallback", side_effect=AssertionError("unexpected fallback")),
          patch.object(agent, "_persist_session")):
        # Returning to A is a new VISIT, not an already-spent session-wide flag.
        for route in (BASE_URL, "https://route-b.example/v1", BASE_URL):
            agent.base_url = route
            reset_route_verdicts(agent)
            history = _history([_item("sent", type=kind)])
            request_history = deepcopy(history)
            transport = agent._get_transport()
            replacement = [{"type": kind, "encrypted_content": "sent"}] if kind == "compaction" else None
            error, body = _wire_error(transport, request_history, replacement=replacement, base_url=route)
            assert any(i.get("encrypted_content") == "sent" for i in body["input"])
            assert _handle(agent, retry, history, request_history, error).action == "continue"
            assert "codex_reasoning_items" not in history[1]
            # A second rejection is terminal, even with no eligible items left.
            verdict = _handle(agent, retry, history, request_history, error)
            assert verdict.action == "return"
            assert verdict.result["failure_reason"] == "invalid_encrypted_content"
            assert verdict.result["failure_retryable"] is False
            assert agent._codex_reasoning_replay_enabled is True
        recover_pool.assert_not_called()
    assert [entry.last_status for entry in pool.entries()] == [None, None]


@pytest.mark.parametrize("pooled", [True, False])
def test_nonoptin_401_keeps_pool_or_singleton_auth_recovery(replay_agent, pooled):
    from providers import get_provider_profile
    from agent.credential_pool import STATUS_EXHAUSTED

    agent, pool = replay_agent
    get_provider_profile(agent.provider).classify_api_error = None
    if not pooled:
        agent._credential_pool = None
    history = _history([_item("valid")])
    request_history = deepcopy(history)
    error, _ = _wire_error(agent._get_transport(), request_history)
    with (patch.object(pool, "_persist"),
          patch.object(pool, "try_refresh_matching", return_value=None),
          patch.object(agent, "_swap_credential", return_value=True) as swap,
          patch.object(agent, "_try_refresh_provider_runtime_credentials", return_value=True) as refresh):
        assert _handle(agent, TurnRetryState(), history, request_history, error).action == "continue"
    if pooled:
        swap.assert_called_once()
        refresh.assert_not_called()
        assert pool.entries()[0].last_status == STATUS_EXHAUSTED
    else:
        refresh.assert_called_once_with(reason="unauthorized")
        swap.assert_not_called()
    assert history[1]["codex_reasoning_items"] == [_item("valid")]

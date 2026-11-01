"""Local key helpers fail once per route, without spending provider recovery budgets."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent.command_token_source import CommandTokenError
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import recover_after_classification
from agent.turn_retry_state import TurnRetryState
from run_agent import AIAgent


def _local_error(wrapped=False):
    error = CommandTokenError("synthetic helper unavailable")
    if wrapped:
        outer = openai.APIConnectionError(request=httpx.Request("POST", "https://synthetic.invalid"))
        outer.__cause__ = error
        return outer
    return error


@pytest.mark.parametrize("wrapped", [False, True])
def test_local_source_classification(wrapped):
    classified = classify_api_error(_local_error(wrapped), provider="openai", model="synthetic")
    assert classified.is_auth
    assert classified.retryable is False
    assert classified.should_fallback is True
    assert classified.should_rotate_credential is False
    assert classified.status_code is None
    assert classified.error_context.get("local_auth_source") is True


def test_local_source_skips_credential_recovery():
    error = _local_error()
    agent = MagicMock()
    agent._recover_with_credential_pool.return_value = (True, True)
    result = recover_after_classification(
        agent, error, classify_api_error(error), TurnRetryState(),
        status_code=None, error_context={}, messages=[], api_messages=[],
    )
    assert result == (False, False)
    agent._recover_with_credential_pool.assert_not_called()


def _make_agent(chain):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="synthetic-key", provider="openai", model="synthetic-primary",
            base_url="https://synthetic.invalid/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True, fallback_model=chain,
        )
    agent.client = MagicMock()
    return agent


@pytest.mark.parametrize("terminal", [False, True])
def test_real_loop_walks_local_failures_after_provider_auth(terminal):
    chain = [{"provider": "openai", "model": f"synthetic-{i}",
              "base_url": "https://synthetic.invalid/v1"} for i in range(3)]
    agent = _make_agent(chain)
    routes = []

    def call(_kwargs):
        routes.append(agent.model)
        if len(routes) == 1:
            error = RuntimeError("synthetic provider refusal")
            error.status_code = 401
            raise error
        if terminal or len(routes) < 4:
            raise _local_error(wrapped=True)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Recovered", tool_calls=None), finish_reason="stop",
        )], model=agent.model, usage=None)

    client = MagicMock()
    client.base_url = "https://synthetic.invalid/v1"
    client.api_key = "synthetic-key"
    client._custom_headers = None
    client.default_headers = None
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_recover_with_credential_pool", return_value=(False, False)) as pool,
        patch("agent.turn_recovery._refresh_credentials_after_401", return_value=False) as refresh,
        patch("agent.turn_api_error.interruptible_backoff_sleep") as backoff,
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=lambda *args, **kw: (client, kw["model"])),
        patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("hello")
    assert routes == ["synthetic-primary", "synthetic-0", "synthetic-1", "synthetic-2"]
    backoff.assert_not_called()
    assert pool.call_count == 1
    assert refresh.call_count == 1
    if terminal:
        assert result["failed"] is True
        assert result["failure_retryable"] is False
        assert "key_cmd" in result["final_response"]
    else:
        assert result["completed"] is True
        assert result["final_response"] == "Recovered"


def test_no_chain_local_source_is_actionable_terminal():
    agent = _make_agent([])
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=_local_error()) as call,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_recover_with_credential_pool", return_value=(False, False)) as pool,
        patch("agent.turn_api_error.interruptible_backoff_sleep") as backoff,
    ):
        result = agent.run_conversation("hello")
    call.assert_called_once()
    pool.assert_not_called()
    backoff.assert_not_called()
    assert result["failed"] is True
    assert result["failure_retryable"] is False
    assert "key_cmd" in result["final_response"]


def test_provider_transport_and_rate_limit_hints_unchanged():
    error = RuntimeError("rate limit exceeded")
    error.status_code = 429
    assert classify_api_error(error).should_rotate_credential is True
    classified = classify_api_error(openai.APIConnectionError(
        request=httpx.Request("POST", "https://synthetic.invalid"),
    ))
    assert classified.reason == FailoverReason.timeout
    assert classified.retryable is True

"""Regression coverage for terminal Z.AI quota fallback."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason
from run_agent import AIAgent


class _ZaiQuotaError(Exception):
    status_code = 429


def _response(content="fallback selected"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        output=[
            SimpleNamespace(
                type="message",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=content)],
            )
        ],
        output_text=content,
        status="completed",
        model="copilot/model",
    )


def _agent(fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="zai-key",
            base_url="https://api.z.ai/api/paas/v4",
            provider="zai",
            api_mode="chat_completions",
            model="glm-5.2",
            fallback_model=fallback_model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    return agent


def test_zai_terminal_quota_rotates_then_selects_copilot_fallback():
    agent = _agent({"provider": "copilot", "model": "gpt-5"})
    quota_error = _ZaiQuotaError("Usage limit reached for 5 hour.")
    relay_attempts = []

    def execute(request, callback, **kwargs):
        relay_attempts.append((request, kwargs))
        if len(relay_attempts) == 1:
            raise quota_error
        return _response()

    fallback_client = MagicMock()
    fallback_client.api_key = "copilot-key"
    fallback_client.base_url = "https://api.githubcopilot.com"
    with (
        patch.object(agent, "_recover_with_credential_pool", return_value=(False, False)) as rotate,
        patch("run_agent._pool_may_recover_from_rate_limit", return_value=True),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "gpt-5"),
        ) as resolve,
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("agent.relay_llm.execute", side_effect=execute),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    rotate.assert_called_once()
    resolve.assert_called_once()
    assert agent._fallback_index == 1
    assert agent.provider == "copilot"
    assert agent.model == "gpt-5"
    assert agent.client is fallback_client
    assert len(relay_attempts) == 2
    assert relay_attempts[0][1]["name"] == "zai"
    assert relay_attempts[1][1]["name"] == "copilot"
    assert result["completed"] is True
    assert result["final_response"] == "fallback selected"


def test_zai_terminal_quota_without_fallback_remains_terminal():
    agent = _agent()
    quota_error = _ZaiQuotaError("Usage limit reached for 5 hour.")

    with (
        patch.object(agent, "_recover_with_credential_pool", return_value=(False, False)),
        patch.object(agent, "_try_activate_fallback", return_value=False) as activate,
        patch("agent.relay_llm.execute", side_effect=quota_error),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    activate.assert_not_called()
    assert result["failed"] is True


def test_successful_credential_rotation_skips_fallback():
    agent = _agent({"provider": "copilot", "model": "gpt-5"})
    quota_error = _ZaiQuotaError("Usage limit reached for 5 hour.")
    attempts = []

    def execute(request, callback, **kwargs):
        attempts.append(kwargs["name"])
        if len(attempts) == 1:
            raise quota_error
        return _response("recovered by rotated credential")

    with (
        patch.object(agent, "_recover_with_credential_pool", side_effect=[(True, False)]) as recover,
        patch.object(agent, "_try_activate_fallback") as activate,
        patch("agent.relay_llm.execute", side_effect=execute),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    recover.assert_called_once()
    activate.assert_not_called()
    assert attempts == ["zai", "zai"]
    assert result["completed"] is True
    assert result["final_response"] == "recovered by rotated credential"


def test_transient_rate_limit_stays_on_pool_recovery_path():
    agent = _agent({"provider": "copilot", "model": "gpt-5"})
    rate_limit = _ZaiQuotaError("rate limit exceeded; retry later")
    attempts = []

    def execute(request, callback, **kwargs):
        attempts.append(kwargs["name"])
        if len(attempts) == 1:
            raise rate_limit
        return _response("recovered after transient limit")

    with (
        patch.object(agent, "_recover_with_credential_pool", side_effect=[(True, False)]) as recover,
        patch("run_agent._pool_may_recover_from_rate_limit", return_value=True),
        patch.object(agent, "_try_activate_fallback") as activate,
        patch("agent.relay_llm.execute", side_effect=execute),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    recover.assert_called_once()
    activate.assert_not_called()
    assert attempts == ["zai", "zai"]
    assert result["completed"] is True


def test_fallback_chain_skips_unusable_entry_and_activates_second():
    agent = _agent(
        [
            {"provider": "unconfigured", "model": "missing"},
            {"provider": "copilot", "model": "gpt-5"},
        ]
    )
    fallback_client = MagicMock()
    fallback_client.api_key = "copilot-key"
    fallback_client.base_url = "https://api.githubcopilot.com"

    with (
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[(None, None), (fallback_client, "gpt-5")],
        ) as resolve,
        patch("agent.credential_pool.load_pool", return_value=None),
    ):
        assert agent._try_activate_fallback(reason=FailoverReason.billing) is True

    assert [call.args[0] for call in resolve.call_args_list] == [
        "unconfigured",
        "copilot",
    ]
    assert agent._fallback_index == 2
    assert agent.provider == "copilot"
    assert agent.model == "gpt-5"
    assert agent.client is fallback_client


def test_failed_fallback_activation_does_not_start_duplicate_outer_traversal():
    agent = _agent({"provider": "unconfigured", "model": "missing"})
    quota_error = _ZaiQuotaError("Usage limit reached for 5 hour.")

    with (
        patch.object(agent, "_recover_with_credential_pool", return_value=(False, False)),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)) as resolve,
        patch("agent.relay_llm.execute", side_effect=quota_error) as execute,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    # The candidate is attempted once; the failed activation must not fall
    # through to the generic non-retryable branch and traverse the chain again.
    resolve.assert_called_once()
    execute.assert_called_once()
    assert agent._fallback_index == 1
    assert result["failed"] is True

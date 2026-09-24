"""Behavior contracts for opt-in same-runtime mid-turn routing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _config():
    return {
        "model_router": {
            "enabled": True,
            "minimum_confidence": 0.9,
            "message_max_chars": 256,
            "routes": {
                "routine": {
                    "provider": "same-provider",
                    "model": "fast-model",
                    "reasoning_effort": "low",
                    "description": "bounded completed tool follow-up",
                },
                "exception": {
                    "provider": "same-provider",
                    "model": "strong-model",
                    "reasoning_effort": "high",
                    "description": "planning, uncertainty, or exceptions",
                },
            },
            "mid_turn": {
                "enabled": True,
                "tool_outcome_max_chars": 128,
                "strong_route": "exception",
            },
        }
    }


def _agent():
    return SimpleNamespace(
        model="base-model",
        provider="same-provider",
        requested_provider="same-provider",
        base_url="https://same.invalid/v1",
        api_mode="chat_completions",
        api_key="base-key",
        reasoning_config={"enabled": True, "effort": "medium"},
        request_overrides={"extra_body": {"base": True}},
    )


def _runtime(provider, model):
    return {
        "provider": provider,
        "requested_provider": provider,
        "base_url": "https://same.invalid/v1",
        "api_key": "base-key",
        "api_mode": "chat_completions",
        "request_overrides": {"extra_body": {"model": model}},
    }


def test_completed_tool_round_applies_compatible_route_and_restores_request_state():
    from agent.midturn_model_router import maybe_apply_midturn_route, restore_midturn_route

    agent = _agent()
    messages = [
        {"role": "user", "content": "Inspect the data."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "inspection complete"},
    ]

    applied = maybe_apply_midturn_route(
        agent,
        messages=messages,
        original_user_message="Inspect the data.",
        config_loader=_config,
        runtime_resolver=_runtime,
        controller=lambda *_args: {"choice": "routine", "confidence": 1.0},
    )

    assert applied is True
    assert agent.model == "fast-model"
    assert agent.reasoning_config == {"enabled": True, "effort": "low"}
    assert agent.request_overrides == {"extra_body": {"model": "fast-model"}}

    restore_midturn_route(agent)

    assert agent.model == "base-model"
    assert agent.reasoning_config == {"enabled": True, "effort": "medium"}
    assert agent.request_overrides == {"extra_body": {"base": True}}


def test_route_with_different_credential_runtime_is_rejected_without_mutation():
    from agent.midturn_model_router import maybe_apply_midturn_route

    agent = _agent()
    primary_runtime = {"provider": "same-provider", "api_key": "base-key"}
    fallback_chain = [{"provider": "fallback", "model": "fallback-model"}]
    agent._primary_runtime = primary_runtime
    agent._fallback_chain = fallback_chain

    def different_credential_runtime(provider, model):
        runtime = _runtime(provider, model)
        runtime["api_key"] = "different-credential"
        return runtime

    applied = maybe_apply_midturn_route(
        agent,
        messages=[{"role": "tool", "content": "completed inspection"}],
        original_user_message="Inspect the data.",
        config_loader=_config,
        runtime_resolver=different_credential_runtime,
        controller=lambda *_args: {"choice": "routine", "confidence": 1.0},
    )

    assert applied is False
    assert agent.model == "base-model"
    assert agent.reasoning_config == {"enabled": True, "effort": "medium"}
    assert agent.request_overrides == {"extra_body": {"base": True}}
    assert agent._primary_runtime is primary_runtime
    assert agent._fallback_chain is fallback_chain


def test_route_with_the_active_private_credential_pool_is_compatible():
    """Runtime pools are intentionally private AIAgent state, but still part of identity."""
    from agent.midturn_model_router import maybe_apply_midturn_route

    agent = _agent()
    pool = object()
    agent._credential_pool = pool

    def pooled_runtime(provider, model):
        runtime = _runtime(provider, model)
        runtime["credential_pool"] = pool
        return runtime

    applied = maybe_apply_midturn_route(
        agent,
        messages=[{"role": "tool", "content": "completed inspection"}],
        original_user_message="Inspect the data.",
        config_loader=_config,
        runtime_resolver=pooled_runtime,
        controller=lambda *_args: {"choice": "routine", "confidence": 1.0},
    )

    assert applied is True
    assert agent.model == "fast-model"


def test_independently_loaded_same_credential_pool_allows_midturn_route():
    """Real pool reloads create new objects; compare credential identity, not object identity."""
    from agent.credential_pool import CredentialPool, PooledCredential
    from agent.midturn_model_router import maybe_apply_midturn_route

    def pool(credential_id: str, token: str = "base-key") -> CredentialPool:
        return CredentialPool("same-provider", [PooledCredential(
            provider="same-provider", id=credential_id, label="test", auth_type="api_key",
            priority=0, source="manual", access_token=token,
        )])

    agent = _agent()
    agent._credential_pool = pool("stable-credential")
    assert agent._credential_pool is not pool("stable-credential")

    def reloaded_runtime(provider, model):
        runtime = _runtime(provider, model)
        runtime["credential_pool"] = pool("stable-credential")
        return runtime

    applied = maybe_apply_midturn_route(
        agent, messages=[{"role": "tool", "content": "completed inspection"}],
        original_user_message="Inspect the data.", config_loader=_config,
        runtime_resolver=reloaded_runtime,
        controller=lambda *_args: {"choice": "routine", "confidence": 1.0},
    )
    assert applied is True
    assert agent.model == "fast-model"


def test_different_pool_credential_identity_is_rejected():
    from agent.credential_pool import CredentialPool, PooledCredential
    from agent.midturn_model_router import maybe_apply_midturn_route

    def pool(credential_id: str) -> CredentialPool:
        return CredentialPool("same-provider", [PooledCredential(
            provider="same-provider", id=credential_id, label="test", auth_type="api_key",
            priority=0, source="manual", access_token="base-key",
        )])

    agent = _agent()
    agent._credential_pool = pool("original-credential")

    def different_pool(provider, model):
        runtime = _runtime(provider, model)
        runtime["credential_pool"] = pool("other-credential")
        return runtime

    assert maybe_apply_midturn_route(
        agent, messages=[{"role": "tool", "content": "completed inspection"}],
        original_user_message="Inspect the data.", config_loader=_config,
        runtime_resolver=different_pool,
        controller=lambda *_args: {"choice": "routine", "confidence": 1.0},
    ) is False
    assert agent.model == "base-model"


def test_midturn_route_400_does_not_activate_cross_provider_fallback_and_restores_runtime():
    """A temporary route must not turn one rejected request into a persistent provider hop."""
    from agent.error_classifier import classify_api_error
    from agent.turn_api_error import settle_unrecovered_error

    class Error400(Exception):
        status_code = 400
        response = None

        def __init__(self):
            super().__init__("Unsupported parameter: 'max_tokens'")
            self.body = {"error": {"message": str(self), "type": "invalid_request_error"}}

    agent = _agent()
    agent.model = "fast-model"
    agent._midturn_route_restore = {
        "model": "base-model",
        "reasoning_config": {"enabled": True, "effort": "medium"},
        "request_overrides": {"extra_body": {"base": True}},
    }
    agent._fallback_chain = [{"provider": "other", "model": "other-model"}]
    agent.activated = []
    agent._has_pending_fallback = lambda: True
    agent._try_activate_fallback = lambda **_kwargs: agent.activated.append(True) or True
    agent._summarize_api_error = lambda error: str(error)
    agent._try_recover_primary_transport = lambda *_args, **_kwargs: False
    error = Error400()
    classified = classify_api_error(error, provider=agent.provider, model=agent.model)
    retry = SimpleNamespace(
        copilot_stale_cred_retry_attempted=False,
        primary_recovery_attempted=False,
        restart_with_redirected_messages=False,
    )

    with (
        patch("agent.conversation_loop._is_copilot_provider", lambda _agent: False),
        patch("agent.turn_api_error.nonretryable_client_error_result", lambda *_args, **_kwargs: {"failed": True}),
    ):
        verdict = settle_unrecovered_error(
            agent, api_error=error, classified=classified, _retry=retry, status_code=400,
            error_msg=str(error), is_context_length_error=False, is_rate_limited=False,
            _is_zai_coding_overload=False, _provider=agent.provider, _base=agent.base_url,
            _model=agent.model, messages=[], api_messages=[], api_kwargs={}, active_system_prompt="",
            conversation_history=None, approx_tokens=0, retry_count=0, max_retries=3,
            compression_attempts=0, api_call_count=1,
        )

    assert verdict.action == "return"
    assert agent.activated == []
    assert agent.model == "base-model"
    assert not hasattr(agent, "_midturn_route_restore")


def test_active_midturn_route_blocks_codex_app_server_fallback_path():
    """The specialized Codex path must obey the same request-local fallback boundary."""
    from agent.turn_recovery import activate_codex_app_server_fallback

    agent = _agent()
    agent._midturn_route_restore = {
        "model": "base-model", "reasoning_config": None, "request_overrides": {},
    }
    activated = []
    agent._has_pending_fallback = lambda: True
    agent._try_activate_fallback = lambda **_kwargs: activated.append(True) or True

    assert activate_codex_app_server_fallback(agent, {"error": "HTTP 429 rate limited"}) is False
    assert activated == []


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent._disable_streaming = True
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.valid_tool_names = {"terminal"}
        return agent


def test_integration_routes_only_after_completed_tool_round_and_restores_cached_agent(loop_agent, monkeypatch):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    loop_agent._use_prompt_caching = True
    prompt_before = loop_agent._cached_system_prompt
    tools_before = loop_agent.tools
    applied_after_tool = []

    def apply_route(agent, *, messages, original_user_message):
        assert messages[-1]["role"] == "tool"
        assert original_user_message == "Inspect the data."
        agent._midturn_route_restore = {
            "model": agent.model,
            "reasoning_config": dict(agent.reasoning_config or {}),
            "request_overrides": dict(agent.request_overrides or {}),
        }
        agent.model = "fast-model"
        agent.reasoning_config = {"enabled": True, "effort": "low"}
        agent.request_overrides = {"extra_body": {"route": "routine"}}
        applied_after_tool.append(True)
        return True

    monkeypatch.setattr("agent.turn_tool_round.maybe_apply_midturn_route", apply_route)
    loop_agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("terminal", '{"command":"true"}', "call-1")],
        ),
        _mock_response(content="Inspection complete.", finish_reason="stop"),
    ]

    def execute_tools(assistant_message, messages, *_args):
        tool_call = assistant_message.tool_calls[0]
        messages.append({
            "role": "tool", "tool_call_id": tool_call.id,
            "name": tool_call.function.name, "content": "completed inspection",
        })

    loop_agent._execute_tool_calls = execute_tools
    with (
        patch.object(loop_agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        result = loop_agent.run_conversation("Inspect the data.")

    assert applied_after_tool == [True]
    assert loop_agent.client.chat.completions.create.call_args_list[1].kwargs["model"] == "fast-model"
    assert loop_agent.client.chat.completions.create.call_args_list[1].kwargs["extra_body"] == {
        "route": "routine",
    }
    assert loop_agent._cached_system_prompt == prompt_before
    assert loop_agent.tools is tools_before
    assert [message["role"] for message in result["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert loop_agent.model != "fast-model"


def test_integration_does_not_route_after_failed_tool_result_persistence(loop_agent, monkeypatch):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    route_calls = []
    monkeypatch.setattr(
        "agent.turn_tool_round.maybe_apply_midturn_route",
        lambda *_args, **_kwargs: route_calls.append(True),
    )
    loop_agent.client.chat.completions.create.return_value = _mock_response(
        content="", finish_reason="tool_calls",
        tool_calls=[_mock_tool_call("terminal", '{"command":"true"}', "call-1")],
    )

    def failed_tool_execution(*_args):
        loop_agent._incremental_persistence_failed = True

    loop_agent._execute_tool_calls = failed_tool_execution
    with (
        patch.object(loop_agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        result = loop_agent.run_conversation("Inspect the data.")

    assert route_calls == []
    assert result["turn_exit_reason"] == "session_persistence_failed"

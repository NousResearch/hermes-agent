import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.claude_agent_runtime import ClaudeProjection, RuntimeFailure
from agent.error_classifier import FailoverReason
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
from run_agent import AIAgent


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "kanban_complete",
                "description": "complete",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _agent(fallback_model=None, *, runtime="claude_agent_sdk", provider="anthropic"):
    with (
        patch("run_agent.get_tool_definitions", return_value=_tools()),
        patch("run_agent.check_toolset_requirements", return_value={}),
    ):
        return AIAgent(
            provider=provider,
            model="claude-sonnet-4-6",
            runtime=runtime,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )


@pytest.fixture(autouse=True)
def _kanban_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "BUILD-392")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(
        "agent.external_runtime.prepare_claude_agent_sdk_runtime",
        lambda agent: setattr(
            agent,
            "_claude_max_attestation",
            SimpleNamespace(included_usage=True, account_key="test-account"),
        ),
    )


def test_claude_runtime_completes_without_duplicating_user_turn():
    agent = _agent()
    agent._claude_max_attestation = SimpleNamespace(included_usage=True)
    projection = ClaudeProjection(
        messages=[{"role": "assistant", "content": "done"}],
        final_text="done",
        session_id="sdk-session",
        usage={
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 3,
        },
    )

    with patch(
        "agent.external_runtime.run_claude_agent_sdk_attempt",
        return_value=projection,
    ) as attempt:
        result = agent.run_conversation("do the card")

    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert result["api_calls"] == 1
    assert result["cost_status"] == "included"
    assert result["estimated_cost_usd"] is None
    assert agent.session_input_tokens == 10
    assert agent.session_output_tokens == 2
    assert agent.session_prompt_tokens == 17
    assert agent.session_total_tokens == 19
    assert result["prompt_tokens"] == 17
    assert result["total_tokens"] == 19
    assert sum(
        message.get("role") == "user" and message.get("content") == "do the card"
        for message in result["messages"]
    ) == 1
    attempt.assert_called_once()


def test_claude_success_honors_kanban_delivery_withholding():
    agent = _agent()
    agent._kanban_delivery_policy = SimpleNamespace(
        withholding=True,
        receipt="Kanban receipt",
        final=lambda _response: "Kanban receipt",
    )
    projection = ClaudeProjection(
        messages=[{"role": "assistant", "content": "raw model prose"}],
        final_text="raw model prose",
    )

    with patch(
        "agent.external_runtime.run_claude_agent_sdk_attempt",
        return_value=projection,
    ):
        result = agent.run_conversation("do the card")

    assert result["final_response"] == "Kanban receipt"


def test_rate_limit_falls_through_to_codex_in_same_worker():
    agent = _agent(
        fallback_model={
            "provider": "openai-codex",
            "model": "gpt-5.4",
            "runtime": "codex_app_server",
        }
    )
    rejected = ClaudeProjection(
        failure=RuntimeFailure(
            FailoverReason.rate_limit,
            "five-hour limit",
            reset_at=1_900_000_000,
        )
    )

    def fake_codex_turn(self, user_input, **kwargs):
        return TurnResult(
            final_text="codex recovered",
            projected_messages=[{"role": "assistant", "content": "codex recovered"}],
            turn_id="codex-turn",
            thread_id="codex-thread",
        )

    with (
        patch(
            "agent.external_runtime.run_claude_agent_sdk_attempt",
            return_value=rejected,
        ),
        patch.object(CodexAppServerSession, "run_turn", fake_codex_turn),
        patch.object(CodexAppServerSession, "ensure_started", return_value="codex-thread"),
    ):
        result = agent.run_conversation("do the card")

    assert result["completed"] is True
    assert result["final_response"] == "codex recovered"
    assert result["api_calls"] == 2
    assert agent.runtime == "codex_app_server"
    assert sum(message.get("role") == "user" for message in result["messages"]) == 1


def test_billing_failure_opens_circuit_falls_back_and_records_usage(caplog):
    agent = _agent(
        fallback_model={
            "provider": "openai-codex",
            "model": "gpt-5.4",
            "runtime": "codex_app_server",
        }
    )
    rejected = ClaudeProjection(
        usage={"input_tokens": 7, "output_tokens": 3},
        failure=RuntimeFailure(FailoverReason.billing, "subscription limit"),
    )

    def fake_codex_turn(self, user_input, **kwargs):
        return TurnResult(final_text="recovered", projected_messages=[])

    with (
        patch(
            "agent.external_runtime.run_claude_agent_sdk_attempt",
            return_value=rejected,
        ),
        patch(
            "agent.runtime_circuit.open_runtime_circuit",
            return_value=1_900_000_000,
        ) as open_circuit,
        patch.object(CodexAppServerSession, "run_turn", fake_codex_turn),
        patch.object(CodexAppServerSession, "ensure_started", return_value="thread"),
    ):
        result = agent.run_conversation("do the card")

    assert result["completed"] is True
    assert agent.runtime == "codex_app_server"
    assert agent.session_input_tokens == 7
    assert agent.session_output_tokens == 3
    open_circuit.assert_called_once()
    events = {
        payload["event"]: payload
        for record in caplog.records
        if record.name == "hermes.runtime_events"
        for payload in [json.loads(record.message)]
    }
    assert {
        "runtime_attempt_start",
        "runtime_attempt_failure",
        "runtime_circuit_open",
        "runtime_fallback_activated",
        "runtime_billing_mode",
    } <= events.keys()
    assert events["runtime_attempt_failure"]["reason"] == "billing"
    assert events["runtime_attempt_failure"]["replay_safe"] is True
    assert events["runtime_circuit_open"]["reset_at"] == 1_900_000_000
    fallback_event = events["runtime_fallback_activated"]
    assert fallback_event["from_provider"] == "anthropic"
    assert fallback_event["from_model"] == "claude-sonnet-4-6"
    assert fallback_event["from_runtime"] == "claude_agent_sdk"
    assert fallback_event["to_provider"] == "openai-codex"
    assert fallback_event["to_model"] == "gpt-5.4"
    assert fallback_event["to_runtime"] == "codex_app_server"
    assert events["runtime_billing_mode"]["billing_mode"] == "subscription_included"


def test_auth_failure_activates_fallback():
    agent = _agent(
        fallback_model={
            "provider": "openai-codex",
            "model": "gpt-5.4",
            "runtime": "codex_app_server",
        }
    )
    rejected = ClaudeProjection(
        failure=RuntimeFailure(FailoverReason.auth, "subscription expired")
    )

    def fake_codex_turn(self, user_input, **kwargs):
        return TurnResult(final_text="recovered", projected_messages=[])

    with (
        patch(
            "agent.external_runtime.run_claude_agent_sdk_attempt",
            return_value=rejected,
        ),
        patch.object(CodexAppServerSession, "run_turn", fake_codex_turn),
        patch.object(CodexAppServerSession, "ensure_started", return_value="thread"),
    ):
        result = agent.run_conversation("do the card")

    assert result["completed"] is True
    assert agent.runtime == "codex_app_server"


def test_rate_limit_falls_through_to_native_hermes_in_same_worker():
    fallback_client = MagicMock()
    fallback_client.base_url = "https://fallback.invalid/v1"
    fallback_client.api_key = "fallback-key"
    fallback_client._custom_headers = None
    fallback_client.default_headers = None
    fallback_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="native recovered", tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="fallback-model",
        usage=None,
    )
    agent = _agent(
        fallback_model={
            "provider": "custom",
            "model": "fallback-model",
            "base_url": "https://fallback.invalid/v1",
        }
    )
    rejected = ClaudeProjection(
        failure=RuntimeFailure(FailoverReason.rate_limit, "limited")
    )

    with (
        patch(
            "agent.external_runtime.run_claude_agent_sdk_attempt",
            return_value=rejected,
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "fallback-model"),
        ),
        patch(
            "agent.moa_loop.aggregate_moa_context",
            return_value="[one advisor result]",
        ) as aggregate,
    ):
        result = agent.run_conversation(
            "do the card",
            moa_config={
                "reference_models": [{"provider": "custom", "model": "advisor"}],
                "aggregator": {"provider": "custom", "model": "synth"},
            },
        )

    assert result["completed"] is True
    assert result["final_response"] == "native recovered"
    assert result["api_calls"] == 2
    assert agent.runtime == "hermes"
    assert sum(message.get("role") == "user" for message in result["messages"]) == 1
    sent = fallback_client.chat.completions.create.call_args.kwargs["messages"]
    assert "Model: fallback-model" in sent[0]["content"]
    assert "Provider: custom" in sent[0]["content"]
    assert "[one advisor result]" in sent[-1]["content"]
    aggregate.assert_called_once()


def test_same_provider_model_native_runtime_fallback_is_not_deduped():
    agent = _agent(
        fallback_model={
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "runtime": "hermes",
        }
    )
    fallback_client = MagicMock()
    fallback_client.base_url = "https://api.anthropic.com"
    fallback_client.api_key = "fallback-key"
    fallback_client._custom_headers = None
    fallback_client.default_headers = None

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fallback_client, "claude-sonnet-4-6"),
    ):
        activated = agent._try_activate_fallback(FailoverReason.auth)

    assert activated is True
    assert agent.runtime == "hermes"


def test_native_moa_reruns_after_tool_result_but_reuses_fallback_bridge_once():
    fallback_client = MagicMock()
    fallback_client.base_url = "https://fallback.invalid/v1"
    fallback_client.api_key = "fallback-key"
    fallback_client._custom_headers = None
    fallback_client.default_headers = None
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="kanban_complete", arguments="{}"),
    )
    fallback_client.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=[tool_call]),
                    finish_reason="tool_calls",
                )
            ],
            model="fallback-model",
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="fallback-model",
            usage=None,
        ),
    ]
    agent = _agent(
        fallback_model={
            "provider": "custom",
            "model": "fallback-model",
            "base_url": "https://fallback.invalid/v1",
        }
    )
    rejected = ClaudeProjection(
        failure=RuntimeFailure(FailoverReason.rate_limit, "limited")
    )

    with (
        patch(
            "agent.external_runtime.run_claude_agent_sdk_attempt",
            return_value=rejected,
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "fallback-model"),
        ),
        patch(
            "agent.moa_loop.aggregate_moa_context",
            side_effect=["[bridge guidance]", "[post-tool guidance]"],
        ) as aggregate,
        patch("run_agent.handle_function_call", return_value='{"success": true}'),
    ):
        result = agent.run_conversation(
            "do the card",
            moa_config={
                "reference_models": [{"provider": "custom", "model": "advisor"}],
                "aggregator": {"provider": "custom", "model": "synth"},
            },
        )

    assert result["completed"] is True
    assert aggregate.call_count == 2
    first_messages = fallback_client.chat.completions.create.call_args_list[0].kwargs[
        "messages"
    ]
    second_messages = fallback_client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert "[bridge guidance]" in next(
        message["content"] for message in reversed(first_messages) if message["role"] == "user"
    )
    assert "[post-tool guidance]" in next(
        message["content"] for message in reversed(second_messages) if message["role"] == "user"
    )


def test_unresolved_tool_side_effect_fails_closed_without_fallback():
    agent = _agent(
        fallback_model={
            "provider": "openai-codex",
            "model": "gpt-5.4",
            "runtime": "codex_app_server",
        }
    )
    unsafe = ClaudeProjection(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "kanban_complete", "arguments": "{}"},
                    }
                ],
            }
        ],
        failure=RuntimeFailure(
            FailoverReason.rate_limit,
            "lost after tool request",
            replay_safe=False,
        ),
    )

    with patch(
        "agent.external_runtime.run_claude_agent_sdk_attempt",
        return_value=unsafe,
    ):
        result = agent.run_conversation("do the card")

    assert result["completed"] is False
    assert result["failed"] is True
    assert agent._fallback_index == 0
    assert any(message.get("tool_calls") for message in result["messages"])
    # The persisted transcript must stay provider-valid: every tool_call id
    # gets a (synthetic) tool result, so no assistant(tool_calls) dangles.
    call_ids = {
        call["id"]
        for message in result["messages"]
        for call in message.get("tool_calls") or []
    }
    result_ids = {
        message.get("tool_call_id")
        for message in result["messages"]
        if message.get("role") == "tool"
    }
    assert call_ids <= result_ids
    synthetic = next(
        message
        for message in result["messages"]
        if message.get("role") == "tool" and message.get("tool_call_id") == "call-1"
    )
    assert "unresolved" in synthetic["content"]


def test_completed_external_side_effect_uses_safe_continuation_on_next_runtime():
    agent = _agent(
        fallback_model={
            "provider": "openai-codex",
            "model": "gpt-5.4",
            "runtime": "codex_app_server",
        }
    )
    completed_then_failed = ClaudeProjection(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "kanban_comment", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": '{"success": true}'},
        ],
        failure=RuntimeFailure(FailoverReason.rate_limit, "limited", replay_safe=True),
    )
    prompts = []

    def fake_codex_turn(self, user_input, **kwargs):
        prompts.append(user_input)
        return TurnResult(
            final_text="continued safely",
            projected_messages=[{"role": "assistant", "content": "continued safely"}],
            turn_id="codex-turn",
            thread_id="codex-thread",
        )

    with (
        patch(
            "agent.external_runtime.run_claude_agent_sdk_attempt",
            return_value=completed_then_failed,
        ),
        patch.object(CodexAppServerSession, "run_turn", fake_codex_turn),
        patch.object(CodexAppServerSession, "ensure_started", return_value="codex-thread"),
    ):
        result = agent.run_conversation("do the card")

    assert result["completed"] is True
    assert prompts and "Do not repeat any completed tool action" in prompts[0]
    assert "Original objective:\ndo the card" in prompts[0]
    assert sum(message.get("role") == "user" for message in result["messages"]) == 1


def test_codex_failure_falls_through_to_claude_in_same_worker():
    agent = _agent(
        runtime="codex_app_server",
        provider="openai-codex",
        fallback_model={
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "runtime": "claude_agent_sdk",
        },
    )
    recovered = ClaudeProjection(
        messages=[{"role": "assistant", "content": "claude recovered"}],
        final_text="claude recovered",
    )

    def fake_codex_turn(self, user_input, **kwargs):
        return TurnResult(error="rate limit exceeded")

    with (
        patch.object(CodexAppServerSession, "run_turn", fake_codex_turn),
        patch.object(CodexAppServerSession, "ensure_started", return_value="thread"),
        patch(
            "agent.external_runtime.run_claude_agent_sdk_attempt",
            return_value=recovered,
        ),
    ):
        result = agent.run_conversation("do the card")

    assert result["completed"] is True
    assert result["final_response"] == "claude recovered"
    assert result["api_calls"] == 2
    assert agent.runtime == "claude_agent_sdk"
    assert sum(message.get("role") == "user" for message in result["messages"]) == 1


def test_codex_completed_tool_effect_continues_to_claude_without_replay():
    agent = _agent(
        runtime="codex_app_server",
        provider="openai-codex",
        fallback_model={
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "runtime": "claude_agent_sdk",
        },
    )
    prompts = []

    def fake_codex_turn(self, user_input, **kwargs):
        return TurnResult(
            error="rate limit exceeded",
            projected_messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "done"},
            ],
        )

    def claude_attempt(agent, *, user_message, effective_task_id):
        prompts.append(user_message)
        return ClaudeProjection(
            messages=[{"role": "assistant", "content": "continued"}],
            final_text="continued",
        )

    with (
        patch.object(CodexAppServerSession, "run_turn", fake_codex_turn),
        patch.object(CodexAppServerSession, "ensure_started", return_value="thread"),
        patch(
            "agent.external_runtime.run_claude_agent_sdk_attempt",
            side_effect=claude_attempt,
        ),
    ):
        result = agent.run_conversation("do the card")

    assert result["completed"] is True
    assert "Do not repeat any completed tool action" in prompts[0]
    assert sum(message.get("role") == "user" for message in result["messages"]) == 1
    assert any(message.get("tool_call_id") == "call-1" for message in result["messages"])


def test_codex_pending_tool_effect_still_blocks_external_fallback():
    agent = _agent(
        runtime="codex_app_server",
        provider="openai-codex",
        fallback_model={
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "runtime": "claude_agent_sdk",
        },
    )

    def fake_codex_turn(self, user_input, **kwargs):
        return TurnResult(
            error="rate limit exceeded",
            projected_messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-pending",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                }
            ],
        )

    with (
        patch.object(CodexAppServerSession, "run_turn", fake_codex_turn),
        patch.object(CodexAppServerSession, "ensure_started", return_value="thread"),
        patch("agent.external_runtime.run_claude_agent_sdk_attempt") as claude_attempt,
    ):
        result = agent.run_conversation("do the card")

    assert result["failed"] is True
    assert "unresolved tool call" in result["error"]
    claude_attempt.assert_not_called()


def test_native_failure_reenters_dispatcher_for_claude_without_duplicate_turn():
    with (
        patch("run_agent.get_tool_definitions", return_value=_tools()),
        patch("run_agent.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://native.invalid/v1",
            provider="custom",
            model="native-model",
            runtime="hermes",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model={
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "runtime": "claude_agent_sdk",
            },
        )

    class RateLimited(Exception):
        status_code = 429

    agent._interruptible_streaming_api_call = MagicMock(
        side_effect=RateLimited("rate limit exceeded")
    )
    recovered = ClaudeProjection(
        messages=[{"role": "assistant", "content": "claude recovered"}],
        final_text="claude recovered",
    )

    with patch(
        "agent.external_runtime.run_claude_agent_sdk_attempt",
        return_value=recovered,
    ):
        result = agent.run_conversation("do the card")

    assert result["completed"] is True
    assert result["final_response"] == "claude recovered"
    assert result["api_calls"] == 2
    assert agent.runtime == "claude_agent_sdk"
    assert sum(message.get("role") == "user" for message in result["messages"]) == 1
    agent._interruptible_streaming_api_call.assert_called_once()


def test_external_chain_exhaustion_returns_one_failed_user_turn():
    agent = _agent(
        fallback_model={
            "provider": "openai-codex",
            "model": "gpt-5.4",
            "runtime": "codex_app_server",
        }
    )
    rejected = ClaudeProjection(
        failure=RuntimeFailure(FailoverReason.rate_limit, "claude limited")
    )

    def fake_codex_turn(self, user_input, **kwargs):
        return TurnResult(error="codex also limited")

    with (
        patch(
            "agent.external_runtime.run_claude_agent_sdk_attempt",
            return_value=rejected,
        ),
        patch.object(CodexAppServerSession, "run_turn", fake_codex_turn),
        patch.object(CodexAppServerSession, "ensure_started", return_value="thread"),
    ):
        result = agent.run_conversation("do the card")

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["api_calls"] == 2
    assert sum(message.get("role") == "user" for message in result["messages"]) == 1


def test_moa_advisor_context_feeds_external_acting_runtime_only_once():
    agent = _agent()
    captured = []

    def attempt(agent, *, user_message, effective_task_id):
        captured.append(user_message)
        return ClaudeProjection(
            messages=[{"role": "assistant", "content": "acted"}],
            final_text="acted",
        )

    with (
        patch(
            "agent.moa_loop.aggregate_moa_context",
            return_value="[private advisor guidance]",
        ) as aggregate,
        patch(
            "agent.external_runtime.run_claude_agent_sdk_attempt",
            side_effect=attempt,
        ),
    ):
        result = agent.run_conversation(
            "do the card",
            moa_config={
                "reference_models": [{"provider": "custom", "model": "advisor"}],
                "aggregator": {"provider": "custom", "model": "synth"},
            },
        )

    assert result["completed"] is True
    assert captured == ["do the card\n\n[private advisor guidance]"]
    aggregate.assert_called_once()
    assert sum(message.get("role") == "user" for message in result["messages"]) == 1


def test_claude_success_preserves_skill_review_cadence():
    agent = _agent()
    agent._skill_nudge_interval = 1
    agent.valid_tool_names.add("skill_manage")
    agent._spawn_background_review = MagicMock()
    projection = ClaudeProjection(
        messages=[{"role": "assistant", "content": "done"}], final_text="done"
    )

    with patch(
        "agent.external_runtime.run_claude_agent_sdk_attempt", return_value=projection
    ):
        result = agent.run_conversation("do the card")

    assert result["completed"] is True
    agent._spawn_background_review.assert_called_once()
    assert agent._spawn_background_review.call_args.kwargs["review_skills"] is True

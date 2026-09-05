"""Behavioral contracts for bidirectional Claude SDK provider fallback."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import model_tools
import run_agent


class RateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate limit exceeded")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "rate limit exceeded"}}


def _response(text, *, tool_calls=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    tool_calls=tool_calls,
                    reasoning_content=None,
                    reasoning=None,
                    reasoning_details=None,
                    refusal=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        model="fallback-chat-model",
        usage=None,
    )


def _agent(*, api_mode="chat_completions", provider="openai-codex", max_iterations=5):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        return run_agent.AIAgent(
            api_key="test-key-12345678",
            base_url="https://primary.invalid/v1",
            provider=provider,
            model="primary-model",
            api_mode=api_mode,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=max_iterations,
        )


def _install_fallback_clients(monkeypatch, clients):
    def resolve(provider, model=None, **_kwargs):
        return clients[provider], model

    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolve)


def _stub_generic_result_building(monkeypatch, agent, text):
    monkeypatch.setattr(agent, "_extract_reasoning", lambda _message: None)
    monkeypatch.setattr(
        agent,
        "_build_assistant_message",
        lambda _message, _finish: {"role": "assistant", "content": text},
    )


def test_sdk_provider_failure_continues_to_generic_with_cumulative_provenance(monkeypatch):
    calls = []
    agent = _agent(api_mode="claude_agent_sdk", provider="claude-agent-sdk")
    agent.model = "claude-sonnet"
    agent._fallback_chain = [{"provider": "openrouter", "model": "fallback-chat-model"}]
    agent._fallback_index = 0

    ordinary = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="fallback")
    ordinary.chat.completions.create.side_effect = (
        lambda **_kwargs: calls.append("generic") or _response("fallback online")
    )
    _install_fallback_clients(monkeypatch, {"openrouter": ordinary})
    _stub_generic_result_building(monkeypatch, agent, "fallback online")
    monkeypatch.setattr(
        agent,
        "_run_claude_agent_sdk_turn",
        lambda **kwargs: calls.append("sdk")
        or {
            "final_response": "expired",
            "messages": kwargs["messages"],
            "api_calls": 1,
            "completed": False,
            "partial": True,
            "failed": True,
            "interrupted": False,
            "error": "API Error: 401 OAuth access token expired",
            "failover_reason": "auth",
            "sdk_effects": {
                "tool": False,
                "streamed": False,
                "projected": False,
                "interrupted": False,
                "mutated": False,
            },
        },
    )

    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("hello")

    assert calls == ["sdk", "generic"]
    assert result["final_response"] == "fallback online"
    assert result["api_calls"] == 2
    assert result["provider"] == "openrouter"
    assert result["model"] == "fallback-chat-model"
    assert [m["role"] for m in result["messages"]].count("user") == 1
    assert "Provider: openrouter" in agent._cached_system_prompt
    assert "Model: fallback-chat-model" in agent._cached_system_prompt


def test_zero_call_sdk_auth_startup_hands_off_without_inventing_a_call(monkeypatch):
    agent = _agent(api_mode="claude_agent_sdk", provider="claude-agent-sdk")
    agent.model = "claude-sonnet"
    agent._fallback_chain = [{"provider": "openrouter", "model": "fallback-chat-model"}]
    agent._fallback_index = 0

    ordinary = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="fallback")
    ordinary.chat.completions.create.return_value = _response("fallback online")
    _install_fallback_clients(monkeypatch, {"openrouter": ordinary})
    _stub_generic_result_building(monkeypatch, agent, "fallback online")
    original_error = "Not signed in to Claude; run `claude auth login`."
    monkeypatch.setattr(
        agent,
        "_run_claude_agent_sdk_turn",
        lambda **kwargs: {
            "final_response": original_error,
            "messages": kwargs["messages"],
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "failed": True,
            "interrupted": False,
            "error": original_error,
            "failover_reason": "auth",
            "sdk_effects": {
                "tool": False,
                "streamed": False,
                "projected": False,
                "interrupted": False,
                "mutated": False,
            },
        },
    )

    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("hello")

    assert result["final_response"] == "fallback online"
    assert result["api_calls"] == 1
    assert ordinary.chat.completions.create.call_count == 1


def test_sdk_chain_spends_max_iterations_and_iteration_budget(monkeypatch):
    for limiter in ("max_iterations", "iteration_budget"):
        calls = []
        agent = _agent(
            api_mode="claude_agent_sdk",
            provider="claude-agent-sdk",
            max_iterations=1 if limiter == "max_iterations" else 5,
        )
        if limiter == "iteration_budget":
            shared_budget = run_agent.IterationBudget(max_total=1)
            monkeypatch.setattr(
                "agent.turn_context.IterationBudget", lambda _maximum: shared_budget
            )
        agent.model = "claude-primary"
        agent._fallback_chain = [
            {"provider": "claude-agent-sdk", "model": "claude-fallback-1"},
            {"provider": "claude-agent-sdk", "model": "claude-fallback-2"},
        ]
        agent._fallback_index = 0
        sdk_client = MagicMock(base_url="", api_key="test")
        _install_fallback_clients(monkeypatch, {"claude-agent-sdk": sdk_client})

        def failed_sdk(**kwargs):
            calls.append(agent.model)
            return {
                "final_response": f"{agent.model} timed out",
                "messages": kwargs["messages"],
                "api_calls": 1,
                "completed": False,
                "partial": True,
                "failed": True,
                "interrupted": False,
                "error": f"{agent.model} timed out",
                "failover_reason": "timeout",
                "sdk_effects": {
                    "tool": False,
                    "streamed": False,
                    "projected": False,
                    "interrupted": False,
                    "mutated": False,
                },
            }

        monkeypatch.setattr(agent, "_run_claude_agent_sdk_turn", failed_sdk)

        result = agent.run_conversation("hello")

        assert calls == ["claude-primary"], limiter
        assert result["api_calls"] == 1
        assert agent.iteration_budget.used == 1
        assert result["model"] == "claude-primary"


def test_generic_rate_limit_dispatches_implicit_sdk_entry_once(monkeypatch):
    calls = []
    agent = _agent()
    agent._fallback_chain = [
        {"provider": "claude-agent-sdk", "model": "claude-sonnet"}
    ]
    agent._fallback_index = 0

    primary = MagicMock(base_url="https://primary.invalid/v1", api_key="primary")
    primary.chat.completions.create.side_effect = RateLimitError()
    agent.client = primary
    sdk_client = MagicMock(base_url="", api_key="claude-subscription-oauth")
    _install_fallback_clients(monkeypatch, {"claude-agent-sdk": sdk_client})
    monkeypatch.setattr(
        agent,
        "_run_claude_agent_sdk_turn",
        lambda **kwargs: calls.append("sdk")
        or {
            "final_response": "SDK fallback",
            "messages": kwargs["messages"],
            "api_calls": 1,
            "completed": True,
            "partial": False,
            "failed": False,
            "interrupted": False,
            "error": None,
        },
    )

    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("first")

    assert result["final_response"] == "SDK fallback"
    assert calls == ["sdk"]
    assert primary.chat.completions.create.call_count == 1
    assert result["api_calls"] == 2
    assert agent.api_mode == "claude_agent_sdk"


def test_generic_sdk_chain_exhaustion_preserves_first_sdk_actionable_error(
    monkeypatch,
):
    agent = _agent()
    agent._fallback_chain = [
        {"provider": "claude-agent-sdk", "model": "claude-sdk-auth"},
        {"provider": "claude-agent-sdk", "model": "claude-sdk-overloaded"},
    ]
    agent._fallback_index = 0

    primary = MagicMock(base_url="https://primary.invalid/v1", api_key="primary")
    primary.chat.completions.create.side_effect = RateLimitError()
    agent.client = primary
    sdk_client = MagicMock(base_url="", api_key="test")
    _install_fallback_clients(monkeypatch, {"claude-agent-sdk": sdk_client})
    attempted = []
    first_sdk_error = "Not signed in to Claude; run `claude auth login`."

    def failed_sdk(**kwargs):
        attempted.append(agent.model)
        if agent.model == "claude-sdk-auth":
            error = first_sdk_error
            reason = "auth"
        else:
            error = "HTTP 503 service overloaded"
            reason = "overloaded"
        return {
            "final_response": error,
            "messages": kwargs["messages"],
            "api_calls": 1,
            "completed": False,
            "partial": True,
            "failed": True,
            "interrupted": False,
            "error": error,
            "failover_reason": reason,
            "sdk_effects": {
                "tool": False,
                "streamed": False,
                "projected": False,
                "interrupted": False,
                "mutated": False,
            },
        }

    monkeypatch.setattr(agent, "_run_claude_agent_sdk_turn", failed_sdk)

    result = agent.run_conversation("hello")

    assert attempted == ["claude-sdk-auth", "claude-sdk-overloaded"]
    assert result["error"] == first_sdk_error
    assert result["final_response"] == first_sdk_error
    assert result["api_calls"] == 3
    assert result["provider"] == "claude-agent-sdk"
    assert result["model"] == "claude-sdk-overloaded"
    assert [m["role"] for m in result["messages"]].count("user") == 1


def test_generic_sdk_chain_success_preserves_successful_result(monkeypatch):
    agent = _agent()
    agent._fallback_chain = [
        {"provider": "claude-agent-sdk", "model": "claude-sdk-auth"},
        {"provider": "claude-agent-sdk", "model": "claude-sdk-success"},
    ]
    agent._fallback_index = 0

    primary = MagicMock(base_url="https://primary.invalid/v1", api_key="primary")
    primary.chat.completions.create.side_effect = RateLimitError()
    agent.client = primary
    sdk_client = MagicMock(base_url="", api_key="test")
    _install_fallback_clients(monkeypatch, {"claude-agent-sdk": sdk_client})
    attempted = []

    def sdk_turn(**kwargs):
        attempted.append(agent.model)
        if agent.model == "claude-sdk-auth":
            error = "Not signed in to Claude; run `claude auth login`."
            return {
                "final_response": error,
                "messages": kwargs["messages"],
                "api_calls": 1,
                "completed": False,
                "partial": True,
                "failed": True,
                "interrupted": False,
                "error": error,
                "failover_reason": "auth",
                "sdk_effects": {
                    "tool": False,
                    "streamed": False,
                    "projected": False,
                    "interrupted": False,
                    "mutated": False,
                },
            }
        return {
            "final_response": "SDK chain recovered",
            "messages": kwargs["messages"],
            "api_calls": 1,
            "completed": True,
            "partial": False,
            "failed": False,
            "interrupted": False,
            "error": None,
        }

    monkeypatch.setattr(agent, "_run_claude_agent_sdk_turn", sdk_turn)

    result = agent.run_conversation("hello")

    assert attempted == ["claude-sdk-auth", "claude-sdk-success"]
    assert result["final_response"] == "SDK chain recovered"
    assert result.get("error") is None
    assert result["completed"] is True
    assert result["api_calls"] == 3
    assert result["provider"] == "claude-agent-sdk"
    assert result["model"] == "claude-sdk-success"
    assert [m["role"] for m in result["messages"]].count("user") == 1


def test_tool_round_skips_sdk_and_continues_next_provider_without_replay_or_charge(
    monkeypatch,
):
    calls = []
    agent = _agent(max_iterations=2)
    agent._fallback_chain = [
        {"provider": "claude-agent-sdk", "model": "claude-sonnet"},
        {"provider": "openrouter", "model": "fallback-chat-model"},
    ]
    agent._fallback_index = 0

    tool_calls = [
        SimpleNamespace(
            id="call-write-once",
            type="function",
            function=SimpleNamespace(name="write_once", arguments="{}"),
        )
    ]
    primary = MagicMock(base_url="https://primary.invalid/v1", api_key="primary")
    primary.chat.completions.create.side_effect = [
        _response(None, tool_calls=tool_calls, finish_reason="tool_calls"),
        RateLimitError(),
    ]
    agent.client = primary
    agent.valid_tool_names = ["write_once"]
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "write_once",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    # The registry dispatch reads ``model_tools.handle_function_call`` at call time
    # (agent/agent_runtime_helpers.py), so that module is the seam — not the run_agent facade.
    monkeypatch.setattr(
        model_tools,
        "handle_function_call",
        lambda name, _args, _task_id, **_kwargs: calls.append(name) or "written once",
    )

    sdk_client = MagicMock(base_url="", api_key="claude-subscription-oauth")
    ordinary = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="fallback")
    ordinary.chat.completions.create.side_effect = (
        lambda **_kwargs: calls.append("generic") or _response("done without replay")
    )
    _install_fallback_clients(
        monkeypatch,
        {"claude-agent-sdk": sdk_client, "openrouter": ordinary},
    )
    monkeypatch.setattr(
        agent,
        "_run_claude_agent_sdk_turn",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("SDK must not replay an executed tool round")
        ),
    )
    _stub_generic_result_building(monkeypatch, agent, "done without replay")

    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("write it once")

    assert result["final_response"] == "done without replay"
    assert calls == ["write_once", "generic"]
    assert sum(call == "write_once" for call in calls) == 1
    assert primary.chat.completions.create.call_count == 2
    # Two real primary calls plus one real safe-provider call; skipped SDK costs zero.
    assert result["api_calls"] == 3


def test_partial_generic_stream_skips_sdk_and_uses_next_safe_provider(monkeypatch):
    calls = []
    agent = _agent()
    agent._fallback_chain = [
        {"provider": "claude-agent-sdk", "model": "claude-sonnet"},
        {"provider": "openrouter", "model": "fallback-chat-model"},
    ]
    agent._fallback_index = 0

    primary = MagicMock(base_url="https://primary.invalid/v1", api_key="primary")

    def fail_after_stream(**_kwargs):
        agent._current_streamed_assistant_text = "visible partial answer"
        raise RateLimitError()

    primary.chat.completions.create.side_effect = fail_after_stream
    agent.client = primary
    sdk_client = MagicMock(base_url="", api_key="claude-subscription-oauth")
    ordinary = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="fallback")
    ordinary.chat.completions.create.side_effect = (
        lambda **_kwargs: calls.append("generic") or _response("safe continuation")
    )
    _install_fallback_clients(
        monkeypatch,
        {"claude-agent-sdk": sdk_client, "openrouter": ordinary},
    )
    monkeypatch.setattr(
        agent,
        "_run_claude_agent_sdk_turn",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("SDK must not replay after visible streamed output")
        ),
    )
    _stub_generic_result_building(monkeypatch, agent, "safe continuation")

    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("hello")

    assert result["final_response"] == "safe continuation"
    assert calls == ["generic"]
    assert result["api_calls"] == 2


def test_sdk_fallback_success_preserves_successful_result(monkeypatch):
    agent = _agent(api_mode="claude_agent_sdk", provider="claude-agent-sdk")
    agent.model = "claude-primary"
    agent._fallback_chain = [
        {"provider": "claude-agent-sdk", "model": "claude-fallback"},
    ]
    agent._fallback_index = 0
    sdk_client = MagicMock(base_url="", api_key="test")
    _install_fallback_clients(monkeypatch, {"claude-agent-sdk": sdk_client})
    attempted = []

    def sdk_turn(**kwargs):
        attempted.append(agent.model)
        if agent.model == "claude-primary":
            return {
                "final_response": "HTTP 429 session limit reached",
                "messages": kwargs["messages"],
                "api_calls": 1,
                "completed": False,
                "partial": True,
                "failed": True,
                "interrupted": False,
                "error": "HTTP 429 session limit reached",
                "failover_reason": "rate_limit",
                "sdk_effects": {
                    "tool": False,
                    "streamed": False,
                    "projected": False,
                    "interrupted": False,
                    "mutated": False,
                },
            }
        return {
            "final_response": "fallback answer",
            "messages": kwargs["messages"],
            "api_calls": 1,
            "completed": True,
            "partial": False,
            "failed": False,
            "interrupted": False,
            "error": None,
        }

    monkeypatch.setattr(agent, "_run_claude_agent_sdk_turn", sdk_turn)

    result = agent.run_conversation("hello")

    assert attempted == ["claude-primary", "claude-fallback"]
    assert result["final_response"] == "fallback answer"
    assert result.get("error") is None
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert result["provider"] == "claude-agent-sdk"
    assert result["model"] == "claude-fallback"
    assert [m["role"] for m in result["messages"]].count("user") == 1


def test_multihop_sdk_exhaustion_preserves_first_actionable_error(monkeypatch):
    agent = _agent(api_mode="claude_agent_sdk", provider="claude-agent-sdk")
    agent.model = "claude-primary"
    agent._fallback_chain = [
        {"provider": "claude-agent-sdk", "model": "claude-fallback-1"},
        {"provider": "claude-agent-sdk", "model": "claude-fallback-2"},
    ]
    agent._fallback_index = 0
    sdk_client = MagicMock(base_url="", api_key="test")
    _install_fallback_clients(monkeypatch, {"claude-agent-sdk": sdk_client})
    outcomes = [
        ("Not signed in to Claude; run `claude auth login`.", "auth", 0),
        ("request timed out while contacting provider", "timeout", 1),
        ("HTTP 503 service overloaded", "overloaded", 1),
    ]
    attempted = []

    def failed_sdk(**kwargs):
        attempted.append(agent.model)
        error, reason, api_calls = outcomes[len(attempted) - 1]
        return {
            "final_response": error,
            "messages": kwargs["messages"],
            "api_calls": api_calls,
            "completed": False,
            "partial": True,
            "failed": True,
            "interrupted": False,
            "error": error,
            "failover_reason": reason,
            "sdk_effects": {
                "tool": False,
                "streamed": False,
                "projected": False,
                "interrupted": False,
                "mutated": False,
            },
        }

    monkeypatch.setattr(agent, "_run_claude_agent_sdk_turn", failed_sdk)

    result = agent.run_conversation("hello")

    assert attempted == ["claude-primary", "claude-fallback-1", "claude-fallback-2"]
    assert result["error"] == outcomes[0][0]
    assert result["final_response"] == outcomes[0][0]
    assert result["api_calls"] == 2
    assert result["provider"] == "claude-agent-sdk"
    assert result["model"] == "claude-fallback-2"


def test_exhausted_sdk_chain_preserves_original_actionable_error_and_count(monkeypatch):
    agent = _agent(api_mode="claude_agent_sdk", provider="claude-agent-sdk")
    agent.model = "claude-sonnet"
    agent._fallback_chain = []
    original_error = "Not signed in to Claude; run `claude auth login`."
    monkeypatch.setattr(
        agent,
        "_run_claude_agent_sdk_turn",
        lambda **kwargs: {
            "final_response": original_error,
            "messages": kwargs["messages"],
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "failed": True,
            "interrupted": False,
            "error": original_error,
            "failover_reason": "auth",
        },
    )

    result = agent.run_conversation("hello")

    assert result["error"] == original_error
    assert result["final_response"] == original_error
    assert result["api_calls"] == 0

"""Current-turn lifecycle regressions for max-iteration auto-continuation.

These tests intentionally exercise the public ``AIAgent.run_conversation``
entrypoint.  A helper that merely resets ``IterationBudget`` in the inner loop
is not sufficient: every continuation must pass through turn_context and
turn_finalizer as a fresh, independently bounded turn.
"""

from __future__ import annotations

import copy
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


MARKER = "[Continuing after max-iteration exhaustion]"


def _tool_definitions() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _tool_response(label: str):
    call = SimpleNamespace(
        id=f"call_{label}_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    message = SimpleNamespace(content="", tool_calls=[call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=None,
    )


def _text_response(content: str):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


def _config(*, enabled: bool, maximum: int) -> dict:
    return {
        "agent": {
            "verify_on_stop": False,
            "api_max_retries": 1,
            "auto_continue_on_max_iterations": {
                "enabled": enabled,
                "max_auto_continues": maximum,
                "prompt": "Continue from the current state without repeating work.",
            },
        }
    }


def _make_agent(*, platform: str = "cli") -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_definitions()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=1,
            platform=platform,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._api_max_retries = 1
    agent._handle_max_iterations = MagicMock(return_value="normal exhaustion summary")
    return agent


def _run_with_responses(agent: AIAgent, config: dict, responses: list):
    requests: list[list[dict]] = []
    pending = list(responses)

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        response = pending.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    agent.client.chat.completions.create.side_effect = create
    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("run_agent.handle_function_call", return_value="tool output"),
    ):
        result = agent.run_conversation("finish the task")
    return result, requests


def _roles_without_system(messages: list[dict]) -> list[str]:
    return [m.get("role") for m in messages if m.get("role") != "system"]


def _assert_provider_valid_roles(messages: list[dict]) -> None:
    roles = _roles_without_system(messages)
    for previous, current in zip(roles, roles[1:]):
        assert (previous, current) != ("tool", "user")
        assert not (previous == current and current in {"user", "assistant"})


@pytest.mark.parametrize("platform", ["cli", "telegram", "tui"])
def test_tool_tail_continues_as_fresh_provider_valid_turn_with_explicit_accounting(
    platform,
):
    agent = _make_agent(platform=platform)

    from agent import conversation_loop, turn_finalizer

    real_build_turn_context = conversation_loop.build_turn_context
    real_finalize_turn = turn_finalizer.finalize_turn
    with (
        patch.object(
            conversation_loop,
            "build_turn_context",
            wraps=real_build_turn_context,
        ) as build_turn_context,
        patch.object(
            turn_finalizer,
            "finalize_turn",
            wraps=real_finalize_turn,
        ) as finalize_turn,
    ):
        result, requests = _run_with_responses(
            agent,
            _config(enabled=True, maximum=1),
            [_tool_response("first"), _text_response("finished")],
        )

    assert result["final_response"] == "finished"
    assert len(requests) == 2
    assert build_turn_context.call_count == 2
    assert finalize_turn.call_count == 2

    second_request = requests[1]
    _assert_provider_valid_roles(second_request)
    assert _roles_without_system(second_request) == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    synthetic_users = [
        m for m in second_request
        if m.get("role") == "user" and str(m.get("content", "")).startswith(MARKER)
    ]
    assert len(synthetic_users) == 1
    assert sum(m.get("role") == "system" for m in requests[0]) == 1
    assert sum(m.get("role") == "system" for m in second_request) == 1
    assert requests[0][0]["content"] == second_request[0]["content"]

    assert result["api_calls"] == 2
    assert result["cycle_api_calls"] == 1
    assert result["api_calls_by_cycle"] == [1, 1]
    assert result["auto_continues_used"] == 1
    agent._handle_max_iterations.assert_not_called()


def test_repeated_exhaustion_uses_independently_bounded_turns_without_state_leakage():
    agent = _make_agent()
    config = _config(enabled=True, maximum=2)

    first_result, first_requests = _run_with_responses(
        agent,
        config,
        [
            _tool_response("first-a"),
            _tool_response("first-b"),
            _text_response("first finished"),
        ],
    )
    second_result, second_requests = _run_with_responses(
        agent,
        config,
        [
            _tool_response("second-a"),
            _tool_response("second-b"),
            _text_response("second finished"),
        ],
    )

    assert first_result["api_calls_by_cycle"] == [1, 1, 1]
    assert second_result["api_calls_by_cycle"] == [1, 1, 1]
    assert first_result["auto_continues_used"] == 2
    assert second_result["auto_continues_used"] == 2
    assert len(first_requests) == len(second_requests) == 3
    for request in [*first_requests, *second_requests]:
        _assert_provider_valid_roles(request)


def test_continuation_limit_falls_back_once_after_last_fresh_turn():
    agent = _make_agent()

    result, requests = _run_with_responses(
        agent,
        _config(enabled=True, maximum=1),
        [_tool_response("first"), _tool_response("second")],
    )

    assert result["final_response"] == "normal exhaustion summary"
    assert len(requests) == 2
    assert result["api_calls"] == 2
    assert result["cycle_api_calls"] == 1
    assert result["api_calls_by_cycle"] == [1, 1]
    assert result["auto_continues_used"] == 1
    agent._handle_max_iterations.assert_called_once()


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"agent": {}},
        _config(enabled=False, maximum=3),
        _config(enabled=True, maximum=0),
        _config(enabled=True, maximum=-1),
    ],
)
def test_disabled_default_zero_and_negative_config_keep_existing_fallback(config):
    agent = _make_agent()

    result, requests = _run_with_responses(
        agent,
        config,
        [_tool_response("only")],
    )

    assert result["final_response"] == "normal exhaustion summary"
    assert len(requests) == 1
    assert result["api_calls"] == 1
    assert "cycle_api_calls" not in result
    assert "api_calls_by_cycle" not in result
    assert "auto_continues_used" not in result
    agent._handle_max_iterations.assert_called_once()


def test_api_error_during_continuation_does_not_start_another_cycle():
    agent = _make_agent()

    result, requests = _run_with_responses(
        agent,
        _config(enabled=True, maximum=3),
        [_tool_response("first"), RuntimeError("provider failed")],
    )

    assert len(requests) == 2
    assert result["auto_continues_used"] == 1
    assert result["failed"] is True
    assert result["api_calls_by_cycle"] == [1, 1]


def test_cleanup_errors_from_an_exhausted_cycle_survive_the_final_cycle():
    agent = _make_agent()
    original_cleanup = agent._drop_trailing_empty_response_scaffolding
    cleanup_calls = 0

    def fail_first_cleanup(messages):
        nonlocal cleanup_calls
        cleanup_calls += 1
        # build_turn_context performs one early persistence cleanup; fail the
        # next call, which is the exhausted turn's finalizer cleanup.
        if cleanup_calls == 2:
            raise RuntimeError("first-cycle cleanup failed")
        return original_cleanup(messages)

    agent._drop_trailing_empty_response_scaffolding = fail_first_cleanup

    result, requests = _run_with_responses(
        agent,
        _config(enabled=True, maximum=1),
        [_tool_response("first"), _text_response("finished")],
    )

    assert len(requests) == 2
    assert result["final_response"] == "finished"
    assert result["cleanup_errors"] == [
        "persist_session: first-cycle cleanup failed"
    ]


def test_preexisting_cancellation_never_auto_continues():
    agent = _make_agent()
    with patch("run_agent._set_interrupt"):
        agent.interrupt()

    result, requests = _run_with_responses(
        agent,
        _config(enabled=True, maximum=3),
        [],
    )

    assert requests == []
    assert result["interrupted"] is True
    assert "auto_continues_used" not in result

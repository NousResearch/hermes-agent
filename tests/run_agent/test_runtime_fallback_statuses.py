"""Runtime fallback policy tests for rate-limit, billing, and overload errors."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.error_classifier import ClassifiedError, FailoverReason
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="ok"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    resp = SimpleNamespace(choices=[choice], model="test/model", usage=None)
    return resp


def _make_error(status_code: int, message: str) -> Exception:
    err = Exception(message)
    err.status_code = status_code
    err.response = None
    return err


def _make_agent(*, fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent._cached_system_prompt = "You are helpful."
        agent.tool_delay = 0
        return agent


def _fallback_client():
    client = MagicMock()
    client.api_key = "fallback-key"
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = _mock_response("fallback success")
    return client


def _run(agent: AIAgent, classify_result: ClassifiedError, primary_effects: list):
    primary_client = MagicMock()
    primary_client.chat.completions.create.side_effect = primary_effects
    agent.client = primary_client

    fallback_client = _fallback_client()

    with (
        patch("run_agent.classify_api_error", return_value=classify_result),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client, "fallback/model")),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch("run_agent.jittered_backoff", return_value=0),
        patch("time.sleep", return_value=None),
    ):
        result = agent.run_conversation("hello")

    return result, primary_client, fallback_client


def test_429_rate_limit_switches_to_fallback_immediately():
    agent = _make_agent(
        fallback_model={"provider": "openrouter", "model": "fallback/model"},
    )
    error = _make_error(429, "rate limit exceeded")
    classified = ClassifiedError(
        reason=FailoverReason.rate_limit,
        status_code=429,
        retryable=True,
        should_fallback=True,
    )

    result, primary_client, fallback_client = _run(agent, classified, [error])

    assert result["completed"] is True
    assert result["final_response"] == "fallback success"
    assert primary_client.chat.completions.create.call_count == 1
    assert fallback_client.chat.completions.create.call_count == 1
    assert agent.provider == "openrouter"
    assert agent.model == "fallback/model"


def test_400_billing_switches_to_fallback_immediately():
    agent = _make_agent(
        fallback_model={"provider": "openrouter", "model": "fallback/model"},
    )
    error = _make_error(400, "out of credits")
    classified = ClassifiedError(
        reason=FailoverReason.billing,
        status_code=400,
        retryable=False,
        should_fallback=True,
    )

    result, primary_client, fallback_client = _run(agent, classified, [error])

    assert result["completed"] is True
    assert result["final_response"] == "fallback success"
    assert primary_client.chat.completions.create.call_count == 1
    assert fallback_client.chat.completions.create.call_count == 1
    assert agent.provider == "openrouter"


def test_503_overload_retries_before_switching_to_fallback():
    agent = _make_agent(
        fallback_model={"provider": "openrouter", "model": "fallback/model"},
    )
    error = _make_error(503, "provider overloaded")
    classified = ClassifiedError(
        reason=FailoverReason.overloaded,
        status_code=503,
        retryable=True,
        should_fallback=True,
    )

    result, primary_client, fallback_client = _run(agent, classified, [error, error, error])

    assert result["completed"] is True
    assert result["final_response"] == "fallback success"
    assert primary_client.chat.completions.create.call_count == 3
    assert fallback_client.chat.completions.create.call_count == 1
    assert agent.provider == "openrouter"

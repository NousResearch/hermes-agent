"""Regression test: Bedrock agents must not fall through to the OpenAI
client branch in handle_max_iterations() (Eddie bot report — "Reached the
maximum iteration of 90 but couldn't summarize. Error: connection error.").

Bedrock agents have agent.client is None / no api_key/base_url configured
(they talk to AWS directly via boto3), so the generic fallback branch that
calls agent._ensure_primary_openai_client().chat.completions.create(...)
always raised a connection error for api_mode == "bedrock_converse". The
fix adds an explicit bedrock_converse branch that routes through the same
transport + boto3 dispatch path the main loop uses.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.chat_completion_helpers import handle_max_iterations


class _FakeTransport:
    def __init__(self, response_content="final summary text"):
        self.build_kwargs_calls = []
        self._response_content = response_content

    def build_kwargs(self, **kwargs):
        self.build_kwargs_calls.append(kwargs)
        return {"model": kwargs["model"], "messages": kwargs["messages"], "__fake__": True}

    def normalize_response(self, response, **_kwargs):
        return SimpleNamespace(content=self._response_content)


def _make_bedrock_agent(transport, dispatch_response):
    agent = MagicMock()
    agent.api_mode = "bedrock_converse"
    agent.max_iterations = 90
    agent.model = "us.anthropic.claude-sonnet-5"
    agent.base_url = ""
    agent.provider = "bedrock"
    agent.max_tokens = 4096
    agent.reasoning_config = None
    agent.request_overrides = {}
    agent.prefill_messages = []
    agent.ephemeral_system_prompt = None
    agent._cached_system_prompt = ""
    agent._bedrock_region = "us-east-1"
    agent._bedrock_guardrail_config = None
    agent.openrouter_min_coding_score = None
    agent._base_url_lower = ""
    agent._should_sanitize_tool_calls.return_value = False
    agent._copy_reasoning_content_for_api.side_effect = lambda msg, api_msg: None
    agent._sanitize_api_messages.side_effect = lambda msgs: msgs
    agent._drop_thinking_only_and_merge_users.side_effect = lambda msgs: msgs
    agent._supports_reasoning_extra_body.return_value = False
    agent._get_transport.return_value = transport
    # If the buggy fallback branch is ever hit again, this raises instead
    # of silently succeeding, so the test fails loudly.
    agent._ensure_primary_openai_client.side_effect = AssertionError(
        "must not build an OpenAI client for bedrock_converse agents"
    )
    return agent


def test_bedrock_agent_does_not_use_openai_client_on_iteration_limit(monkeypatch):
    transport = _FakeTransport(response_content="Did X, Y, Z before hitting the cap.")

    dispatch_mock = MagicMock(return_value="RAW_BEDROCK_RESPONSE")
    monkeypatch.setattr(
        "agent.chat_completion_helpers._dispatch_nonstreaming_api_request",
        dispatch_mock,
    )

    agent = _make_bedrock_agent(transport, dispatch_mock.return_value)
    messages = [{"role": "user", "content": "do the task"}]

    result = handle_max_iterations(agent, messages, api_call_count=90)

    assert result == "Did X, Y, Z before hitting the cap."
    assert "connection error" not in result.lower()
    agent._ensure_primary_openai_client.assert_not_called()
    dispatch_mock.assert_called_once()
    # transport.build_kwargs was called with the bedrock region/guardrail
    assert transport.build_kwargs_calls[0]["region"] == "us-east-1"


def test_bedrock_agent_retry_path_also_avoids_openai_client(monkeypatch):
    # First call returns empty content to force the retry branch.
    transport = _FakeTransport(response_content="")
    call_count = {"n": 0}

    def fake_normalize(response, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return SimpleNamespace(content="")
        return SimpleNamespace(content="Retry summary succeeded.")

    transport.normalize_response = fake_normalize

    dispatch_mock = MagicMock(return_value="RAW_BEDROCK_RESPONSE")
    monkeypatch.setattr(
        "agent.chat_completion_helpers._dispatch_nonstreaming_api_request",
        dispatch_mock,
    )

    agent = _make_bedrock_agent(transport, dispatch_mock.return_value)
    messages = [{"role": "user", "content": "do the task"}]

    result = handle_max_iterations(agent, messages, api_call_count=90)

    assert result == "Retry summary succeeded."
    agent._ensure_primary_openai_client.assert_not_called()
    assert dispatch_mock.call_count == 2

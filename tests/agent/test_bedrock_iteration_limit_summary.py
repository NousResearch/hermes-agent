"""Iteration-limit summary must work on api_mode='bedrock_converse'.

``_SUMMARY_ATTEMPT_BUILDERS`` named ``codex_responses`` and ``anthropic_messages`` and nothing
else, so ``bedrock_converse`` fell through to ``_chat_summary_attempt``, which opens with
``_ensure_primary_openai_client()``. A Bedrock session has no OpenAI credentials, so that raised,
the wrapping ``except Exception`` in ``handle_max_iterations`` swallowed it, and the user got

    I reached the maximum iterations (N) but couldn't summarize. Error: ...

instead of a summary — every single time the iteration limit was hit. These tests drive the real
``BedrockTransport`` and the real ``normalize_converse_response`` so they cover the actual wiring
(kwargs shape, sentinel keys, response normalization), stubbing only the boto3 client.
"""

from unittest.mock import patch

import pytest


def _converse_reply(text):
    """A minimal but real-shaped ``bedrock-runtime.converse()`` response."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 3, "totalTokens": 14},
    }


class _FakeBedrockClient:
    """Records every ``converse()`` call and replies from a scripted list."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= len(self.replies):
            return self.replies[len(self.calls) - 1]
        return self.replies[-1]


@pytest.fixture
def bedrock_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        # Unroutable on purpose. Post-fix nothing here leaves the process, but the PRE-fix path
        # really did reach agent.base_url over the network — that is the defect this file exists
        # for. Port 9 (discard) refuses instantly, so the demonstration that these tests fail
        # without the fix stays deterministic and offline instead of depending on a real endpoint.
        base_url="http://127.0.0.1:9/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cached_system_prompt = "SYS"
    agent.api_mode = "bedrock_converse"
    agent.model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    agent._bedrock_region = "us-east-2"
    return agent


def _run(agent, client, messages=None):
    """Drive handle_max_iterations with a stubbed bedrock runtime client."""
    from agent.chat_completion_helpers import handle_max_iterations
    from agent.transports.bedrock import BedrockTransport

    if messages is None:
        messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    with patch.object(agent, "_get_transport", return_value=BedrockTransport()), \
         patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client):
        return handle_max_iterations(agent, messages, 5), messages


class TestBedrockIterationLimitSummary:
    def test_summary_is_produced_over_converse(self, bedrock_agent):
        """The regression: a Bedrock session gets a real summary, not the 'couldn't summarize'
        error string."""
        client = _FakeBedrockClient(_converse_reply("SUMMARY"))
        out, _ = _run(bedrock_agent, client)

        assert out == "SUMMARY"
        assert "couldn't summarize" not in out
        assert len(client.calls) == 1

    def test_no_openai_client_is_ever_built(self, bedrock_agent):
        """The pre-fix path did not merely fail — it built an OpenAI-wire request carrying the
        whole conversation and addressed it to ``agent.base_url`` with a Bedrock modelId, i.e. it
        sent a Bedrock user's conversation to a non-Bedrock endpoint. Make touching that client
        fatal rather than merely unused."""
        client = _FakeBedrockClient(_converse_reply("SUMMARY"))

        def _boom(*a, **kw):
            raise AssertionError("bedrock summary must not build an OpenAI client")

        with patch.object(bedrock_agent, "_ensure_primary_openai_client", side_effect=_boom):
            out, _ = _run(bedrock_agent, client)

        assert out == "SUMMARY"

    def test_converse_kwargs_are_well_formed(self, bedrock_agent):
        """The dispatch must receive transport-built kwargs and strip both sentinel keys before
        they reach boto3, which rejects unknown params."""
        client = _FakeBedrockClient(_converse_reply("SUMMARY"))
        _run(bedrock_agent, client)

        sent = client.calls[0]
        assert sent["modelId"] == bedrock_agent.model
        assert "__bedrock_converse__" not in sent
        assert "__bedrock_region__" not in sent
        # Converse takes the system prompt in its own top-level field, never as role='system'.
        assert all(m["role"] != "system" for m in sent["messages"])

    def test_summary_offers_no_tools(self, bedrock_agent):
        """The summary prompt says 'without calling any more tools' — sending toolConfig invites
        the model to do exactly that."""
        bedrock_agent.tools = [{
            "type": "function",
            "function": {
                "name": "read_file", "description": "read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }]
        client = _FakeBedrockClient(_converse_reply("SUMMARY"))
        _run(bedrock_agent, client)

        assert "toolConfig" not in client.calls[0]

    def test_empty_first_reply_falls_through_to_the_retry_path(self, bedrock_agent):
        """Bedrock needs the branch in *both* paths: an empty summary retries once, and that retry
        must not land on the OpenAI branch either."""
        client = _FakeBedrockClient(_converse_reply(""), _converse_reply("SECOND-TRY"))

        def _boom(*a, **kw):
            raise AssertionError("bedrock summary retry must not build an OpenAI client")

        with patch.object(bedrock_agent, "_ensure_primary_openai_client", side_effect=_boom):
            out, _ = _run(bedrock_agent, client)

        assert out == "SECOND-TRY"
        assert len(client.calls) == 2

    def test_summary_is_appended_to_history(self, bedrock_agent):
        """A successful summary is the turn's assistant message, so downstream persistence and the
        next turn's prefix both see it."""
        client = _FakeBedrockClient(_converse_reply("SUMMARY"))
        _, messages = _run(bedrock_agent, client)

        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "SUMMARY"

    def test_make_client_hook_is_fatal_not_silent(self):
        """``_dispatch_nonstreaming_api_request``'s bedrock branch never calls ``make_client``.
        Pin that contract so a refactor which starts calling it cannot quietly pass None to an
        SDK."""
        from agent.chat_completion_helpers import _bedrock_summary_no_client

        with pytest.raises(AssertionError, match="own boto3 client"):
            _bedrock_summary_no_client("some_reason", kind="openai")

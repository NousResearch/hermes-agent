"""OpenRouter response-cache bypass on unchanged-request retry paths.

OpenRouter's response cache replays a stored full response for a request that
looks identical to a previous one. Three retry paths resend the request
byte-identically (empty content, invalid tool-call JSON, incomplete
<REASONING_SCRATCHPAD>); without an explicit ``X-OpenRouter-Cache: false``
bypass the retried call can be handed back the very response it is retrying
(#70387). These tests pin the bypass on all three paths.
"""

from secrets import token_hex
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import uuid

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
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


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    """Return a SimpleNamespace mimicking a tool call object."""
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    """Return a SimpleNamespace mimicking an OpenAI ChatCompletion response."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model")


@pytest.fixture(autouse=True)
def _mock_plugin_discovery(monkeypatch):
    # Tool definitions are supplied by these unit fixtures; rescanning bundled
    # plugins for each isolated test home adds no coverage.
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch(
            "model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key=token_hex(16),
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def _drive_turn(agent, responses):
    """Run one conversation with scripted responses; capture request kwargs."""
    responses = iter(responses)
    request_kwargs = []

    def _create(**kwargs):
        request_kwargs.append(kwargs)
        return next(responses)

    agent.client.chat.completions.create.side_effect = _create

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.retry_utils.jittered_backoff", lambda *args, **kwargs: 0.0),
    ):
        result = agent.run_conversation("answer me")
    return result, request_kwargs


class TestOpenRouterRetryCacheBypass:
    def _setup_agent(self, agent):
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False

    def test_invalid_json_retry_bypasses_openrouter_cache(self, agent):
        """An invalid-JSON retry resends the request unchanged, so it must bypass
        the response cache instead of replaying the broken tool call."""
        self._setup_agent(agent)
        # Ends in "}" so the truncation guard does not fire, but not valid JSON.
        broken = _mock_response(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(arguments='{"query": }')],
        )
        good = _mock_response(content="Recovered.", finish_reason="stop")

        result, request_kwargs = _drive_turn(agent, [broken, good])

        assert result["final_response"] == "Recovered."
        assert "X-OpenRouter-Cache" not in request_kwargs[0].get("extra_headers", {})
        assert request_kwargs[1]["extra_headers"]["X-OpenRouter-Cache"] == "false"

    def test_incomplete_scratchpad_retry_bypasses_openrouter_cache(self, agent):
        """An incomplete-scratchpad retry resends the request unchanged, so it
        must bypass the response cache instead of replaying the truncated output."""
        self._setup_agent(agent)
        broken = _mock_response(
            content="<REASONING_SCRATCHPAD>\nthe model ran out of tokens mid",
            finish_reason="stop",
        )
        good = _mock_response(content="Completed reasoning.", finish_reason="stop")

        result, request_kwargs = _drive_turn(agent, [broken, good])

        assert result["final_response"] == "Completed reasoning."
        assert "X-OpenRouter-Cache" not in request_kwargs[0].get("extra_headers", {})
        assert request_kwargs[1]["extra_headers"]["X-OpenRouter-Cache"] == "false"

    def test_invalid_json_retry_non_openrouter_sets_no_bypass_header(self, agent):
        """Off OpenRouter the retry must not grow the cache-bypass header."""
        self._setup_agent(agent)
        agent.base_url = "https://api.example.com/v1"
        agent._base_url_lower = agent.base_url.lower()
        broken = _mock_response(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(arguments='{"query": }')],
        )
        good = _mock_response(content="Recovered.", finish_reason="stop")

        result, request_kwargs = _drive_turn(agent, [broken, good])

        assert result["final_response"] == "Recovered."
        assert "X-OpenRouter-Cache" not in request_kwargs[1].get("extra_headers", {})

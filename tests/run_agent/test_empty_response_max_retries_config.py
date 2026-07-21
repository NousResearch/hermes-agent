"""Tests for agent.empty_response_max_retries config surface.

Closes #58670 — make the hardcoded empty-response retry count (``< 3`` in
``conversation_loop.py``) user-configurable so providers with higher
empty-response rates under load can be tuned without patching source.

Covers both the init/config surface and the conversation-loop retry guard
so a regression that ignores the configured value would fail these tests.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def _mock_response(content=None, finish_reason="stop"):
    """Return a SimpleNamespace mimicking an OpenAI ChatCompletion response."""
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
        function_call=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(empty_response_max_retries=None):
    """Build an AIAgent with a mocked config.load_config that returns a
    config tree containing the given agent.empty_response_max_retries
    (or default)."""
    cfg = {"agent": {}}
    if empty_response_max_retries is not None:
        cfg["agent"]["empty_response_max_retries"] = empty_response_max_retries

    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


def _setup_conversation_agent(agent):
    """Common setup matching TestRunConversation._setup_agent."""
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    # Local-ish base URL avoids provider-specific empty-response side paths.
    agent.base_url = "http://127.0.0.1:1234/v1"
    # No fallback chain so exhausted retries surface empty_response_exhausted.
    agent._fallback_chain = []
    agent._fallback_index = 0
    agent._fallback_activated = False


def test_default_empty_response_max_retries_is_three():
    """No config override → legacy default of 3 retries preserved."""
    agent = _make_agent()
    assert agent._empty_response_max_retries == 3


def test_empty_response_max_retries_honors_config_override():
    """Setting agent.empty_response_max_retries in config propagates."""
    agent = _make_agent(empty_response_max_retries=1)
    assert agent._empty_response_max_retries == 1

    agent2 = _make_agent(empty_response_max_retries=6)
    assert agent2._empty_response_max_retries == 6


def test_empty_response_max_retries_allows_zero_to_disable():
    """0 is a valid value — disable empty-response retries entirely so the
    agent fails over / surfaces "No reply" immediately."""
    agent = _make_agent(empty_response_max_retries=0)
    assert agent._empty_response_max_retries == 0


def test_empty_response_max_retries_clamps_negative_to_zero():
    """Negative values are meaningless for a retry ceiling → clamp to 0."""
    agent = _make_agent(empty_response_max_retries=-3)
    assert agent._empty_response_max_retries == 0


def test_empty_response_max_retries_falls_back_on_invalid_value():
    """Garbage values in config don't crash agent init — fall back to 3."""
    agent = _make_agent(empty_response_max_retries="not-a-number")
    assert agent._empty_response_max_retries == 3


def test_conversation_loop_zero_ceiling_skips_empty_retries():
    """Configured ceiling of 0 must not re-call the API after the first empty.

    Exercises the conversation_loop retry guard (not just agent_init) so a
    regression that hardcodes ``< 3`` again would fail this test.
    """
    agent = _make_agent(empty_response_max_retries=0)
    _setup_conversation_agent(agent)
    assert agent._empty_response_max_retries == 0

    empty_resp = _mock_response(content=None, finish_reason="stop")
    # Only one empty response should be consumed — further empties would
    # raise StopIteration if the loop incorrectly retried.
    agent.client.chat.completions.create.side_effect = [empty_resp]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("answer me")

    assert result["completed"] is True
    assert result.get("turn_exit_reason") == "empty_response_exhausted" or (
        result["final_response"] != "(empty)" and "No reply:" in result["final_response"]
    )
    assert result["api_calls"] == 1  # original only; 0 retries
    assert agent.client.chat.completions.create.call_count == 1


def test_conversation_loop_nondefault_ceiling_controls_api_call_count():
    """A non-default positive ceiling (2) must yield 1 original + 2 retries.

    Asserts API-call count and terminal empty_response_exhausted behavior at
    the conversation_loop guard, not only the init-time attribute.
    """
    agent = _make_agent(empty_response_max_retries=2)
    _setup_conversation_agent(agent)
    assert agent._empty_response_max_retries == 2

    empty_resp = _mock_response(content=None, finish_reason="stop")
    # 3 responses: 1 original + 2 retries, all empty. Extra empties would
    # only be consumed if the ceiling were ignored (e.g. still hardcoded 3).
    agent.client.chat.completions.create.side_effect = [
        empty_resp,
        empty_resp,
        empty_resp,
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("answer me")

    assert result["completed"] is True
    assert result["final_response"] != "(empty)"
    assert "No reply:" in result["final_response"]
    assert result["api_calls"] == 3  # 1 original + 2 retries
    assert agent.client.chat.completions.create.call_count == 3

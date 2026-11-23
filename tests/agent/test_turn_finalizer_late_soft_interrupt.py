"""A soft interrupt that carries user text (``agent.interrupt(text)``, the gateway's priority
follow-up under ``busy_input_mode: interrupt``) and lands after the final assistant message has
already completed stops nothing: the turn ran to its end. Its text is then the user's next
message, so the finalizer hands it back the way it hands back a leftover /steer instead of
dropping it with ``clear_interrupt()``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

FOLLOW_UP = "actually use Python"


def _tool_definitions(*names):
    return [
        {"type": "function", "function": {"name": name, "description": f"{name} tool",
                                           "parameters": {"type": "object", "properties": {}}}}
        for name in names
    ]


def _text_response(text):
    message = SimpleNamespace(content=text, reasoning_content=None, reasoning=None, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_definitions("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=10,
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = lambda **kwargs: _text_response("stale answer")
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = True
    return agent


def _run_turn(agent):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("first prompt")


def _interrupt_once_the_response_is_built(agent, **interrupt_kwargs):
    """Interrupt the first time a complete provider response is turned into its message row: the
    provider call has returned and its ``finally`` has run; the finalizer has not run yet."""
    build = agent._build_assistant_message
    fired = []

    def build_then_interrupt(assistant_message, finish_reason="stop", *args, **kwargs):
        row = build(assistant_message, finish_reason, *args, **kwargs)
        if not fired:
            fired.append(agent.interrupt(FOLLOW_UP, **interrupt_kwargs))
        return row

    return patch.object(agent, "_build_assistant_message", build_then_interrupt), fired


def test_soft_interrupt_after_the_final_response_is_handed_back_as_the_next_turn():
    agent = _agent()
    interrupting, fired = _interrupt_once_the_response_is_built(agent)

    with interrupting:
        result = _run_turn(agent)

    assert fired == [True]
    assert result["completed"] is True and result["interrupted"] is False
    assert result["final_response"] == "stale answer"
    assert result["pending_steer"] == FOLLOW_UP
    assert agent._interrupt_requested is False and agent._interrupt_message is None


def test_hard_interrupt_after_the_final_response_is_not_handed_back():
    """A hard stop names why the turn ended; it is not a message to run as the next turn."""
    agent = _agent()
    interrupting, fired = _interrupt_once_the_response_is_built(agent, hard_cancel=True)

    with interrupting:
        result = _run_turn(agent)

    assert fired == [True]
    assert result.get("pending_steer") is None
    assert agent._interrupt_requested is False and agent._interrupt_message is None

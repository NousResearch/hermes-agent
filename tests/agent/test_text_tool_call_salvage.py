"""A tool call the provider serialized as TEXT must resume the tool loop, not end the turn.

Some OpenRouter backends intermittently fail to lift a model's native tool-call markup into
the OpenAI ``tool_calls`` array and emit it as text — on interleaved-reasoning models, into
the reasoning channel. The turn then presents as a clean text answer with zero tool calls and
the loop finishes mid-task while reporting success.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.turn_response_intake import normalize_model_response
from run_agent import AIAgent

# A real leak, byte-for-byte from a session that died this way: the whole call rendered as
# DSML inside the reasoning channel while content came back empty.
DSML_LEAK = (
    "  responseInstalled. Now: close popup, then act.\n\n"
    "<\uff5cDSML\uff5ctool_calls>\n"
    "<\uff5cDSML\uff5cinvoke name=\"terminal\">\n"
    "<\uff5cDSML\uff5cparameter name=\"command\" string=\"true\">echo hello</\uff5cDSML\uff5cparameter>\n"
    "<\uff5cDSML\uff5cparameter name=\"timeout\" string=\"false\">8</\uff5cDSML\uff5cparameter>\n"
    "</\uff5cDSML\uff5cinvoke>\n"
    "</\uff5cDSML\uff5ctool_calls>"
)


def _response(*, content, reasoning):
    """A provider response with no structured tool calls and a clean ``stop``."""
    message = SimpleNamespace(
        content=content, tool_calls=None, reasoning_content=reasoning, reasoning=reasoning,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model", usage=None,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id="text-tool-call-salvage",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance.valid_tool_names = {"terminal", "read_file"}
    return instance


def _intake(agent, response):
    return normalize_model_response(
        agent, response=response, messages=[], api_messages=[], conversation_history=[],
        api_call_count=1, api_duration=0.1, api_start_time=0.0, api_request_id="req-1",
        effective_task_id=None, turn_id="turn-1",
    )


def test_reasoning_borne_tool_call_resumes_the_tool_loop(agent):
    """Markup leaked into reasoning becomes a structured call the tool round can run.

    The loop dispatches on ``assistant_message.tool_calls``, so this is the contract that
    decides "run the tool" versus "end the turn and call it complete".
    """
    verdict = _intake(agent, _response(content="", reasoning=DSML_LEAK))

    calls = verdict.assistant_message.tool_calls
    assert calls, "leaked tool call was not salvaged — the turn would end mid-task"
    assert [c.function.name for c in calls] == ["terminal"]

    import json
    assert json.loads(calls[0].function.arguments)["command"] == "echo hello"
    # A row carrying tool calls must not still claim the provider's original "stop".
    assert verdict.finish_reason == "tool_calls"


def test_salvage_refuses_unoffered_tools_and_unclosed_markup(agent):
    """Recovery refuses rather than guesses, so it cannot invent or half-run a call.

    An unoffered name means the text was not a real call from this turn's toolset; an unclosed
    block means the arguments may be truncated, and running half a command is worse than
    ending the turn.
    """
    unoffered = DSML_LEAK.replace('name="terminal"', 'name="rm_everything"')
    assert not _intake(agent, _response(content="", reasoning=unoffered)).assistant_message.tool_calls

    unclosed = (
        "<\uff5cDSML\uff5cinvoke name=\"terminal\">\n"
        "<\uff5cDSML\uff5cparameter name=\"command\" string=\"true\">rm -rf /tmp/half"
    )
    assert not _intake(agent, _response(content="", reasoning=unclosed)).assistant_message.tool_calls

    # Prose that merely mentions tools is untouched and still reads as the answer.
    prose = "I could invoke the terminal tool here, but the build already passed."
    verdict = _intake(agent, _response(content=prose, reasoning=""))
    assert not verdict.assistant_message.tool_calls
    assert verdict.assistant_message.content == prose

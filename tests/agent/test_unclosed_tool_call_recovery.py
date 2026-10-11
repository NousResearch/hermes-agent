"""Regression tests for unclosed text tool-call recovery.

Some OpenAI-compatible servers return a model's tool-call markup as plain content when their
tool parser cannot read it (GLM style ``<tool_call>get_weather<arg_key>city</arg_key>
<arg_value>Paris`` with no ``</tool_call>``, or a JSON body cut short), with
``finish_reason="stop"`` and no ``tool_calls``. Before the fix the loop took that as the final
answer: ``strip_think_blocks`` drops a cut block that starts a line (#101899), so the turn ended
on the prose before it, an inline opener reached the user as raw markup, and a bare fragment took
a blind empty retry. In every case the call never ran and the model was never told.

The fix re-prompts with a corrective nudge, sharing the dropped-tool-call nudge's ephemeral flag
and 3-consecutive budget. Text that only mentions ``<tool_call>`` is a normal answer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _UNCLOSED_TOOLCALL_NUDGE_CONTENT
from agent.turn_final_response import ends_in_unclosed_tool_call

GLM_CALL = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris"
ANSWER = "It is sunny in Paris."


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (same shape as test_dropped_tool_call_recovery's)."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def _run(agent, *responses):
    agent.valid_tool_names = {"get_weather"}
    agent.client.chat.completions.create.side_effect = list(responses)
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        # Unfixed code sends the bare fragment down the empty-retry backoff; keep that instant.
        patch("agent.turn_empty_response.interruptible_backoff_sleep", return_value=None),
    ):
        return agent.run_conversation("What's the weather in Paris?")


def _sent_messages(agent):
    return [c.kwargs["messages"] for c in agent.client.chat.completions.create.call_args_list]


def _nudge_count(agent):
    return sum(
        1 for msgs in _sent_messages(agent) for m in msgs
        if m.get("role") == "user" and m.get("content") == _UNCLOSED_TOOLCALL_NUDGE_CONTENT
    )


def _stop(content):
    from tests.agent.test_run_agent import _mock_response
    return _mock_response(content=content, finish_reason="stop")


class TestUnclosedToolCallRecovery:
    @pytest.mark.parametrize("content", [
        pytest.param(GLM_CALL, id="bare-glm-fragment"),
        pytest.param("Let me look that up.\n" + GLM_CALL, id="glm-after-prose-line"),
        pytest.param("Checking the forecast. " + GLM_CALL, id="glm-inline-after-prose"),
        pytest.param('Checking. <tool_call>{"name": "get_weather", "arguments": {"city": "Par', id="json-body"),
        pytest.param("Here we go: <tool_call>get_weather", id="bare-function-name"),
    ])
    def test_unclosed_call_reprompts_with_corrective_nudge(self, loop_agent, content):
        result = _run(loop_agent, _stop(content), _stop(ANSWER))

        sent = _sent_messages(loop_agent)
        assert len(sent) == 2, "An unclosed tool call must be re-elicited, not end the turn."
        # The model sees its own broken call, then the nudge naming the failure (alternation kept).
        assert sent[1][-1] == {"role": "user", "content": _UNCLOSED_TOOLCALL_NUDGE_CONTENT}
        assert sent[1][-2]["role"] == "assistant"
        assert result["final_response"] == ANSWER

    def test_nudge_pair_is_not_left_in_the_transcript(self, loop_agent):
        result = _run(loop_agent, _stop("Checking the forecast. " + GLM_CALL), _stop(ANSWER))

        assert result["completed"] is True
        assert _nudge_count(loop_agent) == 1, "the nudge must have been sent before it is cleaned up"
        assert not [m for m in result["messages"] if isinstance(m, dict) and m.get("_dropped_toolcall_nudge")]
        # The assistant row carrying the broken call is scaffolding too, not part of the answer.
        assert not [m for m in result["messages"] if GLM_CALL in str(m.get("content") or "")]
        assert not [
            m for m in result["messages"]
            if isinstance(m, dict) and m.get("content") == _UNCLOSED_TOOLCALL_NUDGE_CONTENT
        ]

    @pytest.mark.parametrize("content", [
        pytest.param(
            "Calling it.\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            id="closed-block",
        ),
        pytest.param("Wrap every call in a <tool_call> tag and close it.", id="prose-mention"),
        pytest.param("The tag looks like `<tool_call>get_weather<arg_key>city`.", id="inline-code"),
        pytest.param("Example:\n```\n" + GLM_CALL + "\n```\nThat is the format.", id="fenced-code"),
        pytest.param("Example:\n```xml\n" + GLM_CALL, id="unclosed-fence"),
    ])
    def test_text_that_only_mentions_tool_call_is_a_final_answer(self, loop_agent, content):
        _run(loop_agent, _stop(content), _stop("unexpected second call"))

        assert len(_sent_messages(loop_agent)) == 1
        assert _nudge_count(loop_agent) == 0

    def test_cap_reached_falls_through_to_final_text(self, loop_agent):
        broken = "Checking the forecast. " + GLM_CALL
        result = _run(loop_agent, *[_stop(broken) for _ in range(9)], _stop(ANSWER))

        # 1 initial call + 3 re-prompts; the 4th broken reply ends the turn as today.
        sent = _sent_messages(loop_agent)
        assert len(sent) == 4
        assert [m.get("content") for m in sent[-1]].count(_UNCLOSED_TOOLCALL_NUDGE_CONTENT) == 3
        assert result["final_response"].startswith("Checking the forecast.")

    def test_tool_calls_finish_reason_keeps_the_dropped_call_nudge(self, loop_agent):
        from agent.conversation_loop import _DROPPED_TOOLCALL_NUDGE_CONTENT
        from tests.agent.test_run_agent import _mock_response

        _run(
            loop_agent,
            _mock_response(content="Checking the forecast. " + GLM_CALL, finish_reason="tool_calls"),
            _stop(ANSWER),
        )

        assert _sent_messages(loop_agent)[1][-1]["content"] == _DROPPED_TOOLCALL_NUDGE_CONTENT
        assert _nudge_count(loop_agent) == 0

    def test_length_finish_reason_is_left_to_the_continuation_path(self, loop_agent):
        from tests.agent.test_run_agent import _mock_response

        _run(
            loop_agent,
            _mock_response(content="Checking the forecast. " + GLM_CALL, finish_reason="length"),
            *[_stop(ANSWER) for _ in range(4)],
        )

        assert _nudge_count(loop_agent) == 0


@pytest.mark.parametrize("text, expected", [
    (GLM_CALL, True),
    ("Text.\n" + GLM_CALL, True),
    ("<tool_call>get_weather<arg_key>city</arg_key>", True),
    ('<tool_call>\n{"name": "get_weather", "arguments": {', True),
    ("<tool_call>{'name': 'get_weather'", True),
    ("<TOOL_CALL>get_weather<ARG_KEY>city", True),
    ("<ns:tool_call>get_weather<arg_key>city", True),
    ("<tool_call>get_weather", True),
    ("<tool_call>get_weather\n", True),
    ("<think>maybe <tool_call>x</think>Now: <tool_call>get_weather<arg_key>city", True),
    ("<tool_call>get_weather<arg_key>city</arg_key></tool_call>", False),
    ("Done.<tool_call>a<arg_key>b</arg_key></tool_call> Then I said <tool_call> again", False),
    ("<tool_call>unknown_tool", False),
    ("Use <tool_call> tags", False),
    ("Calling now <tool_call>", False),
    ("<tool_call>this is just prose after a tag", False),
    ('<tool_call>{"city": "Paris"', False),
    ("<think>I could emit <tool_call>get_weather<arg_key>city</think>Sunny.", False),
    ("`<tool_call>get_weather<arg_key>city`", False),
    ("```\n<tool_call>get_weather<arg_key>city\n```", False),
    ("~~~\n<tool_call>get_weather\n~~~", False),
    ("plain answer", False),
    (None, False),
])
def test_ends_in_unclosed_tool_call(text, expected):
    assert ends_in_unclosed_tool_call(text, {"get_weather", "a"}) is expected


def test_bare_name_needs_an_offered_tool():
    assert ends_in_unclosed_tool_call("<tool_call>get_weather", set()) is False
    # Argument markup is a call whatever the tool list says.
    assert ends_in_unclosed_tool_call(GLM_CALL, set()) is True

"""A tool call leaked into the reply as its bare arguments is re-prompted, not shown as the answer.

gpt-oss (Harmony) on Ollama sometimes writes the intended call into the content channel: the reply is
``{"command": "date", ...}`` with ``finish_reason=stop`` and no ``tool_calls`` (2026-10-16, ~1 in 12
replays of one Hermes request). It now takes the existing dropped-tool-call re-prompt path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_final_response import leaked_tool_args_target

TERMINAL = {"type": "function", "function": {"name": "terminal", "description": "Run a shell command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                           "background": {"type": "boolean"}, "timeout": {"type": "integer"}},
                           "required": ["command"]}}}
READ_FILE = {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
SHELL_TOO = {"type": "function", "function": {"name": "shell", "description": "Also a shell.",
             "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
LEAK = '{\n  "command": "date \'+%A\'",\n  "background": false,\n  "timeout": 30\n}'
PLAN = "The user wants the weekday. We should call terminal with date."


class TestDetector:
    def test_real_leak_shape(self):
        assert leaked_tool_args_target(LEAK, [TERMINAL, READ_FILE], PLAN) == "terminal"

    def test_reasoning_must_name_the_tool(self):
        assert leaked_tool_args_target(LEAK, [TERMINAL], "Return the JSON the user asked for.") is None

    def test_no_reasoning_is_not_a_leak(self):
        assert leaked_tool_args_target(LEAK, [TERMINAL], None) is None

    def test_ambiguous_tools_not_a_leak(self):
        assert leaked_tool_args_target('{"command": "date"}', [TERMINAL, SHELL_TOO], "call terminal or shell") is None

    def test_unknown_key_not_a_leak(self):
        assert leaked_tool_args_target('{"command": "date", "color": "red"}', [TERMINAL], PLAN) is None

    def test_missing_required_key_not_a_leak(self):
        assert leaked_tool_args_target('{"timeout": 30}', [TERMINAL], PLAN) is None

    @pytest.mark.parametrize("text", ["Today is Friday.", "", "[1, 2]", "{not json}", '{"command": "date"} and more'])
    def test_non_object_replies(self, text):
        assert leaked_tool_args_target(text, [TERMINAL], PLAN) is None


def _response(content, finish_reason, reasoning=None):
    msg = SimpleNamespace(content=content, tool_calls=None)
    if reasoning is not None:
        msg.reasoning = reasoning
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
                           model="gpt-oss:120b", usage=None)


@pytest.fixture()
def agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[TERMINAL]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(api_key="ollama", base_url="http://10.69.20.2:11434/v1", provider="custom",
                    model="gpt-oss:120b", quiet_mode=True, skip_context_files=True, skip_memory=True)
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _run(agent, side_effect):
    agent.client.chat.completions.create.side_effect = side_effect
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("What weekday is it? Use the terminal.")


def _sent(agent, i):
    call = agent.client.chat.completions.create.call_args_list[i]
    return call.kwargs.get("messages") or call.args[0].get("messages")


def test_leaked_call_is_reprompted(agent):
    result = _run(agent, [_response(LEAK, "stop", PLAN), _response("Today is Friday.", "stop")])
    assert agent.client.chat.completions.create.call_count == 2, "leaked call was shown as the answer"
    last = _sent(agent, 1)[-1]
    assert last["role"] == "user" and "tool call" in last["content"], last
    assert result["final_response"] == "Today is Friday."
    assert not any(m.get("content") == LEAK for m in result["messages"] if isinstance(m, dict)), \
        "leaked arguments must not stay in the durable transcript"


def test_plain_json_answer_is_left_alone(agent):
    result = _run(agent, [_response('{"command": "date"}', "stop", "The user asked for this JSON verbatim.")])
    assert agent.client.chat.completions.create.call_count == 1
    assert result["final_response"] == '{"command": "date"}'

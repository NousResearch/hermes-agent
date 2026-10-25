"""A plugin that goes live while a turn runs is told to the model in that turn, once.

Onboarding is one long turn: the plugins card blocks inside ``setup_beat`` while the user installs. In
a live run the go-live notice waited for the next user message and arrived six minutes after the
install, after the model had already failed to use the plugin. The notice now rides the next tool
result (a new row, so the cached prefix is untouched); the next user message carries only what no
tool result did, and nothing is told twice.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.plugin_live_notes import queue_plugin_notice
from run_agent import AIAgent
from tools.budget_config import BudgetConfig

NOTE = "[System note: plugin blender was installed and is live in this chat now.]"
OLD_NOTE = "[System note: plugin blender (an older install) is live.]"
TOOL_OUTPUT = json.dumps({"results": ["a web page about Blender scripting"]})


@pytest.fixture(autouse=True)
def _no_background_titles(monkeypatch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)


def _agent():
    home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (home / "logs").mkdir(parents=True, exist_ok=True)
    tool = {"type": "function", "function": {"name": "web_search", "description": "search",
                                             "parameters": {"type": "object", "properties": {}}}}
    with (
        patch("model_tools.get_tool_definitions", return_value=[tool]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _response(content="", tool_call_id=None):
    calls = [SimpleNamespace(id=tool_call_id, type="function",
                             function=SimpleNamespace(name="web_search", arguments="{}"))] if tool_call_id else None
    message = SimpleNamespace(content=content, tool_calls=calls)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _script(agent, *responses):
    """Answer the model calls in order; return the request each call was sent."""
    requests, pending = [], list(responses)

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        return pending.pop(0)

    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = create
    return requests


def _run(agent, text, history=None, tool_result=lambda: TOOL_OUTPUT):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", side_effect=lambda *a, **k: tool_result()),
    ):
        return agent.run_conversation(text, conversation_history=history)


def _sent_text(message) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else json.dumps(content)


def _install_during_tool():
    """The plugins card: the user installs (twice: an update) while the tool blocks."""
    return lambda agent: (queue_plugin_notice(agent, "blender", OLD_NOTE),
                          queue_plugin_notice(agent, "blender", NOTE), TOOL_OUTPUT)[-1]


def test_a_plugin_installed_during_a_tool_is_told_in_its_result_and_not_again_next_turn():
    agent = _agent()
    requests = _script(agent, _response(tool_call_id="c1"), _response("done"), _response("gg back"))
    install = _install_during_tool()

    first = _run(agent, "set me up", tool_result=lambda: install(agent))
    _run(agent, "gg", history=first["messages"])

    in_turn = requests[1]  # the model call right after the tool result
    tool_row = next(m for m in in_turn if m.get("role") == "tool")
    # After the tool's output and outside the untrusted-data wrapper a web result gets: Hermes' own note.
    assert TOOL_OUTPUT in tool_row["content"]
    assert tool_row["content"].endswith(f"</untrusted_tool_result>\n\n{NOTE}")
    assert OLD_NOTE not in json.dumps(in_turn)  # a reinstall replaces the plugin's older notice

    next_turn = requests[2]
    assert next_turn[-1]["role"] == "user" and NOTE not in _sent_text(next_turn[-1])
    assert json.dumps(next_turn).count(NOTE.replace('"', '\\"')) == 1


def test_a_notice_no_tool_result_carried_rides_the_next_user_message_once():
    agent = _agent()
    requests = _script(agent, _response("hi"), _response("still here"))

    queue_plugin_notice(agent, "blender", NOTE)
    first = _run(agent, "hello")
    _run(agent, "and now?", history=first["messages"])

    assert NOTE in _sent_text(requests[0][-1])
    assert NOTE not in _sent_text(requests[1][-1])


def test_the_notice_survives_the_turn_budget_spilling_its_tool_result():
    agent = _agent()
    requests = _script(agent, _response(tool_call_id="c1"), _response("done"))
    big = "x" * 20_000
    install = _install_during_tool()

    with patch("agent.tool_executor._budget_for_agent", return_value=BudgetConfig(turn_budget=16_000)):
        _run(agent, "set me up", tool_result=lambda: (install(agent), big)[-1])

    tool_row = next(m for m in requests[1] if m.get("role") == "tool")
    assert big not in tool_row["content"]  # spilled to a file, a preview stays inline
    assert tool_row["content"].endswith(NOTE)

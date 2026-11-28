"""A write_file stale-write refusal stays between the tool and the model (#155997).

The refusal is a precondition, not a failed edit: nothing was written, and its text tells the model to
read the file and write again. The turn-end file-mutation verifier footer must not turn it into chat text.
Driven through the real turn loop and the real write_file / read_file handlers, with a scripted model.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _call(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(id=f"call_{uuid.uuid4().hex[:8]}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _response(content=None, tool_calls=None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _agent() -> AIAgent:
    tools = [{"type": "function", "function": {"name": n, "description": n,
                                               "parameters": {"type": "object", "properties": {}}}}
             for n in ("write_file", "read_file", "patch")]
    with (
        patch("model_tools.get_tool_definitions", return_value=tools),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", max_iterations=10,
                        quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli")
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run_turn(agent: AIAgent, script: list) -> dict:
    replies = iter(script)
    with patch.object(agent, "_interruptible_api_call", side_effect=lambda *a, **k: next(replies)):
        return agent.run_conversation("Put 'new' into notes.md", task_id=f"task-{uuid.uuid4().hex[:8]}")


@pytest.fixture
def notes(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.delenv("HERMES_FILE_MUTATION_VERIFIER", raising=False)
    path = tmp_path / "notes.md"
    path.write_text("written by the user\n", encoding="utf-8")
    return path


def test_refused_write_the_agent_then_settles_without_rewriting_adds_no_footer(notes):
    final = "notes.md already says that, so I left it unchanged."
    result = _run_turn(_agent(), [
        _response(tool_calls=[_call("write_file", {"path": str(notes), "content": "new\n"})]),
        _response(tool_calls=[_call("read_file", {"path": str(notes)})]),
        _response(content=final),
    ])

    refusal = json.loads(next(m["content"] for m in result["messages"] if m.get("role") == "tool"))
    assert refusal.get("stale_write_blocked") is True  # the guard did refuse, and the file is untouched
    assert notes.read_text(encoding="utf-8") == "written by the user\n"
    assert result["final_response"] == final


def test_a_failed_edit_in_the_same_turn_is_still_reported(notes):
    other = notes.with_name("other.md")
    other.write_text("unrelated\n", encoding="utf-8")
    result = _run_turn(_agent(), [
        _response(tool_calls=[_call("write_file", {"path": str(notes), "content": "new\n"}),
                              _call("patch", {"mode": "replace", "path": str(other),
                                              "old_string": "not in the file", "new_string": "new"})]),
        _response(content="Updated both files."),
    ])

    footer = result["final_response"].split("Updated both files.", 1)[1]
    assert "other.md` — [patch]" in footer
    assert "[write_file]" not in footer and "Refusing to overwrite" not in footer

"""The sequential executor turns a repeat skill_view of an edited skill into a transcript-verified
delta, and leaves the result whole when the transcript lacks the baseline."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.tool_dispatch_helpers import make_tool_result_message
from tests.agent.test_tool_call_incremental_persistence import _make_agent, _mock_tool_call

_BODY = "".join(f"## Section {i}\n\n- Rule {i}: keep ids.\n\n" for i in range(1, 300))


def _result(body):
    return json.dumps({"success": True, "name": "demo", "description": "d", "content": body,
                       "path": "demo/SKILL.md", "linked_files": None})


def _run(agent, messages, function_result):
    tool_calls = [_mock_tool_call(name="skill_view", arguments=json.dumps({"name": "demo"}), call_id="call_new")]
    agent._flush_messages_to_session_db = MagicMock()
    with patch("model_tools.handle_function_call", return_value=function_result):
        agent._execute_tool_calls_sequential(SimpleNamespace(content="", tool_calls=tool_calls), messages, "task-1")
    return json.loads(messages[-1]["content"])


def _with_baseline(body):
    return [{"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_old", "type": "function",
                 "function": {"name": "skill_view", "arguments": json.dumps({"name": "demo"})}}]},
            make_tool_result_message("skill_view", _result(body), "call_old")]


def test_sequential_commit_serves_a_delta_against_the_visible_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    edited = _BODY.replace("- Rule 77: keep ids.", "- Rule 77: keep ids, never names.")
    out = _run(_make_agent(), _with_baseline(_BODY), _result(edited))
    assert out.get("delta") is True and "+- Rule 77: keep ids, never names." in out["diff"]


def test_sequential_commit_keeps_full_content_without_a_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    edited = _BODY.replace("- Rule 77: keep ids.", "- Rule 77: keep ids, never names.")
    out = _run(_make_agent(), [], _result(edited))
    assert out.get("delta") is None and out["content"] == edited

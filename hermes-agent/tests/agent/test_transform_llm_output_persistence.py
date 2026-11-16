"""#44239: what ``transform_llm_output`` shows the user is what the session stores and replays.

The hook used to fire after the final assistant row had been flushed to SQLite, so
``final_response`` carried the rewritten text while ``messages[-1]``, the SQLite row and the
next turn's replay kept the raw model text.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent


def _rewriting_hook(calls):
    def invoke_hook(hook_name, **kwargs):
        calls.append(hook_name)
        if hook_name == "transform_llm_output":
            return ["REWRITTEN:" + kwargs["response_text"]]
        return []
    return invoke_hook


def _fake_completion(text):
    def create(**kwargs):
        msg = SimpleNamespace(content=text, tool_calls=None, reasoning=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15), model="fake/model",
        )
    return create


@pytest.fixture
def db_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    db = SessionDB(db_path=tmp_path / ".hermes" / "state.db")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="fake/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli",
            session_id="sess-44239", session_db=db,
        )
    agent.client = MagicMock()
    return agent, db


def test_transformed_reply_is_the_stored_and_replayed_text(db_agent, monkeypatch):
    agent, db = db_agent
    calls = []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _rewriting_hook(calls))
    agent.client.chat.completions.create = _fake_completion("RAW MODEL TEXT")

    result = agent.run_conversation("hello")

    last_assistant = next(m for m in reversed(result["messages"]) if m.get("role") == "assistant")
    stored = [r["content"] for r in db.get_messages("sess-44239") if r["role"] == "assistant"]
    assert result["final_response"] == "REWRITTEN:RAW MODEL TEXT"
    assert last_assistant["content"] == result["final_response"]
    assert stored == [result["final_response"]]
    assert result["response_transformed"] is True
    assert result["pre_transform_response"] == "RAW MODEL TEXT"
    # Exactly once per turn: a second firing would nest the prefix.
    assert calls.count("transform_llm_output") == 1


def test_recovery_path_tail_row_carries_transformed_text(db_agent, monkeypatch):
    """A recovery ``break`` returns ``final_response`` with no closing assistant row; the row
    finalize_turn appends must be the transformed text, and the hook still fires once."""
    from agent.turn_finalizer import finalize_turn

    agent, _db = db_agent
    calls = []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _rewriting_hook(calls))
    agent._persist_session = lambda *a, **k: None
    agent._current_turn_id = "turn-r"
    messages = [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"ok": True})},
    ]

    result = finalize_turn(
        agent, final_response="RECOVERED TEXT", api_call_count=1, interrupted=False, failed=False,
        messages=messages, conversation_history=None, effective_task_id="task-1", turn_id="turn-r",
        user_message="do a thing", original_user_message="do a thing", _should_review_memory=False,
        _turn_exit_reason="partial_stream_recovery",
    )

    assert result["final_response"].startswith("REWRITTEN:RECOVERED TEXT")
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "REWRITTEN:RECOVERED TEXT"
    assert calls.count("transform_llm_output") == 1

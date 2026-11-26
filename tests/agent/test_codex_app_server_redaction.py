"""Codex app-server rows reach state.db redacted, like the chat path (#43666).

``build_assistant_message`` masks ``content`` and the unsigned reasoning copies before a chat
completion row is stored; the app-server runtime builds its rows in ``CodexEventProjector`` and
flushes them from ``_persist_projected_messages`` instead, so the same scrub has to run there.
Redaction off keeps every row verbatim.
"""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.codex_runtime import run_codex_app_server_turn
from agent.transports.codex_event_projector import CodexEventProjector
from hermes_state import SessionDB
from run_agent import AIAgent

_SECRET = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
_THOUGHT = f"I will use the token {_SECRET} to call the API."
_REPLY = f"Done. Your token is {_SECRET}."


def _completed(item: dict) -> dict:
    return {"method": "item/completed", "params": {"item": item}}


def _projected_turn(*items: dict) -> SimpleNamespace:
    projector = CodexEventProjector()
    rows: list[dict] = []
    final_text = None
    for item in items:
        projection = projector.project(_completed(item))
        rows.extend(projection.messages)
        final_text = projection.final_text if projection.final_text is not None else final_text
    return SimpleNamespace(interrupted=False, error=None, thread_id="thread-1", turn_id="turn-1",
                           projected_messages=rows, tool_iterations=0, final_text=final_text, should_retire=False)


def _tool_then_reply_turn() -> SimpleNamespace:
    return _projected_turn(
        {"type": "reasoning", "id": "r1", "summary": [_THOUGHT], "content": []},
        {"type": "commandExecution", "id": "c1", "command": "gh auth status", "cwd": "/w",
         "aggregatedOutput": "ok", "exitCode": 0},
        {"type": "reasoning", "id": "r2", "summary": [], "content": [_THOUGHT]},
        {"type": "agentMessage", "id": "a1", "text": _REPLY},
    )


def _run(tmp_path, turn) -> tuple[dict, list[tuple]]:
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        db.create_session(session_id="s1", source="cli", model="codex")
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, session_db=db, session_id="s1")
        agent._session_db_created = True
        agent._codex_session = MagicMock()
        agent._codex_session.run_turn.return_value = turn
        agent.tool_progress_callback = None
        messages = [{"role": "user", "content": "check my token"}]
        agent._flush_messages_to_session_db(messages)  # turn-start flush owns the user row
        result = run_codex_app_server_turn(agent, user_message="check my token",
                                           original_user_message="check my token", messages=messages,
                                           effective_task_id="task-1")
    finally:
        db.close()
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT role, content, reasoning, tool_calls FROM messages ORDER BY id").fetchall()
    con.close()
    return result, rows


def test_codex_reasoning_and_content_persist_redacted(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True, raising=False)
    result, rows = _run(tmp_path, _tool_then_reply_turn())

    assert [r[0] for r in rows] == ["user", "assistant", "tool", "assistant"]
    _user, tool_call_row, _tool, reply_row = rows
    assert tool_call_row[2] and _SECRET not in tool_call_row[2]  # reasoning stashed on the tool round
    assert reply_row[2] and _SECRET not in reply_row[2]
    assert reply_row[1] and _SECRET not in reply_row[1]
    assert "gh auth status" in tool_call_row[3]  # tool-call arguments stay verbatim (#43083)
    assert not any(_SECRET in str(field) for row in rows for field in row)
    # The in-memory history the agent keeps is the stored copy too.
    assert not any(_SECRET in str(m.get("content")) + str(m.get("reasoning")) for m in result["messages"])


def test_redaction_off_keeps_codex_rows_verbatim(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)
    _result, rows = _run(tmp_path, _tool_then_reply_turn())

    _user, tool_call_row, _tool, reply_row = rows
    assert tool_call_row[2] == _THOUGHT
    assert reply_row[2] == _THOUGHT
    assert reply_row[1] == _REPLY


@pytest.mark.parametrize("held", [False, True])
def test_final_reply_with_a_secret_stays_the_visible_final_row(tmp_path, monkeypatch, held):
    """The scrub runs after the reply transform, which finds the final row by its raw text: a held
    turn hides only the earlier commentary, never the (now masked) final answer."""
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True, raising=False)
    turn = _projected_turn({"type": "agentMessage", "id": "a0", "text": "Checking."},
                           {"type": "agentMessage", "id": "a1", "text": _REPLY})
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        db.create_session(session_id="s1", source="cli", model="codex")
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, session_db=db, session_id="s1")
        agent._session_db_created = True
        agent._codex_session = MagicMock()
        agent._codex_session.run_turn.return_value = turn
        agent.tool_progress_callback = None
        agent._hold_stream = held
        result = run_codex_app_server_turn(agent, user_message="hi", original_user_message="hi",
                                           messages=[{"role": "user", "content": "hi"}], effective_task_id="t")
    finally:
        db.close()

    commentary, final = [m for m in result["messages"] if m["role"] == "assistant"]
    assert _SECRET not in final["content"]
    assert "display_kind" not in final
    assert commentary.get("display_kind") == ("hidden" if held else None)

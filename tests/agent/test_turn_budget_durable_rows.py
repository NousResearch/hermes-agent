"""A tool result the per-turn budget spills must replay as the stub it was sent as.

Each tool result is flushed to the session store as it lands; the turn budget then swaps the
largest ones for ``<persisted-output>`` previews in memory. The gateway, the TUI resume path and
``--resume`` rebuild history from the store, so a row left holding the full body comes back on the
next turn: the request grows by everything the budget removed (past ``context_length`` in the live
repro) and its cached prefix diverges at the first spilled row.
"""

import contextlib
import os
import sqlite3
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.budget_config import BudgetConfig
from tools.tool_result_storage import PERSISTED_OUTPUT_TAG

_LARGE = "L" * 4_000


def _tc(name, arguments, call_id):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": n,
                                              "parameters": {"type": "object", "properties": {}}}}
            for n in names]


@pytest.fixture()
def agent_and_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs("web_search", "terminal")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            session_db=db, session_id=f"sess-{uuid.uuid4().hex[:8]}",
        )
    agent.client = MagicMock()
    yield agent, db
    db.close()


def _run_batch(agent, calls):
    messages = [{"role": "user", "content": "read them"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]}]
    budget = BudgetConfig(default_result_size=100_000, turn_budget=6_000, preview_size=64)

    def fake_handle(name, args, task_id, **kwargs):
        return _LARGE if "large" in kwargs["tool_call_id"] else "small"

    with (
        patch("model_tools.handle_function_call", side_effect=fake_handle),
        patch("agent.tool_executor._budget_for_agent", return_value=budget),
    ):
        agent._execute_tool_calls(SimpleNamespace(content="", tool_calls=calls), messages, "task-1")
    return messages


@pytest.mark.parametrize(
    "calls",
    [
        [_tc("web_search", '{"query":"a"}', "p-large-1"), _tc("web_search", '{"query":"b"}', "p-large-2"),
         _tc("web_search", '{"query":"c"}', "p-small")],
        [_tc("terminal", '{"command":"a"}', "s-large-1"), _tc("terminal", '{"command":"b"}', "s-large-2"),
         _tc("terminal", '{"command":"c"}', "s-small")],
        [_tc("web_search", '{"query":"a"}', "m-large-1"), _tc("terminal", '{"command":"b"}', "m-large-2"),
         _tc("terminal", '{"command":"c"}', "m-small")],
    ],
    ids=["parallel", "sequential", "mixed"],
)
def test_replayed_history_carries_the_budget_stub_that_was_sent(agent_and_db, calls):
    agent, db = agent_and_db
    messages = _run_batch(agent, calls)

    live = {m["tool_call_id"]: m["content"] for m in messages if m.get("role") == "tool"}
    spilled = [cid for cid, content in live.items() if PERSISTED_OUTPUT_TAG in content]
    assert spilled, "the fixture must make the turn budget spill at least one result"

    # The gateway's live-replay read (gateway/session_transcript.load_transcript).
    replay = db.get_messages_as_conversation(agent.session_id, repair_alternation=True)
    replayed = {m["tool_call_id"]: m["content"] for m in replay if m.get("role") == "tool"}
    assert replayed == live


def test_spilled_full_body_stays_searchable_and_next_flush_appends_once(agent_and_db):
    agent, db = agent_and_db
    calls = [_tc("web_search", '{"query":"a"}', "q-large-1"), _tc("web_search", '{"query":"b"}', "q-large-2")]
    messages = _run_batch(agent, calls)

    with contextlib.closing(sqlite3.connect(db.db_path)) as conn:
        full_bodies = conn.execute(
            "SELECT active FROM messages WHERE session_id = ? AND role = 'tool' AND instr(content, ?) > 0 "
            "ORDER BY id", (agent.session_id, _LARGE)).fetchall()
    # The result the budget kept inline, plus the archived twin of the one it spilled.
    assert full_bodies == [(1,), (0,)]

    messages.append({"role": "assistant", "content": "done"})
    assert agent._flush_messages_to_session_db(messages) is True
    replay = db.get_messages_as_conversation(agent.session_id, repair_alternation=True)
    assert [m["role"] for m in replay] == ["user", "assistant", "tool", "tool", "assistant"]



def test_a_process_that_lost_the_turn_lease_leaves_the_rows_alone(agent_and_db):
    agent, db = agent_and_db
    assert db.try_acquire_session_turn_lease(agent.session_id, "process-a", ttl_seconds=60)
    agent._active_session_turn_lease_holder = "process-a"
    rewrite = db.rewrite_pruned_rows

    def taken_over_first(*args, **kwargs):
        # Another process takes the turn after A flushed the full rows and before A's rewrite commits.
        db.release_session_turn_lease(agent.session_id, "process-a")
        assert db.try_acquire_session_turn_lease(agent.session_id, "process-b", ttl_seconds=60)
        return rewrite(*args, **kwargs)

    db.rewrite_pruned_rows = taken_over_first
    calls = [_tc("web_search", '{"query":"a"}', "l-large-1"), _tc("web_search", '{"query":"b"}', "l-large-2")]
    _run_batch(agent, calls)

    stored = [m["content"] for m in db.get_messages_as_conversation(agent.session_id) if m.get("role") == "tool"]
    assert stored and all(_LARGE in body and PERSISTED_OUTPUT_TAG not in body for body in stored), (
        "a rewrite under another process's lease must not land")

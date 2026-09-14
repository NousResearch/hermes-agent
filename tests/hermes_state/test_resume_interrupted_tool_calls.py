"""A session killed mid-tool must tell the resumed model what was in flight (#99869).

The agent loop persists the assistant ``tool_calls`` turn before the tools run,
so a crash / SIGKILL / watchdog kill leaves the durable transcript ending with
that turn and no result rows. On resume, ``repair_message_sequence`` Pass 2
used to prune those unanswered calls — the model resumed with no trace that a
``write_file`` had been in progress, and trusted stale on-disk state.

``_answer_interrupted_tool_calls`` gives each trailing unanswered call an
explicit synthetic result instead, and ONLY the trailing run: an unanswered
call followed by later turns is the compression-displacement shape Pass 2
exists for.
"""

import json

import pytest

from agent.context_compressor import _DB_PERSISTED_MARKER
from hermes_state import INTERRUPTED_TOOL_CALL_RESULT, SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _call(call_id, name="write_file", **args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _tool_results(messages):
    return [m for m in messages if m["role"] == "tool"]


def test_trailing_unanswered_call_gets_interrupted_result(db):
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "cap the loop-state files")
    db.append_message(
        "s1", "assistant", "Applying cap.",
        tool_calls=[_call("call_1", path="loop-state/ci-sweeper.md")],
    )
    # Process dies here: no tool result row, no end_session().

    model, display = db.get_resume_conversations("s1")

    assert model[-2]["role"] == "assistant"
    assert model[-2]["tool_calls"][0]["function"]["name"] == "write_file"
    stub = model[-1]
    assert stub["role"] == "tool"
    assert stub["tool_call_id"] == "call_1"
    assert stub["tool_name"] == "write_file"
    assert stub["content"] == INTERRUPTED_TOOL_CALL_RESULT
    # Never re-flushed as a real row; a repeat resume regenerates it.
    assert stub[_DB_PERSISTED_MARKER] is True
    # Display projection keeps the historical shape.
    assert _tool_results(display) == []


def test_partially_answered_batch_only_stubs_the_missing_calls(db):
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "cap both files")
    db.append_message(
        "s1", "assistant", None,
        tool_calls=[_call("call_a", path="a.md"), _call("call_b", path="b.md")],
    )
    db.append_message("s1", "tool", "ok", tool_name="write_file", tool_call_id="call_a")
    # Dies during call_b.

    model, _ = db.get_resume_conversations("s1")

    results = _tool_results(model)
    assert [r["tool_call_id"] for r in results] == ["call_a", "call_b"]
    assert results[0]["content"] == "ok"
    assert results[1]["content"] == INTERRUPTED_TOOL_CALL_RESULT


def test_mid_transcript_unanswered_call_is_still_pruned(db):
    """Not trailing => compression-displacement shape, Pass 2 semantics unchanged."""
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "first")
    db.append_message("s1", "assistant", "old", tool_calls=[_call("call_old")])
    db.append_message("s1", "user", "second")
    db.append_message("s1", "assistant", "done")

    model, _ = db.get_resume_conversations("s1")

    assert _tool_results(model) == []
    assert not any(m.get("tool_calls") for m in model)


def test_clean_transcript_is_untouched(db):
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "hi")
    db.append_message("s1", "assistant", "x", tool_calls=[_call("call_1")])
    db.append_message("s1", "tool", "ok", tool_name="write_file", tool_call_id="call_1")
    db.append_message("s1", "assistant", "all done")
    db.end_session("s1", "cli_close")

    model, _ = db.get_resume_conversations("s1")

    assert [r["content"] for r in _tool_results(model)] == ["ok"]

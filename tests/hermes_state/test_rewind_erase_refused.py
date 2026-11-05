"""The owner-side durable erase of a safety-refused turn (``erase_refused_turn_from_transcript``).

Regression for the "Hermes Bot stuck sessions" report: a refusal is deterministic for the whole
conversation, so a refused turn left in the durable transcript re-trips the filter on every later
message — including after a reload. CLI and ACP own a direct ``SessionDB``, so they erase the whole
refused turn (user row + any tool rounds + the refusal) owner-side after the turn. This pins that the
erase removes EXACTLY the last turn, never a prior accepted one, and fails safe.
"""
from hermes_state import SessionDB
from hermes_state_rewind import erase_refused_turn_from_transcript


def test_erase_removes_only_the_last_refused_turn_including_tool_rounds(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        # An accepted exchange, then a refused turn that ran a tool before the refusal.
        db.append_message("s", "user", "hello")
        db.append_message("s", "assistant", "hi there")
        db.append_message("s", "user", "the request the filter blocked")
        db.append_message("s", "assistant", "", tool_calls=[{"id": "t1", "type": "function",
                          "function": {"name": "shell", "arguments": "{}"}}])
        db.append_message("s", "tool", "ran", tool_call_id="t1")

        assert erase_refused_turn_from_transcript(db, "s") is True

        active = [(m["role"], m["content"]) for m in db.get_messages("s") if m["role"] != "session_meta"]
        # The whole refused turn is gone — the user ask, the tool round, everything after it — and
        # only the prior accepted exchange survives.
        assert active == [("user", "hello"), ("assistant", "hi there")]
    finally:
        db.close()


def test_erase_is_best_effort_and_a_no_op_without_a_target():
    # No DB / no session id: never raises, reports nothing erased.
    assert erase_refused_turn_from_transcript(None, "s") is False
    assert erase_refused_turn_from_transcript(object(), "") is False


def test_erase_swallows_a_rewind_failure(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        # No user turn to rewind -> rewind_user_turn raises; the helper must not propagate it.
        assert erase_refused_turn_from_transcript(db, "s") is False
    finally:
        db.close()

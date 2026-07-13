import pytest

import hermes_undo
from hermes_state import SessionDB
from hermes_undo import compute_half_turn_target


@pytest.fixture()
def db(tmp_path, monkeypatch):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(hermes_undo, "_session_db", session_db)
    hermes_undo.clear_state()
    yield session_db
    session_db.close()
    hermes_undo.clear_state()


def _session(db, sid="s1"):
    db.create_session(sid, source="cli")
    return sid


def _ids(messages):
    return {m["role"]: m["id"] for m in messages}


def test_tool_call_assistant_run_is_one_half_turn(db):
    sid = _session(db)
    u = db.append_message(sid, "user", "run tools")
    a = db.append_message(
        sid,
        "assistant",
        None,
        tool_calls=[{"id": "c1"}, {"id": "c2"}],
    )
    t1 = db.append_message(sid, "tool", "one", tool_call_id="c1")
    t2 = db.append_message(sid, "tool", "two", tool_call_id="c2")
    final = db.append_message(sid, "assistant", "done")
    msgs = db.get_messages(sid)

    assert compute_half_turn_target(msgs, 1) == a
    assert compute_half_turn_target(msgs, 2) == u
    assert [t1, t2, final] == [m["id"] for m in msgs[2:]]


def test_partial_assistant_tool_turn_starts_at_assistant(db):
    sid = _session(db)
    u = db.append_message(sid, "user", "run")
    a = db.append_message(sid, "assistant", None, tool_calls=[{"id": "c1"}])
    db.append_message(sid, "tool", "result", tool_call_id="c1")

    msgs = db.get_messages(sid)
    assert compute_half_turn_target(msgs, 1) == a
    assert compute_half_turn_target(msgs, 2) == u


def test_adjacent_multimodal_and_text_user_rows_are_one_half_turn(db):
    sid = _session(db)
    u1 = db.append_message(sid, "user", "first")
    a = db.append_message(sid, "assistant", "answer")
    mm = db.append_message(
        sid,
        "user",
        [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": "x"}],
    )
    text = db.append_message(sid, "user", "more detail")

    msgs = db.get_messages(sid)
    assert compute_half_turn_target(msgs, 1) == mm
    assert compute_half_turn_target(msgs, 2) == a
    assert text > mm > a > u1


def test_other_roles_are_not_counted_or_targeted_and_clamp_to_oldest_user(db):
    sid = _session(db)
    system = db.append_message(sid, "system", "rules")
    first_user = db.append_message(sid, "user", "one")
    db.append_message(sid, "assistant", "two")
    db.append_message(sid, "user", "three")

    msgs = db.get_messages(sid)
    assert compute_half_turn_target(msgs, 4) == first_user
    assert compute_half_turn_target(msgs, 99) != system


def test_zero_non_other_returns_none_and_undo_noops_without_touching_stacks(db):
    sid = _session(db)
    db.append_message(sid, "system", "rules only")
    state = hermes_undo.get_state(sid)
    state.redo_stack.append(hermes_undo.UndoOp(n=1, rewound_ids=[999]))

    assert compute_half_turn_target(db.get_messages(sid), 1) is None
    result = hermes_undo.undo(sid, 1)

    assert result["rewound_ids"] == []
    assert result["prefill_text"] is None
    assert "nothing to undo" in result["message"]
    assert state.undo_stack == []
    assert len(state.redo_stack) == 1
    assert [m["role"] for m in db.get_messages(sid)] == ["system"]


def test_tool_row_monotonicity_holds_for_normal_data_and_violation_raises(db):
    sid = _session(db)
    db.append_message(sid, "user", "run")
    assistant = db.append_message(
        sid, "assistant", None, tool_calls=[{"id": "c1"}]
    )
    tool = db.append_message(sid, "tool", "result", tool_call_id="c1")
    assert tool > assistant

    bad = _session(db, "bad")
    db.append_message(bad, "user", "run")
    orphan = db.append_message(bad, "tool", "too early", tool_call_id="bad-call")
    owner = db.append_message(
        bad, "assistant", None, tool_calls=[{"id": "bad-call"}]
    )
    assert orphan < owner
    with pytest.raises(ValueError, match="orphan"):
        db.rewind_to_message(bad, owner, require_user_role=False)

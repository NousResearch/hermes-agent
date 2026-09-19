"""Rewind across an in-memory user-merge (#115493).

The pre-request belt (``repair_message_sequence``) merges consecutive user rows in
memory after both were already flushed, so the warm history holds FEWER user turns
than the durable transcript while describing the same logical transcript. The rewind
guard must see through that merge instead of raising ``session history changed``.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handle = SessionDB(db_path=tmp_path / "state.db")
    yield handle
    handle.close()


def _seed_merged_tail(db: SessionDB, sid: str) -> None:
    """A durable ``user;user`` tail: both rows flushed (active) before the merge ran."""
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "q1")
    db.append_message(sid, "assistant", "a1")
    db.append_message(sid, "user", "q2")
    db.append_message(sid, "user", "q3")


def _merged_warm(db: SessionDB, sid: str):
    """What the caller holds after the pre-request belt merged the tail in memory."""
    from agent.agent_runtime_helpers import repair_message_sequence
    warm = db.get_messages_as_conversation(sid)
    assert repair_message_sequence(None, warm) == 1
    return warm


def _active(db: SessionDB, sid: str):
    return [(r[1], r[2]) for r in db._conn.execute(
        "SELECT id, role, content, active FROM messages WHERE session_id = ? ORDER BY id",
        (sid,)).fetchall() if r[3]]


def test_retry_newest_merged_turn_rewinds_both_physical_rows(db):
    sid = "merged-retry"
    _seed_merged_tail(db, sid)
    warm = _merged_warm(db, sid)
    assert [m["role"] for m in warm] == ["user", "assistant", "user"]

    outcome = db.rewind_user_turn(sid, -1, warm_history=warm, require_retryable=True)

    assert outcome.live_text == "q2\n\nq3"
    assert [(m["role"], m.get("content")) for m in outcome.prefix] == [
        ("user", "q1"), ("assistant", "a1")]
    assert outcome.turns_undone == 1
    assert outcome.rewound_count == 2
    assert _active(db, sid) == [("user", "q1"), ("assistant", "a1")]


def test_undo_older_turn_with_merged_tail_archives_from_target(db):
    sid = "merged-undo-older"
    _seed_merged_tail(db, sid)
    warm = _merged_warm(db, sid)

    outcome = db.rewind_user_turn(sid, 0, warm_history=warm)

    assert outcome.prefix == []
    assert outcome.turns_undone == 2
    assert _active(db, sid) == []


def test_genuine_divergence_still_raises_and_changes_nothing(db):
    sid = "merged-diverged"
    _seed_merged_tail(db, sid)
    warm = _merged_warm(db, sid)
    before = _active(db, sid)
    warm.append({"role": "user", "content": "q-unflushed"})

    with pytest.raises(RuntimeError, match="session history changed"):
        db.rewind_user_turn(sid, -1, warm_history=warm, require_retryable=True)
    assert _active(db, sid) == before


def test_gateway_rewind_without_warm_history_heals_merged_tail(db):
    sid = "merged-gateway"
    _seed_merged_tail(db, sid)

    outcome = db.rewind_user_turn(sid, -1)

    assert outcome.live_text == "q2\n\nq3"
    assert _active(db, sid) == [("user", "q1"), ("assistant", "a1")]

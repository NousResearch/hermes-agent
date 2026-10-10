"""One ACTIVE result row per (session_id, tool_call_id) across archive_and_compact.

In-place compaction archives the live rows (active=0, compacted=1) and inserts
the compacted set as fresh active rows, so a carried tool exchange legitimately
exists once per compaction GENERATION on disk. The live set must still hold one
result per tool call — a second active copy replays the exchange on resume.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB, TranscriptInvariantError


def _call(tc_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": tc_id, "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"}}],
    }


def _result(tc_id: str, text: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": text}


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("S", source="test")
    yield d
    d.close()


def _append(db, msgs):
    for m in msgs:
        db.append_message("S", m["role"], content=m.get("content"),
                          tool_calls=m.get("tool_calls"), tool_call_id=m.get("tool_call_id"))


def _rows(db, active=None):
    sql = "SELECT role, tool_call_id, active, compacted FROM messages WHERE session_id = 'S'"
    if active is not None:
        sql += f" AND active = {int(active)}"
    return [tuple(r) for r in db._conn.execute(sql + " ORDER BY id").fetchall()]


def _active_results(db, tc_id):
    return db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='S' AND active=1 "
        "AND role='tool' AND tool_call_id=?", (tc_id,)).fetchone()[0]


def test_carried_tail_keeps_one_active_result_per_tool_call(db):
    """The carried tail re-inserts the exchange; the old copy is archived."""
    _append(db, [{"role": "user", "content": "go"}, _call("t1"), _result("t1")])

    db.archive_and_compact("S", [{"role": "user", "content": "[summary]"}, _call("t1"), _result("t1")])

    assert _active_results(db, "t1") == 1
    # Both generations stay on disk: that is the design, not the defect.
    assert db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='S' AND role='tool' AND tool_call_id='t1'"
    ).fetchone()[0] == 2


def test_compaction_that_introduces_duplicate_result_rolls_back(db):
    _append(db, [{"role": "user", "content": "go"}, _call("t1"), _result("t1")])
    before = _rows(db)

    with pytest.raises(TranscriptInvariantError, match="t1"):
        db.archive_and_compact("S", [{"role": "user", "content": "[summary]"}, _call("t1"),
                                     _result("t1"), _result("t1")])

    assert _rows(db) == before, "the invariant must fire inside the txn so nothing is archived"
    assert db._conn.execute("SELECT message_count FROM sessions WHERE id='S'").fetchone()[0] == 3


def test_concurrent_tail_clone_cannot_double_carry_a_result(db):
    """A row after the watermark is re-sequenced by clone; carrying it in the compacted set too is a dup."""
    _append(db, [{"role": "user", "content": "go"}, _call("t1")])
    watermark = db.get_active_message_watermark("S")
    _append(db, [_result("t1")])  # arrived during the slow summary

    with pytest.raises(TranscriptInvariantError):
        db.archive_and_compact("S", [{"role": "user", "content": "[summary]"}, _call("t1"), _result("t1")],
                               watermark=watermark)
    assert _active_results(db, "t1") == 1

    # Without the double carry the same compaction commits and keeps exactly one result.
    db.archive_and_compact("S", [{"role": "user", "content": "[summary]"}, _call("t1")], watermark=watermark)
    assert _active_results(db, "t1") == 1


def test_reused_provider_id_occurrences_carry_through_compaction(db):
    """Providers reuse index ids (``terminal:0``): two call/result occurrences are two logical calls."""
    _append(db, [_call("terminal:0"), _result("terminal:0", "a"), _call("terminal:0"), _result("terminal:0", "b")])
    assert _active_results(db, "terminal:0") == 2

    db.archive_and_compact("S", [{"role": "user", "content": "[summary]"}, _call("terminal:0"),
                                 _result("terminal:0", "a"), _call("terminal:0"), _result("terminal:0", "b")])
    assert _active_results(db, "terminal:0") == 2


def test_new_occurrence_of_a_reused_provider_id_is_not_a_duplicate(db):
    """Providers reuse index ids: a second ``terminal:0`` call/result pair is a new logical occurrence."""
    _append(db, [{"role": "user", "content": "go"}, _call("terminal:0"), _result("terminal:0", "a")])
    assert _active_results(db, "terminal:0") == 1

    db.archive_and_compact("S", [{"role": "user", "content": "[summary]"},
                                 _call("terminal:0"), _result("terminal:0", "a"),
                                 _call("terminal:0"), _result("terminal:0", "b")])
    assert _active_results(db, "terminal:0") == 2


def test_double_carry_is_caught_when_the_provider_id_is_already_reused(db):
    """Two legitimate ``terminal:0`` occurrences, then the later result carried twice: 2 -> 3 must raise."""
    _append(db, [{"role": "user", "content": "go"}, _call("terminal:0"), _result("terminal:0", "a"),
                 _call("terminal:0")])
    watermark = db.get_active_message_watermark("S")
    _append(db, [_result("terminal:0", "b")])  # arrived during the slow summary
    assert _active_results(db, "terminal:0") == 2
    before = _rows(db)

    with pytest.raises(TranscriptInvariantError, match="terminal:0"):
        db.archive_and_compact("S", [{"role": "user", "content": "[summary]"},
                                     _call("terminal:0"), _result("terminal:0", "a"),
                                     _call("terminal:0"), _result("terminal:0", "b")],
                               watermark=watermark)
    assert _rows(db) == before


def test_legacy_duplicate_result_carried_unchanged_does_not_wedge_compaction(db):
    """A live set that ALREADY holds two results for one call (a legacy row) may be carried forward as-is."""
    _append(db, [{"role": "user", "content": "go"}, _call("t1"), _result("t1", "a"), _result("t1", "a")])
    assert _active_results(db, "t1") == 2

    db.archive_and_compact("S", [{"role": "user", "content": "[summary]"}, _call("t1"),
                                 _result("t1", "a"), _result("t1", "a")])
    assert _active_results(db, "t1") == 2

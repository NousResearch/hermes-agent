"""Integration tests for SessionDB per-message token aggregation.

Exercises the bit-packed token_count decode *in SQLite* (the arithmetic
right-shift masking is easy to get wrong), the legacy/packed mix, and the
compression-lineage recursive CTE.
"""
import pytest

from hermes_state import SessionDB
from hermes_token_codec import pack_assistant_tokens, pack_input_tokens


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "tok_state.db")
    yield session_db
    session_db.close()


def _seed_turn(db, session_id, *, in_tok, cache_tok, out_tok, reason_tok):
    """Append a user (packed input) + assistant (packed output) pair."""
    db.append_message(
        session_id=session_id,
        role="user",
        content="hi",
        token_count=pack_input_tokens(in_tok, cache_tok),
    )
    db.append_message(
        session_id=session_id,
        role="assistant",
        content="hello",
        token_count=pack_assistant_tokens(out_tok, reason_tok),
    )


def test_single_message_decode_round_trips(db):
    db.create_session(session_id="s1", source="cli")
    mid = db.append_message(
        session_id="s1",
        role="assistant",
        content="x",
        token_count=pack_assistant_tokens(900, 120),
    )
    assert db.get_message_tokens(mid) == {
        "input": 0, "output": 900, "cache_read": 0, "reasoning": 120,
    }


def test_single_message_missing_row(db):
    assert db.get_message_tokens(999999) == {
        "input": 0, "output": 0, "cache_read": 0, "reasoning": 0,
    }


def test_session_totals_packed(db):
    db.create_session(session_id="s1", source="cli")
    _seed_turn(db, "s1", in_tok=1000, cache_tok=400, out_tok=300, reason_tok=50)
    _seed_turn(db, "s1", in_tok=2000, cache_tok=900, out_tok=600, reason_tok=70)
    totals = db.get_session_message_token_totals("s1")
    assert totals["input"] == 3000
    assert totals["cache_read"] == 1300
    assert totals["output"] == 900
    assert totals["reasoning"] == 120
    assert totals["messages"] == 4


def test_session_totals_mixes_legacy_and_packed(db):
    db.create_session(session_id="s1", source="cli")
    # Legacy assistant row: raw non-negative count -> attributed to output.
    db.append_message(session_id="s1", role="assistant", content="legacy", token_count=500)
    # Legacy user row: non-negative count is ambiguous -> ignored.
    db.append_message(session_id="s1", role="user", content="legacy-u", token_count=999)
    # Packed turn on top.
    _seed_turn(db, "s1", in_tok=1000, cache_tok=400, out_tok=300, reason_tok=50)
    totals = db.get_session_message_token_totals("s1")
    assert totals["output"] == 500 + 300   # legacy assistant + packed output
    assert totals["input"] == 1000
    assert totals["cache_read"] == 400
    assert totals["reasoning"] == 50


def test_session_totals_handles_null_token_count(db):
    db.create_session(session_id="s1", source="cli")
    db.append_message(session_id="s1", role="user", content="no tokens")
    totals = db.get_session_message_token_totals("s1")
    assert totals == {
        "input": 0, "output": 0, "cache_read": 0, "reasoning": 0, "messages": 1,
    }


def test_session_totals_empty_session(db):
    db.create_session(session_id="s1", source="cli")
    totals = db.get_session_message_token_totals("s1")
    assert totals == {
        "input": 0, "output": 0, "cache_read": 0, "reasoning": 0, "messages": 0,
    }


def test_conversation_totals_spans_compression_split(db):
    # Root compresses into a continuation child; totals must cover both.
    db.create_session(session_id="root", source="cli")
    _seed_turn(db, "root", in_tok=1000, cache_tok=200, out_tok=300, reason_tok=40)
    db.end_session("root", end_reason="compression")

    child = db.get_session("root")  # ensure root exists
    assert child is not None
    db.create_session(
        session_id="cont",
        source="cli",
        parent_session_id="root",
    )
    _seed_turn(db, "cont", in_tok=5000, cache_tok=1500, out_tok=800, reason_tok=90)

    # Session-scoped totals only see their own session.
    root_only = db.get_session_message_token_totals("root")
    assert root_only["input"] == 1000
    assert root_only["output"] == 300

    # Conversation totals span the lineage.
    convo = db.get_conversation_message_token_totals("root")
    assert convo["input"] == 6000
    assert convo["cache_read"] == 1700
    assert convo["output"] == 1100
    assert convo["reasoning"] == 130
    assert convo["messages"] == 4


def test_saturated_values_decode_in_sql(db):
    # Values exceeding the field width are clamped at pack time; verify the
    # SQL decode returns the clamped maxima (no sign/overflow corruption).
    from hermes_token_codec import _V1_MAX, _V2_MAX
    db.create_session(session_id="s1", source="cli")
    db.append_message(
        session_id="s1",
        role="user",
        content="big",
        token_count=pack_input_tokens(_V1_MAX + 10_000, _V2_MAX + 10_000),
    )
    totals = db.get_session_message_token_totals("s1")
    assert totals["input"] == _V1_MAX
    assert totals["cache_read"] == _V2_MAX

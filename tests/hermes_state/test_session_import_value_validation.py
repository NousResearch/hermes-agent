"""Session-import boundary validation for values later read by session APIs.

``import_sessions`` used shallow type casts at the persistence boundary: ``float()`` accepted
``inf``/``NaN`` (a JSON ``1e309``), ``int()`` accepted integers SQLite cannot bind (outside the
signed 64-bit range), ``model_config`` was only checked for being a JSON *object* after its
intentional decode (non-finite values inside it persisted verbatim), and a literal string
carrying the internal ``\\x00json:`` content marker was decoded as structured content on read.
Bad values therefore reached sqlite3 binding or persisted as ``inf``, breaking session APIs
with HTTP 500. Regression for #63954 (validation shape reused from PR #63956 by @Jupiter363).
"""

from __future__ import annotations

import json

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


def _payload(session_id: str, **overrides) -> dict:
    session = {"id": session_id, "messages": [{"role": "user", "content": "hello"}]}
    session.update(overrides)
    return session


def _rejected(db: SessionDB, session: dict) -> str:
    """Import must reject atomically (structured 400 contract) and name the field."""
    result = db.import_sessions([session])
    assert result["ok"] is False, "unsafe import was accepted"
    assert result["imported"] == 0
    assert db.get_session(session["id"]) is None, "a rejected import still wrote a row"
    error = result["errors"][0]["error"]
    assert error, "rejection carried no error message"
    return error


def test_import_rejects_non_finite_session_floats(db):
    for field in ("started_at", "ended_at", "estimated_cost_usd", "actual_cost_usd"):
        # 1e309 decodes to float('inf'); NaN arrives the same way (json NaN syntax).
        for value in (1e309, float("nan")):
            error = _rejected(db, _payload(f"bad-{field}-{value}", **{field: value}))
            assert "non-finite" in error or "timestamp" in error, (field, error)


def test_import_rejects_non_finite_nested_model_config(db):
    error = _rejected(db, _payload("nested", model_config='{"temperature": 1e309}'))
    assert "non-finite number at session.model_config" in error
    # Also as a decoded dict, not only as a JSON string.
    error = _rejected(db, _payload("nested-dict", model_config={"temperature": float("inf")}))
    assert "non-finite" in error


def test_import_rejects_non_finite_reasoning_json(db):
    session = _payload(
        "reasoning",
        messages=[{"role": "assistant", "content": "hi",
                   "reasoning_details": '[{"temperature": 1e309}]'}],
    )
    error = _rejected(db, session)
    assert "non-finite number at session.messages[0].reasoning_details" in error


def test_import_rejects_int_columns_outside_sqlite_range(db):
    huge = 1 << 63
    for field in ("input_tokens", "output_tokens", "api_call_count", "cache_read_tokens",
                  "cache_write_tokens", "reasoning_tokens"):
        assert "outside SQLite's integer range" in _rejected(db, _payload(f"huge-{field}", **{field: huge})), field


def test_import_rejects_oversized_scalar_message_content(db):
    error = _rejected(db, _payload("huge-content", messages=[
        {"role": "user", "content": 1 << 63}]))
    assert "content is outside SQLite's integer range" in error
    error = _rejected(db, _payload("inf-content", messages=[
        {"role": "user", "content": float("inf")}]))
    assert "non-finite number at session.messages[0].content" in error


def test_import_rejects_oversized_token_count_and_bad_message_timestamp(db):
    assert "token_count" in _rejected(db, _payload("huge-tokens", messages=[
        {"role": "user", "content": "hi", "token_count": 1 << 63}]))
    assert "timestamp" in _rejected(db, _payload("bad-ts", messages=[
        {"role": "user", "content": "hi", "timestamp": 1e309}]))


def test_reserved_content_prefix_literal_round_trips_verbatim(db):
    """A literal string imitating the internal ``\\x00json:`` marker must NOT be decoded
    as structured content on read — it comes back as the exact literal string."""
    marker = SessionDB._CONTENT_JSON_PREFIX
    literal = marker + "not actually json"
    db.import_sessions([_payload("marker", messages=[{"role": "user", "content": literal}])])
    assert db.get_messages("marker")[0]["content"] == literal
    # And the encoded form of a marker-prefixed string still decodes to the literal.
    encoded = SessionDB._encode_content(literal)
    assert encoded.startswith(marker)
    assert SessionDB._decode_content(encoded) == literal


def test_decode_content_fails_closed_on_malformed_marker_rows(db):
    malformed = SessionDB._CONTENT_JSON_PREFIX + "[{not json"
    assert SessionDB._decode_content(malformed) == malformed


def test_normal_export_round_trips_unchanged(db):
    """The primary compatibility contract: an ordinary session exports and re-imports
    with content, timestamps and counters intact."""
    db.create_session(session_id="rt-src", source="cli", model="test-model")
    db.append_message("rt-src", "user", "hello world", token_count=5)
    db.append_message("rt-src", "assistant", [{"type": "text", "text": "hi"}],
                      token_count=7, finish_reason="stop",
                      reasoning_details=[{"type": "summary", "text": "because"}])
    db.end_session("rt-src", "done")
    payload = json.loads(json.dumps(db.export_session("rt-src")))

    result = db.import_sessions([dict(payload, id="rt-copy")])
    assert result["ok"] is True and result["imported"] == 1

    restored = db.export_session("rt-copy")
    for message in (payload["messages"], restored["messages"]):
        message.sort(key=lambda m: m.get("timestamp") or 0)
    for original, copy in zip(payload["messages"], restored["messages"]):
        assert copy["content"] == original["content"]
        assert copy.get("token_count") == original.get("token_count")
        assert copy["role"] == original["role"]
    assert restored["message_count"] == payload["message_count"]
    assert restored["tool_call_count"] == payload["tool_call_count"]
    assert restored["ended_at"] is not None

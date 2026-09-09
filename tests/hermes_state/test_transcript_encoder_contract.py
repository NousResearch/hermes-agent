"""Tests pinning the transcript-write JSON encoder contract.

``_JSON_ENCODER`` is a module-level shared encoder used on the hot transcript
write paths (_encode_content / display-metadata / reasoning / tool_calls
serialization). Its safety argument is **byte-identity with json.dumps
defaults** — the stored bytes in state.db must not change, and
``ensure_ascii=True`` is load-bearing (it escapes lone surrogates so the
string is bindable to sqlite TEXT; see _encode_content's comment).

These tests pin that contract so a future "optimization" (e.g. switching to
ensure_ascii=False or compact separators, as a prior patch proposed) cannot
silently change what is persisted.
"""
import json

import pytest

import hermes_state_messages as hsm
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("s1", "cli")
    return d


class TestEncoderByteIdentity:
    def test_encoder_is_byte_identical_to_json_dumps(self):
        """The shared encoder must produce exactly the bytes json.dumps
        (default separators, ensure_ascii=True) produces."""
        samples = [
            "hello",
            "héllo wörld",
            "line1\nline2\ttabbed",
            'quote " inside',
            "lone surrogate: \ud800",  # ensure_ascii escapes it: bindable
            {"b": 1, "a": [1, 2, {"c": None, "d": True}]},
            [],
            {},
            1.5,
            None,
            True,
            {"content": "multi\nline", "nested": {"deep": ["héllo", 2]}},
            [{"id": "call_1", "function": {"name": "f", "arguments": "{\"k\": 1}"}}],
        ]
        for s in samples:
            assert hsm._JSON_ENCODER.encode(s) == json.dumps(s), repr(s)[:60]

    def test_encoder_escapes_lone_surrogates_like_dumps(self):
        """ensure_ascii=True must escape a lone surrogate (bindable to sqlite
        TEXT) — not raise, not emit the raw surrogate."""
        value = "bad \ud800 surrogate"
        encoded = hsm._JSON_ENCODER.encode(value)
        assert "\ud800" not in encoded
        assert encoded == json.dumps(value)
        # And it round-trips through sqlite binding without error.
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (x TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (encoded,))

    def test_encoder_uses_ascii_ensure_not_compact_separators(self):
        """Guard against silently swapping in compact separators (",", ":") —
        that changes stored bytes (no space after ',')."""
        value = {"a": 1, "b": [1, 2]}
        assert hsm._JSON_ENCODER.encode(value) == '{"a": 1, "b": [1, 2]}'


class TestStoredBytesUnchanged:
    def test_roundtrip_content_and_metadata_through_db(self, db):
        """Write a message with unicode + tool_calls + display_metadata, read it
        back, and confirm the values survive — i.e. the shared encoder did not
        change what is persisted."""
        db.append_message(
            "s1", "assistant", "héllo — naïve café",
            tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": "{\"k\": \"v\"}"}}],
            display_metadata={"surface": "test", "note": "unicode ✓"},
        )
        msgs = db.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "héllo — naïve café"
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "f"
        assert msgs[0]["display_metadata"]["surface"] == "test"

    def test_raw_stored_text_is_ascii_escaped_like_dumps(self, db):
        """Pin the storage format: plain strings are stored raw (surrogate-scrubbed);
        STRUCTURED content (list/dict, e.g. multimodal parts) is stored as
        ``_CONTENT_JSON_PREFIX + json.dumps(...)`` — the ensure_ascii-escaped form.
        The shared encoder must not change either."""
        db.append_message("s1", "user", "héllo")
        structured = [{"type": "text", "text": "héllo ✓", "meta": {"k": "v"}}]
        db.append_message("s1", "assistant", structured)
        import sqlite3
        with db._read_ctx() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                ("s1",)).fetchall()
        by_role = {r["role"]: r["content"] for r in rows}
        # Plain string: stored raw (sanitized), no JSON wrapping.
        assert by_role["user"] == "héllo"
        # Structured content: prefix + json.dumps-identical bytes (é escaped as
        # \u00e9, the surrogate-safe bindable form).
        expected = SessionDB._CONTENT_JSON_PREFIX + json.dumps(structured)
        assert by_role["assistant"] == expected

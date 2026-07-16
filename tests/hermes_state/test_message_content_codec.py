"""Synthetic tests for the versioned messages.content storage codec."""

import json

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session("codec", source="test")
    yield session_db
    session_db.close()


def _raw_content(db: SessionDB, message_id: int):
    with db._lock:
        return db._conn.execute(
            "SELECT content FROM messages WHERE id = ?", (message_id,)
        ).fetchone()["content"]


@pytest.mark.parametrize(
    ("content", "storage_prefix"),
    [
        (
            [{"type": "text", "text": "caf\u00e9 \u732b"}],
            SessionDB._CONTENT_V2_JSON_PREFIX,
        ),
        (
            {"parts": [{"text": "line\x00break"}]},
            SessionDB._CONTENT_V2_JSON_PREFIX,
        ),
        (b"plain bytes", SessionDB._CONTENT_V2_BYTES_PREFIX),
        (b"before\x00after", SessionDB._CONTENT_V2_BYTES_PREFIX),
    ],
)
def test_structured_and_bytes_round_trip_as_printable_nul_free_text(
    db, content, storage_prefix
):
    message_id = db.append_message("codec", role="tool", content=content)

    raw = _raw_content(db, message_id)
    assert isinstance(raw, str)
    assert raw.startswith(storage_prefix)
    assert "\x00" not in raw
    assert db.get_messages("codec")[0]["content"] == content
    assert db.get_messages_as_conversation("codec")[0]["content"] == content


def test_json_storage_is_deterministic_and_compact():
    content = {"z": [1, 2], "unicode": "\u732b"}
    expected = SessionDB._CONTENT_V2_JSON_PREFIX + json.dumps(
        content, separators=(",", ":")
    )
    assert SessionDB._encode_content(content) == expected
    assert SessionDB._encode_content(content) == expected


def test_plain_text_and_null_keep_native_storage(db):
    plain_id = db.append_message("codec", role="user", content="searchable caf\u00e9")
    null_id = db.append_message("codec", role="assistant", content=None)

    assert _raw_content(db, plain_id) == "searchable caf\u00e9"
    assert _raw_content(db, null_id) is None
    assert [msg["content"] for msg in db.get_messages("codec")] == [
        "searchable caf\u00e9",
        None,
    ]
    assert any(row["id"] == plain_id for row in db.search_messages("searchable"))


@pytest.mark.parametrize("content", [7, 1.5])
def test_numeric_scalars_keep_existing_text_affinity_behavior(db, content):
    message_id = db.append_message("codec", role="tool", content=content)

    # messages.content has TEXT affinity, so SQLite has always returned numeric
    # bindings as their text representation.
    assert _raw_content(db, message_id) == str(content)
    assert db.get_messages("codec")[0]["content"] == str(content)


@pytest.mark.parametrize(
    "content",
    [
        SessionDB._CONTENT_V2_PREFIX + "unknown:payload",
        SessionDB._CONTENT_V2_JSON_PREFIX + '[{"type":"text"}]',
        SessionDB._CONTENT_V2_STRING_PREFIX + '"literal"',
        SessionDB._CONTENT_V2_BYTES_PREFIX + "00ff",
    ],
)
def test_prefix_looking_plain_strings_are_escaped_and_round_trip(db, content):
    message_id = db.append_message("codec", role="user", content=content)

    raw = _raw_content(db, message_id)
    assert raw.startswith(SessionDB._CONTENT_V2_STRING_PREFIX)
    assert "\x00" not in raw
    assert db.get_messages("codec")[0]["content"] == content


def test_legacy_nul_prefix_string_collision_uses_existing_sanitization(db):
    content = SessionDB._CONTENT_JSON_PREFIX + '{"parts":[]}'
    message_id = db.append_message("codec", role="tool", content=content)

    raw = _raw_content(db, message_id)
    assert raw.startswith(SessionDB._CONTENT_V2_STRING_PREFIX)
    assert "\x00" not in raw
    assert db.get_messages("codec")[0]["content"] == content.replace("\x00", "\ufffd")


def test_embedded_nul_in_plain_string_is_replaced_and_never_stored(db):
    message_id = db.append_message("codec", role="tool", content="before\x00after")

    raw = _raw_content(db, message_id)
    assert raw.startswith(SessionDB._CONTENT_V2_STRING_PREFIX)
    assert "\x00" not in raw
    assert db.get_messages("codec")[0]["content"] == "before\ufffdafter"


@pytest.mark.parametrize(
    "content", [[{"type": "text", "text": "legacy"}], {"parts": []}]
)
def test_legacy_nul_json_encoding_remains_readable(content):
    raw = SessionDB._CONTENT_JSON_PREFIX + json.dumps(content)
    assert SessionDB._decode_content(raw) == content


@pytest.mark.parametrize(
    "raw",
    [
        SessionDB._CONTENT_V2_PREFIX + "x:unknown",
        SessionDB._CONTENT_V2_JSON_PREFIX + "{not-json",
        SessionDB._CONTENT_V2_JSON_PREFIX + '"wrong-type"',
        SessionDB._CONTENT_V2_STRING_PREFIX + "{not-json",
        SessionDB._CONTENT_V2_STRING_PREFIX + "[]",
        SessionDB._CONTENT_V2_BYTES_PREFIX + "not-hex",
        SessionDB._CONTENT_V2_BYTES_PREFIX + "0",
        SessionDB._CONTENT_V2_BYTES_PREFIX + "00 FF",
        SessionDB._CONTENT_JSON_PREFIX + "{not-json",
    ],
)
def test_malformed_or_unknown_encoded_values_remain_raw_strings(raw):
    assert SessionDB._decode_content(raw) == raw


def test_replace_messages_uses_the_same_codec(db):
    content = [{"type": "text", "text": "replacement \u732b"}]
    db.replace_messages(
        "codec",
        [
            {"role": "user", "content": content},
            {"role": "tool", "content": b"\x00replacement"},
        ],
    )

    with db._lock:
        rows = db._conn.execute(
            "SELECT content FROM messages WHERE session_id = ? ORDER BY id", ("codec",)
        ).fetchall()
    assert all(isinstance(row["content"], str) for row in rows)
    assert all("\x00" not in row["content"] for row in rows)
    assert [msg["content"] for msg in db.get_messages("codec")] == [
        content,
        b"\x00replacement",
    ]

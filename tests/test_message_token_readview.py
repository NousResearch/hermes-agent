"""Read-side flattening of bit-packed message token_count.

get_messages / get_messages_as_conversation must never surface the raw
negative packed sentinel to display consumers, while re-persist paths
(flatten_tokens=False) keep the raw value for lossless round-trips.
"""
import pytest

from hermes_state import SessionDB
from hermes_token_codec import pack_assistant_tokens, pack_input_tokens


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "readview.db")
    yield session_db
    session_db.close()


def _seed(db):
    db.create_session(session_id="s1", source="cli")
    db.append_message(session_id="s1", role="user", content="q",
                      token_count=pack_input_tokens(8000, 6000))
    db.append_message(session_id="s1", role="assistant", content="a",
                      token_count=pack_assistant_tokens(900, 120))
    db.append_message(session_id="s1", role="user", content="legacy-none")  # NULL


def test_get_messages_flattens_by_default(db):
    _seed(db)
    msgs = db.get_messages("s1")
    user, asst, none_row = msgs[0], msgs[1], msgs[2]

    # No raw negative sentinel anywhere.
    assert all((m["token_count"] is None or m["token_count"] >= 0) for m in msgs)

    # User row: scalar suppressed, buckets carry input/cache.
    assert user["token_count"] is None
    assert user["tokens"] == {"input": 8000, "output": 0, "cache_read": 6000, "reasoning": 0}

    # Assistant row: scalar == legacy output, buckets carry output/reasoning.
    assert asst["token_count"] == 900
    assert asst["tokens"] == {"input": 0, "output": 900, "cache_read": 0, "reasoning": 120}

    # NULL row: zero buckets.
    assert none_row["token_count"] is None
    assert none_row["tokens"] == {"input": 0, "output": 0, "cache_read": 0, "reasoning": 0}


def test_get_messages_raw_when_opted_out(db):
    _seed(db)
    raw = db.get_messages("s1", flatten_tokens=False)
    # Packed rows keep their negative sentinel; no `tokens` view attached.
    assert raw[0]["token_count"] < 0
    assert raw[1]["token_count"] < 0
    assert "tokens" not in raw[0]
    assert "tokens" not in raw[1]


def test_conversation_format_attaches_tokens_without_scalar(db):
    _seed(db)
    convo = db.get_messages_as_conversation("s1")
    # Replay format never carries a scalar token_count key...
    assert all("token_count" not in m for m in convo)
    # ...but exposes the flattened view for display consumers.
    assert convo[0]["tokens"]["input"] == 8000
    assert convo[0]["tokens"]["cache_read"] == 6000
    assert convo[1]["tokens"]["output"] == 900
    assert convo[1]["tokens"]["reasoning"] == 120


def test_export_session_decodes_tokens(db):
    # Export is for analysis (no re-import reads token_count) — it must decode,
    # never emit the raw negative packed sentinel.
    _seed(db)
    exported = db.export_session("s1")
    msgs = exported["messages"]
    assert all((m["token_count"] is None or m["token_count"] >= 0) for m in msgs)
    assert msgs[0]["tokens"]["input"] == 8000
    assert msgs[1]["tokens"]["output"] == 900


def test_fork_style_roundtrip_preserves_packed_accounting(db):
    # Mirror the api_server fork path: get_messages(flatten_tokens=False)
    # then replace_messages must preserve packed token_count losslessly.
    _seed(db)
    db.create_session(session_id="fork", source="cli")
    raw = db.get_messages("s1", flatten_tokens=False)
    db.replace_messages("fork", raw)

    # The forked session decodes to the same buckets as the source.
    src = db.get_session_message_token_totals("s1")
    forked = db.get_session_message_token_totals("fork")
    assert forked["input"] == src["input"] == 8000
    assert forked["cache_read"] == src["cache_read"] == 6000
    assert forked["output"] == src["output"] == 900
    assert forked["reasoning"] == src["reasoning"] == 120

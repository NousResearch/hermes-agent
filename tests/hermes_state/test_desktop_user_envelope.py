"""Desktop MessageDocument envelope persistence and agent replay unwrapping."""

import pytest

from hermes_state import SessionDB, _unwrap_desktop_user_content


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def test_unwrap_desktop_user_content_extracts_wire_text():
    envelope = {
        "text": "hi @file:`foo.txt`",
        "document": [{"type": "text", "value": "hi"}],
        "document_version": 1,
    }
    assert _unwrap_desktop_user_content(envelope) == "hi @file:`foo.txt`"
    assert _unwrap_desktop_user_content("plain") == "plain"


def test_envelope_round_trip_and_conversation_replay(db):
    sid = "sess-envelope"
    db.create_session(sid, source="cli")
    envelope = {
        "text": "look @file:`bar.ts`",
        "document": [
            {"type": "text", "value": "look "},
            {
                "type": "attachment",
                "id": "attachment-1",
                "kind": "file",
                "path": "bar.ts",
                "displayName": "bar.ts",
            },
        ],
        "document_version": 1,
    }
    db.append_message(sid, role="user", content=envelope)

    stored = db.get_messages(sid)
    assert isinstance(stored[0]["content"], dict)
    assert stored[0]["content"]["document_version"] == 1

    conversation = db.get_messages_as_conversation(sid)
    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"] == "look @file:`bar.ts`"

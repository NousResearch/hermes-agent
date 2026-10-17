"""Round-trip tests for the structured reasoning columns.

get_messages() returns reasoning_details / codex_reasoning_items /
codex_message_items as the raw TEXT stored in their columns (it only
hydrates content and tool_calls). Callers that feed those rows straight
back into a write — the POST /api/sessions/{id}/fork handler pipes
get_messages() into replace_messages() — must not re-encode that TEXT,
or the forked session replays with reasoning fields decoding to strings
and every isinstance(..., list) consumer silently drops them.
"""
import pytest

from hermes_state import SessionDB


REASONING_DETAILS = [
    {"type": "reasoning.text", "text": "compare both branches first", "format": "unknown"}
]
CODEX_REASONING_ITEMS = [
    {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque-blob"}
]
CODEX_MESSAGE_ITEMS = [
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "done"}],
    }
]


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed(db, sid="src"):
    """Session with one assistant message carrying all three reasoning fields."""
    db.create_session(sid, source="cli")
    db.append_message(sid, role="user", content="hi")
    db.append_message(
        sid,
        role="assistant",
        content="done",
        reasoning_details=REASONING_DETAILS,
        codex_reasoning_items=CODEX_REASONING_ITEMS,
        codex_message_items=CODEX_MESSAGE_ITEMS,
    )


def _fork(db, src, dst):
    """The fork handler's copy step: raw get_messages rows into replace_messages."""
    db.create_session(dst, source="cli")
    db.replace_messages(dst, db.get_messages(src))


def _assistant(conversation):
    return next(m for m in conversation if m["role"] == "assistant")


class TestDirectWrite:
    """Live-runtime path: structured values in, structured values back."""

    def test_reasoning_fields_hydrate_as_structures(self, db):
        _seed(db)
        msg = _assistant(db.get_messages_as_conversation("src"))
        assert msg["reasoning_details"] == REASONING_DETAILS
        assert msg["codex_reasoning_items"] == CODEX_REASONING_ITEMS
        assert msg["codex_message_items"] == CODEX_MESSAGE_ITEMS


class TestForkRoundTrip:
    """get_messages -> replace_messages must keep the stored TEXT intact."""

    def test_reasoning_details_survive_fork(self, db):
        _seed(db)
        _fork(db, "src", "fork")
        msg = _assistant(db.get_messages_as_conversation("fork"))
        assert msg["reasoning_details"] == REASONING_DETAILS

    def test_codex_reasoning_items_survive_fork(self, db):
        _seed(db)
        _fork(db, "src", "fork")
        msg = _assistant(db.get_messages_as_conversation("fork"))
        assert msg["codex_reasoning_items"] == CODEX_REASONING_ITEMS

    def test_codex_message_items_survive_fork(self, db):
        _seed(db)
        _fork(db, "src", "fork")
        msg = _assistant(db.get_messages_as_conversation("fork"))
        assert msg["codex_message_items"] == CODEX_MESSAGE_ITEMS

    def test_fork_of_fork_stays_stable(self, db):
        # Each extra round-trip used to add another encoding layer.
        _seed(db)
        _fork(db, "src", "fork1")
        _fork(db, "fork1", "fork2")
        msg = _assistant(db.get_messages_as_conversation("fork2"))
        assert msg["reasoning_details"] == REASONING_DETAILS
        assert msg["codex_reasoning_items"] == CODEX_REASONING_ITEMS
        assert msg["codex_message_items"] == CODEX_MESSAGE_ITEMS


class TestAppendMessageRoundTrip:
    """append_message accepts a stored row's already-serialized TEXT too."""

    def test_string_value_not_double_encoded(self, db):
        _seed(db)
        row = next(m for m in db.get_messages("src") if m["role"] == "assistant")
        db.create_session("copy", source="cli")
        db.append_message(
            "copy",
            role="assistant",
            content="done",
            reasoning_details=row["reasoning_details"],
        )
        msg = _assistant(db.get_messages_as_conversation("copy"))
        assert msg["reasoning_details"] == REASONING_DETAILS

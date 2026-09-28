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


class TestSharedReasoningStoredOnce:
    """reasoning-content providers (DeepSeek, Kimi) hand back the same text as both
    ``reasoning`` and ``reasoning_content``; it lands on disk once (#125273)."""

    TEXT = "compare both branches first"

    def _columns(self, db, sid):
        with db._read_ctx() as conn:
            return conn.execute(
                "SELECT reasoning, reasoning_content FROM messages "
                "WHERE session_id = ? AND role = 'assistant'", (sid,)).fetchone()

    def _append(self, db, sid, **reasoning):
        db.create_session(sid, source="cli")
        db.append_message(sid, role="user", content="hi")
        db.append_message(sid, role="assistant", content="done", **reasoning)

    def test_identical_text_is_stored_once(self, db):
        self._append(db, "s", reasoning=self.TEXT, reasoning_content=self.TEXT)
        assert tuple(self._columns(db, "s")) == (None, self.TEXT)

    def test_both_fields_come_back_on_every_read_path(self, db):
        self._append(db, "s", reasoning=self.TEXT, reasoning_content=self.TEXT)
        for msg in (_assistant(db.get_messages_as_conversation("s")), _assistant(db.get_messages("s"))):
            assert msg["reasoning"] == self.TEXT
            assert msg["reasoning_content"] == self.TEXT

    def test_differing_text_keeps_both_columns(self, db):
        self._append(db, "s", reasoning="summary + " + self.TEXT, reasoning_content=self.TEXT)
        assert tuple(self._columns(db, "s")) == ("summary + " + self.TEXT, self.TEXT)

    def test_reasoning_only_stays_without_reasoning_content(self, db):
        self._append(db, "s", reasoning=self.TEXT)
        msg = _assistant(db.get_messages_as_conversation("s"))
        assert msg["reasoning"] == self.TEXT
        assert "reasoning_content" not in msg

    def test_blank_pad_does_not_grow_reasoning(self, db):
        # Thinking-mode tool-call pad: reasoning_content=" " with no reasoning at all.
        self._append(db, "s", reasoning_content=" ")
        msg = _assistant(db.get_messages_as_conversation("s"))
        assert msg["reasoning_content"] == " "
        assert "reasoning" not in msg

    def test_reasoning_content_alone_does_not_grow_reasoning(self, db):
        # Tool-call merge / partial-stream stub: non-blank reasoning_content, no reasoning of its own.
        self._append(db, "s", reasoning_content=self.TEXT)
        for msg in (_assistant(db.get_messages_as_conversation("s")), _assistant(db.get_messages("s"))):
            assert msg["reasoning_content"] == self.TEXT
            assert not msg.get("reasoning")

    def test_rows_written_before_the_fix_still_read_back(self, db):
        self._append(db, "s", reasoning="x", reasoning_content="y")
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET reasoning = ?, reasoning_content = ? WHERE role = 'assistant'",
            (self.TEXT, self.TEXT)))
        msg = _assistant(db.get_messages_as_conversation("s"))
        assert (msg["reasoning"], msg["reasoning_content"]) == (self.TEXT, self.TEXT)

    def test_fork_keeps_one_copy(self, db):
        self._append(db, "src", reasoning=self.TEXT, reasoning_content=self.TEXT)
        _fork(db, "src", "fork")
        assert tuple(self._columns(db, "fork")) == (None, self.TEXT)
        msg = _assistant(db.get_messages_as_conversation("fork"))
        assert (msg["reasoning"], msg["reasoning_content"]) == (self.TEXT, self.TEXT)

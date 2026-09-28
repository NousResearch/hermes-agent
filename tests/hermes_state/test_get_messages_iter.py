"""Tests for SessionDB.get_messages_iter — the streaming counterpart to get_messages.

Covers:
- Ascending stream == get_messages(session_id) (chronological, oldest-first).
- Descending stream (latest=True) == reversed(get_messages(session_id)) (newest-first),
  the order the run_topics "last assistant message" caller relies on.
- Early break does not require materializing the full transcript (bounded fetch).
- The motivating use case: scanning newest-first finds the NEWEST assistant message
  (the bug the original stash's per-batch reversal would have introduced found the
  oldest-of-the-batch instead).
- after_id is incompatible with latest/offset (ValueError), matching get_messages.
- Empty session yields nothing.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    """A SessionDB with one session and a known message sequence."""
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("s1", "cli")
    # Insert in order; ids are assigned ascending.
    d.append_message("s1", "user", "u1")
    d.append_message("s1", "assistant", "a1")
    d.append_message("s1", "user", "u2")
    d.append_message("s1", "assistant", "a2")
    d.append_message("s1", "user", "u3")
    return d


def _contents(msgs):
    return [(m["role"], m["content"]) for m in msgs]


class TestGetMessagesIter:
    def test_ascending_matches_get_messages(self, db):
        expected = _contents(db.get_messages("s1"))
        got = _contents(list(db.get_messages_iter("s1")))
        assert got == expected
        # chronological, oldest-first
        assert got[0] == ("user", "u1")
        assert got[-1] == ("user", "u3")

    def test_latest_true_yields_newest_first(self, db):
        # get_messages_iter(latest=True) yields newest-first (DESC) for tail
        # scanning — NOT the chronological order get_messages(latest=True)
        # returns. Equivalent to reversed(get_messages(...)).
        chronological = db.get_messages("s1")
        got = list(db.get_messages_iter("s1", latest=True))
        assert got == list(reversed(chronological))
        assert got[0]["content"] == "u3"  # newest first
        assert got[-1]["content"] == "u1"

    def test_find_newest_assistant_via_latest_scan(self, db):
        # The run_topics use case: scan newest-first, break on the first
        # assistant message with content. That must be the NEWEST assistant
        # ("a2"), not the oldest ("a1") — the regression the original
        # per-batch-reversal design would have introduced.
        found = None
        for m in db.get_messages_iter("s1", latest=True):
            if m["role"] == "assistant" and m["content"]:
                found = m["content"]
                break
        assert found == "a2"

    def test_early_break_does_not_exhaust(self, db):
        # A caller that breaks after the first yield should not be forced to
        # read the rest; the generator simply stops. (Correctness proxy: the
        # first yielded value in latest=True is the newest message.)
        gen = db.get_messages_iter("s1", latest=True)
        first = next(gen)
        assert first["content"] == "u3"
        gen.close()  # release the pooled read connection
        # DB is still usable afterwards (connection returned to pool).
        assert _contents(db.get_messages("s1"))[0] == ("user", "u1")

    def test_after_id_incompatible_with_latest(self, db):
        with pytest.raises(ValueError):
            list(db.get_messages_iter("s1", after_id=1, latest=True))

    def test_after_id_incompatible_with_offset(self, db):
        with pytest.raises(ValueError):
            list(db.get_messages_iter("s1", after_id=1, offset=1))

    def test_empty_session_yields_nothing(self, db):
        db.create_session("empty", "cli")
        assert list(db.get_messages_iter("empty")) == []
        assert list(db.get_messages_iter("empty", latest=True)) == []

    def test_decoded_fields_match_get_messages(self, db):
        # tool_calls / display_metadata decoded identically to get_messages.
        full = db.get_messages("s1")
        streamed = list(db.get_messages_iter("s1"))
        assert len(full) == len(streamed)
        for f, s in zip(full, streamed):
            assert f["role"] == s["role"]
            assert f["content"] == s["content"]
            assert f.get("tool_calls") == s.get("tool_calls")
            assert f.get("display_metadata") == s.get("display_metadata")

    def test_limit_caps_newest_first_stream(self, db):
        # limit with latest=True takes the N newest rows, yielded newest-first.
        got = _contents(list(db.get_messages_iter("s1", latest=True, limit=2)))
        assert got == [("user", "u3"), ("assistant", "a2")]

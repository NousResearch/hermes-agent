"""Message reactions: persistence, tapback semantics, and cache safety.

Behavior contracts, not snapshots — these assert how reactions must RELATE to
the transcript (one per author, announced once, never mutating history), so
they survive refactors of where reactions are stored.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return SessionDB(db_path=tmp_path / "state.db")


@pytest.fixture
def session(db):
    key = db.create_session("react-test", "test")
    db.append_message(key, "user", "how do i center a div")
    db.append_message(key, "assistant", "use flexbox")
    rows = [m["_row_id"] for m in db.get_messages_as_conversation(key, include_row_ids=True)]

    return key, rows


def test_conversation_rows_carry_durable_row_id(session, db):
    """Every projected message exposes its messages.id — reactions key off it."""
    key, rows = session

    assert all(isinstance(r, int) for r in rows)
    assert rows == sorted(rows), "row ids must follow insertion order"
    assert len(set(rows)) == len(rows), "row ids must be unique"


def test_one_reaction_per_author(session, db):
    """A second emoji from the same author REPLACES the first (iOS Tapback)."""
    key, rows = session

    db.set_message_reaction(key, rows[0], "\u2764\ufe0f", author="user")
    reactions = db.set_message_reaction(key, rows[0], "\U0001f602", author="user")

    assert [r["emoji"] for r in reactions] == ["\U0001f602"]


def test_repeating_an_emoji_retracts_it(session, db):
    """Tapping the live reaction again clears it."""
    key, rows = session

    db.set_message_reaction(key, rows[0], "\U0001f44d", author="user")
    reactions = db.set_message_reaction(key, rows[0], "\U0001f44d", author="user")

    assert reactions == []
    assert db.get_message_reactions(key, rows[0]) == []


def test_authors_are_independent(session, db):
    """User and agent each hold their own slot on the same message."""
    key, rows = session

    db.set_message_reaction(key, rows[0], "\u2764\ufe0f", author="user")
    reactions = db.set_message_reaction(key, rows[0], "\U0001f525", author="agent")

    assert {r["author"] for r in reactions} == {"user", "agent"}

    remaining = db.set_message_reaction(key, rows[0], None, author="user")
    assert [r["author"] for r in remaining] == ["agent"]


def test_rejects_rows_outside_the_session(session, db):
    """A row id from another conversation is never writable."""
    key, rows = session
    other = db.create_session("other", "test")
    db.append_message(other, "user", "elsewhere")

    assert db.set_message_reaction(key, 9999, "\u2764\ufe0f") is None
    assert db.set_message_reaction("no-such-session", rows[0], "\u2764\ufe0f") is None


def test_clearing_every_reaction_leaves_no_metadata(session, db):
    """An empty reaction set removes the key instead of persisting `[]`."""
    key, rows = session

    db.set_message_reaction(key, rows[0], "\u2764\ufe0f", author="user")
    db.set_message_reaction(key, rows[0], None, author="user")

    message = db.get_messages_as_conversation(key)[0]
    assert "display_metadata" not in message


def test_reactions_survive_reload(session, db):
    """Reactions are durable, not in-memory display state."""
    key, rows = session
    db.set_message_reaction(key, rows[1], "\U0001f525", author="agent")

    reopened = SessionDB(db_path=db.db_path)
    assert [r["emoji"] for r in reopened.get_message_reactions(key, rows[1])] == ["\U0001f525"]


def test_unseen_reactions_are_taken_exactly_once(session, db):
    """The model is told about a reaction on ONE turn, never twice."""
    key, rows = session
    db.set_message_reaction(key, rows[1], "\u2764\ufe0f", author="user")

    first = db.take_unseen_reactions(key, author="user")
    assert [e["emoji"] for e in first] == ["\u2764\ufe0f"]
    assert first[0]["row_id"] == rows[1]
    assert first[0]["text"] == "use flexbox"

    assert db.take_unseen_reactions(key, author="user") == []


def test_a_new_reaction_becomes_unseen_again(session, db):
    """Replacing a seen reaction re-arms the announcement."""
    key, rows = session
    db.set_message_reaction(key, rows[1], "\u2764\ufe0f", author="user")
    db.take_unseen_reactions(key, author="user")

    db.set_message_reaction(key, rows[1], "\U0001f525", author="user")

    assert [e["emoji"] for e in db.take_unseen_reactions(key, author="user")] == ["\U0001f525"]


def test_take_unseen_filters_by_author(session, db):
    """The agent's own reactions are never fed back to it as user input."""
    key, rows = session
    db.set_message_reaction(key, rows[0], "\U0001f60a", author="agent")

    assert db.take_unseen_reactions(key, author="user") == []


def test_reacting_never_mutates_message_content(session, db):
    """CACHE SAFETY: reacting must not rewrite any already-sent message.

    Rewriting a past message would invalidate the provider's cached prefix for
    the whole conversation — the reason reactions ride display_metadata and are
    announced on the NEXT turn instead.
    """
    key, rows = session
    before = [m["content"] for m in db.get_messages_as_conversation(key)]

    db.set_message_reaction(key, rows[0], "\u2764\ufe0f", author="user")
    db.set_message_reaction(key, rows[1], "\U0001f525", author="agent")
    db.take_unseen_reactions(key, author="user")

    after = [m["content"] for m in db.get_messages_as_conversation(key)]
    assert before == after


def test_latest_user_message_is_the_agents_default_target(session, db):
    """The agent reacts to "the message that triggered me" without an id."""
    key, rows = session
    assert db.latest_user_message_row_id(key) == rows[0]

    db.append_message(key, "user", "thanks!")
    newest = db.get_messages_as_conversation(key, include_row_ids=True)[-1]["_row_id"]

    assert db.latest_user_message_row_id(key) == newest


def test_row_id_is_opt_in_and_never_reaches_the_provider(session, db):
    """Only include_row_ids=True consumers see _row_id — and it's underscore-
    prefixed so transports strip it before the wire even for them. Default
    consumers (ACP restore, export) get the transcript in its historical shape.

    ``_db_persisted`` is the other sanctioned underscore key: stamped on every
    loaded row (#92231) so a flush can never re-append a resumed transcript.
    Like ``_row_id`` it is stripped before the wire by every transport.
    """
    key, _rows = session

    for message in db.get_messages_as_conversation(key):
        assert "_row_id" not in message
        assert message.get("_db_persisted") is True

    for message in db.get_messages_as_conversation(key, include_row_ids=True):
        assert "_row_id" in message
        assert all(
            not k.startswith("_") or k in {"_row_id", "_db_persisted"}
            for k in message
        )

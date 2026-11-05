"""Message reactions: persistence, tapback semantics, and cache safety.

Behavior contracts, not snapshots — these assert how reactions must RELATE to
the transcript (one per author, announced once, never mutating history), so
they survive refactors of where reactions are stored.
"""

import pytest

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import ConversationRequest, NewMessage
from persistence.conversation.sessions import SessionLineageKind
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import conversation, message_reactions
from tests.fixtures.storage_seed import seed_messages, seed_session

_WITH_ROW_IDS = ConversationRequest(include_row_ids=True)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = open_persistence(profile_scope(home=tmp_path))
    yield store
    store.close()


@pytest.fixture
def session(db):
    key = seed_session(db, "react-test", "test")
    seed_messages(db, key, {"role": "user", "content": "how do i center a div"},
                  {"role": "assistant", "content": "use flexbox"})
    rows = [m["_row_id"] for m in conversation(db, key, _WITH_ROW_IDS)]

    return key, rows


def test_conversation_rows_carry_durable_row_id(session, db):
    """Every projected message exposes its messages.id — reactions key off it."""
    _key, rows = session

    assert all(isinstance(r, int) for r in rows)
    assert rows == sorted(rows), "row ids must follow insertion order"
    assert len(set(rows)) == len(rows), "row ids must be unique"


def test_one_reaction_per_author(session, db):
    """A second emoji from the same author REPLACES the first (iOS Tapback)."""
    key, rows = session

    db.messages.set_reaction(key, rows[0], "\u2764\ufe0f", author="user")
    reactions = db.messages.set_reaction(key, rows[0], "\U0001f602", author="user")

    assert [r.emoji for r in reactions] == ["\U0001f602"]


def test_repeating_an_emoji_retracts_it(session, db):
    """Tapping the live reaction again clears it."""
    key, rows = session

    db.messages.set_reaction(key, rows[0], "\U0001f44d", author="user")
    reactions = db.messages.set_reaction(key, rows[0], "\U0001f44d", author="user")

    assert reactions == ()
    assert message_reactions(db, key, rows[0]) == ()


def test_authors_are_independent(session, db):
    """User and agent each hold their own slot on the same message."""
    key, rows = session

    db.messages.set_reaction(key, rows[0], "\u2764\ufe0f", author="user")
    reactions = db.messages.set_reaction(key, rows[0], "\U0001f525", author="agent")

    assert {r.author for r in reactions} == {"user", "agent"}

    remaining = db.messages.set_reaction(key, rows[0], None, author="user")
    assert [r.author for r in remaining] == ["agent"]


def test_rejects_rows_outside_the_session(session, db):
    """A row id from another conversation is never writable."""
    key, rows = session
    other = seed_session(db, "other", "test")
    seed_messages(db, other, {"role": "user", "content": "elsewhere"})

    assert db.messages.set_reaction(key, 9999, "\u2764\ufe0f", author="user") is None
    assert db.messages.set_reaction("no-such-session", rows[0], "\u2764\ufe0f", author="user") is None


def test_clearing_every_reaction_leaves_no_metadata(session, db):
    """An empty reaction set removes the key instead of persisting `[]`."""
    key, rows = session

    db.messages.set_reaction(key, rows[0], "\u2764\ufe0f", author="user")
    db.messages.set_reaction(key, rows[0], None, author="user")

    message = conversation(db, key)[0]
    assert "display_metadata" not in message


def test_reactions_survive_reload(session, db, tmp_path):
    """Reactions are durable, not in-memory display state."""
    key, rows = session
    db.messages.set_reaction(key, rows[1], "\U0001f525", author="agent")
    db.close()

    reopened = open_persistence(profile_scope(home=tmp_path))
    try:
        assert [r.emoji for r in message_reactions(reopened, key, rows[1])] == ["\U0001f525"]
    finally:
        reopened.close()


def test_unseen_reactions_are_taken_exactly_once(session, db):
    """The model is told about a reaction on ONE turn, never twice."""
    key, rows = session
    db.messages.set_reaction(key, rows[1], "\u2764\ufe0f", author="user")

    first = db.messages.take_unseen_reactions(key, author="user")
    assert [e.emoji for e in first] == ["\u2764\ufe0f"]
    assert first[0].row_id == rows[1]
    assert first[0].text == "use flexbox"

    assert db.messages.take_unseen_reactions(key, author="user") == ()


def test_a_new_reaction_becomes_unseen_again(session, db):
    """Replacing a seen reaction re-arms the announcement."""
    key, rows = session
    db.messages.set_reaction(key, rows[1], "\u2764\ufe0f", author="user")
    db.messages.take_unseen_reactions(key, author="user")

    db.messages.set_reaction(key, rows[1], "\U0001f525", author="user")

    assert [e.emoji for e in db.messages.take_unseen_reactions(key, author="user")] == ["\U0001f525"]


def test_take_unseen_filters_by_author(session, db):
    """The agent's own reactions are never fed back to it as user input."""
    key, rows = session
    db.messages.set_reaction(key, rows[0], "\U0001f60a", author="agent")

    assert db.messages.take_unseen_reactions(key, author="user") == ()


def test_reacting_never_mutates_message_content(session, db):
    """CACHE SAFETY: reacting must not rewrite any already-sent message.

    Rewriting a past message would invalidate the provider's cached prefix for
    the whole conversation — the reason reactions ride display_metadata and are
    announced on the NEXT turn instead.
    """
    key, rows = session
    before = [m["content"] for m in conversation(db, key)]

    db.messages.set_reaction(key, rows[0], "\u2764\ufe0f", author="user")
    db.messages.set_reaction(key, rows[1], "\U0001f525", author="agent")
    db.messages.take_unseen_reactions(key, author="user")

    after = [m["content"] for m in conversation(db, key)]
    assert before == after


def test_latest_user_message_is_the_agents_default_target(session, db):
    """The agent reacts to "the message that triggered me" without an id."""
    key, rows = session
    assert db.messages.latest_row_id(key, role="user") == rows[0]

    seed_messages(db, key, {"role": "user", "content": "thanks!"})
    newest = conversation(db, key, _WITH_ROW_IDS)[-1]["_row_id"]

    assert db.messages.latest_row_id(key, role="user") == newest

    # Role-targeting contract: a newer ASSISTANT message must not become the
    # agent's default target — it always means the latest USER message.
    seed_messages(db, key, {"role": "assistant", "content": "you're welcome"})
    assert db.messages.latest_row_id(key, role="user") == newest
    assert db.messages.latest_row_id(key, role="assistant") != newest


def test_row_id_is_opt_in_and_never_reaches_the_provider(session, db):
    """Only include_row_ids=True consumers see _row_id — and it's underscore-
    prefixed so transports strip it before the wire even for them. Default
    consumers (ACP restore, export) get the transcript in its historical shape.

    ``_db_persisted`` is the other sanctioned underscore key: stamped on every
    loaded row (#92231) so a flush can never re-append a resumed transcript.
    Like ``_row_id`` it is stripped before the wire by every transport.
    """
    key, _rows = session

    for message in conversation(db, key):
        assert "_row_id" not in message
        assert message.get("_db_persisted") is True

    for message in conversation(db, key, _WITH_ROW_IDS):
        assert "_row_id" in message
        assert all(
            not k.startswith("_") or k in {"_row_id", "_db_persisted"}
            for k in message
        )


@pytest.fixture
def compacted(session, db):
    """The parent segment ended by compaction; ``react-tip`` is the continuation the client now holds."""
    parent, rows = session
    assert db.coordination.try_acquire_compression_lock(session_id=parent, holder="w", ttl_seconds=60)
    db.lineage.publish_compression_child(parent, CompressionChild(session_id="react-tip", source="test"),
                                         [NewMessage.from_live({"role": "user", "content": "summary"})],
                                         compression_lock_holder="w")
    return parent, rows, "react-tip"


def test_ancestor_rows_react_through_the_continuation_key(compacted, db):
    """A display resume shows the whole compression lineage with row ids, so a row in an ended parent
    segment must react, read back and announce through the CONTINUATION key the client holds (#80670)."""
    _parent, rows, tip = compacted
    assert rows[1] in [m.row_id for m in db.messages.read_resume_conversations(tip).display]

    assert db.messages.set_reaction(tip, rows[1], "👍", author="user") == message_reactions(db, tip, rows[1])
    assert message_reactions(db, tip, rows[1])[0].emoji == "👍"
    assert [p.row_id for p in db.messages.take_unseen_reactions(tip)] == [rows[1]]


def test_lineage_scope_still_rejects_foreign_and_branch_rows(compacted, db):
    """Lineage widening must not turn row ids into a global lookup: an unrelated session's row and a row
    of an explicit /branch copy (which keeps its own rows) stay foreign to the continuation."""
    parent, rows, tip = compacted
    other = seed_session(db, "elsewhere", "test")
    [foreign] = seed_messages(db, other, {"role": "user", "content": "other conversation"})
    branch = seed_session(db, "branch", "test", parent_session_id=parent, lineage_kind=SessionLineageKind.BRANCH)

    assert db.messages.set_reaction(tip, foreign, "👍", author="user") is None
    assert db.messages.set_reaction(branch, rows[0], "👍", author="user") is None
    assert message_reactions(db, branch, rows[0]) == ()

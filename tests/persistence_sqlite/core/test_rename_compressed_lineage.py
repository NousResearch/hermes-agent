"""A user rename addressed to an ended compression segment lands on the conversation's live tip (#152624).

The SQLite backend is driven through the storage interface: the lineage is published with
``lineage.publish_compression_child`` and each rename is a ``runtime.mutate`` sidebar command.
"""
from __future__ import annotations

from typing import Any

import pytest

from persistence import AuthorityRefusal
from persistence.authority import runtime as rt
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE, SessionQuery, SessionWindow, TitleSource
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


@pytest.fixture
def db(tmp_path):
    store = open_persistence(profile_scope(home=tmp_path))
    yield store
    store.close()


def _compressed(db: Any, root_title: str | None = None) -> int:
    seed_session(db, "chat", "desktop", title=root_title)
    seed_messages(db, "chat", {"role": "user", "content": "hi"})
    db.lineage.publish_compression_child(
        "chat", CompressionChild(session_id="chat-2", source="desktop"),
        [NewMessage.from_live({"role": "user", "content": "hi"})], require_compression_lease=False)
    db.sessions.set_auto_title("chat-2", "Plugin para sessoes", source=TitleSource.LLM)
    return db.runtime.begin_epoch(instance_id="owner")


def mutate(db: Any, epoch: int, sid: str, title: str, request_id: str, expected_revision: int) -> dict:
    return db.runtime.mutate(epoch, principal_id="human", session_id=sid, request_id=request_id,
                             expected_revision=expected_revision, command=rt.Sidebar(title=title), owner=None).result


def _rename(db: Any, epoch: int, sid: str, title: str, request_id: str) -> dict:
    return mutate(db, epoch, sid, title, request_id, db.sessions.get(sid).runtime_revision)


def _title(db: Any, sid: str) -> str | None:
    return db.sessions.get(sid).title


def _listed(db: Any) -> dict:
    rows = [entry.as_list_row() for entry in db.sessions.list_sessions(SessionQuery(), SessionWindow(limit=50))]
    [row] = [s for s in rows if "chat" in s.get("_lineage_ids", [s["id"]])]
    return row


def test_rename_through_ended_segment_is_what_the_listing_shows(db):
    epoch = _compressed(db)
    receipt = _rename(db, epoch, "chat", "SESSION TALK", "r1")
    assert receipt["title"] == "SESSION TALK"
    assert _title(db, "chat-2") == "SESSION TALK"
    assert _title(db, "chat") is None
    assert _listed(db)["title"] == "SESSION TALK"
    # The receipt's revision is the addressed row's, so the client's next fence on that id holds.
    assert receipt["revision"] == db.sessions.get("chat").runtime_revision == 1
    assert _rename(db, epoch, "chat", "Second name", "r2")["title"] == "Second name"
    assert _title(db, "chat-2") == "Second name"


def test_name_held_by_ended_ancestor_moves_to_the_tip(db):
    epoch = _compressed(db, root_title="Old name")
    _rename(db, epoch, "chat", "Old name", "r1")
    assert _title(db, "chat-2") == "Old name"
    assert _title(db, "chat") is None
    # A rename sent to the tip itself (the classic CLI path) is unchanged.
    assert _rename(db, epoch, "chat-2", "Tip name", "r2")["title"] == "Tip name"
    assert _title(db, "chat-2") == "Tip name"


def test_a_tip_rename_makes_an_older_copy_of_any_segment_stale(db):
    """Client A read the ended segment, client B then renamed the live tip: A's rename (now landing
    on that same tip) must refuse instead of silently overwriting B's name."""
    epoch = _compressed(db)
    stale = db.sessions.get("chat").runtime_revision
    _rename(db, epoch, "chat-2", "B name", "b1")
    with pytest.raises(AuthorityRefusal, match="revision_conflict"):
        mutate(db, epoch, "chat", "A name", "a1", stale)
    assert _title(db, "chat-2") == "B name"


def test_clearing_through_an_ended_segment_clears_the_listed_name(db):
    """A rotation cut off before the title copy leaves the name on the root, and the untitled tip
    lists under it: a clear must remove that fallback too, or the sidebar keeps the old name."""
    epoch = _compressed(db, root_title="Old name")
    db.sessions.set_title("chat-2", "")  # the copy never happened: the tip is untitled
    assert _rename(db, epoch, "chat", "", "c1")["title"] is None
    assert _listed(db)["title"] is None


def test_a_name_freed_from_another_chat_advances_that_chats_revision(db):
    """Taking a name an abandoned empty chat held clears it there too: that row's revision must move,
    or a client's older copy of it still passes the fence."""
    seed_session(db, "ghost", "desktop", title="Shared name", end_reason="user_exit")
    epoch = _compressed(db)
    before = db.sessions.get("ghost").runtime_revision
    _rename(db, epoch, "chat", "Shared name", "g1")
    assert (_title(db, "ghost"), _title(db, "chat-2")) == (None, "Shared name")
    assert db.sessions.get("ghost").runtime_revision > before


def test_clearing_through_a_canonical_bot_chat_ancestor_keeps_its_name(db):
    """The hidden canonical Bot Chat is found by its exact name: a clear must not reach past the
    untitled tip and empty it on the ancestor, or the bot loses its conversation."""
    epoch = _compressed(db, root_title=CANONICAL_BOT_CHAT_TITLE)
    db.sessions.set_hidden("chat", True)
    db.sessions.set_title("chat-2", "")  # the rotation's title copy never happened
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        _rename(db, epoch, "chat", "", "b1")
    assert db.sessions.resolve_by_title(CANONICAL_BOT_CHAT_TITLE) is not None

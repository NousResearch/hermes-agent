"""Every display projection of a compacted session must agree.

In-place compaction archives earlier turns as ``active=0, compacted=1`` rows.
They are durable display history — the user's own conversation, still on disk.
#80680 taught the REST transcript read to include them, but three GATEWAY
display projections kept filtering ``active = 1``:

- ``get_resume_conversations()``   — what ``session.resume`` ships
- ``get_ancestor_display_prefix()`` — the ancestor lineage prefix
- ``get_messages_as_conversation()`` — the warm-session payload on tab switch

So the same conversation read four ways gave two different answers: REST showed
everything, the gateway cut the transcript off at the compaction boundary. The
user sees their chat "vanish" down to a summary plus a couple of carried-forward
turns, and a resumed agent that cannot see its own completed work starts it over
(#92080, #93618, #68321).

These tests assert the INVARIANT — all display reads of one session return the
same transcript — rather than any particular row count, and pin the two things
that must NOT grow with it: the model-fed projection stays compressed, and
soft-deleted Undo/Rewind rows stay hidden.
"""

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message,
    archive_and_compact,
    assert_resume_safe,
    get_ancestor_display_prefix,
    get_messages,
    get_messages_as_conversation,
    get_resume_conversations,
    rewind_to_message,
)
from persistence.sqlite.conversation._store.sessions import create_session, end_session


@pytest.fixture
def db(tmp_path):
    return StoreHolder(tmp_path / "state.db")


def _resume_count(db, sid, *, tip_only=False):
    """The rows a resume would load: the resume guard's count under a limit no test reaches."""
    return assert_resume_safe(db, sid, max_messages=1_000_000, tip_only=tip_only)


def _compact_in_place(db, sid, *, epochs=3, turns=4, tail_count=2):
    """Drive *sid* through repeated in-place compaction, like a long chat."""
    create_session(db, sid, source="desktop")
    for epoch in range(epochs):
        for i in range(turns):
            append_message(db, sid, "user", f"e{epoch} user {i}")
            append_message(db, sid, "assistant", f"e{epoch} assistant {i}")
        live = get_messages_as_conversation(db, sid)
        archive_and_compact(
            db,
            sid,
            [{"role": "user", "content": f"[summary {epoch}]"}] + live[-tail_count:],
            tail_count=tail_count,
        )
    return sid


def _texts(messages):
    return [(m["role"], m["content"]) for m in messages]


def _rest_display(db, sid):
    """The read that was already correct — the parity reference."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in get_messages(db, sid, include_compacted=True)
    ]


class TestDisplayProjectionParity:
    def test_resume_display_matches_the_rest_transcript(self, db):
        sid = _compact_in_place(db, "chat")

        _, display = get_resume_conversations(db, sid)

        assert _texts(display) == _texts(_rest_display(db, sid))

    def test_warm_session_display_matches_the_rest_transcript(self, db):
        """The read behind ``_live_visible_history`` (switching back to a tab)."""
        sid = _compact_in_place(db, "chat")

        warm = get_messages_as_conversation(
            db,
            sid, include_ancestors=True, include_row_ids=True, include_compacted=True
        )

        assert _texts(warm) == _texts(_rest_display(db, sid))

    def test_pre_compaction_turns_survive_in_the_resume_transcript(self, db):
        """The user's own first turn is still there after several compactions."""
        sid = _compact_in_place(db, "chat")

        _, display = get_resume_conversations(db, sid)

        assert ("user", "e0 user 0") in _texts(display)
        assert ("assistant", "e0 assistant 0") in _texts(display)

    def test_display_read_dedupes_carried_forward_tail(self, db):
        """Each logical message appears once, not once per compaction epoch."""
        sid = _compact_in_place(db, "chat", epochs=4, tail_count=2)

        _, display = get_resume_conversations(db, sid)
        seen = _texts(display)

        assert len(seen) == len(set(seen))


class TestModelProjectionStaysCompressed:
    def test_model_history_excludes_archived_rows(self, db):
        """Compaction must still do its job: the model gets the compressed set."""
        sid = _compact_in_place(db, "chat")

        model, display = get_resume_conversations(db, sid)

        assert len(model) < len(display)
        assert ("user", "e0 user 0") not in _texts(model)

    def test_model_history_matches_the_active_only_read(self, db):
        sid = _compact_in_place(db, "chat")

        model, _ = get_resume_conversations(db, sid)
        active_only = get_messages_as_conversation(db, sid, repair_alternation=True)

        assert _texts(model) == _texts(active_only)


class TestSoftDeletedRowsStayHidden:
    def test_rewound_rows_are_excluded_from_the_display_projections(self, db):
        """Undo/Rewind rows (active=0, compacted=0) are NOT display history."""
        sid = "chat"
        create_session(db, sid, source="desktop")
        append_message(db, sid, "user", "kept")
        append_message(db, sid, "assistant", "kept reply")
        append_message(db, sid, "user", "taken back")
        append_message(db, sid, "assistant", "taken back reply")

        rewind_target = next(
            m for m in reversed(get_messages(db, sid)) if m["role"] == "user"
        )
        rewind_to_message(db, sid, rewind_target["id"])

        _, display = get_resume_conversations(db, sid)
        warm = get_messages_as_conversation(
            db,
            sid, include_ancestors=True, include_compacted=True
        )

        for projection in (display, warm):
            contents = [c for _, c in _texts(projection)]
            assert "taken back" not in contents
            assert "kept" in contents


class TestAncestorPrefix:
    def test_prefix_includes_a_compacted_ancestor_s_archived_rows(self, db):
        """A compression ROTATION's parent still shows its pre-compaction turns."""
        parent, child = "parent", "child"
        create_session(db, parent, source="desktop")
        for i in range(3):
            append_message(db, parent, "user", f"P user {i}")
            append_message(db, parent, "assistant", f"P assistant {i}")
        archive_and_compact(db, parent, [{"role": "user", "content": "[parent summary]"}])
        # A real rotation stamps the parent end_reason='compression' (publish_compression_child).
        end_session(db, parent, "compression")

        create_session(db, child, source="desktop", parent_session_id=parent)
        append_message(db, child, "user", "C user 0")
        append_message(db, child, "assistant", "C assistant 0")

        prefix = get_ancestor_display_prefix(db, child)
        _, display = get_resume_conversations(db, child)

        assert ("user", "P user 0") in _texts(prefix)
        assert ("user", "P user 0") in _texts(display)
        # The child's own turns belong to the tip, never the ancestor prefix.
        assert ("user", "C user 0") not in _texts(prefix)

    def test_explicit_branch_has_no_ancestor_prefix(self, db):
        """A /branch copy owns its transcript; the live parent must not leak in."""
        sid = _compact_in_place(db, "chat")
        create_session(
            db,
            "branch",
            source="desktop",
            parent_session_id=sid,
            model_config={"_branched_from": sid},
        )
        append_message(db, "branch", "user", "branch turn")

        assert get_ancestor_display_prefix(db, "branch") == []

        _, display = get_resume_conversations(db, "branch")
        assert _texts(display) == [("user", "branch turn")]


class TestResumeGuardBoundsWhatResumeLoads:
    def test_guard_counts_the_rows_the_display_read_materializes(self, db):
        """The guard must not undercount: it bounds an in-memory materialization."""
        sid = _compact_in_place(db, "chat", epochs=4)

        _, display = get_resume_conversations(db, sid)

        assert _resume_count(db, sid) >= len(display)

    def test_guard_rejects_a_lineage_over_the_limit(self, db):
        from persistence.sqlite.conversation._store.core import SessionResumeTooLargeError

        sid = _compact_in_place(db, "chat", epochs=4)

        with pytest.raises(SessionResumeTooLargeError):
            assert_resume_safe(db, sid, max_messages=2)

    def test_tip_only_guard_still_bounds_only_the_live_tip(self, db):
        """The #4130 carve-out: a healthy compacted chat must stay resumable.

        A well-compressed conversation is exactly the shape compression is
        meant to produce. Counting its archive against a tip-sized budget is
        what stranded Bot Chats on "Waking up…"; ``tip_only`` callers never
        materialize the archive, so they keep the active-only bound.
        """
        sid = _compact_in_place(db, "chat", epochs=4)

        tip_count = _resume_count(db, sid, tip_only=True)

        assert tip_count < _resume_count(db, sid)
        assert assert_resume_safe(db, sid, max_messages=tip_count, tip_only=True)

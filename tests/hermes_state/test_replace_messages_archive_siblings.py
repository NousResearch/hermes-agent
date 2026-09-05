"""Sibling-site regression tests for the #80216 bug class.

#80216 fixed /retry destroying soft-archived (``active=0/compacted=1``)
in-place-compaction rows because ``rewrite_transcript`` defaulted to a
destructive full ``replace_messages``.  The same class existed at two more
sites, fixed here:

- ``acp_adapter/session.py`` ``_persist`` (non-owned-agent branch): probed
  ``has_archived_messages`` and FAILED OPEN into the destructive replace on
  any probe error (and could race a concurrent ``archive_and_compact``).
  Now passes ``active_only=True`` unconditionally.
- ``tui_gateway/methods_prompt.py`` edit/regenerate truncation: bare
  ``replace_messages`` deleted the archived transcript on every
  edit/regenerate of a compacted session.  Now ``active_only=True``.

Behavior contract on a fresh (never-compacted) session: every row is
``active=1``, so the active-only replace is identical to the full replace —
also pinned below.
"""

from types import SimpleNamespace

import pytest

from acp_adapter import session as acp_session
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.fixture
def state_db(tmp_path):
    """A real SessionDB on a temp state.db."""
    return SessionDB(tmp_path / "state.db")


def _seed_compacted_session(db, session_id: str) -> None:
    """Create a session with archived (active=0/compacted=1) + live rows."""
    db.create_session(session_id, "test")
    msgs = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "another old question"},
        {"role": "assistant", "content": "another old answer"},
    ]
    db.append_messages_batch(session_id, msgs)
    db.archive_and_compact(
        session_id,
        [
            {"role": "assistant", "content": "summary of old turns"},
            {"role": "user", "content": "live question"},
            {"role": "assistant", "content": "live answer"},
        ],
    )


def _archived_count(db, session_id: str) -> int:
    return sum(
        1
        for m in db.get_messages(session_id, include_inactive=True)
        if not m["active"]
    )


class TestAcpPersistPreservesArchives:
    def test_persist_nonowned_branch_keeps_archived_rows(
        self, state_db, monkeypatch
    ):
        """The ACP _persist fallback replace must not delete archived rows."""
        monkeypatch.setattr(acp_session, "_register_task_cwd", lambda *_args: None)
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(),
            db=state_db,
        )
        state = manager.create_session()
        sid = state.session_id
        state_db.append_messages_batch(
            sid,
            [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "another old question"},
                {"role": "assistant", "content": "another old answer"},
            ],
        )
        state_db.archive_and_compact(
            sid,
            [
                {"role": "assistant", "content": "summary of old turns"},
                {"role": "user", "content": "live question"},
                {"role": "assistant", "content": "live answer"},
            ],
        )
        assert _archived_count(state_db, sid) == 4

        # A probe failure must not turn the fallback into a destructive full
        # replace. Drive the public manager save path so this pins the actual
        # _persist wiring rather than the shape of its source code.
        def fail_if_probed(*_args, **_kwargs):
            raise OSError("transient archive lookup failure")

        monkeypatch.setattr(state_db, "has_archived_messages", fail_if_probed)
        state.history = [
            {"role": "user", "content": "rewritten"},
            {"role": "assistant", "content": "rewritten answer"},
        ]
        manager.save_session(sid)

        assert _archived_count(state_db, sid) == 4
        live = [
            m for m in state_db.get_messages_as_conversation(sid)
            if m.get("role") in ("user", "assistant")
        ]
        assert [m["content"] for m in live] == ["rewritten", "rewritten answer"]

    def test_fresh_session_active_only_equals_full_replace(self, state_db):
        """On a never-compacted session active_only=True must behave exactly
        like the historical full replace (the safety claim the unconditional
        switch rests on)."""
        sid = "acp-fresh"
        state_db.create_session(sid, "test")
        state_db.append_messages_batch(
            sid,
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )
        state_db.replace_messages(
            sid, [{"role": "user", "content": "only"}], active_only=True
        )
        rows = [
            m for m in state_db.get_messages_as_conversation(sid)
            if m.get("role") in ("user", "assistant")
        ]
        assert [m["content"] for m in rows] == ["only"]
        assert _archived_count(state_db, sid) == 0


class TestTuiPromptTruncationPreservesArchives:
    def test_truncation_write_keeps_archived_rows(self, state_db):
        """Exercise active-only replacement on a compacted session and preserve
        its archived-row count."""
        sid = "tui-compacted"
        _seed_compacted_session(state_db, sid)
        assert _archived_count(state_db, sid) == 4

        truncated = [{"role": "user", "content": "kept head"}]
        state_db.replace_messages(sid, truncated, active_only=True)

        assert _archived_count(state_db, sid) == 4
        live = [
            m for m in state_db.get_messages_as_conversation(sid)
            if m.get("role") in ("user", "assistant")
        ]
        assert [m["content"] for m in live] == ["kept head"]


class TestArchiveDroppedIsRecoverable:
    """`active_only=True` protects rows archived EARLIER; it still DELETEs the
    live ones it replaces.

    That is the last write standing between a mis-aimed rewind and permanent
    loss, and all three reported incidents (#70516, #80763, #82756) ended
    there with an empty WAL, no `active=0` rows and an FTS entry dropped in
    sync. `archive_dropped=True` keeps the replaced turns on disk under the
    same "the user took it back" marking `rewind_to_message` uses.
    """

    def test_dropped_turns_survive_as_inactive_rows(self, state_db):
        sid = "archive-dropped"
        state_db.create_session(sid, "test")
        state_db.append_messages_batch(
            sid,
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second reply"},
            ],
        )

        state_db.replace_messages(
            sid,
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first reply"},
            ],
            active_only=True,
            archive_dropped=True,
        )

        # The live transcript is exactly what a destructive replace would leave.
        live = [
            m for m in state_db.get_messages_as_conversation(sid)
            if m.get("role") in ("user", "assistant")
        ]
        assert [m["content"] for m in live] == ["first", "first reply"]

        # …but the dropped turns are still readable instead of gone.
        recovered = [
            m["content"]
            for m in state_db.get_messages(sid, include_inactive=True)
            if not m["active"]
        ]
        assert "second" in recovered
        assert "second reply" in recovered

    def test_archived_rows_use_rewind_marking_not_compaction(self, state_db):
        """compacted=0 keeps abandoned turns out of session search results.

        `archive_and_compact` marks its rows compacted=1 precisely so they stay
        discoverable; a rewound turn is one the user took back, so it must
        carry the `rewind_to_message` marking instead.
        """
        sid = "archive-marking"
        state_db.create_session(sid, "test")
        state_db.append_messages_batch(
            sid,
            [
                {"role": "user", "content": "keep"},
                {"role": "assistant", "content": "drop me"},
            ],
        )

        state_db.replace_messages(
            sid,
            [{"role": "user", "content": "keep"}],
            active_only=True,
            archive_dropped=True,
        )

        archived = [
            m for m in state_db.get_messages(sid, include_inactive=True)
            if not m["active"]
        ]
        assert archived, "the replaced rows must still be on disk"
        assert all(not m.get("compacted") for m in archived)

    def test_default_stays_destructive(self, state_db):
        """The other three callers must be untouched by the new parameter."""
        sid = "archive-default"
        state_db.create_session(sid, "test")
        state_db.append_messages_batch(
            sid,
            [
                {"role": "user", "content": "gone"},
                {"role": "assistant", "content": "also gone"},
            ],
        )

        state_db.replace_messages(sid, [{"role": "user", "content": "fresh"}])

        assert not [
            m for m in state_db.get_messages(sid, include_inactive=True)
            if not m["active"]
        ]

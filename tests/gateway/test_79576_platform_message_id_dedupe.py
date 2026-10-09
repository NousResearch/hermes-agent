"""Regression tests for issue #79576 follow-up — ``has_platform_message_id``
visibility contract.

Scope note (2026-09-09): main ``3114916ee4`` (fix #104653) landed the
accepted-input **ownership marker** approach for inbound-turn dedupe — the
gateway's exception path and restart recovery now gate on
``has_gateway_input_owner``, and main's contract tests
(``tests/agent/test_codex_echo_ownership.py``,
``tests/gateway/test_failure_writer_ownership.py``) require that two
independently accepted turns sharing a platform_message_id BOTH survive.
That supersedes this PR's original in-transaction DB dedupe, which is why
these tests no longer assert one-row-per-platform-id.

What remains (and what these tests pin down) is the **visibility contract**
of the surviving probe, ``SessionDB.has_platform_message_id`` — the
gateway's transient-failure retry guard (#47237, ``run_turn.py``) and the
restart drain-window recovery dedup (``agent/session_persistence.py``) both
consult it before re-persisting an inbound turn. Before this fix the probe
counted ANY row carrying the id — including soft-archived rewind/undo rows
(``active=0, compacted=0``) — so a platform redelivery of a turn the user
had taken back was silently swallowed instead of re-persisted.

The contract mirrors ``has_gateway_input_owner`` (main ``3114916ee4``):

- live rows (``active = 1``) count — the original #47237 guard behavior;
- compaction-archived rows (``active = 0, compacted = 1``) count —
  re-inserting a summarized-away turn would resurrect it after in-place
  compaction;
- rewind/undo rows (``active = 0, compacted = 0``) do NOT count — the user
  took the turn back, so a redelivery of the same id re-persists.
"""

import pytest

from hermes_state import SessionDB


def _make_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", "cli")
    return db


def _archive_rows(db, session_id, *, compacted):
    """Flip every live row of the session to active=0 with the given flag."""
    def _do(conn):
        conn.execute(
            "UPDATE messages SET active = 0, compacted = ? "
            "WHERE session_id = ? AND active = 1",
            (1 if compacted else 0, session_id),
        )

    db._execute_write(_do)  # noqa: SLF001 - test-only helper


def _unarchive_rows(db, session_id):
    def _do(conn):
        conn.execute(
            "UPDATE messages SET active = 1, compacted = 0 "
            "WHERE session_id = ? AND active = 0",
            (session_id,),
        )

    db._execute_write(_do)  # noqa: SLF001 - test-only helper


class TestHasPlatformMessageIdVisibility:
    """The probe only counts rows the transcript would still show."""

    def test_live_row_counts(self, tmp_path):
        db = _make_db(tmp_path)
        db.append_message(
            session_id="s1", role="user", content="hello",
            platform_message_id="msg-1",
        )
        assert db.has_platform_message_id("s1", "msg-1") is True

    def test_absent_id_is_false(self, tmp_path):
        db = _make_db(tmp_path)
        db.append_message(
            session_id="s1", role="user", content="hello",
            platform_message_id="msg-1",
        )
        assert db.has_platform_message_id("s1", "msg-other") is False

    def test_rewound_row_does_not_count(self, tmp_path):
        """Redelivery of a rewound turn must re-persist: the probe must not
        let an archived rewind/undo row shadow a legitimate re-insert.

        Regression: before the fix the probe counted ANY row with the id, so
        the gateway retry guard swallowed the redelivery and the user's
        re-sent message never reached the transcript."""
        db = _make_db(tmp_path)
        db.append_message(
            session_id="s1", role="user", content="taken back",
            platform_message_id="msg-9",
        )
        assert db.has_platform_message_id("s1", "msg-9") is True
        _archive_rows(db, "s1", compacted=False)  # rewind/undo rows
        assert db.get_messages("s1") == []  # nothing live anymore
        assert db.has_platform_message_id("s1", "msg-9") is False

    def test_compacted_row_still_counts(self, tmp_path):
        """A summarized-away turn stays visible to the probe — re-inserting
        it after in-place compaction would resurrect the turn."""
        db = _make_db(tmp_path)
        db.append_message(
            session_id="s1", role="user", content="summarized away",
            platform_message_id="msg-9",
        )
        _archive_rows(db, "s1", compacted=True)
        assert db.get_messages("s1") == []
        assert db.has_platform_message_id("s1", "msg-9") is True

    def test_visibility_round_trip(self, tmp_path):
        """Un-archiving restores visibility — the contract tracks the row's
        current state, not a one-time latch."""
        db = _make_db(tmp_path)
        db.append_message(
            session_id="s1", role="user", content="taken back",
            platform_message_id="msg-9",
        )
        _archive_rows(db, "s1", compacted=False)
        assert db.has_platform_message_id("s1", "msg-9") is False
        _unarchive_rows(db, "s1")
        assert db.has_platform_message_id("s1", "msg-9") is True

    def test_scoped_per_session(self, tmp_path):
        """An id archived in one session never shadows another session's
        probe — the guard is (session_id, platform_message_id) scoped."""
        db = _make_db(tmp_path)
        db.create_session("s2", "cli")
        db.append_message(
            session_id="s1", role="user", content="hello",
            platform_message_id="msg-123",
        )
        _archive_rows(db, "s1", compacted=False)
        assert db.has_platform_message_id("s1", "msg-123") is False
        db.append_message(
            session_id="s2", role="user", content="hello",
            platform_message_id="msg-123",
        )
        assert db.has_platform_message_id("s2", "msg-123") is True


class TestRedeliveryAfterRewind:
    """End-to-end statement of the user-visible behavior: a platform
    redelivery of a rewound turn lands in the transcript again."""

    def test_redelivery_reinserts_new_live_row(self, tmp_path):
        db = _make_db(tmp_path)
        first_id = db.append_message(
            session_id="s1", role="user", content="taken back",
            platform_message_id="msg-9",
        )
        _archive_rows(db, "s1", compacted=False)
        assert db.get_messages("s1") == []

        # The retry guard consults the probe; with the fix it does not skip.
        assert db.has_platform_message_id("s1", "msg-9") is False
        second_id = db.append_message(
            session_id="s1", role="user", content="redelivered",
            platform_message_id="msg-9",
        )
        assert second_id != first_id  # NEW row, not the archived one
        live = db.get_messages("s1")
        assert len(live) == 1
        assert live[0]["content"] == "redelivered"

    def test_redelivery_after_compaction_is_guarded(self, tmp_path):
        """The same path through a compaction-archived row stays blocked:
        the probe still answers True so the guard skips the re-insert."""
        db = _make_db(tmp_path)
        db.append_message(
            session_id="s1", role="user", content="summarized away",
            platform_message_id="msg-9",
        )
        _archive_rows(db, "s1", compacted=True)
        assert db.has_platform_message_id("s1", "msg-9") is True
        assert db.get_messages("s1") == []  # nothing resurrected


class TestSessionStoreWrapperContract:
    """The gateway's SessionStore wrapper (gateway/session.py composing
    SessionTranscriptMixin) is the seam the retry guard actually calls —
    its thin wrapper must inherit the visibility contract from SessionDB,
    not re-implement it."""

    @pytest.fixture()
    def store(self, tmp_path, monkeypatch):
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore

        db = _make_db(tmp_path)
        store = SessionStore(tmp_path, GatewayConfig())
        monkeypatch.setattr(store, "_db_for_session_id", lambda sid: db)
        store._db = db
        return store

    def test_wrapper_delegates_visibility(self, store, tmp_path):
        db = store._db
        db.append_message(
            session_id="s1", role="user", content="taken back",
            platform_message_id="msg-9",
        )
        _archive_rows(db, "s1", compacted=False)
        assert store.has_platform_message_id("s1", "msg-9") is False
        _unarchive_rows(db, "s1")
        assert store.has_platform_message_id("s1", "msg-9") is True

    def test_wrapper_false_without_db(self, tmp_path, monkeypatch):
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore

        store = SessionStore(tmp_path, GatewayConfig())
        monkeypatch.setattr(store, "_db_for_session_id", lambda sid: None)  # in-memory session
        # Must not raise and must not count: the guard skips nothing.
        assert store.has_platform_message_id("s1", "msg-9") is False

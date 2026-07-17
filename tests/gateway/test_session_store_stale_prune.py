"""Tests for SessionStore._prune_stale_sessions_locked — crash self-healing.

When a gateway crashes (exit code 1) the graceful shutdown path is skipped and
sessions.json is left pointing at sessions already ended in state.db. On the
next startup _ensure_loaded_locked calls _prune_stale_sessions_locked to detect
and remove those stale routing entries before get_or_create_session() can reuse
them and silently route incoming messages into a closed session (#52804).
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(key: str, session_id: str) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=key,
        session_id=session_id,
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )


def _make_entry_with_origin(key: str, session_id: str) -> SessionEntry:
    entry = _make_entry(key, session_id)
    entry.origin = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5140768830",
        chat_type="dm",
        user_id="5140768830",
        user_name="João",
    )
    return entry


def _make_store_with_db(tmp_path, db_mock) -> SessionStore:
    """Build a SessionStore with a mock SessionDB, bypassing disk load."""
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db_mock
    store._loaded = True
    return store


def _db_returning(rows: dict) -> MagicMock:
    """SessionDB mock where get_session maps session_id -> row dict."""
    db = MagicMock()
    db.get_session.side_effect = lambda sid: rows.get(sid)
    return db


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

class TestPruneStaleSessionsLocked:
    def test_prunes_ended_session(self, tmp_path):
        db = _db_returning({"sid_dm": {"end_reason": "agent_close", "id": "sid_dm"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["dm_key"] = _make_entry("dm_key", "sid_dm")

        store._prune_stale_sessions_locked()

        assert "dm_key" not in store._entries

    def test_keeps_live_session(self, tmp_path):
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["live_key"] = _make_entry("live_key", "sid_live")

        store._prune_stale_sessions_locked()

        assert "live_key" in store._entries

    def test_keeps_session_absent_from_db(self, tmp_path):
        """Entry for a session_id not in state.db (legacy) is left alone."""
        db = _db_returning({})
        store = _make_store_with_db(tmp_path, db)
        store._entries["legacy_key"] = _make_entry("legacy_key", "sid_legacy")

        store._prune_stale_sessions_locked()

        assert "legacy_key" in store._entries

    def test_prunes_multiple_stale_entries(self, tmp_path):
        db = _db_returning({
            "sid_a": {"end_reason": "agent_close", "id": "sid_a"},
            "sid_b": {"end_reason": "session_reset", "id": "sid_b"},
            "sid_c": {"end_reason": None, "id": "sid_c"},  # alive — keep
        })
        store = _make_store_with_db(tmp_path, db)
        store._entries["key_a"] = _make_entry("key_a", "sid_a")
        store._entries["key_b"] = _make_entry("key_b", "sid_b")
        store._entries["key_c"] = _make_entry("key_c", "sid_c")

        store._prune_stale_sessions_locked()

        assert "key_a" not in store._entries
        assert "key_b" not in store._entries
        assert "key_c" in store._entries

    def test_repoints_stale_compression_parent_to_latest_live_child(self, tmp_path):
        """Compression-ended parents should recover their live child mapping.

        A gateway crash can leave sessions.json pointing at the pre-compression
        parent (end_reason='compression') even though the agent already rotated
        into a live child session. If the child has gateway peer metadata, the
        startup prune pass must repoint the route instead of deleting it, or
        restart auto-resume and queued follow-ups have no session to continue.
        """
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning({
            "sid_parent": {"end_reason": "compression", "id": "sid_parent"},
        })
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_child",
            "started_at": 1782744974.0,
        }
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key in store._entries
        assert store._entries[key].session_id == "sid_child"
        db.find_latest_gateway_session_for_peer.assert_called_once()
        db.reopen_session.assert_called_once_with("sid_child")

    def test_prunes_stale_entry_when_recovery_only_finds_same_ended_session(self, tmp_path):
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning({"sid_parent": {"end_reason": "agent_close", "id": "sid_parent"}})
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_parent",
            "started_at": 1782744974.0,
        }
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key not in store._entries

    def test_no_resurrect_ended_session_with_different_id(self, tmp_path):
        """#61993 regression: a stale routing entry whose latest peer row is
        ALSO ended (different id, e.g. an explicit ``/new`` reset or a
        compression-ended parent whose newest sibling is ended) must NOT be
        reopened and resurrected — recovery must drop the stale entry so the
        next message mints a fresh session.

        The bug: ``_recover_session_from_db`` reopened the ended row (clearing
        end_reason) and repointed the route to it, silently undoing the user's
        reset. The fix peeks the latest peer row's liveness *before* recovery
        and only recovers genuinely live sessions.
        """
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning({"sid_ended": {"end_reason": "session_reset", "id": "sid_ended"}})
        # Latest recoverable peer row is a DIFFERENT `agent_close` session — the
        # only end_reason the production finder returns. It must NOT be reopened,
        # and it must be superseded so the next inbound message can't resurrect
        # it via the per-message recovery path (#62012).
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_ended_v2",
            "end_reason": "agent_close",
            "started_at": 1782744974.0,
        }
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_ended")

        store._prune_stale_sessions_locked()

        assert key not in store._entries
        # Recovery must never reopen an ended session...
        db.reopen_session.assert_not_called()
        # ...and the recoverable row is finalized so it can't be resurrected.
        db.mark_session_superseded.assert_called_once_with("sid_ended_v2")

    def test_noop_when_db_is_none(self, tmp_path):
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        store._loaded = True
        store._entries["key"] = _make_entry("key", "sid_x")

        store._prune_stale_sessions_locked()  # must not raise

        assert "key" in store._entries

    def test_noop_when_no_entries(self, tmp_path):
        db = MagicMock()
        store = _make_store_with_db(tmp_path, db)

        store._prune_stale_sessions_locked()

        db.get_session.assert_not_called()

    def test_db_error_is_non_fatal(self, tmp_path):
        db = MagicMock()
        db.get_session.side_effect = Exception("DB locked")
        store = _make_store_with_db(tmp_path, db)
        store._entries["key"] = _make_entry("key", "sid_x")

        store._prune_stale_sessions_locked()  # must not raise

        assert "key" in store._entries  # safe fallback — keep on error

    def test_sessions_json_rewritten_after_pruning(self, tmp_path):
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["stale_key"] = _make_entry("stale_key", "sid_stale")

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()
            mock_save.assert_called_once()

    def test_sessions_json_not_rewritten_when_nothing_pruned(self, tmp_path):
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["live_key"] = _make_entry("live_key", "sid_live")

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()
            mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: _ensure_loaded_locked calls _prune_stale_sessions_locked
# ---------------------------------------------------------------------------

class TestEnsureLoadedCallsPrune:
    def test_stale_entry_pruned_during_load(self, tmp_path):
        entry = _make_entry("dm_key", "sid_stale")
        (tmp_path / "sessions.json").write_text(
            json.dumps({"dm_key": entry.to_dict()}, indent=2), encoding="utf-8"
        )
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db

        store._ensure_loaded()

        assert "dm_key" not in store._entries

    def test_live_entry_survives_load(self, tmp_path):
        entry = _make_entry("active_key", "sid_live")
        (tmp_path / "sessions.json").write_text(
            json.dumps({"active_key": entry.to_dict()}, indent=2), encoding="utf-8"
        )
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db

        store._ensure_loaded()

        assert "active_key" in store._entries


# ---------------------------------------------------------------------------
# End-to-end with a real SessionDB: startup prune + next inbound message
# ---------------------------------------------------------------------------

class TestRealDbStartupThenNextMessage:
    def test_reset_then_restart_then_message_starts_fresh_session(self, tmp_path):
        """#62012 e2e: an ended (`agent_close`) session must NOT be resurrected
        by the FIRST inbound message after a gateway restart.

        This exercises the actual production path teknium1 flagged: the prune
        drops the stale routing key, but without superseding the recoverable
        row the very next ``get_or_create_session`` re-queries the finder,
        reopens the ``agent_close`` row, and resurrects the history. With the
        fix the row is superseded, so the finder returns nothing and a fresh
        session id is minted.
        """
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="5140768830",
            chat_type="dm",
            user_id="5140768830",
            user_name="João",
        )
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        # ---- Session 1: created, given a message, then ended as agent_close ----
        key = store._generate_session_key(source)
        old_entry = store.get_or_create_session(source)
        old_id = old_entry.session_id
        # Persist a durable row + peer + a message so the finder deems it
        # recoverable, then end it the way graceful shutdown does.
        db.create_session(
            old_id,
            source=source.platform.value,
            user_id=source.user_id,
            session_key=key,
            chat_id=source.chat_id,
            chat_type=source.chat_type,
        )
        db.record_gateway_session_peer(
            old_id,
            source=source.platform.value,
            user_id=source.user_id,
            session_key=key,
            chat_id=source.chat_id,
            chat_type=source.chat_type,
            thread_id=source.thread_id,
        )
        db.append_message(old_id, "user", "hello from the oversized session")
        db.end_session(old_id, "agent_close")

        # Sanity: the finder considers this row recoverable before the prune.
        assert db.find_latest_gateway_session_for_peer(
            source=source.platform.value,
            user_id=source.user_id,
            session_key=key,
            chat_id=source.chat_id,
            chat_type=source.chat_type,
            thread_id=source.thread_id,
        ) is not None

        # ---- Restart: stale sessions.json entry still points at the ended id --
        store._entries[key] = _make_entry_with_origin(key, old_id)
        store._prune_stale_sessions_locked()
        assert key not in store._entries

        # The recoverable row is now superseded -> finder returns nothing.
        assert db.find_latest_gateway_session_for_peer(
            source=source.platform.value,
            user_id=source.user_id,
            session_key=key,
            chat_id=source.chat_id,
            chat_type=source.chat_type,
            thread_id=source.thread_id,
        ) is None

        # ---- Next inbound message: must mint a FRESH session, not resurrect ---
        new_entry = store.get_or_create_session(source)
        assert new_entry.session_id != old_id


class TestRecoveredEntryPreservesOriginalTimestamp:
    def test_recovered_entry_updated_at_is_original_not_now(self, tmp_path):
        """#62012 follow-up (lucianosillem): a genuinely-recovered live session
        must keep its original `updated_at`, not be stamped `now`. Otherwise
        `_should_reset()` sees a "fresh today" session and silently skips the
        daily/idle reset policy, defeating the reset entirely.
        """
        from hermes_state import SessionDB
        from datetime import datetime as _dt

        db = SessionDB(db_path=tmp_path / "state.db")
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="5140768830", chat_type="dm",
            user_id="5140768830", user_name="João",
        )
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="daily"))
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        key = store._generate_session_key(source)
        past = _dt.now().timestamp() - 86400  # last active yesterday
        row = {
            "id": "sid_old",
            "session_key": key,
            "started_at": past,
            "ended_at": past,       # logically ended yesterday, but a live
            "end_reason": None,     # (recoverable) row — the recovery scenario
            "source": source.platform.value,
            "user_id": source.user_id,
            "chat_id": source.chat_id,
            "chat_type": source.chat_type,
        }
        recovered = store._create_entry_from_recovered_row(
            row=row, session_key=key, source=source, now=_dt.now(),
        )
        # updated_at reflects the ORIGINAL timestamp (yesterday), NOT now —
        # so _should_reset() correctly detects the daily-reset boundary.
        assert recovered.updated_at is not None
        assert abs(recovered.updated_at.timestamp() - past) < 2
        assert recovered.updated_at.timestamp() < _dt.now().timestamp() - 3600


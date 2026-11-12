"""A deliberately archived gateway session must not keep receiving turns (#154410).

A chat-backed platform adapter keeps one stable ``chat_id`` and therefore one
Hermes session. Archiving that session from the Desktop sidebar hides it from
the conversation list, but the gateway session router kept routing new inbound
turns to it: archived is sidebar state, not an "ended" signal, so the turn ran
and the reply was delivered into a conversation the user can no longer see.
The next turn after a deliberate archive must start a fresh session instead.

The archive's provenance decides. A DELIBERATE archive (user, CLI, API:
``archived=1, auto_archived=0``) retires the row for routing; the idle
sweep's archive (``archived=1, auto_archived=1``) is undone by re-activation
and the row stays routable (#117713). A row the old reapers ended with a
recoverable reason (``ws_orphan_reap``) and archived (#92687) is still
recoverable: the accidental end stamp, not a user's intent, is what put it
there. Same predicate as the Bot Chat title release (#139779).
"""

from __future__ import annotations

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


def _make_store(tmp_path):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    # Isolate the SQLite transcript store so we exercise per-session_id
    # transcripts without touching the developer's real state.db.
    store._db = SessionDB(db_path=tmp_path / "state.db")
    return store


def _make_source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")


def _flags(store, session_id):
    row = store._db.get_session(session_id)
    return (row["archived"], row["auto_archived"], row["end_reason"])


def _prune_routing_entry(store, session_key):
    """Drop the routing entry, as a pruned sessions.json / cold start would:
    the durable state.db row is the only thing left for the key."""
    with store._lock:
        store._entries.pop(session_key, None)
    # _save_entries / peek_session_id take _lock themselves (a plain
    # threading.Lock, not re-entrant), so they must run outside it.
    store._save_entries()
    assert store.peek_session_id(session_key) is None


class TestArchivedSessionRouting:
    """#154410: routing must not reuse a deliberately archived session."""

    def test_deliberately_archived_session_starts_fresh(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()

        entry = store.get_or_create_session(source)
        old_id = entry.session_id
        store._db.append_message(old_id, "user", "hello")
        assert _flags(store, old_id) == (0, 0, None)  # precondition: live row

        # The Desktop sidebar archive: the DELIBERATE archive.
        assert store._db.set_session_archived(old_id, True)
        assert _flags(store, old_id) == (1, 0, None)

        resolved = store.get_or_create_session(source)

        assert resolved.session_id != old_id, (
            "A deliberately archived session must not keep receiving turns; "
            "the next turn must start a fresh session."
        )
        # The old conversation is retired, not destroyed.
        old = store._db.get_session(old_id)
        assert old is not None and old["archived"] == 1
        assert len(store.load_transcript(old_id)) == 1
        # The fresh session is visible.
        assert store._db.get_session(resolved.session_id)["archived"] == 0

    def test_auto_archived_session_is_still_reused(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()

        entry = store.get_or_create_session(source)
        session_id = entry.session_id
        # The idle sweep's stamp: archived by the sweep, not by the user.
        store._db._write_sql(
            "UPDATE sessions SET archived = 1, auto_archived = 1 WHERE id = ?",
            (session_id,),
        )
        assert _flags(store, session_id) == (1, 1, None)

        resolved = store.get_or_create_session(source)

        # #117713: a sweep archive is undone by re-activation, so the live
        # chat keeps its session instead of forking a new one.
        assert resolved.session_id == session_id

    def test_cold_recovery_does_not_resurrect_deliberately_archived_row(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()

        entry = store.get_or_create_session(source)
        old_id = entry.session_id
        session_key = entry.session_key
        store._db.append_message(old_id, "user", "hello")
        assert store._db.set_session_archived(old_id, True)

        _prune_routing_entry(store, session_key)

        resolved = store.get_or_create_session(source)

        assert resolved.session_id != old_id, (
            "Cold recovery must not resurrect a deliberately archived row; "
            "the peer finder must skip it just as the live route does."
        )
        old = store._db.get_session(old_id)
        assert old is not None and old["archived"] == 1
        assert len(store.load_transcript(old_id)) == 1

    def test_recoverable_reaper_row_is_still_recovered(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()

        entry = store.get_or_create_session(source)
        session_id = entry.session_id
        session_key = entry.session_key
        store._db.append_message(session_id, "user", "hello")
        # Pre-#139779 reaper state: ended by the ws-orphan reaper, archived
        # by it too -- an accident, not a user's deliberate archive (#92687).
        store._db.end_session(session_id, "ws_orphan_reap")
        store._db._write_sql(
            "UPDATE sessions SET archived = 1, auto_archived = 0 WHERE id = ?",
            (session_id,),
        )
        assert _flags(store, session_id) == (1, 0, "ws_orphan_reap")

        _prune_routing_entry(store, session_key)

        resolved = store.get_or_create_session(source)

        assert resolved.session_id == session_id, (
            "A row ended only by a recoverable reaper stamp must stay "
            "recoverable even though it is archived; the deliberate-archive "
            "exclusion must not swallow the legacy reaper path."
        )
        assert len(store.load_transcript(session_id)) == 1

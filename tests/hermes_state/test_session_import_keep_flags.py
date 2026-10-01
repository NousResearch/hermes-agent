"""A restored or adopted session keeps the user's durable flags.

``import_sessions`` (the dashboard import of a ``hermes sessions export`` backup, and stranded-session
adoption) restored ``archived`` but not ``pinned`` or ``hidden``. Pinned is the "keep" flag the startup
prune and the stale-archive sweep both exempt, so an old pinned session came back unpinned and the next
startup deleted it; an adopted Bot Mode chat came back visible and no longer canonical.
"""

from __future__ import annotations

import json
import time

from hermes_state import SessionDB

SESSION_ID = "20260301_120000_abc123"


def test_restored_pinned_session_survives_the_startup_prune(tmp_path):
    source = SessionDB(db_path=tmp_path / "source.db")
    target = SessionDB(db_path=tmp_path / "target.db")
    try:
        source.create_session(SESSION_ID, source="cli")
        source.append_message(SESSION_ID, "user", "keep this one")
        source.end_session(SESSION_ID, "user_exit")
        old = time.time() - 200 * 86400
        source._conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?", (old, old, SESSION_ID))
        source._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (old, SESSION_ID))
        source._conn.commit()
        source.set_session_pinned(SESSION_ID, True)
        payload = json.loads(json.dumps(source.export_all(include_inactive=True)))

        assert target.import_sessions(payload)["imported"] == 1
        target.maybe_auto_prune_and_vacuum(retention_days=90, vacuum=False)

        restored = target.get_session(SESSION_ID)
        assert restored is not None, "the startup prune deleted a session the user pinned to keep"
        assert restored["pinned"]
    finally:
        source.close()
        target.close()


def test_adopted_bot_chat_stays_the_hidden_canonical_chat(tmp_path):
    """Stranded-session adoption imports the donor's Bot Mode chat into the profile store; Bot Mode
    chats are hidden, and hidden is what keeps the canonical one out of listings and the stale sweep."""
    donor = SessionDB(db_path=tmp_path / "default.db")
    profile = SessionDB(db_path=tmp_path / "profile.db")
    try:
        donor.create_session(SESSION_ID, source="tui")
        donor.set_session_title(SESSION_ID, SessionDB.CANONICAL_BOT_CHAT_TITLE)
        donor.set_session_hidden(SESSION_ID, True)
        donor.append_message(SESSION_ID, "user", "hi bot")

        assert profile.adopt_session_lineage_from(donor, SESSION_ID)["adopted"]

        assert profile.get_session(SESSION_ID)["hidden"]
        assert SESSION_ID not in {s["id"] for s in profile.list_sessions_rich(limit=50)}
        assert profile.archive_stale_sessions(idle_days=0) == 0
    finally:
        donor.close()
        profile.close()

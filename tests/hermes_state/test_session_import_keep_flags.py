"""A restored or adopted session keeps the user's durable flags.

``import_sessions`` (the dashboard import of a ``hermes sessions export`` backup, and stranded-session
adoption) restored ``archived`` but not ``pinned`` or ``hidden``. Pinned is the "keep" flag the startup
prune and the stale-archive sweep both exempt, so an old pinned session came back unpinned and the next
startup deleted it; an adopted Bot Mode chat came back visible and no longer canonical; and without
auto_archived a chat the idle sweep archived came back as a manual archive that resume never undoes.
"""

from __future__ import annotations

import json
import time

import pytest

from hermes_state import SessionDB

SESSION_ID = "20260301_120000_abc123"


def test_restored_sessions_keep_the_pin_and_the_sweeps_archive_provenance(tmp_path):
    """A pinned session survives the startup prune; a chat the idle sweep archived still comes
    back on resume, which only a sweep archive (not the user's own) does."""
    source = SessionDB(db_path=tmp_path / "source.db")
    target = SessionDB(db_path=tmp_path / "target.db")
    swept = "20260301_130000_def456"
    try:
        for sid, age_days in ((SESSION_ID, 200), (swept, 30)):
            source.create_session(sid, source="cli")
            source.append_message(sid, "user", "keep this one")
            source.end_session(sid, "user_exit")
            old = time.time() - age_days * 86400
            source._conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?", (old, old, sid))
            source._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (old, sid))
        source._conn.commit()
        source.set_session_pinned(SESSION_ID, True)
        assert source.archive_stale_sessions(7) == 1
        payload = json.loads(json.dumps(source.export_all(include_inactive=True)))
        # An export that predates the durable flags still imports as an ordinary, visible session.
        payload.append({"id": "legacy", "source": "cli", "messages": [{"role": "user", "content": "old"}]})

        assert target.import_sessions(payload)["imported"] == 3
        target.maybe_auto_prune_and_vacuum(retention_days=90, vacuum=False)

        restored = target.get_session(SESSION_ID)
        assert restored is not None, "the startup prune deleted a session the user pinned to keep"
        assert restored["pinned"]
        assert target.get_session(swept)["archived"]
        target.reopen_session(swept)
        assert not target.get_session(swept)["archived"], "a restored sweep archive became a manual one"
        legacy = target.get_session("legacy")
        assert all(legacy[flag] == 0 for flag in ("archived", "auto_archived", "pinned", "hidden"))
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
        with pytest.raises(ValueError, match="canonical Bot Chat"):
            profile.set_session_title(SESSION_ID, "renamed")
        assert donor.get_messages(SESSION_ID)[0]["content"] == "hi bot"

        # Retrying adoption must not restore the donor's retirement archive over the local copy.
        again = profile.adopt_session_lineage_from(donor, SESSION_ID)
        assert again["adopted"] and again["imported"] == 0 and SESSION_ID in again["skipped_ids"], again
        assert not profile.get_session(SESSION_ID)["archived"]
    finally:
        donor.close()
        profile.close()

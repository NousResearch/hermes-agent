"""A restored or adopted session keeps the user's durable flags.

``import_sessions`` (the dashboard import of a ``hermes sessions export`` backup, and stranded-session
adoption) restored ``archived`` but not ``pinned`` or ``hidden``. Pinned is the "keep" flag the startup
prune and the stale-archive sweep both exempt, so an old pinned session came back unpinned and the next
startup deleted it; an adopted Bot Mode chat came back visible and no longer canonical; and without
auto_archived a chat the idle sweep archived came back as a manual archive that resume never undoes.

The import is the SQLite store's; the startup prune, the idle sweep and the adoption are the helpers that
run them over the store's verbs.
"""

from __future__ import annotations

import json
import time

import pytest

import lineage_adoption
import session_retention
from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session, end_session, get_session, list_sessions_rich, reopen_session, set_session_hidden,
    set_session_pinned,
)
from persistence.sqlite.conversation._store.titles import set_session_title

SESSION_ID = "20260301_120000_abc123"


def test_restored_sessions_keep_the_pin_and_the_sweeps_archive_provenance(backend, scope_a, scope_b, tmp_path):
    """A pinned session survives the startup prune; a chat the idle sweep archived still comes
    back on resume, which only a sweep archive (not the user's own) does."""
    source_store, target_store = backend.open(scope_a), backend.open(scope_b)
    source, target = source_store.holder(), target_store.holder()
    swept = "20260301_130000_def456"
    for sid, age_days in ((SESSION_ID, 200), (swept, 30)):
        create_session(source, sid, source="cli")
        append_message(source, sid, "user", "keep this one")
        end_session(source, sid, "user_exit")
        old = time.time() - age_days * 86400
        source._write_sql("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?", (old, old, sid))
        source._write_sql("UPDATE messages SET timestamp = ? WHERE session_id = ?", (old, sid))
    set_session_pinned(source, SESSION_ID, True)
    assert session_retention.run_auto_archive(source_store, idle_days=7)["archived"] == 1
    payload = json.loads(json.dumps([{**export.session.as_dict(), "messages": [m.as_wire_row() for m in export.messages]}
                                     for export in source_store.maintenance.export_all(include_inactive=True)]))
    # An export that predates the durable flags still imports as an ordinary, visible session.
    payload.append({"id": "legacy", "source": "cli", "messages": [{"role": "user", "content": "old"}]})
    # A hand-edited export's string "0" flags are coerced like the int columns, not by truthiness,
    # and a live row is never marked as the sweep's archive.
    flags = ("archived", "auto_archived", "pinned", "hidden")
    payload.append({"id": "edited", "source": "cli", **{flag: "0" for flag in flags}, "auto_archived": 1})

    assert target_store.maintenance.import_sessions(payload).imported == 4
    session_retention.run_auto_prune_and_vacuum(target_store, None, store_home=scope_b.profile_dir,
                                                media_root=str(tmp_path / "media"), retention_days=90, vacuum=False)

    restored = get_session(target, SESSION_ID)
    assert restored is not None, "the startup prune deleted a session the user pinned to keep"
    assert restored["pinned"]
    assert get_session(target, swept)["archived"]
    reopen_session(target, swept)
    assert not get_session(target, swept)["archived"], "a restored sweep archive became a manual one"
    for sid in ("legacy", "edited"):
        assert all(get_session(target, sid)[flag] == 0 for flag in flags), sid


def test_adopted_bot_chat_stays_the_hidden_canonical_chat(backend, scope_a, scope_b):
    """Stranded-session adoption imports the donor's Bot Mode chat into the profile store; Bot Mode
    chats are hidden, and hidden is what keeps the canonical one out of listings and the stale sweep."""
    donor_store, profile_store = backend.open(scope_a), backend.open(scope_b)
    donor, profile = donor_store.holder(), profile_store.holder()
    create_session(donor, SESSION_ID, source="tui")
    set_session_title(donor, SESSION_ID, CANONICAL_BOT_CHAT_TITLE)
    set_session_hidden(donor, SESSION_ID, True)
    append_message(donor, SESSION_ID, "user", "hi bot")

    assert lineage_adoption.adopt_session_lineage(profile_store, donor_store, SESSION_ID)["adopted"]

    assert get_session(profile, SESSION_ID)["hidden"]
    assert SESSION_ID not in {s["id"] for s in list_sessions_rich(profile, limit=50)}
    assert session_retention.run_auto_archive(profile_store, idle_days=0)["archived"] == 0
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        set_session_title(profile, SESSION_ID, "renamed")

"""A chat the idle sweep archived must reappear once it is live again (#117713).

The sidebar admits a compression lineage by its ROOT's ``archived`` flag. When
``sessions.auto_archive`` hid an idle lineage and the chat later resumed and
compressed, the fresh tip was inserted with ``archived=0`` under the archived
root, so the live chat stayed invisible. A DELIBERATE archive (user/API/CLI)
must keep hiding the chat through the same re-activation.
"""
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    return SessionDB(tmp_path / "state.db")


def _sidebar_ids(db):
    """Same flags as the desktop sidebar slice (tips projected from roots)."""
    return {
        row["id"]
        for row in db.list_sessions_rich(
            limit=50, offset=0, min_message_count=1, include_archived=False,
            archived_only=False, order_by_last_active=True, compact_rows=True, include_pinned=True,
        )
    }


def _compress(db, parent, child):
    holder = f"holder-{child}"
    assert db.try_acquire_compression_lock(parent, holder, ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="telegram", system_prompt="p",
        messages=[{"role": "user", "content": f"summary for {child}"}], compression_lock_holder=holder)


def _flags(db, *ids):
    return {sid: (db.get_session(sid)["archived"], db.get_session(sid)["auto_archived"]) for sid in ids}


def _lineage(db):
    """root -> mid (compression) -> tip, all with messages."""
    db.create_session("root", "telegram")
    db.append_message("root", "user", "hello")
    _compress(db, "root", "mid")
    _compress(db, "mid", "tip")
    db.append_message("tip", "user", "latest")


def _sweep(db):
    time.sleep(0.02)
    return db.archive_stale_sessions(0)


def test_compression_after_auto_archive_unhides_the_live_chat(db):
    _lineage(db)
    assert _sweep(db) == 1
    assert _flags(db, "root", "mid", "tip") == {s: (1, 1) for s in ("root", "mid", "tip")}
    assert not _sidebar_ids(db)

    # A new inbound message resumes the chat and compression publishes a fresh tip.
    _compress(db, "tip", "tip2")

    assert _flags(db, "root", "mid", "tip", "tip2") == {
        s: (0, 0) for s in ("root", "mid", "tip", "tip2")}
    assert _sidebar_ids(db) == {"tip2"}


def test_compression_keeps_a_manually_archived_chat_hidden(db):
    _lineage(db)
    assert db.set_session_archived("tip", True)

    _compress(db, "tip", "tip2")

    # The new tip inherits the deliberate archive instead of creating a mixed lineage.
    assert _flags(db, "root", "mid", "tip", "tip2") == {
        s: (1, 0) for s in ("root", "mid", "tip", "tip2")}
    assert not _sidebar_ids(db)
    # ...so a later idle sweep has no archived=0 tip to relabel as sweep-owned.
    assert _sweep(db) == 0
    _compress(db, "tip2", "tip3")
    assert not _sidebar_ids(db)


def test_manual_archive_overrides_sweep_provenance(db):
    _lineage(db)
    assert _sweep(db) == 1
    assert db.set_session_archived("root", True)  # user archives it deliberately afterwards
    assert _flags(db, "root", "mid", "tip") == {s: (1, 0) for s in ("root", "mid", "tip")}

    _compress(db, "tip", "tip2")
    assert not _sidebar_ids(db)


def test_sweep_does_not_relabel_a_manually_archived_ancestor(db):
    """Legacy mixed lineage (pre-fix): manually archived root, unarchived continuation tip."""
    _lineage(db)
    db._write_sql("UPDATE sessions SET archived = 1 WHERE id IN ('root', 'mid')", ())
    assert _sweep(db) == 1  # the tip is still archived=0, so it is a sweep candidate
    assert _flags(db, "root", "mid", "tip") == {"root": (1, 0), "mid": (1, 0), "tip": (1, 1)}

    _compress(db, "tip", "tip2")
    assert not _sidebar_ids(db)
    assert db.get_session("root")["archived"] == 1


def test_inbound_message_unhides_auto_archived_but_not_manual(db):
    """A resumed gateway DM (no compression, ended_at NULL) reappears on its next inbound message (#133307)."""
    db.create_session("dm", "signal")
    db.append_message("dm", "user", "hello")
    assert _sweep(db) == 1
    assert _flags(db, "dm") == {"dm": (1, 1)}
    assert not _sidebar_ids(db)

    db.append_message("dm", "user", "still here")

    assert _flags(db, "dm") == {"dm": (0, 0)}
    assert _sidebar_ids(db) == {"dm"}


def test_inbound_message_keeps_manual_archive_hidden(db):
    db.create_session("dm", "signal")
    db.append_message("dm", "user", "hello")
    assert db.set_session_archived("dm", True)

    db.append_message("dm", "user", "still here")

    assert _flags(db, "dm") == {"dm": (1, 0)}
    assert not _sidebar_ids(db)


def test_resume_reopen_unhides_auto_archived_but_not_manual(db):
    db.create_session("auto", "cli")
    db.append_message("auto", "user", "a")
    db.end_session("auto", "cli_close")
    db.create_session("manual", "cli")
    db.append_message("manual", "user", "m")
    db.end_session("manual", "cli_close")
    assert db.set_session_archived("manual", True)
    assert _sweep(db) == 1
    assert _flags(db, "auto", "manual") == {"auto": (1, 1), "manual": (1, 0)}

    db.reopen_session("auto")
    db.reopen_session("manual")

    assert _flags(db, "auto", "manual") == {"auto": (0, 0), "manual": (1, 0)}
    assert _sidebar_ids(db) == {"auto"}


def test_unarchive_clears_sweep_provenance(db):
    _lineage(db)
    assert _sweep(db) == 1
    assert db.set_session_archived("tip", False)
    assert _flags(db, "root", "mid", "tip") == {s: (0, 0) for s in ("root", "mid", "tip")}

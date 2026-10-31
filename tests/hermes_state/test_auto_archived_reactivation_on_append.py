"""A sweep-hidden chat is live again the moment a new message lands (#133307).

``_unarchive_auto_archived_lineage`` used to run only from ``reopen_session()``
(which needs ``ended_at`` set) and compression publish. A long-lived gateway DM
session (``ended_at IS NULL`` — one persistent session per contact) that idled
past the threshold and then received new inbound messages kept ``archived=1``:
invisible to every sessions-list sort even though ``last_activity_at`` was
fresh. Appending a message now re-activates the lineage; a DELIBERATE archive
still hides the chat through the same mixed-provenance refusal.
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


def _flags(db, *ids):
    return {sid: (db.get_session(sid)["archived"], db.get_session(sid)["auto_archived"]) for sid in ids}


def _compress(db, parent, child):
    holder = f"holder-{child}"
    assert db.try_acquire_compression_lock(parent, holder, ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="signal", system_prompt="p",
        messages=[{"role": "user", "content": f"summary for {child}"}], compression_lock_holder=holder)


def _lineage(db):
    """root -> mid (compression) -> tip, all with messages."""
    db.create_session("root", "signal")
    db.append_message("root", "user", "hello")
    _compress(db, "root", "mid")
    _compress(db, "mid", "tip")
    db.append_message("tip", "user", "latest")


def _sweep(db):
    time.sleep(0.02)
    return db.archive_stale_sessions(0)


def test_append_unhides_a_never_ended_dm_session(db):
    """The #133307 shape: a persistent DM row (``ended_at IS NULL``) that never reopens."""
    db.create_session("dm", "signal")
    db.append_message("dm", "user", "hi")
    assert _sweep(db) == 1
    assert _flags(db, "dm") == {s: (1, 1) for s in ("dm",)}
    assert not _sidebar_ids(db)

    # New inbound activity: no reopen_session happens (the row never ended), just an append.
    db.append_message("dm", "user", "back again")

    assert _flags(db, "dm") == {s: (0, 0) for s in ("dm",)}
    assert _sidebar_ids(db) == {"dm"}


def test_append_unhides_the_whole_compression_lineage(db):
    _lineage(db)
    assert _sweep(db) == 1
    assert not _sidebar_ids(db)

    db.append_message("tip", "user", "new inbound")

    # The sidebar admits a lineage by its ROOT's flag: everything must be un-hidden.
    assert _flags(db, "root", "mid", "tip") == {s: (0, 0) for s in ("root", "mid", "tip")}
    assert _sidebar_ids(db) == {"tip"}


def test_batch_append_reactivates_the_sweep_hidden_chat(db):
    """The agent's turn flush lands rows via append_messages_batch (#133307's other leg)."""
    _lineage(db)
    assert _sweep(db) == 1

    inserted = db.append_messages_batch(
        "tip", [{"role": "assistant", "content": "reply"}])

    assert inserted == 1
    assert _flags(db, "root", "mid", "tip") == {s: (0, 0) for s in ("root", "mid", "tip")}
    assert _sidebar_ids(db) == {"tip"}


def test_append_keeps_a_manually_archived_chat_hidden(db):
    _lineage(db)
    assert db.set_session_archived("tip", True)
    assert _sweep(db) == 0  # already archived deliberately: nothing to relabel

    db.append_message("tip", "user", "still hidden")

    assert _flags(db, "root", "mid", "tip") == {s: (1, 0) for s in ("root", "mid", "tip")}
    assert not _sidebar_ids(db)


def test_branch_seed_into_a_new_row_leaves_the_archived_parent_alone(db):
    """Branch copies append into a FRESH row (archived=0), so the copy must not
    resurrect the sweep-hidden parent the branch was taken from."""
    db.create_session("root", "signal")
    db.append_message("root", "user", "hello")
    assert _sweep(db) == 1
    assert not _sidebar_ids(db)

    db.create_session("branch", "signal", model_config={"_branched_from": "root"},
                      parent_session_id="root")
    db.append_messages_batch("branch", [{"role": "user", "content": "hello"}])

    assert _flags(db, "root") == {s: (1, 1) for s in ("root",)}  # parent stays hidden
    assert _flags(db, "branch") == {s: (0, 0) for s in ("branch",)}
    assert _sidebar_ids(db) == {"branch"}

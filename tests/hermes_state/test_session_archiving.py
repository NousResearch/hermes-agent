import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _compression_pair(db: SessionDB):
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.commit()


def _compression_chain(db: SessionDB) -> None:
    """root --compression--> mid --compression--> tip(open): the shape a long-lived messaging chat
    reaches after context compression fires repeatedly."""
    base = time.time() - 300
    db.create_session("root", source="cli", chat_id="dm-1")
    db.create_session("mid", source="cli", parent_session_id="root", chat_id="dm-1")
    db.create_session("tip", source="cli", parent_session_id="mid", chat_id="dm-1")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', "
        "message_count = 1 WHERE id IN ('root', 'mid')",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'", (base + 20,)
    )
    db._conn.commit()


def test_archiving_compression_tip_archives_projected_root(db):
    _compression_pair(db)

    assert db.set_session_archived("tip", True) is True

    assert db.get_session("root")["archived"] == 1
    assert db.get_session("tip")["archived"] == 1
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True, archived_only=True)] == ["tip"]


def test_archiving_a_finished_middle_segment_keeps_the_conversation_visible(db):
    """A hide is lineage-wide only when it starts at the chain's tail.

    Listings resolve the chain ROOT and project it onto its tip, so fanning a hide upward from a
    *finished* segment reaches the root and takes the whole conversation out of every
    ``archived = 0`` listing — including a chat whose tip is open and still being written to
    (#115489). Archiving a mid segment is a node-scoped act and must leave root and tip alone.
    """
    _compression_chain(db)

    assert db.set_session_archived("mid", True) is True

    assert db.get_session("mid")["archived"] == 1
    assert db.get_session("root")["archived"] == 0
    assert db.get_session("tip")["archived"] == 0
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["tip"]


def test_a_message_appended_to_an_archived_chain_restores_it(db):
    """A chat swept while it was idle comes back by itself.

    The idle auto-archive sweep hides the whole lineage (the root is what listings read), so a
    conversation that resumes afterwards stayed invisible — the Desktop sidebar filters ``archived``
    — while the gateway kept answering on the open tip. A message landing on the chain is the proof
    that the conversation is alive, so it clears the soft-hide in the same transaction (#115489).
    """
    _compression_pair(db)
    assert db.set_session_archived("tip", True) is True
    assert db.list_sessions_rich(order_by_last_active=True) == []

    db.append_message(session_id="tip", role="user", content="new inbound DM")

    assert db.get_session("root")["archived"] == 0
    assert db.get_session("tip")["archived"] == 0
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["tip"]


def test_reopening_an_accident_ended_tip_restores_the_archived_chain(db):
    """Resuming a session ended by an automatic cleanup undoes the archive that rode along.

    ``agent_close`` / ``startup_orphan_reap`` / ``ws_orphan_reap`` mean "a runtime went away", not
    "the conversation is over" — the row stays recoverable, so the lineage-wide archive stamped with
    it is collateral and a revive must restore visibility (#115489).
    """
    _compression_chain(db)
    db._conn.execute("UPDATE sessions SET end_reason = 'startup_orphan_reap' WHERE id = 'tip'")
    db._conn.commit()
    assert db.set_session_archived("tip", True) is True
    assert db.get_session("root")["archived"] == 1

    db.reopen_session("tip")

    assert db.get_session("tip")["end_reason"] is None
    assert db.get_session("tip")["archived"] == 0
    assert db.get_session("root")["archived"] == 0
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["tip"]


def test_reopening_leaves_a_deliberate_archive_alone(db):
    """Only accidents resurrect: a user archive (no end reason) survives a resume."""
    _compression_pair(db)
    assert db.set_session_archived("tip", True) is True

    db.reopen_session("tip")

    assert db.get_session("tip")["archived"] == 1
    assert db.get_session("root")["archived"] == 1


def test_idle_sweep_skips_a_tip_holding_a_live_turn_lease(db):
    """The automatic sweep never hides a running turn.

    The lease row is the proof a turn is live (the reported chat held one while the
    sidebar hid it), so the idle sweep skips it; once the lease is released the same
    sweep archives the idle chain (#115489).
    """
    _compression_chain(db)
    assert db.try_acquire_session_turn_lease("tip", "gateway-pid") is True

    assert db.archive_stale_sessions(0) == 0
    assert db.get_session("tip")["archived"] == 0
    assert db.get_session("root")["archived"] == 0

    db.release_session_turn_lease("tip", "gateway-pid")

    assert db.archive_stale_sessions(0) == 1
    assert db.get_session("tip")["archived"] == 1
    assert db.get_session("root")["archived"] == 1


def test_unarchiving_compression_tip_unarchives_projected_root(db):
    _compression_pair(db)
    db.set_session_archived("tip", True)

    assert db.set_session_archived("tip", False) is True

    assert db.get_session("root")["archived"] == 0
    assert db.get_session("tip")["archived"] == 0
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["tip"]


def test_archived_only_view_includes_hidden_archived_sessions(db):
    """The archived-only view is the recovery surface: a session that is both
    archived and hidden (Bot Mode marks its sessions hidden) must appear
    there, otherwise it is unreachable from every UI list (#90946)."""
    db.create_session("plain", source="cli")
    db.create_session("both", source="cli")
    assert db.set_session_hidden("both", True) is True
    assert db.set_session_archived("both", True) is True

    # Default list: hidden rows stay excluded (unchanged behaviour)...
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["plain"]
    # ...and the archived-only view must surface the archived+hidden row.
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True, archived_only=True)] == ["both"]

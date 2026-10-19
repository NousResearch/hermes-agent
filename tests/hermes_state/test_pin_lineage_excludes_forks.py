"""Lineage-wide flag writes (pin, archive, hidden, read) cover one conversation only.

A branch, delegate, or reset fork of a compression-ended segment is a separate
conversation. Pinning the main chat must not pin (and so resurface) an archived
fork, and unpinning a fork must not unpin the main chat.
"""
from __future__ import annotations

import pytest

from hermes_state import SessionDB

MARKERS = ["_branched_from", "_delegate_from", "_reset_from"]


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _chat_with_fork(db: SessionDB, marker: str) -> None:
    """``main`` compresses into ``main-2``; ``fork`` branches off ``main`` and is archived."""
    db.create_session("main", source="desktop")
    db.end_session("main", "compression")
    db.create_session("main-2", source="desktop", parent_session_id="main")
    db.create_session("fork", source="desktop", parent_session_id="main", model_config={marker: "main"})
    db._conn.execute("UPDATE sessions SET archived = 1 WHERE id = 'fork'")
    db._conn.commit()


def _flags(db: SessionDB, column: str) -> dict[str, int]:
    return {sid: db.get_session(sid)[column] for sid in ("main", "main-2", "fork")}


@pytest.mark.parametrize("marker", MARKERS)
@pytest.mark.parametrize("pin_target", ["main", "main-2"])
def test_pinning_the_chat_leaves_an_archived_fork_unpinned(db: SessionDB, marker: str, pin_target: str) -> None:
    _chat_with_fork(db, marker)

    assert db.set_session_pinned(pin_target, True)

    assert _flags(db, "pinned") == {"main": 1, "main-2": 1, "fork": 0}
    assert db.get_session("fork")["archived"] == 1


@pytest.mark.parametrize("marker", MARKERS)
def test_unpinning_a_fork_keeps_the_main_chat_pinned(db: SessionDB, marker: str) -> None:
    _chat_with_fork(db, marker)
    db._conn.execute("UPDATE sessions SET pinned = 1")
    db._conn.commit()

    assert db.set_session_pinned("fork", False)

    assert _flags(db, "pinned") == {"main": 1, "main-2": 1, "fork": 0}


@pytest.mark.parametrize("marker", MARKERS)
def test_archiving_the_chat_leaves_a_live_fork_visible(db: SessionDB, marker: str) -> None:
    _chat_with_fork(db, marker)
    db._conn.execute("UPDATE sessions SET archived = 0 WHERE id = 'fork'")
    db._conn.commit()

    assert db.set_session_archived("main", True)

    assert _flags(db, "archived") == {"main": 1, "main-2": 1, "fork": 0}


def test_continuation_with_an_inherited_foreign_marker_still_joins_the_pin(db: SessionDB) -> None:
    """Continuations copy model_config verbatim, so a marker naming ANOTHER row is not a fork."""
    db.create_session("main", source="desktop")
    db.end_session("main", "compression")
    db.create_session("main-2", source="desktop", parent_session_id="main",
                      model_config={"_branched_from": "some-earlier-parent"})

    assert db.set_session_pinned("main", True)

    assert db.get_session("main-2")["pinned"] == 1

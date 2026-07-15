"""Tests for SessionStore.rewind_session — the gateway /undo [N] primitive.

The gateway /undo backs up N user turns by soft-deleting the truncated rows
in state.db (active=0, kept for audit, hidden from re-prompts/search) via
SessionDB.rewind_to_message, rather than the old hard rewrite_transcript.
load_transcript returns only the active view. See issue #21910.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import SessionDB
from gateway.config import GatewayConfig
from gateway.session import SessionStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    s = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    s._db = db  # use the same DB instance the fixture seeds
    return s


def _seed(store, sid, source="telegram", turns=3):
    store._db.create_session(sid, source=source)
    for i in range(1, turns + 1):
        store._db.append_message(sid, "user", f"q{i}")
        store._db.append_message(sid, "assistant", f"a{i}")
    return sid


def test_rewind_default_one_turn(store):
    sid = _seed(store, "gw-1")
    res = store.rewind_session(sid)
    assert res["turns_undone"] == 1
    assert res["target_text"] == "q3"
    assert res["rewound_count"] == 2  # q3 + a3
    active = store.load_transcript(sid)
    assert [m["role"] for m in active] == ["user", "assistant", "user", "assistant"]


def test_rewind_n_turns(store):
    sid = _seed(store, "gw-2")
    res = store.rewind_session(sid, 2)
    assert res["turns_undone"] == 2
    assert res["target_text"] == "q2"
    assert res["rewound_count"] == 4  # q2,a2,q3,a3
    assert len(store.load_transcript(sid)) == 2  # q1,a1


def test_rewind_soft_deletes_rows_for_audit(store):
    sid = _seed(store, "gw-3")
    store.rewind_session(sid, 1)
    all_rows = store._db.get_messages(sid, include_inactive=True)
    assert len(all_rows) == 6  # nothing hard-deleted
    assert sum(1 for r in all_rows if r["active"] == 1) == 4
    assert store._db.get_session(sid)["rewind_count"] == 1


def test_rewind_reconciles_message_count(store):
    sid = _seed(store, "gw-mc")
    assert store._db.get_session(sid)["message_count"] == 6
    store.rewind_session(sid, 1)
    # message_count must follow the active (live) transcript, not the
    # pre-rewind total. q3 + a3 went inactive, so four rows remain.
    sess = store._db.get_session(sid)
    assert sess["message_count"] == 4
    assert sess["message_count"] == len(store.load_transcript(sid))
    # A message sent after the undo counts up from the reconciled value.
    store._db.append_message(sid, "user", "q3-again")
    assert store._db.get_session(sid)["message_count"] == 5


def test_rewind_and_restore_reconcile_session_counts(store):
    sid = "gw-tc"
    store._db.create_session(sid, source="telegram")
    store._db.append_message(sid, "user", "q1")
    store._db.append_message(sid, "assistant", "a1")
    store._db.append_message(sid, "user", "q2")
    store._db.append_message(
        sid, "assistant", "a2", tool_calls=[{"id": "t1"}, {"id": "t2"}]
    )
    q3_id = store._db.append_message(sid, "user", "q3")
    store._db.append_message(sid, "assistant", "a3", tool_calls=[{"id": "t3"}])

    store.rewind_session(sid, 1)
    rewound = store._db.get_session(sid)
    assert rewound["message_count"] == 4
    assert rewound["tool_call_count"] == 2

    assert store._db.restore_rewound(sid, q3_id) == 2
    restored = store._db.get_session(sid)
    assert restored["message_count"] == 6
    assert restored["tool_call_count"] == 3
    assert len(store.load_transcript(sid)) == 6


def test_rewind_clamps_to_oldest_turn(store):
    sid = _seed(store, "gw-4", turns=2)
    res = store.rewind_session(sid, 99)
    assert res["target_text"] == "q1"
    assert len(store.load_transcript(sid)) == 0


def test_rewind_empty_session_returns_none(store):
    store._db.create_session("gw-5", source="discord")
    assert store.rewind_session("gw-5") is None


def test_rewind_clamps_negative_count_to_one(store):
    sid = _seed(store, "gw-6")
    res = store.rewind_session(sid, -5)
    assert res["turns_undone"] == 1
    assert res["target_text"] == "q3"

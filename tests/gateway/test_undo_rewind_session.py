"""Tests for SessionStore.rewind_session — the gateway /undo [N] primitive.

The gateway /undo backs up N half-turns by soft-deleting rows in state.db
(active=0, kept for audit, hidden from re-prompts/search) via the shared
undo core. load_transcript returns only the active view. See issue #21910.
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
    assert res["rewound_ids"]
    assert res["prefill_text"] == "q3"
    assert len(res["rewound_ids"]) == 1  # a3
    active = store.load_transcript(sid)
    assert [m["role"] for m in active] == ["user", "assistant", "user", "assistant", "user"]


def test_rewind_n_turns(store):
    sid = _seed(store, "gw-2")
    res = store.rewind_session(sid, 2)
    assert len(res["rewound_ids"]) == 2  # q3,a3
    assert res["prefill_text"] is None
    assert len(store.load_transcript(sid)) == 4  # q1,a1,q2,a2


def test_rewind_soft_deletes_rows_for_audit(store):
    sid = _seed(store, "gw-3")
    store.rewind_session(sid, 1)
    all_rows = store._db.get_messages(sid, include_inactive=True)
    assert len(all_rows) == 6  # nothing hard-deleted
    assert sum(1 for r in all_rows if r["active"] == 1) == 5
    assert store._db.get_session(sid)["rewind_count"] == 1


def test_rewind_clamps_to_oldest_turn(store):
    sid = _seed(store, "gw-4", turns=2)
    res = store.rewind_session(sid, 99)
    assert res["rewound_ids"]
    assert res["prefill_text"] is None
    assert len(store.load_transcript(sid)) == 0


def test_rewind_empty_session_returns_none(store):
    store._db.create_session("gw-5", source="discord")
    assert store.rewind_session("gw-5") is None


def test_rewind_clamps_negative_count_to_one(store):
    sid = _seed(store, "gw-6")
    res = store.rewind_session(sid, -5)
    assert len(res["rewound_ids"]) == 1
    assert res["prefill_text"] == "q3"

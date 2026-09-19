"""Regression tests: preloaded skill names persist on the session row.

`hermes chat -s <skill>` / oneshot preload sessions were indistinguishable
from plain sessions in state.db (preload content lives only in the runtime
system prompt), blinding per-skill usage analytics. The fix carries the
preloaded names in ``sessions.origin_json`` under ``preload_skills`` — no
schema change (origin_json is an existing COALESCE'd TEXT column).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hermes_state import SessionDB  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "state.db"
    store = SessionDB(db_path=db_path)
    yield store, db_path


def _origin_json(db_path, session_id):
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT origin_json FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row[0]
    finally:
        conn.close()


def test_create_session_without_preload_keeps_origin_json_null(db):
    store, db_path = db
    store.create_session("sess-plain", "cli")
    assert _origin_json(db_path, "sess-plain") is None


def test_preload_skills_persisted_in_origin_json(db):
    store, db_path = db
    store.create_session(
        "sess-pre", "cli", preload_skills=["hermes-dojo", "lark-im"]
    )
    raw = _origin_json(db_path, "sess-pre")
    assert raw is not None
    data = json.loads(raw)
    assert data["preload_skills"] == ["hermes-dojo", "lark-im"]


def test_preload_merges_into_existing_origin_json(db):
    store, db_path = db
    existing = json.dumps({"gateway": "feishu", "chat_id": "oc_x"})
    store.create_session("sess-merge", "gateway", origin_json=existing,
                         preload_skills=["lark-im"])
    data = json.loads(_origin_json(db_path, "sess-merge"))
    # both survive
    assert data["gateway"] == "feishu"
    assert data["chat_id"] == "oc_x"
    assert data["preload_skills"] == ["lark-im"]


def test_preload_does_not_clobber_existing_preload_field(db):
    store, db_path = db
    existing = json.dumps({"preload_skills": ["old-skill"]})
    store.create_session("sess-keep", "cli", origin_json=existing,
                         preload_skills=["new-skill"])
    data = json.loads(_origin_json(db_path, "sess-keep"))
    # setdefault: first write wins, never clobbered by a later row insert
    assert data["preload_skills"] == ["old-skill"]


def test_ensure_session_forwards_preload(db):
    store, db_path = db
    store.ensure_session("sess-ensure", source="cli",
                         preload_skills=["a", "b"])
    data = json.loads(_origin_json(db_path, "sess-ensure"))
    assert data["preload_skills"] == ["a", "b"]

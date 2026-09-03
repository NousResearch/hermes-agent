"""Gateway SessionStore must not FTS-repair a replaced state.db (#89332)."""

import json
import os
import shutil

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from hermes_state import SessionDB


def _assert_diverted(tmp_path, sid, needle):
    pending = list((tmp_path / "pending_messages").glob("pending-*.json"))
    assert pending, "expected pending_messages/pending-*.json spool"
    spooled = False
    for path in pending:
        payload = json.loads(path.read_text(encoding="utf-8"))
        message = (payload.get("data") or {}).get("message") or {}
        if needle in str(message.get("content", "")):
            spooled = True
            break
    assert spooled, f"{needle!r} missing from pending spool"
    jsonl = tmp_path / "sessions" / f"{sid}.jsonl"
    assert jsonl.is_file()
    assert needle in jsonl.read_text(encoding="utf-8")


def test_replaced_state_db_diverts_pending_without_fts_rebuild(tmp_path, monkeypatch):
    import hermes_state

    live = tmp_path / "state.db"
    other = tmp_path / "other.db"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", live)

    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    sid = "gw-replaced"
    store._db.create_session(session_id=sid, source="cli")
    store.append_to_transcript(
        sid, {"role": "user", "content": "before", "timestamp": 1.0}
    )
    if store._db._db_file_identity is None:
        store.close_all_db_handles()
        pytest.skip("filesystem does not expose st_dev/st_ino")

    alt = SessionDB(db_path=other)
    alt.create_session("other", "cli")
    alt.close()
    os.replace(other, live)

    store.append_to_transcript(
        sid, {"role": "user", "content": "after-replace", "timestamp": 2.0}
    )

    assert store._db._db_replaced is True
    # No FTS surgery ran on either layer: state-level fail-open never
    # detached, and the gateway one-shot rebuild was not consumed.
    assert store._db._fts_enabled is True
    assert store._db._fts_stale is False
    assert store._fts_rebuild_attempted is False
    _assert_diverted(tmp_path, sid, "after-replace")
    store.close_all_db_handles()


def test_copyfile_replaced_state_db_diverts_pending_without_fts_rebuild(
    tmp_path, monkeypatch
):
    import hermes_state

    live = tmp_path / "state.db"
    other = tmp_path / "other.db"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", live)

    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    sid = "gw-cp-replaced"
    store._db.create_session(session_id=sid, source="cli")
    store.append_to_transcript(
        sid, {"role": "user", "content": "before-cp", "timestamp": 1.0}
    )
    if not store._db._db_file_application_id:
        store.close_all_db_handles()
        pytest.skip("generation stamp not recorded")

    alt = SessionDB(db_path=other)
    alt.create_session("other", "cli")
    alt.close()
    shutil.copyfile(other, live)

    store.append_to_transcript(
        sid, {"role": "user", "content": "after-cp", "timestamp": 2.0}
    )

    assert store._db._db_replaced is True
    # No FTS surgery ran on either layer: state-level fail-open never
    # detached, and the gateway one-shot rebuild was not consumed.
    assert store._db._fts_enabled is True
    assert store._db._fts_stale is False
    assert store._fts_rebuild_attempted is False
    _assert_diverted(tmp_path, sid, "after-cp")
    store.close_all_db_handles()

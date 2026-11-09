"""Gateway SessionStore must not FTS-repair a replaced state.db (#89332)."""

import json
import os
import shutil

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session
from tests.fixtures.storage_holder import raw_holder


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


@pytest.mark.platforms("posix")
def test_replaced_state_db_diverts_pending_without_fts_rebuild(tmp_path, monkeypatch):
    live = tmp_path / "state.db"
    other = tmp_path / "other"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    sid = "gw-replaced"
    seed_session(store.persistence, sid, "cli")
    store.append_to_transcript(
        sid, {"role": "user", "content": "before", "timestamp": 1.0}
    )
    if raw_holder(store.persistence)._db_file_identity is None:
        store.close_all_db_handles()
        pytest.skip("filesystem does not expose st_dev/st_ino")

    with open_persistence(profile_scope(home=other)) as alt:
        seed_session(alt, "other", "cli")
    os.replace(other / "state.db", live)

    store.append_to_transcript(
        sid, {"role": "user", "content": "after-replace", "timestamp": 2.0}
    )

    holder = raw_holder(store.persistence)  # the handle the transcript writes went through
    assert holder._db_replaced is True
    # No FTS surgery ran on either layer: state-level fail-open never
    # detached, and the gateway one-shot rebuild was not consumed.
    assert holder._fts_enabled is True
    assert holder._fts_stale is False
    assert store._fts_rebuild_last_attempt_at is None
    _assert_diverted(tmp_path, sid, "after-replace")
    store.close_all_db_handles()


def test_copyfile_replaced_state_db_diverts_pending_without_fts_rebuild(
    tmp_path, monkeypatch
):
    live = tmp_path / "state.db"
    other = tmp_path / "other"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    sid = "gw-cp-replaced"
    seed_session(store.persistence, sid, "cli")
    store.append_to_transcript(
        sid, {"role": "user", "content": "before-cp", "timestamp": 1.0}
    )
    if not raw_holder(store.persistence)._db_file_application_id:
        store.close_all_db_handles()
        pytest.skip("generation stamp not recorded")

    with open_persistence(profile_scope(home=other)) as alt:
        seed_session(alt, "other", "cli")
    shutil.copyfile(other / "state.db", live)

    store.append_to_transcript(
        sid, {"role": "user", "content": "after-cp", "timestamp": 2.0}
    )

    holder = raw_holder(store.persistence)  # the handle the transcript writes went through
    assert holder._db_replaced is True
    # No FTS surgery ran on either layer: state-level fail-open never
    # detached, and the gateway one-shot rebuild was not consumed.
    assert holder._fts_enabled is True
    assert holder._fts_stale is False
    assert store._fts_rebuild_last_attempt_at is None
    _assert_diverted(tmp_path, sid, "after-cp")
    store.close_all_db_handles()

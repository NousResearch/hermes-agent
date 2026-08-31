"""File-identity guard on SessionDB writes (#89332).

When state.db is replaced out-of-band under a live handle, in-place FTS
rebuild / fail-open cannot help: they operate on a generation mismatch.
The store must fail loudly instead of limping.
"""

import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from hermes_state import (
    SessionDB,
    StateDbReplacedError,
    classify_persistence_error,
    divert_session_transcript_jsonl,
)


def _make_db(path: Path, session_id: str, content: str) -> SessionDB:
    db = SessionDB(db_path=path)
    db.create_session(session_id, "cli")
    db.append_message(session_id, role="user", content=content)
    return db


def _require_identity(db: SessionDB) -> None:
    if db._db_file_identity is None:
        pytest.skip("filesystem does not expose st_dev/st_ino for identity checks")


def test_replace_with_new_inode_fails_loudly_without_fts_repair(tmp_path):
    live = tmp_path / "state.db"
    other = tmp_path / "other.db"
    db = _make_db(live, "live-sess", "original")
    _require_identity(db)
    alt = _make_db(other, "other-sess", "replacement")
    alt.close()

    recorded = db._db_file_identity
    assert recorded is not None
    os.replace(other, live)
    assert _stat_changed(live, recorded)

    with pytest.raises(StateDbReplacedError, match="replaced underneath"):
        db.append_message("live-sess", role="user", content="after-replace")

    assert db._db_replaced is True
    # No FTS surgery ran: fail-open never detached the indexes.
    assert db._fts_enabled is True
    assert db._fts_stale is False
    db.close()


def test_second_write_after_halt_does_not_attempt_repair(tmp_path):
    live = tmp_path / "state.db"
    other = tmp_path / "other.db"
    db = _make_db(live, "s", "a")
    _require_identity(db)
    alt = _make_db(other, "t", "b")
    alt.close()
    os.replace(other, live)
    with pytest.raises(StateDbReplacedError):
        db.append_message("s", role="user", content="first")
    with pytest.raises(StateDbReplacedError):
        db.append_message("s", role="user", content="second")
    assert db._fts_enabled is True
    assert db._fts_stale is False
    db.close()


def test_same_file_fts_corruption_still_fails_open(tmp_path):
    """Identity guard must not disable genuine in-file FTS recovery.

    Since 18ac3c4fb6 the live write path never rebuilds FTS in place; the
    recovery contract is the fail-open detach (stale marker + triggers
    dropped) followed by a successful canonical retry.
    """
    db = _make_db(tmp_path / "state.db", "s1", "hello world")
    _require_identity(db)
    identity = db._db_file_identity
    raw = sqlite3.connect(str(tmp_path / "state.db"))
    raw.execute(
        "UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()
    db.append_message("s1", role="user", content="healed append")
    assert db._db_file_identity == identity
    assert db._db_replaced is False
    # Fail-open detach ran: canonical write landed, FTS marked stale.
    assert db._fts_stale is True
    assert db._fts_enabled is False
    db.close()


def test_classify_replaced_is_not_disk_or_fts_repair():
    err = StateDbReplacedError(
        "FATAL: state.db was replaced underneath the gateway; refusing further writes"
    )
    assert classify_persistence_error(err) == "replaced"
    assert classify_persistence_error(str(err)) == "replaced"


def test_new_sessiondb_on_replaced_path_records_new_identity(tmp_path):
    live = tmp_path / "state.db"
    other = tmp_path / "other.db"
    db = _make_db(live, "s", "a")
    old_id = db._db_file_identity
    _require_identity(db)
    db.close()
    alt = _make_db(other, "t", "b")
    alt.close()
    os.replace(other, live)
    reopened = SessionDB(db_path=live)
    try:
        assert reopened._db_file_identity != old_id
        reopened.append_message("t", role="user", content="adopted after reopen")
        assert reopened._db_replaced is False
    finally:
        reopened.close()


def test_fts_scoped_error_on_replaced_file_skips_fts_fail_open(tmp_path):
    """Even FTS-provenance corruption must not authorize surgery on a
    replaced file. (A generic malformed error never reaches fail-open at
    all since the provenance classifier of #99652 rejects it earlier.)"""
    live = tmp_path / "state.db"
    other = tmp_path / "other.db"
    db = _make_db(live, "s", "a")
    _require_identity(db)
    alt = _make_db(other, "t", "b")
    alt.close()
    os.replace(other, live)

    with pytest.raises(StateDbReplacedError):
        db._enter_fts_fail_open(
            sqlite3.DatabaseError(
                'fts5: corrupt structure record for table "messages_fts"'
            )
        )
    assert db._fts_enabled is True
    assert db._fts_stale is False
    db.close()


def test_copyfile_same_inode_fails_loudly_without_fts_repair(tmp_path):
    """``cp`` keeps st_ino; generation stamp must still halt (#89332)."""
    live = tmp_path / "state.db"
    other = tmp_path / "other.db"
    db = _make_db(live, "live-sess", "original")
    alt = _make_db(other, "other-sess", "replacement")
    live_app = db._db_file_application_id
    other_app = alt._db_file_application_id
    if not live_app or not other_app:
        alt.close()
        db.close()
        pytest.skip("generation stamp not recorded on this filesystem")
    assert live_app != other_app
    recorded = db._db_file_identity
    alt.close()
    shutil.copyfile(other, live)
    if recorded is not None:
        st = os.stat(live)
        assert (st.st_dev, st.st_ino) == recorded
    with pytest.raises(StateDbReplacedError, match="replaced underneath"):
        db.append_message("live-sess", role="user", content="after-cp")
    assert db._db_replaced is True
    assert db._fts_enabled is True
    assert db._fts_stale is False
    db.close()


def test_divert_session_transcript_jsonl_appends(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = divert_session_transcript_jsonl(
        "sess-jsonl",
        [{"role": "user", "content": "hello-jsonl"}],
    )
    assert path == tmp_path / "sessions" / "sess-jsonl.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[-1])["content"] == "hello-jsonl"
    assert divert_session_transcript_jsonl("sess-jsonl", []) is None


def _stat_changed(path: Path, recorded) -> bool:
    st = os.stat(path)
    return (st.st_dev, st.st_ino) != recorded

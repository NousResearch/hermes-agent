"""Session-file cleanup follows only the transaction attempt that actually committed.

``_execute_write`` rolls SQLite back and re-runs the whole callback after a lock error, but it cannot roll
back Python state. A deletion-id list kept outside the callback therefore still carried the rolled-back
attempt's ids, and the post-commit sweep unlinked transcript files of rows the successful retry kept
(a write guard acquired between attempts, or a row that stopped matching).

The interleaving here is real: rollback-journal mode (``database.journal_mode: delete``), a reader holding a
SHARED lock so the first COMMIT fails with ``database is locked`` after every DELETE ran, then the reader is
released and the state is changed through a second SessionDB before the writer's retry.
"""

import json
import os
import sqlite3
from contextlib import closing

import pytest
import hermes_yaml as yaml

from hermes_state import SessionDB
from hermes_state_errors import SessionActiveWriteGuardError

OLD = 1_000_000.0  # far past any retention window
HOLDER = f"pid={os.getpid()}:cmp=retry-cleanup"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"database": {"journal_mode": "delete"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "sessions").mkdir()
    return home


def _open(home):
    db = SessionDB(db_path=home / "state.db")
    assert db._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    return db


def _write_files(home, ids):
    for sid in ids:
        (home / "sessions" / f"{sid}.jsonl").write_text(json.dumps({"session_id": sid}), encoding="utf-8")


def _files(home, ids):
    return {sid: (home / "sessions" / f"{sid}.jsonl").exists() for sid in ids}


def _seed_chain(db, prefix):
    chain = [prefix, f"{prefix}-2", f"{prefix}-3"]
    db.create_session(prefix, source="cli")
    db.append_message(prefix, "user", f"original {prefix}", timestamp=OLD)
    for parent, child in zip(chain, chain[1:]):
        db.publish_compression_child(
            parent_session_id=parent, child_session_id=child, source="cli",
            messages=[{"role": "user", "content": "summary", "timestamp": OLD}],
            require_compression_lease=False,
        )
    db.end_session(chain[-1], "done")
    db._execute_write(lambda c: c.executemany(
        "UPDATE sessions SET started_at = ?, last_activity_at = ?, ended_at = ? WHERE id = ?",
        [(OLD, OLD, OLD + 1, sid) for sid in chain],
    ))
    return chain


def _fail_first_commit(db, home, between_attempts):
    """Make the writer's first COMMIT fail on a real SHARED lock; run *between_attempts* before the retry."""
    reader = sqlite3.connect(home / "state.db")
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM sessions").fetchone()
    original = db._sleep_before_write_retry
    events = []

    def interleave(deadline, patience):
        reader.rollback()
        reader.close()
        db._sleep_before_write_retry = original
        events.append("retry")
        between_attempts()
        return original(deadline, patience)

    db._sleep_before_write_retry = interleave
    return events


def _prune(db, home):
    return db.prune_sessions(
        older_than_days=None, last_active_before=OLD + 10, exclude_active_write_guards=True,
        sessions_dir=home / "sessions",
    )


@pytest.mark.parametrize("mode", ["guard-between-attempts", "guard-before-prune", "no-guard"])
def test_prune_retry_removes_files_only_for_committed_deletions(home, mode):
    with closing(_open(home)) as db, closing(_open(home)) as sibling:
        kept = _seed_chain(db, "kept")
        idle = _seed_chain(db, "idle")
        _write_files(home, kept + idle)
        before = {sid: (home / "sessions" / f"{sid}.jsonl").read_bytes() for sid in kept}

        def acquire():
            assert sibling.try_acquire_compression_lock("kept-3", HOLDER, ttl_seconds=300.0)
            # The rolled-back first attempt left every row and file in place.
            assert all(sibling.get_session(sid) for sid in kept + idle)
            assert all(_files(home, kept + idle).values())

        if mode == "guard-before-prune":
            acquire()
        events = _fail_first_commit(db, home, acquire if mode == "guard-between-attempts" else lambda: None)

        pruned = _prune(db, home)

        assert events == ["retry"]  # the first COMMIT really failed and the callback re-ran
        if mode == "no-guard":
            assert pruned == 6
            assert not any(db.get_session(sid) for sid in kept + idle)
            assert not any(_files(home, kept + idle).values())
            return
        assert pruned == 3
        assert db.get_compression_lineage("kept-3") == kept
        assert db.get_messages("kept")  # original transcript rows kept
        assert all(_files(home, kept).values())
        assert {sid: (home / "sessions" / f"{sid}.jsonl").read_bytes() for sid in kept} == before
        assert not any(db.get_session(sid) for sid in idle)
        assert not any(_files(home, idle).values())


def test_prune_retry_that_deletes_nothing_removes_no_files(home):
    """Zero-deletion retry (early return): every row is guarded by the time the callback re-runs."""
    with closing(_open(home)) as db, closing(_open(home)) as sibling:
        a = _seed_chain(db, "a")
        b = _seed_chain(db, "b")
        _write_files(home, a + b)

        def acquire_both():
            assert sibling.try_acquire_compression_lock("a-3", HOLDER, ttl_seconds=300.0)
            assert sibling.try_acquire_compression_lock("b-3", HOLDER, ttl_seconds=300.0)

        events = _fail_first_commit(db, home, acquire_both)
        assert _prune(db, home) == 0
        assert events == ["retry"]
        assert all(db.get_session(sid) for sid in a + b)
        assert all(_files(home, a + b).values())


def test_delete_session_refused_on_retry_keeps_files(home):
    with closing(_open(home)) as db, closing(_open(home)) as sibling:
        db.create_session("target", source="cli")
        _write_files(home, ["target"])
        events = _fail_first_commit(
            db, home, lambda: sibling.try_acquire_compression_lock("target", HOLDER, ttl_seconds=300.0))

        with pytest.raises(SessionActiveWriteGuardError):
            db.delete_session("target", sessions_dir=home / "sessions", exclude_active_write_guards=True)

        assert events == ["retry"]
        assert db.get_session("target") is not None
        assert _files(home, ["target"]) == {"target": True}


def test_verified_delete_refused_on_retry_keeps_files(home):
    """``sessions export --delete-after-verified`` shape: the retry sees transcript drift and returns False."""
    with closing(_open(home)) as db, closing(_open(home)) as sibling:
        db.create_session("target", source="cli")
        db.append_message("target", "user", "exported line")
        _write_files(home, ["target"])
        with db._read_ctx() as conn:
            snapshot = {"target": db._display_messages_from_conn(conn, "target")}
        events = _fail_first_commit(db, home, lambda: sibling.append_message("target", "user", "written after export"))

        assert db.delete_session(
            "target", sessions_dir=home / "sessions", expected_delete_ids=["target"],
            expected_display_messages=snapshot, exclude_active_write_guards=True,
        ) is False

        assert events == ["retry"]
        assert db.get_session("target") is not None
        assert _files(home, ["target"]) == {"target": True}


def test_delete_sessions_retry_reports_and_cleans_only_the_committed_attempt(home):
    with closing(_open(home)) as db, closing(_open(home)) as sibling:
        for sid in ("guarded", "free"):
            db.create_session(sid, source="cli")
        _write_files(home, ["guarded", "free"])
        events = _fail_first_commit(
            db, home, lambda: sibling.try_acquire_compression_lock("guarded", HOLDER, ttl_seconds=300.0))
        skipped: list = []

        deleted = db.delete_sessions(
            ["guarded", "free"], sessions_dir=home / "sessions", exclude_active_write_guards=True,
            skipped_ids=skipped,
        )

        assert events == ["retry"]
        assert deleted == 1
        assert skipped == ["guarded"]
        assert db.get_session("guarded") is not None and db.get_session("free") is None
        assert _files(home, ["guarded", "free"]) == {"guarded": True, "free": False}


def test_delete_sessions_skip_report_is_not_duplicated_across_retries(home):
    """``skipped_ids`` is caller-visible output: a rolled-back attempt must not append to it either."""
    with closing(_open(home)) as db, closing(_open(home)) as sibling:
        for sid in ("guarded", "free"):
            db.create_session(sid, source="cli")
        assert sibling.try_acquire_compression_lock("guarded", HOLDER, ttl_seconds=300.0)
        _fail_first_commit(db, home, lambda: None)
        skipped: list = ["from-caller"]

        assert db.delete_sessions(["guarded", "free"], exclude_active_write_guards=True, skipped_ids=skipped) == 1
        assert skipped == ["from-caller", "guarded"]


def test_delete_empty_sessions_retry_keeps_files_of_rows_that_gained_messages(home):
    with closing(_open(home)) as db, closing(_open(home)) as sibling:
        for sid in ("revived", "empty"):
            db.create_session(sid, source="cli")
            db.end_session(sid, "done")
        _write_files(home, ["revived", "empty"])
        events = _fail_first_commit(db, home, lambda: sibling.append_message("revived", "user", "back again"))

        assert db.delete_empty_sessions(sessions_dir=home / "sessions") == 1

        assert events == ["retry"]
        assert db.get_session("revived") is not None and db.get_session("empty") is None
        assert _files(home, ["revived", "empty"]) == {"revived": True, "empty": False}

"""Read-only session-list opens must stay disposable.

Two properties that a "keep the read-only handle for the process lifetime"
optimisation silently destroys. Both are asserted against the real
``_open_session_db_at_path`` read path the sidebar poll uses, because both
failures are invisible in a unit test that mocks the store.

1. **The store on disk is the truth.** Recovering a corrupt ``state.db``
   is a file swap (``mv state.db state.db.corrupt-…; cp -a recovered.db
   state.db``) performed while the backend is stopped, but a poll can also
   race a restore. A reader pinned to the old inode keeps serving
   pre-recovery rows forever, so the user "recovers" and still sees the
   broken list.

2. **Forensic backup must stay reachable.** ``offline_file_access`` refuses
   raw byte access while ANY tracked connection is registered for the path,
   because a raw ``close()`` would cancel this process's POSIX advisory locks
   (howtocorrupt §2.2). ``_backup_db_file`` (the copy taken BEFORE a malformed
   store is repaired) and ``_db_fingerprint`` (the repair-attempt ledger key)
   both go through it. A never-closed list reader makes both fail for the rest
   of the process, so a repair runs without its forensic backup and the
   ledger degrades to a size-only key.
"""

from __future__ import annotations

import shutil

from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access
from hermes_cli.web_server import _open_session_db_at_path
from hermes_state import SessionDB, _db_fingerprint


def _ids(db) -> list:
    return [row["id"] for row in db.list_sessions_rich(limit=10, compact_rows=True)]


def test_poll_observes_a_replaced_state_db(tmp_path):
    db_path = tmp_path / "state.db"
    old = SessionDB(db_path=db_path)
    old.create_session("before-recovery", source="cli")
    old.close()

    first = _open_session_db_at_path(db_path, read_only=True)
    try:
        assert _ids(first) == ["before-recovery"]
    finally:
        first.close()

    # `hermes sessions recover` writes a clean database, which the operator
    # then installs over the corrupt one.
    recovered = tmp_path / "recovered-state.db"
    rebuilt = SessionDB(db_path=recovered)
    rebuilt.create_session("after-recovery", source="cli")
    rebuilt.close()

    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            sidecar.unlink()
    db_path.unlink()
    shutil.copy2(recovered, db_path)

    second = _open_session_db_at_path(db_path, read_only=True)
    try:
        assert _ids(second) == ["after-recovery"]
    finally:
        second.close()


def test_poll_leaves_forensic_backup_reachable(tmp_path):
    db_path = tmp_path / "state.db"
    writer = SessionDB(db_path=db_path)
    writer.create_session("s1", source="cli")
    writer.close()

    baseline = _db_fingerprint(db_path)
    assert baseline is not None

    poll = _open_session_db_at_path(db_path, read_only=True)
    try:
        assert _ids(poll) == ["s1"]
    finally:
        poll.close()

    # The raw-copy path a malformed-store repair takes before it touches
    # anything must still be permitted after the poll.
    try:
        with offline_file_access(db_path, what="forensic-backup"):
            pass
    except LiveConnectionError as exc:  # pragma: no cover - failure detail
        raise AssertionError(
            f"a session-list poll left a tracked connection open: {exc}"
        ) from exc

    assert _db_fingerprint(db_path) == baseline

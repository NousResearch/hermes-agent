"""Shared harness for the lost-WAL-generation tests (#105670 / #105726).

Helpers for pinning WAL mode, taking a writer's ``-wal``/``-shm`` generation away, minting a second
generation through the path from another process, and driving a long-lived "gateway" writer child that
serves stdin commands so a test can stop it at the exact split-brain moment.
"""

import contextlib
import json
import os
import queue
import sqlite3
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import hermes_state
import hermes_state_wal
from hermes_state import SessionDB


def pin_wal(monkeypatch) -> None:
    """Pin WAL so this host's vulnerable SQLite still matches production topology."""
    monkeypatch.setattr(
        hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False
    )
    monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: "wal")


def make_db(path: Path, session_id: str, content: str) -> SessionDB:
    db = SessionDB(db_path=path)
    db.create_session(session_id, "cli")
    db.append_message(session_id, role="user", content=content)
    return db


def require_wal(db: SessionDB) -> Path:
    if not db._wal_active:
        db.close()
        pytest.skip("WAL not active on this filesystem")
    wal = Path(os.fspath(db.db_path) + "-wal")
    if not wal.exists():
        db.close()
        pytest.skip("WAL sidecar missing after first write")
    return wal


def lose_sidecars(db_path: Path, *, rename: bool) -> None:
    """Take the -wal/-shm generation away from the writer, as the field incident did."""
    for suffix in ("-wal", "-shm"):
        side = Path(str(db_path) + suffix)
        if not side.exists():
            continue
        if rename:
            side.rename(db_path.with_name("retired" + side.name))
        else:
            side.unlink()


def write_second_generation(db_path: Path, n_rows: int) -> int:
    """From a process that does NOT already hold this db open, mint a fresh WAL generation through the path
    (as any non-hermes opener would), write ``n_rows`` messages and checkpoint them into the main file.
    Returns the message count on the path afterwards. MUST run in a different process from the writer that
    holds the deleted generation — two live handles on one db in one process collide on the -shm."""
    conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode")  # header says WAL -> a fresh -wal/-shm generation on the path
        sessions = [r[0] for r in conn.execute("SELECT id FROM sessions ORDER BY id").fetchall()]
        conn.execute("BEGIN IMMEDIATE")
        for i in range(n_rows):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (sessions[i % len(sessions)], "assistant", "gen2 " + "y" * 2000 + f" #{i}", 1.0 + i),
            )
        conn.execute("COMMIT")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()


def integrity_ok_conn(conn: sqlite3.Connection) -> bool:
    try:
        return [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()] == ["ok"]
    except sqlite3.DatabaseError:
        return False  # a badly torn file fails the pragma itself


def integrity_ok_path(db_path: Path) -> bool:
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        return integrity_ok_conn(conn)
    finally:
        conn.close()


def message_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        return conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()


# The long-lived "gateway" writer A: opens state.db in WAL mode, seeds + checkpoints history, leaves a few
# frames in its WAL (autocheckpoint off so they stay there), then serves stdin commands (write / close /
# quit) so the test can drive close() at the exact split-brain moment. Runs in its OWN process — the second
# generation is written from the test process, so the two live handles never collide on one -shm. Returns
# normally on quit, exercising normal interpreter cleanup as well as explicit close and GC.
#
# Events: ``ready`` | ``skip`` (WAL not active) | ``write`` {refused, artifact} | ``closed`` {error}.
_GATEWAY_CHILD = textwrap.dedent(
    """
    import gc, json, os, sys
    from pathlib import Path
    repo, hermes_home, db_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, repo)
    os.environ["HERMES_HOME"] = hermes_home
    import hermes_state_wal
    if hermes_state_wal.is_sqlite_wal_reset_vulnerable():
        hermes_state_wal.is_sqlite_wal_reset_vulnerable = lambda version_info=None: False
    hermes_state_wal.resolve_journal_mode = lambda: "wal"
    from hermes_state import DeletedWalGenerationError, SessionDB

    def emit(**e):
        sys.stdout.write(json.dumps(e) + "\\n"); sys.stdout.flush()

    db = SessionDB(db_path=Path(db_path))
    if not db._wal_active:
        emit(event="skip"); sys.exit(3)
    for sid in ("gw-0", "gw-1", "gw-2", "gw-3"):
        db.create_session(sid, "cli")
        db.append_message(sid, role="user", content="seed")
    db._try_wal_checkpoint()  # history checkpointed into state.db, like a `sessions optimize` pass
    db._conn.execute("PRAGMA wal_autocheckpoint=0")
    for sid in ("gw-0", "gw-1", "gw-2", "gw-3"):
        db.append_message(sid, role="assistant", content="uncheckpointed " + "x" * 3000)
    emit(event="ready")

    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "write":
            try:
                db.append_message("gw-0", role="user", content="post-loss turn")
                emit(event="write", refused=False, artifact=None)
            except DeletedWalGenerationError:
                capture = db._retired_generation_capture
                emit(event="write", refused=True, artifact=None if capture is None else str(capture))
        elif cmd == "close":
            try:
                db.close()
                emit(event="closed", error=None)
            except Exception as exc:
                emit(event="closed", error=repr(exc))
            del db
            gc.collect()
        elif cmd in ("break-setconfig", "interrupt-setconfig", "exit-setconfig"):
            import sqlite3

            if cmd == "break-setconfig":
                # Force a real native setconfig call to reject its operation.
                sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE = -1
            else:
                interruption = KeyboardInterrupt if cmd == "interrupt-setconfig" else SystemExit

                def interrupted_setconfig(self, *args, _error=interruption):
                    raise _error(73)  # preserve SystemExit's requested status too

                # Override the Python TrackedConnection subclass, preserving the real
                # connection object and its still-enabled native close checkpoint.
                type(db._conn).setconfig = interrupted_setconfig
            emit(event="broken", what=cmd)
        elif cmd in ("break-capture", "break-copy-memory", "break-copy-interrupt"):
            import hermes_state
            import hermes_state_dbfile
            from hermes_state_dbfile import RetiredGenerationCaptureError

            error, message = {
                "break-capture": (RetiredGenerationCaptureError, "no space left on device"),
                "break-copy-memory": (MemoryError, "capture copy ran out of memory"),
                "break-copy-interrupt": (KeyboardInterrupt, "capture copy interrupted"),
            }[cmd]

            def refuse(*args, _error=error, _message=message, **kwargs):
                raise _error(_message)

            if cmd == "break-capture":
                hermes_state.capture_retired_wal_generation = refuse
            else:
                # Keep the real capture routine and its exception handling in the path.
                hermes_state_dbfile._copy_range = refuse
            emit(event="broken", what=cmd)
        elif cmd == "release":
            from hermes_state_registry import release_or_close
            release_or_close(db)  # the production close path: swallows close() errors
            emit(event="released", open=db._conn is not None)
        elif cmd == "quit":
            break
    """
)


class GatewayWriter:
    """Handle on a running gateway writer child; see :func:`gateway_writer`."""

    def __init__(self, proc: subprocess.Popen, path: Path, stderr_path: Path):
        self._proc = proc
        self.path = path
        self._stderr_path = stderr_path
        self._events: "queue.Queue[str | None]" = queue.Queue()
        self._reader = threading.Thread(target=self._read_events, daemon=True)
        self._reader.start()

    def _read_events(self):
        try:
            for line in self._proc.stdout:
                self._events.put(line)
        finally:
            self._events.put(None)

    def stderr_text(self) -> str:
        return self._stderr_path.read_text(encoding="utf-8")

    def next_event(self, name: str) -> dict:
        try:
            line = self._events.get(timeout=20)
        except queue.Empty:
            pytest.fail(f"writer timed out waiting for {name!r}\n" + self.stderr_text())
        assert line is not None, (
            f"writer exited early (rc={self._proc.poll()}) waiting for {name!r}\n" + self.stderr_text()
        )
        event = json.loads(line)
        if name == "ready" and event.get("event") == "skip":
            pytest.skip("WAL not active on this filesystem")
        assert event.get("event") == name, event
        return event

    def send(self, command: str) -> None:
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

    def wait_exit(self, timeout: float = 20) -> int:
        return self._proc.wait(timeout=timeout)

    def _teardown(self) -> None:
        proc = self._proc
        with contextlib.suppress(BrokenPipeError):
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._reader.join(timeout=5)
        proc.stdout.close()


@contextlib.contextmanager
def gateway_writer(tmp_path: Path):
    """Spawn the gateway writer child on ``tmp_path / "state.db"`` and yield a :class:`GatewayWriter`.

    The child is torn down (stdin closed, then wait → terminate → kill) on exit from the block."""
    repo_root = os.path.dirname(os.path.abspath(hermes_state.__file__))
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    path = tmp_path / "state.db"
    stderr_path = tmp_path / "writer-stderr.log"
    with stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            [sys.executable, "-c", _GATEWAY_CHILD, repo_root, str(hermes_home), str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, encoding="utf-8", bufsize=1,
            env={**os.environ, "HERMES_STATE_DB_GUARD_BYPASS": "1"},
        )
        gw = GatewayWriter(proc, path, stderr_path)
        try:
            yield gw
        finally:
            gw._teardown()

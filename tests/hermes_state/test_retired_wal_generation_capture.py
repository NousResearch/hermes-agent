"""Durable capture of a lost WAL generation (#105670 follow-up).

After ``DeletedWalGenerationError`` the retired frames survive only in an unlinked inode this process
keeps open, and the prescribed remediation ("stop the writers, then reopen") lets the kernel drop it.
These tests lose the sidecars for real and check that the exact generation is captured next to the
database at the first halt (or at ``close()``), located by inode rather than by pathname, and that a
transaction committed only in the retired WAL is recoverable from the capture alone, including after
the writer process has exited.
"""

import contextlib
import hashlib
import json
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import hermes_state
import hermes_state_wal
from hermes_state import DeletedWalGenerationError, SessionDB
from hermes_state_dbfile import (
    RETIRED_GENERATION_MANIFEST, RetiredGenerationCaptureError, capture_retired_wal_generation,
)

FD_DIRECTORY = "/proc/self/fd" if sys.platform.startswith("linux") else "/dev/fd"
not_windows = pytest.mark.skipif(sys.platform == "win32", reason="a held sidecar cannot be unlinked on Windows")


@pytest.fixture
def force_wal(monkeypatch):
    """Pin WAL so this host's vulnerable SQLite still matches production topology."""
    monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False)
    monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: "wal")


def _make_db(path: Path, session_id: str, content: str) -> SessionDB:
    db = SessionDB(db_path=path)
    db.create_session(session_id, "cli")
    db.append_message(session_id, role="user", content=content)
    return db


def _require_wal(db: SessionDB) -> Path:
    if not db._wal_active:
        db.close()
        pytest.skip("WAL not active on this filesystem")
    wal = Path(str(db.db_path) + "-wal")
    if not wal.exists():
        db.close()
        pytest.skip("WAL sidecar missing after first write")
    return wal


def _lose_sidecars(db_path: Path, *, rename: bool) -> None:
    """Take the -wal/-shm generation away from the writer, as the field incident did."""
    for suffix in ("-wal", "-shm"):
        side = Path(str(db_path) + suffix)
        if not side.exists():
            continue
        if rename:
            side.rename(db_path.with_name("retired" + side.name))
        else:
            side.unlink()


def _descriptor_for(identity: tuple) -> int:
    for name in os.listdir(FD_DIRECTORY):
        try:
            fd = int(name)
            st = os.fstat(fd)
        except (ValueError, OSError):
            continue
        if (st.st_dev, st.st_ino) == identity:
            return fd
    pytest.fail("the owning SQLite connection discarded its WAL inode")


def _descriptor_contents(fd: int) -> bytes:
    return os.pread(fd, os.fstat(fd).st_size, 0)


def _wal_only_sentinel(db: SessionDB, session_id: str) -> str:
    """Checkpoint history, then commit one row that lives only in the WAL."""
    db._conn.execute("PRAGMA wal_autocheckpoint=0")
    db._try_wal_checkpoint()
    sentinel = "committed only in the retired WAL " + "x" * 3000
    db.append_message(session_id, role="assistant", content=sentinel)
    return sentinel


def _manifest(artifact: Path) -> dict:
    return json.loads((artifact / RETIRED_GENERATION_MANIFEST).read_text(encoding="utf-8"))


def _recover(artifact: Path, name: str, workdir: Path) -> sqlite3.Connection:
    """Open the captured image + WAL as a fresh database, nothing else from the original path."""
    workdir.mkdir()
    shutil.copyfile(artifact / name, workdir / name)
    shutil.copyfile(artifact / (name + "-wal"), workdir / (name + "-wal"))
    return sqlite3.connect(str(workdir / name))


def _count(conn: sqlite3.Connection, content_like: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM messages WHERE content LIKE ?", (content_like,)).fetchone()[0]


def _integrity_ok(conn: sqlite3.Connection) -> bool:
    try:
        return [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()] == ["ok"]
    except sqlite3.DatabaseError:
        return False


# ── Capture at the first halt ───────────────────────────────────────────────────────────────────────


@not_windows
@pytest.mark.parametrize("rename", [False, True], ids=["unlink", "rename"])
def test_halt_captures_the_exact_retired_generation(tmp_path, force_wal, rename):
    path = tmp_path / "state.db"
    db = _make_db(path, "gw-0", "seed")
    _require_wal(db)
    sentinel = _wal_only_sentinel(db, "gw-0")
    identity = db._db_sidecar_identity["-wal"]
    _lose_sidecars(path, rename=rename)
    original = _descriptor_contents(_descriptor_for(identity))

    with pytest.raises(DeletedWalGenerationError):
        db.append_message("gw-0", role="user", content="after the loss")

    artifact = db._retired_generation_capture
    assert artifact is not None and artifact.is_dir() and artifact.parent == tmp_path
    manifest = _manifest(artifact)
    assert manifest["trigger"] == "halt"
    assert tuple(manifest["wal"]["identity"]) == identity
    captured = (artifact / "state.db-wal").read_bytes()
    assert captured == original and captured
    assert manifest["wal"]["sha256"] == hashlib.sha256(captured).hexdigest()
    assert manifest["main"]["mode"] == "copied" and manifest["main"]["header"]["valid"]
    assert (artifact / "state.db").stat().st_size == manifest["main"]["bytes"]

    recovered = _recover(artifact, "state.db", tmp_path / "recovered")
    try:
        assert _count(recovered, sentinel) == 1
        assert _integrity_ok(recovered)
    finally:
        recovered.close()

    db.close()
    assert db._conn is None
    assert db._retired_generation_capture == artifact  # captured once, not again at close


@not_windows
def test_close_captures_when_the_loss_is_first_seen_at_close(tmp_path, force_wal):
    path = tmp_path / "state.db"
    db = _make_db(path, "gw-0", "seed")
    _require_wal(db)
    sentinel = _wal_only_sentinel(db, "gw-0")
    _lose_sidecars(path, rename=False)

    db.close()  # no write in between: close() itself must notice the loss and capture

    artifact = db._retired_generation_capture
    assert artifact is not None and _manifest(artifact)["trigger"] == "close"
    assert db._conn is None
    recovered = _recover(artifact, "state.db", tmp_path / "recovered")
    try:
        assert _count(recovered, sentinel) == 1
    finally:
        recovered.close()


@not_windows
def test_close_refuses_to_settle_without_a_capture(tmp_path, force_wal, monkeypatch):
    path = tmp_path / "state.db"
    db = _make_db(path, "gw-0", "seed")
    _require_wal(db)
    sentinel = _wal_only_sentinel(db, "gw-0")
    _lose_sidecars(path, rename=False)

    def refuse(*args, **kwargs):
        raise RetiredGenerationCaptureError("no space left on device")

    monkeypatch.setattr(hermes_state, "capture_retired_wal_generation", refuse)
    with pytest.raises(RetiredGenerationCaptureError, match="no space left"):
        db.close()
    assert db._conn is not None, "shutdown must not settle while the retired generation is uncaptured"
    assert db._retired_generation_capture is None

    monkeypatch.undo()
    db.close()
    artifact = db._retired_generation_capture
    assert artifact is not None and db._conn is None
    recovered = _recover(artifact, "state.db", tmp_path / "recovered")
    try:
        assert _count(recovered, sentinel) == 1
    finally:
        recovered.close()


@not_windows
def test_capture_selects_the_recorded_inode_not_the_pathname(tmp_path, force_wal):
    """A second deleted WAL under the same pathname belongs to another owner: it must be neither
    captured as ours nor touched."""
    path = tmp_path / "state.db"
    db = _make_db(path, "gw-0", "seed")
    wal = _require_wal(db)
    sentinel = _wal_only_sentinel(db, "gw-0")
    identity = db._db_sidecar_identity["-wal"]
    _lose_sidecars(path, rename=False)
    original = _descriptor_contents(_descriptor_for(identity))

    other = sqlite3.connect(str(tmp_path / "other.db"))
    try:
        other.execute("PRAGMA journal_mode=WAL")
        other.execute("CREATE TABLE independent (content TEXT)")
        other.execute("INSERT INTO independent VALUES ('another owner committed this')")
        other.commit()
        Path(str(tmp_path / "other.db") + "-wal").rename(wal)  # same pathname as our lost WAL
        other_identity = (wal.stat().st_dev, wal.stat().st_ino)
        wal.unlink()
        assert other_identity != identity
        other_before = _descriptor_contents(_descriptor_for(other_identity))

        with pytest.raises(DeletedWalGenerationError):
            db.append_message("gw-0", role="user", content="after the loss")

        artifact = db._retired_generation_capture
        assert (artifact / "state.db-wal").read_bytes() == original
        assert tuple(_manifest(artifact)["wal"]["identity"]) == identity
        assert _descriptor_contents(_descriptor_for(other_identity)) == other_before
        recovered = _recover(artifact, "state.db", tmp_path / "recovered")
        try:
            assert _count(recovered, sentinel) == 1
        finally:
            recovered.close()
        db.close()
    finally:
        other.close()


def test_capture_refuses_to_guess_by_pathname(tmp_path, force_wal):
    path = tmp_path / "state.db"
    db = _make_db(path, "gw-0", "seed")
    try:
        with pytest.raises(RetiredGenerationCaptureError, match="pathname"):
            capture_retired_wal_generation(path, sidecar_identity={}, trigger="test")
        with pytest.raises(RetiredGenerationCaptureError, match="no longer holds"):
            capture_retired_wal_generation(path, sidecar_identity={"-wal": (1, 1)}, trigger="test")
        assert not list(tmp_path.glob("state.db.retired-wal-*"))
    finally:
        db.close()


# ── Recoverable after the writer process has exited ──────────────────────────────────────────────────
#
# The gateway writer A runs in its OWN process, seeds + checkpoints history, leaves rows only in its WAL
# and then serves stdin commands. The parent takes A's sidecars away, drives one refused write (halt +
# capture), mints and checkpoints a newer generation through the path from THIS process, then has A
# close and exit normally. The retired rows must be recoverable from the capture alone afterwards.

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
        elif cmd == "quit":
            break
    """
)


def _write_second_generation(db_path: Path, n_rows: int) -> int:
    """From a process that does NOT hold this db open, mint a fresh WAL generation through the path,
    write ``n_rows`` messages and checkpoint them into the main file."""
    conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode")
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


def _assert_retired_rows_recoverable_after_exit(tmp_path, *, rename):
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
        events: "queue.Queue[str | None]" = queue.Queue()

        def read_events():
            try:
                for line in proc.stdout:
                    events.put(line)
            finally:
                events.put(None)

        threading.Thread(target=read_events, daemon=True).start()

        def next_event(name):
            try:
                line = events.get(timeout=20)
            except queue.Empty:
                pytest.fail(f"writer timed out waiting for {name!r}\n" + stderr_path.read_text(encoding="utf-8"))
            assert line is not None, (
                f"writer exited early (rc={proc.poll()}) waiting for {name!r}\n"
                + stderr_path.read_text(encoding="utf-8"))
            event = json.loads(line)
            if name == "ready" and event.get("event") == "skip":
                pytest.skip("WAL not active on this filesystem")
            assert event.get("event") == name, event
            return event

        def send(command):
            proc.stdin.write(command + "\n")
            proc.stdin.flush()

        try:
            next_event("ready")
            _lose_sidecars(path, rename=rename)

            send("write")
            refused = next_event("write")
            assert refused["refused"] is True
            artifact = Path(refused["artifact"])
            assert artifact.is_dir() and _manifest(artifact)["trigger"] == "halt"

            expected = _write_second_generation(path, n_rows=400)

            send("close")
            assert next_event("closed")["error"] is None
            send("quit")
            assert proc.wait(timeout=20) == 0, stderr_path.read_text(encoding="utf-8")
        finally:
            with contextlib.suppress(BrokenPipeError):
                proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            proc.stdout.close()

    # The writer is gone and its unlinked WAL inode with it. Recovery must come from the capture alone.
    recovered = _recover(artifact, "state.db", tmp_path / "recovered")
    try:
        assert _count(recovered, "uncheckpointed %") == 4, "retired WAL-only rows lost across process exit"
        assert _integrity_ok(recovered), "captured image + WAL do not form a consistent database"
    finally:
        recovered.close()

    # The newer generation at the path is untouched too: setconfig switched SQLite's close-time
    # checkpoint off, or the handle was retired unclosed where it could not be.
    live = sqlite3.connect(str(path))
    try:
        assert _integrity_ok(live) and live.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == expected
    finally:
        live.close()


@pytest.mark.linux_only
def test_retired_rows_recoverable_after_process_exit(tmp_path):
    _assert_retired_rows_recoverable_after_exit(tmp_path, rename=False)


@pytest.mark.macos_only
def test_retired_rows_recoverable_after_process_exit_with_renamed_sidecars(tmp_path):
    _assert_retired_rows_recoverable_after_exit(tmp_path, rename=True)

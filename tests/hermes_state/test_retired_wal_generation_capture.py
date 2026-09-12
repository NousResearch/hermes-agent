"""Durable capture of a lost WAL generation (#105670 follow-up).

After ``DeletedWalGenerationError`` the retired frames survive only in an unlinked inode this process
keeps open, and the prescribed remediation ("stop the writers, then reopen") lets the kernel drop it.
These tests lose the sidecars for real and check that the exact generation is captured next to the
database at the first halt (or at ``close()``), located by inode rather than by pathname, and that a
transaction committed only in the retired WAL is recoverable from the capture alone, including after
the writer process has exited.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

import hermes_state
from hermes_state import DeletedWalGenerationError, SessionDB, _close_time_checkpoint_configurable
from hermes_state_dbfile import (
    RETIRED_GENERATION_MANIFEST, RetiredGenerationCaptureError, capture_retired_wal_generation,
)
from tests.hermes_state._wal_generation_harness import (
    gateway_writer, integrity_ok_conn, lose_sidecars, make_db, pin_wal, require_wal, write_second_generation,
)

FD_DIRECTORY = "/proc/self/fd" if sys.platform.startswith("linux") else "/dev/fd"
not_windows = pytest.mark.skipif(sys.platform == "win32", reason="a held sidecar cannot be unlinked on Windows")


@pytest.fixture
def force_wal(monkeypatch):
    pin_wal(monkeypatch)


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


# ── Capture at the first halt ───────────────────────────────────────────────────────────────────────


@not_windows
@pytest.mark.parametrize("rename", [False, True], ids=["unlink", "rename"])
def test_halt_captures_the_exact_retired_generation(tmp_path, force_wal, rename):
    path = tmp_path / "state.db"
    db = make_db(path, "gw-0", "seed")
    require_wal(db)
    sentinel = _wal_only_sentinel(db, "gw-0")
    identity = db._db_sidecar_identity["-wal"]
    lose_sidecars(path, rename=rename)
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
        assert integrity_ok_conn(recovered)
    finally:
        recovered.close()

    db.close()
    assert db._conn is None
    assert db._retired_generation_capture == artifact  # captured once, not again at close


@not_windows
def test_close_captures_when_the_loss_is_first_seen_at_close(tmp_path, force_wal):
    path = tmp_path / "state.db"
    db = make_db(path, "gw-0", "seed")
    require_wal(db)
    sentinel = _wal_only_sentinel(db, "gw-0")
    lose_sidecars(path, rename=False)

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
    db = make_db(path, "gw-0", "seed")
    require_wal(db)
    sentinel = _wal_only_sentinel(db, "gw-0")
    lose_sidecars(path, rename=False)

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
def test_failed_capture_still_pins_the_handle_and_surfaces_through_the_registry(tmp_path, force_wal, monkeypatch, caplog):
    """Production closes go through hermes_state_registry.release_or_close, which swallows close()
    errors. A failed capture must still (a) log above DEBUG and (b) on runtimes without setconfig take
    the retention pin, so an interpreter exit before the retry cannot checkpoint the stale frames."""
    from hermes_state_registry import release_or_close

    path = tmp_path / "state.db"
    db = make_db(path, "gw-0", "seed")
    require_wal(db)
    _wal_only_sentinel(db, "gw-0")
    lose_sidecars(path, rename=False)
    pins = []
    retires_unclosed = not _close_time_checkpoint_configurable()
    if retires_unclosed:
        monkeypatch.setattr(db, "_retire_connection", pins.append)

    def refuse(*args, **kwargs):
        raise RetiredGenerationCaptureError("no space left on device")

    monkeypatch.setattr(hermes_state, "capture_retired_wal_generation", refuse)
    with caplog.at_level("ERROR"):
        release_or_close(db)  # must not raise
    assert db._conn is not None
    assert "no space left on device" in caplog.text
    if retires_unclosed:  # no setconfig: the pin must already be taken
        assert pins == [db._conn]
        monkeypatch.undo()
        monkeypatch.setattr(db, "_retire_connection", pins.append)
        db.close()  # retry succeeds and must not pin a second time
        assert pins == [pins[0]]
    else:
        monkeypatch.undo()
        db.close()
    assert db._conn is None and db._retired_generation_capture is not None


@not_windows
def test_capture_selects_the_recorded_inode_not_the_pathname(tmp_path, force_wal):
    """A second deleted WAL under the same pathname belongs to another owner: it must be neither
    captured as ours nor touched."""
    path = tmp_path / "state.db"
    db = make_db(path, "gw-0", "seed")
    wal = require_wal(db)
    sentinel = _wal_only_sentinel(db, "gw-0")
    identity = db._db_sidecar_identity["-wal"]
    lose_sidecars(path, rename=False)
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
    db = make_db(path, "gw-0", "seed")
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


def _assert_retired_rows_recoverable_after_exit(tmp_path, *, rename):
    with gateway_writer(tmp_path) as gw:
        path = gw.path
        gw.next_event("ready")
        lose_sidecars(path, rename=rename)

        gw.send("write")
        refused = gw.next_event("write")
        assert refused["refused"] is True
        artifact = Path(refused["artifact"])
        assert artifact.is_dir() and _manifest(artifact)["trigger"] == "halt"

        expected = write_second_generation(path, n_rows=400)

        gw.send("close")
        assert gw.next_event("closed")["error"] is None
        gw.send("quit")
        assert gw.wait_exit(timeout=20) == 0, gw.stderr_text()

    # The writer is gone and its unlinked WAL inode with it. Recovery must come from the capture alone.
    recovered = _recover(artifact, "state.db", tmp_path / "recovered")
    try:
        assert _count(recovered, "uncheckpointed %") == 4, "retired WAL-only rows lost across process exit"
        assert integrity_ok_conn(recovered), "captured image + WAL do not form a consistent database"
    finally:
        recovered.close()

    # The newer generation at the path is untouched too: setconfig switched SQLite's close-time
    # checkpoint off, or the handle was retired unclosed where it could not be.
    live = sqlite3.connect(str(path))
    try:
        assert integrity_ok_conn(live) and live.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == expected
    finally:
        live.close()


@pytest.mark.linux_only
def test_retired_rows_recoverable_after_process_exit(tmp_path):
    _assert_retired_rows_recoverable_after_exit(tmp_path, rename=False)


@pytest.mark.macos_only
def test_retired_rows_recoverable_after_process_exit_with_renamed_sidecars(tmp_path):
    _assert_retired_rows_recoverable_after_exit(tmp_path, rename=True)

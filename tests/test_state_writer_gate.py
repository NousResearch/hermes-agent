"""Cross-process writer coordination for state.db (see #103339).

SQLite owns row concurrency; the gate owns file-structure concurrency:
ordinary row writes from any number of processes proceed, while structural
work (repair surgery, second-connection checkpoints) refuses under a live
writer instead of corrupting the WAL.
"""
import multiprocessing as mp
import os
import sys
import time
import uuid
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_errors import StateDbWriterHeldError
from hermes_state_repair import repair_state_db_schema
from hermes_state_writergate import (
    OwnerToken,
    acquire_writer_gate,
    release_writer_gate,
    writer_gate_holder,
)


def _hold_gate_child(db_path_str, ready, release):
    """Second process: take the gate and hold it until released."""
    acquire_writer_gate(Path(db_path_str))
    ready.set()
    release.wait(60)


def _write_child(db_path_str, queue):
    """Second process: try one real SessionDB write, report the outcome."""
    try:
        db = SessionDB(db_path=Path(db_path_str))
        sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
        db.append_message(sid, role="user", content="second writer hello")
        db.close()
        queue.put("wrote")
    except Exception as exc:  # noqa: BLE001 — the refusal IS the assertion
        queue.put(f"{type(exc).__name__}: {exc}")


def _build_db_child(db_path_str, queue):
    """Second process: build a healthy db, then exit (gate dies with it)."""
    try:
        db = SessionDB(db_path=Path(db_path_str))
        sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
        db.append_message(sid, role="user", content="hello")
        db.close()
        queue.put("built")
    except Exception as exc:  # noqa: BLE001
        queue.put(f"{type(exc).__name__}: {exc}")


def _corrupt_duplicate_fts(db_path: Path) -> None:
    """Duplicate messages_fts row in sqlite_master (raw sqlite: no gate)."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute(
        "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) "
        "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master "
        "WHERE name='messages_fts'"
    )
    conn.commit()
    conn.close()


def _hold_surgery_child(db_path_str, ready, release):
    """Second process: hold the GLOBAL structural lock as repair (surgery)."""
    from hermes_state_writergate import OwnerToken, acquire_writer_gate

    acquire_writer_gate(
        Path(db_path_str), role="repair", owner=OwnerToken("surgery"), exclusive=True)
    ready.set()
    release.wait(60)


def _presence_file(db_path: Path, pid: int) -> Path:
    return db_path.with_name(f"{db_path.name}.writer.{pid}.lock")


def _spawn(fn, *args):
    # "spawn", not fork: a genuinely separate process shares neither memory
    # nor fds — exactly the gateway-vs-CLI shape. Fork presence safety is
    # pinned by test_forked_child_cannot_remove_parent_presence below.
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=fn, args=args)
    proc.start()
    return proc


def _snapshot(db_path: Path) -> bytes:
    return db_path.read_bytes()


def test_concurrent_row_writes_proceed_under_sqlite(tmp_path):
    """Row writes are SQLite's job: a second process announces presence and
    writes; both land (the pinned conformance cell proves exactly-once)."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.append_message(sid, role="user", content="holder hello")

    queue = mp.get_context("spawn").Queue()
    proc = _spawn(_write_child, str(db_path), queue)
    try:
        outcome = queue.get(timeout=60)
    finally:
        proc.join(timeout=60)
    assert outcome == "wrote", outcome
    db.append_message(sid, role="assistant", content="holder again")
    row = db._read_one("SELECT COUNT(*) FROM sessions")
    assert row is not None and row[0] == 2
    row = db._read_one("SELECT COUNT(*) FROM messages")
    assert row is not None and row[0] == 3
    db.close()


def test_same_process_second_handle_writes_freely(tmp_path):
    """Same-process writers (registry, threads, sub-agents) never self-lock."""
    db_path = tmp_path / "state.db"
    db1 = SessionDB(db_path=db_path)
    db2 = SessionDB(db_path=db_path)
    sid = db1.create_session(session_id=str(uuid.uuid4()), source="cli")
    db2.append_message(sid, role="user", content="via second handle")
    db1.append_message(sid, role="assistant", content="via first handle")
    row = db1._read_one("SELECT COUNT(*) FROM messages")
    assert row is not None and row[0] == 2
    db1.close()
    db2.close()


def test_read_only_unaffected_by_foreign_gate(tmp_path):
    """Reads never touch the gate: a foreign holder blocks writes, not reads."""
    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        assert writer_gate_holder(db_path) is not None  # foreign holder visible
        ro = SessionDB(db_path=db_path, read_only=True)
        row = ro._read_one("SELECT COUNT(*) FROM sessions")
        ro.close()
        assert row is not None and row[0] == 1
    finally:
        release.set()
        holder.join(timeout=60)


def test_repair_refuses_live_gated_db_without_touching_file(tmp_path):
    """Repair under a live writer returns REFUSED and modifies nothing."""
    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        before = _snapshot(db_path)
        report = repair_state_db_schema(db_path, backup=False)
        assert report.get("repaired") is False
        assert "REFUSED" in (report.get("error") or ""), report
        assert _snapshot(db_path) == before
        assert report.get("backup_path") is None
        assert list(tmp_path.glob("*.backup*")) == []
    finally:
        release.set()
        holder.join(timeout=60)
    # Gate free again: no longer refused (healthy db needs no repair either way).
    report = repair_state_db_schema(db_path, backup=False)
    assert "REFUSED" not in (report.get("error") or ""), report


def test_doctor_checkpoint_skipped_under_live_writer(tmp_path):
    """`doctor --fix` never checkpoints a WAL a live writer owns."""
    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _state_db_wal

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    wal_path = tmp_path / "state.db-wal"
    fd = os.open(str(wal_path), os.O_WRONLY | os.O_CREAT)
    try:
        os.ftruncate(fd, 60 * 1024 * 1024)  # sparse: big size, no disk cost
    finally:
        os.close(fd)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        f = Finding()
        _state_db_wal(f, True, db_path)
        assert f.fixed == 0
        assert any("gateway" in issue for issue in f.issues), f.issues
        assert wal_path.stat().st_size == 60 * 1024 * 1024
    finally:
        release.set()
        holder.join(timeout=60)


def test_close_releases_gate_for_other_processes(tmp_path):
    """A writer that closed does not pin the gate: later processes proceed."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.append_message(sid, role="user", content="hello")
    db.close()
    assert writer_gate_holder(db_path) is None
    queue = mp.get_context("spawn").Queue()
    proc = _spawn(_write_child, str(db_path), queue)
    try:
        outcome = queue.get(timeout=60)
    finally:
        proc.join(timeout=60)
    assert outcome == "wrote", outcome


def test_gate_free_when_nobody_holds(tmp_path):
    """Probe is silent on a free gate; same-process acquire is idempotent."""
    db_path = tmp_path / "state.db"
    assert writer_gate_holder(db_path) is None
    acquire_writer_gate(db_path)
    acquire_writer_gate(db_path)  # idempotent, no self-lock
    assert writer_gate_holder(db_path) is None  # ours reads as free


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork test")
def test_forked_child_cannot_remove_parent_presence(tmp_path):
    """After fork(), the child announces under its own pid and its close()
    must not disturb the parent's presence (unlink is own-pid-only)."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.append_message(sid, role="user", content="holder hello")
    parent_file = _presence_file(db_path, os.getpid())
    assert parent_file.exists()

    pid = os.fork()
    if pid == 0:  # child: own presence ok; close must spare the parent file
        try:
            from hermes_state_writergate import OwnerToken, acquire_writer_gate

            acquire_writer_gate(db_path, owner=OwnerToken("fork"))
            child_file = _presence_file(db_path, os.getpid())
            announced = child_file.exists()
            db.close()
            ok = announced and parent_file.exists() and not child_file.exists()
            os._exit(20 if ok else 10)
        except Exception:  # noqa: BLE001
            os._exit(30)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 20
    assert parent_file.exists()
    db.append_message(sid, role="assistant", content="parent still writes")
    db.close()


def test_stale_presence_litter_ignored_and_reaped(tmp_path):
    """A crashed writer's presence file (dead pid, old, flock-free) neither
    blocks repair nor survives it."""
    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    stale = _presence_file(db_path, 1 << 30)
    stale.write_bytes(b'{"pid": 1073741824, "role": "writer"}')
    stamp = time.time() - 120.0
    os.utime(stale, (stamp, stamp))
    report = repair_state_db_schema(db_path, backup=False)
    assert "REFUSED" not in (report.get("error") or ""), report
    assert not stale.exists()


def test_row_write_refused_while_surgery_holds_global(tmp_path):
    """Fail-closed in both directions: a row write starting mid-surgery is
    refused, and succeeds again once surgery releases."""
    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    surgery = _spawn(_hold_surgery_child, str(db_path), ready, release)
    # Open BEFORE surgery starts (open-time gates only fire while a surgery
    # holds the global lock); the write below lands mid-surgery.
    db = SessionDB(db_path=db_path)
    try:
        assert ready.wait(timeout=60)
        with pytest.raises(StateDbWriterHeldError, match="structural operation"):
            db.create_session(session_id=str(uuid.uuid4()), source="cli")
    finally:
        release.set()
        surgery.join(timeout=60)
    db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.close()


def test_open_refuses_init_repair_under_live_gate(tmp_path):
    """SessionDB() on a corrupt db held by a live writer raises instead of
    repairing: the INIT self-heal delegates to the gated repair (proven
    REFUSED by test_repair_refuses_live_gated_db_without_touching_file), so
    the open fails with the original error and the file is untouched."""
    import sqlite3

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)
    _corrupt_duplicate_fts(db_path)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        before = _snapshot(db_path)
        with pytest.raises(sqlite3.DatabaseError):
            SessionDB(db_path=db_path)
        assert _snapshot(db_path) == before
    finally:
        release.set()
        holder.join(timeout=60)


def test_doctor_repair_path_refuses_live_gate(tmp_path):
    """`doctor --fix` schema repair delegates to the gated repair: REFUSED
    under a live writer, recorded as failed-issue, file untouched."""
    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _repair_state_db

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)
    _corrupt_duplicate_fts(db_path)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        before = _snapshot(db_path)
        f = Finding()
        _repair_state_db(f, True, db_path, "schema")
        assert f.fixed == 0
        assert any("malformed" in issue for issue in f.issues), f.issues
        assert _snapshot(db_path) == before
    finally:
        release.set()
        holder.join(timeout=60)


def test_doctor_checkpoint_runs_when_gate_free(tmp_path):
    """Free gate: `doctor --fix` checkpoints the WAL and reports fixed."""
    import os

    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _state_db_wal

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    wal_path = tmp_path / "state.db-wal"
    fd = os.open(str(wal_path), os.O_WRONLY | os.O_CREAT)
    try:
        os.ftruncate(fd, 60 * 1024 * 1024)  # sparse
    finally:
        os.close(fd)

    f = Finding()
    _state_db_wal(f, True, db_path)
    assert f.fixed == 1, f.issues
    assert not any("gateway" in issue for issue in f.issues), f.issues


def _probe_child(db_path_str, queue):
    """Second process: report the foreign structural probe (None or holder)."""
    from hermes_state_writergate import writer_gate_holder

    try:
        queue.put(("holder", writer_gate_holder(Path(db_path_str))))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _foreign_holder(db_path):
    """writer_gate_holder() as seen by a genuinely separate process."""
    queue = mp.get_context("spawn").Queue()
    proc = _spawn(_probe_child, str(db_path), queue)
    try:
        kind, value = queue.get(timeout=60)
    finally:
        proc.join(timeout=60)
    assert kind == "holder", value
    return value


def test_second_handle_pins_presence_until_it_closes(tmp_path):
    """Lifetime invariant: with two writable handles, closing the first must
    not clear presence — only the second close may (blocker 2)."""
    db_path = tmp_path / "state.db"
    db1 = SessionDB(db_path=db_path)
    db1.create_session(session_id=str(uuid.uuid4()), source="cli")
    db2 = SessionDB(db_path=db_path)
    db2.create_session(session_id=str(uuid.uuid4()), source="cli")
    db1.close()
    assert _foreign_holder(db_path) is not None
    db2.close()
    assert _foreign_holder(db_path) is None


def test_structural_take_refuses_writer_announced_after_probe(tmp_path):
    """Deterministic probe->take interleaving: a writer announcing after our
    free probe still refuses the structural take (blocker 1)."""
    from hermes_state_writergate import OwnerToken, acquire_writer_gate

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    assert _foreign_holder(db_path) is None  # probe: free
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)  # writer announces after the probe
        token = OwnerToken("late-take")
        with pytest.raises(StateDbWriterHeldError):
            acquire_writer_gate(db_path, role="repair", owner=token, exclusive=True)
    finally:
        release.set()
        holder.join(timeout=60)
    token = OwnerToken("late-take")
    acquire_writer_gate(db_path, role="repair", owner=token, exclusive=True)
    release_writer_gate(db_path, token)


def test_open_time_mutations_refuse_during_surgery(tmp_path):
    """Constructor DDL/generation writes refuse while a surgery holds the
    global lock, and open cleanly once it releases (blocker 3)."""
    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    surgery = _spawn(_hold_surgery_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        with pytest.raises(StateDbWriterHeldError, match="structural"):
            SessionDB(db_path=db_path)
    finally:
        release.set()
        surgery.join(timeout=60)
    db = SessionDB(db_path=db_path)
    db.close()

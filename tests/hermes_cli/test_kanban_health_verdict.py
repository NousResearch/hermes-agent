"""Durable healthy-verdict cache for the kanban DB guard (``<db>.health.json``).

``connect()`` / ``init_db()`` used to run ``PRAGMA integrity_check`` on the first
connect of EVERY process: seconds of CPU and disk on a large board, serialized on
the init flock, and the processes that timed out of the bounded acquire then ran
the walk anyway, unlocked (the "kanban init lock … not acquired within 10s"
stampede). The verdict is now remembered beside the board, bound to the exact
bytes it describes, so:

* the page walk is paid at most once per TTL per board;
* a connect that finds a verdict for these exact bytes does no work at all;
* a connect that finds the bytes moved pays an O(1) structural probe, not a walk,
  and keeps the walk's TTL slot (a live board's main-file mtime moves on every
  WAL checkpoint, so a bare mtime bump must not re-walk);
* a connect that did not get the init lock never walks and never writes;
* a new/replaced board is walked before the first verdict exists, and damage
  still fails closed.

These are behaviour contracts: each drives the public guard through ``connect()``
and asserts how many walks happened, never that a particular helper is called.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


@pytest.fixture
def walks(monkeypatch):
    """Every ``PRAGMA integrity_check`` page walk the guard performs."""
    calls: list[Path] = []
    original = kbc._probe_for_corruption

    def counted(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(kbc, "_probe_for_corruption", counted)
    return calls


@pytest.fixture
def probes(monkeypatch):
    """Every structural (schema-header) probe the guard performs."""
    calls: list[Path] = []
    original = kbc._structural_probe

    def counted(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(kbc, "_structural_probe", counted)
    return calls


def _db_path() -> Path:
    return kb.kanban_db_path(board="default")


def _build_board(db_path: Path, tasks: int = 6) -> None:
    """A real board with rows, created the way the CLI creates one."""
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(db_path=db_path)
    conn = kbc.connect(db_path=db_path)
    try:
        for i in range(tasks):
            kb.create_task(conn, title=f"task-{i}")
    finally:
        conn.close()  # not the `with` form: __exit__ commits, it does not close


def _record_verdict(db_path: Path) -> dict:
    """Let the guard walk the board and hand back the verdict it stored."""
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kbc.connect(db_path=db_path).close()
    return json.loads(kbc._health_verdict_path(db_path.resolve()).read_text(encoding="utf-8"))


def _bump_identity(db_path: Path, *, seconds: int = 5) -> None:
    """Move the file's mtime without touching its bytes (a WAL checkpoint)."""
    st = db_path.stat()
    os.utime(db_path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


def _hold_init_lock(db_path: Path):
    """Hold the board's init flock from another thread — (release, thread)."""
    holding = threading.Event()
    release = threading.Event()

    def _holder():
        with kbc._cross_process_init_lock(db_path):
            holding.set()
            release.wait(timeout=10)

    thread = threading.Thread(target=_holder, daemon=True)
    thread.start()
    assert holding.wait(timeout=5), "holder thread never acquired the init lock"
    return release, thread


def _garbage_db(path: Path) -> None:
    """Valid SQLite header, garbage pages — the non-index corruption class."""
    header = b"SQLite format 3\x00" + b"\x10\x00\x02\x02\x00\x40\x20\x20"
    header += b"\x00\x00\x00\x0c\x00\x00\x23\x46\x00\x00\x00\x00"
    header = header.ljust(100, b"\x00")
    path.write_bytes(header + b"definitely not a valid sqlite page \x00\x01\x02\x03" * 64)
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))


def test_first_healthy_connect_walks_and_records_the_verdict(kanban_home, walks):
    """A board with no verdict yet is walked in full, then vouched for on disk."""
    db_path = _db_path()
    _build_board(db_path)
    verdict_path = kbc._health_verdict_path(db_path.resolve())
    assert not verdict_path.exists(), "a board we just created must not be vouched for yet"

    verdict = _record_verdict(db_path)

    assert len(walks) == 1
    assert verdict["verdict"] == "ok"
    assert verdict["version"] == kbc._HEALTH_VERDICT_VERSION
    assert verdict["db"] == db_path.name
    assert verdict["size"] == db_path.stat().st_size
    assert isinstance(verdict["schema_version"], int)
    assert verdict["checked_at"] > 0


def test_a_connect_for_these_exact_bytes_does_no_work(kanban_home, walks, probes):
    """The warm path reuses the verdict: no page walk and no probe at all."""
    db_path = _db_path()
    _build_board(db_path)
    _record_verdict(db_path)
    walks.clear()
    probes.clear()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kbc.connect(db_path=db_path).close()

    assert walks == []
    assert probes == []


def test_moved_bytes_cost_a_structural_probe_not_a_page_walk(kanban_home, walks, probes):
    """A write/checkpoint moves the file; that is a probe, not a 500 MB walk."""
    db_path = _db_path()
    _build_board(db_path)
    before = _record_verdict(db_path)
    walks.clear()
    probes.clear()
    _bump_identity(db_path)

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kbc.connect(db_path=db_path).close()

    after = json.loads(kbc._health_verdict_path(db_path.resolve()).read_text(encoding="utf-8"))
    assert walks == [], "moving the bytes must not re-run the page walk inside the TTL"
    assert probes, "the moved file is structurally probed"
    assert after["mtime_ns"] == db_path.stat().st_mtime_ns, "the refreshed verdict binds the new bytes"
    assert after["checked_at"] == before["checked_at"], (
        "a refresh must not buy time: the walk keeps its TTL slot")


def test_an_expired_verdict_forces_the_walk(kanban_home, walks, monkeypatch):
    db_path = _db_path()
    _build_board(db_path)
    _record_verdict(db_path)
    monkeypatch.setattr(kbc, "_HEALTH_VERDICT_TTL_SECONDS", 0)
    walks.clear()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kbc.connect(db_path=db_path).close()

    assert len(walks) == 1


def test_a_replaced_board_never_inherits_the_verdict(kanban_home, walks):
    """Same size, same mtime, different file: the swap must not look vouched-for."""
    db_path = _db_path()
    _build_board(db_path)
    _record_verdict(db_path)
    # A byte-identical copy with a fresh inode: copy2 keeps size AND mtime, so only
    # the inode can tell the guard this is no longer the file it vouched for.
    replacement = db_path.parent / "replacement.db"
    shutil.copy2(db_path, replacement)
    os.replace(replacement, db_path)
    walks.clear()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kbc.connect(db_path=db_path).close()

    assert len(walks) == 1, "a new inode is a new board: it is walked before being trusted"


def test_corruption_still_fails_closed_with_a_stored_verdict(kanban_home):
    """A verdict that predates the damage cannot paper over it."""
    db_path = _db_path()
    _build_board(db_path)
    _record_verdict(db_path)
    _garbage_db(db_path)

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with pytest.raises(kbc.KanbanDbCorruptError):
        kbc.connect(db_path=db_path)
    assert not kbc._health_verdict_path(db_path.resolve()).exists(), (
        "a refused file must not keep a verdict that could vouch for it")
    assert list(db_path.parent.glob(f"{db_path.name}.corrupt.*.bak")), "pre-touch bytes are preserved"


def test_a_broken_verdict_file_is_not_evidence_of_health(kanban_home, walks):
    db_path = _db_path()
    _build_board(db_path)
    kbc._health_verdict_path(db_path.resolve()).write_text("{not json", encoding="utf-8")
    walks.clear()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kbc.connect(db_path=db_path).close()

    assert len(walks) == 1


def test_a_lockless_connect_walks_nothing_and_writes_nothing(kanban_home, walks, monkeypatch):
    """The timeout fallback must not add another page walk to the stampede."""
    monkeypatch.setattr(kbc, "_INIT_LOCK_TIMEOUT_SECONDS", 0.3)
    db_path = _db_path()
    _build_board(db_path)  # ...and no verdict exists yet, so nothing could be reused
    walks.clear()

    release, thread = _hold_init_lock(db_path)
    try:
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with kbc.connect(db_path=db_path) as conn:
            conn.execute("SELECT 1").fetchone()
    finally:
        release.set()
        thread.join(timeout=5)

    assert walks == [], "a process that did not get the init lock must not walk the board"
    assert not kbc._health_verdict_path(db_path.resolve()).exists(), (
        "a process that did not get the init lock must not write the verdict")


def test_a_lockless_connect_still_refuses_a_corrupt_board(kanban_home, monkeypatch):
    """Skipping the walk is not skipping the guard: damage fails closed anyway."""
    monkeypatch.setattr(kbc, "_INIT_LOCK_TIMEOUT_SECONDS", 0.3)
    db_path = _db_path()
    _build_board(db_path)
    _garbage_db(db_path)

    release, thread = _hold_init_lock(db_path)
    try:
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with pytest.raises(kbc.KanbanDbCorruptError):
            kbc.connect(db_path=db_path)
    finally:
        release.set()
        thread.join(timeout=5)


def test_a_missing_board_gets_no_verdict(kanban_home, walks):
    """A board we create is not vouched for: the next process walks it in full."""
    db_path = _db_path()
    _build_board(db_path)
    assert not kbc._health_verdict_path(db_path.resolve()).exists()
    assert walks == []


def test_repair_records_the_verdict_it_earned(kanban_home, walks):
    """:func:`repair_db` walks; that walk is worth caching like any other."""
    db_path = _db_path()
    _build_board(db_path)
    walks.clear()

    result = kbc.repair_db(db_path=db_path)

    assert result.status == "ok"
    assert len(walks) == 1
    verdict = json.loads(kbc._health_verdict_path(db_path.resolve()).read_text(encoding="utf-8"))
    assert verdict["verdict"] == "ok"
    assert verdict["size"] == db_path.stat().st_size


def test_the_verdict_is_not_stored_for_a_zero_byte_board(kanban_home, walks):
    """An empty board file has nothing to vouch for; it gets a verdict once real."""
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.touch()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    kbc.connect(db_path=db_path).close()

    assert walks == []
    # The empty file was schema-initialised by this connect, so no verdict is
    # written for it: the first process to open the finished board walks it.
    assert not kbc._health_verdict_path(db_path.resolve()).exists()

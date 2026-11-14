"""A gateway never runs a full VACUUM; ``vacuum_after_prune`` reclaims space while serving.

The constructor's state.db pass ran ``maybe_auto_prune_and_vacuum(vacuum=True)`` before the control
socket and adapters came up. VACUUM holds the write lock for the whole rewrite, so a multi-GB store on
NFS held the boot for over twenty minutes and operators turned ``vacuum_after_prune`` off — after which
the file could only shrink with a stop, ``hermes sessions optimize`` and a start. Now the constructor
only prunes, and the housekeeping tick gives freed pages back in short incremental batches.
"""

from __future__ import annotations

import itertools
import threading
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from hermes_state import SessionDB

_CONFIG = (
    "sessions:\n"
    "  auto_prune: true\n"
    "  retention_days: 0\n"
    "  min_interval_hours: 0\n"
    "  min_vacuum_interval_days: 0\n"
    "  vacuum_after_prune: true\n"
)


class _Ticks:
    """Stop event that lets the housekeeping loop run exactly ``n`` ticks with no sleeping (the
    state.db chore runs every 60th tick)."""

    def __init__(self, n: int):
        self.n, self.left = 0, n

    def is_set(self):
        return self.n >= self.left

    def wait(self, timeout=None):
        self.n += 1
        return True


def _seed(home, *, rows: int = 60) -> None:
    """An ended session holding enough bytes that pruning frees far more than the 25% ratio gate."""
    (home / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("old", "cli")
        for _ in range(rows):
            db.append_message("old", "user", "x" * 20_000)
        db.end_session("old", "done")
    finally:
        db.close()


@pytest.fixture
def home(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return tmp_path


def _startup_pass():
    """The constructor's state.db phase itself (``GatewayRunner._init_session_db``), on a bare runner."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._session_db_init_error = None
    gateway_run.GatewayRunner._init_session_db(runner)
    for handle in list(runner._session_db_handles.values()):
        from hermes_state_registry import release_or_close
        release_or_close(getattr(handle, "_db", handle))


def test_startup_maintenance_never_enters_a_full_vacuum(home, monkeypatch):
    """A VACUUM that never returns must not hold the constructor: the pass prunes and returns."""
    _seed(home)
    entered, release = threading.Event(), threading.Event()

    def _hung_vacuum(self):
        entered.set()
        release.wait(30)
        return 0

    monkeypatch.setattr(SessionDB, "vacuum", _hung_vacuum)
    done = threading.Event()
    worker = threading.Thread(target=lambda: (_startup_pass(), done.set()), daemon=True)
    worker.start()
    try:
        assert done.wait(15), "startup state.db maintenance is still running: it is blocked in VACUUM"
        assert not entered.is_set(), "startup maintenance entered a full VACUUM"
    finally:
        release.set()
        worker.join(5)

    db = SessionDB(db_path=home / "state.db")
    try:
        assert db.get_session("old") is None, "startup pass no longer prunes"
    finally:
        db.close()


def test_startup_maintenance_leaves_the_loop_ticking(home, monkeypatch):
    """Loop-tick check: ``GatewayRunner()`` (and this pass) runs on the owner loop thread, so a
    VACUUM inside it froze every coroutine for the whole rewrite. A stand-in VACUUM that holds for
    several seconds must not show up as a gap between loop ticks."""
    import asyncio

    _seed(home)
    hold_s = 6.0
    release = threading.Event()
    timer = threading.Timer(hold_s, release.set)
    monkeypatch.setattr(SessionDB, "vacuum", lambda self: release.wait(30) or 0)

    async def _scenario():
        loop = asyncio.get_running_loop()
        ticks = [loop.time()]

        async def _ticker():
            for _ in range(40):
                await asyncio.sleep(0.01)  # health: allow HX013 -- the tick cadence IS the loop-liveness probe
                ticks.append(loop.time())

        ticker = asyncio.ensure_future(_ticker())
        await asyncio.sleep(0)
        loop.call_soon(_startup_pass)  # synchronous on the loop thread, as the constructor is
        await ticker
        return max(b - a for a, b in itertools.pairwise(ticks))

    timer.start()
    try:
        worst_gap = asyncio.run(_scenario())
    finally:
        release.set()
        timer.cancel()
    assert worst_gap < hold_s / 2, f"the owner loop froze for {worst_gap:.1f}s in startup maintenance"


def test_housekeeping_tick_reclaims_freed_space_without_a_full_vacuum(home, monkeypatch):
    """On an INCREMENTAL store the serving gateway gives pruned pages back to the disk itself."""
    _seed(home)
    vacuums = []
    monkeypatch.setattr(SessionDB, "vacuum", lambda self: vacuums.append(True) or 0)
    db = SessionDB(db_path=home / "state.db")
    try:
        assert db._conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2  # new stores start INCREMENTAL
        pages_before = db._conn.execute("PRAGMA page_count").fetchone()[0]
    finally:
        db.close()

    gateway_run._start_gateway_housekeeping(_Ticks(60), interval=0)

    db = SessionDB(db_path=home / "state.db")
    try:
        assert db.get_session("old") is None
        pages_after = db._conn.execute("PRAGMA page_count").fetchone()[0]
        free_after = db._conn.execute("PRAGMA freelist_count").fetchone()[0]
        assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        db.close()
    assert vacuums == [], "the serving gateway ran a full VACUUM"
    assert pages_after < pages_before // 2, (pages_before, pages_after)
    assert free_after < 256

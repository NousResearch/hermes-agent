"""A store that lags SCHEMA_SQL must stay browsable (#140315).

#106742 added four tables and three columns to SCHEMA_SQL under schema_version 31. A named
profile whose state.db had not been opened writable since the update lost its whole session
list on every read-only surface: the browse probe hit ``no such table`` and answered 503
"schema is unavailable or corrupt", and an idle profile never got the writable open that would
have caught it up. These tests build the real pre-merge shape from the pre-#106742 SCHEMA_SQL
(trimmed to what that release declared) and browse it through the real read-only path.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest
from fastapi import HTTPException

from hermes_cli.web_server_sessions import _open_session_db_at_path
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION

_ADDED_TABLES = ("runtime_epoch", "session_admissions", "worker_executions", "worker_receipts")
_ADDED_COLUMNS = (("sessions", "runtime_revision"), ("sessions", "runtime_generation"),
                  ("async_delegations", "owner_execution_id"))


def _pre_merge_store(path, *, version=31):
    """The shape a store kept by the release before #106742 has: built by SessionDB, then the
    tables/columns #106742 added removed again, stamped with the version that release wrote."""
    with SessionDB(db_path=path) as db:
        db.create_session("kept", source="cli")
        db.append_message("kept", role="user", content="history survives")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("DROP TRIGGER IF EXISTS sessions_runtime_ledger_delete_guard")
        for table in reversed(_ADDED_TABLES):
            conn.execute(f"DROP TABLE {table}")
        conn.execute("DROP INDEX IF EXISTS idx_sessions_local_created_started")
        for table, column in _ADDED_COLUMNS:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        conn.execute("UPDATE schema_version SET version = ?", (version,))
        conn.commit()
    return path


def _tables(path):
    with closing(sqlite3.connect(path)) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _profile_home(tmp_path, name="scribe"):
    home = tmp_path / "profiles" / name
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    return home


def test_unowned_lagging_store_browses_after_one_reconcile(tmp_path):
    path = _pre_merge_store(_profile_home(tmp_path) / "state.db")

    db = _open_session_db_at_path(path, read_only=True)
    try:
        assert [row["id"] for row in db.list_sessions_rich(limit=10, compact_rows=True)] == ["kept"]
        assert [m["content"] for m in db.get_messages("kept")] == ["history survives"]
    finally:
        db.close()
    assert set(_ADDED_TABLES) <= _tables(path)

    # Once: the second browse is a pure read-only open (no write, no new DDL).
    before = path.read_bytes()
    db = _open_session_db_at_path(path, read_only=True)
    db.close()
    assert path.read_bytes() == before


def test_owned_lagging_store_is_not_written_and_names_the_remedy(tmp_path):
    from gateway.runtime_ownership import ProfileOwnership

    home = _profile_home(tmp_path)
    path = _pre_merge_store(home / "state.db")
    before = path.read_bytes()
    owner = ProfileOwnership()  # what a running gateway holds on its profile
    owner.reserve([home])
    try:
        with pytest.raises(HTTPException) as caught:
            _open_session_db_at_path(path, read_only=True)
    finally:
        owner.close()
    assert caught.value.status_code == 503
    detail = str(caught.value.detail)
    assert "running gateway owns it" in detail
    assert "gateway restart" in detail
    assert "corrupt" not in detail
    assert path.read_bytes() == before
    assert not set(_ADDED_TABLES) & _tables(path)


def test_corrupt_store_still_returns_503(tmp_path):
    """Corruption is not drift: the open fails with the corrupt-store 503 and nothing is written."""
    path = tmp_path / "state.db"
    with SessionDB(db_path=path) as db:
        db.create_session("kept", source="cli")
    with closing(sqlite3.connect(path)) as conn:
        if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    with open(path, "r+b") as fh:  # sqlite_master (page 1 b-tree) overwritten past the header
        fh.seek(100)
        fh.write(b"\xde\xad\xbe\xef" * ((page_size - 100) // 4))
    before = path.read_bytes()

    with pytest.raises(HTTPException) as caught:
        _open_session_db_at_path(path, read_only=True).close()
    assert caught.value.status_code == 503
    assert caught.value.detail["error"] == "state_db_corrupt"
    assert path.read_bytes() == before


def test_missing_history_table_is_refused_not_recreated(tmp_path):
    path = _pre_merge_store(tmp_path / "state.db")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DROP TABLE messages")
        conn.commit()
    before = path.read_bytes()
    with pytest.raises(HTTPException) as caught:
        _open_session_db_at_path(path, read_only=True)
    assert caught.value.status_code == 503
    assert "sessions recover" in str(caught.value.detail)
    assert path.read_bytes() == before


def test_version_bump_migration_is_additive_and_idempotent(tmp_path):
    path = _pre_merge_store(tmp_path / "state.db")
    assert SCHEMA_VERSION > 31

    for _ in range(2):
        with SessionDB(db_path=path) as db:
            assert db._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
            assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    assert set(_ADDED_TABLES) <= _tables(path)
    with closing(sqlite3.connect(path)) as conn:
        for table, column in _ADDED_COLUMNS:
            assert column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    # A browse-reconciled store (still stamped 31) also upgrades cleanly on the owner's open.
    other = _pre_merge_store(tmp_path / "other.db")
    _open_session_db_at_path(other, read_only=True).close()
    with SessionDB(db_path=other) as db:
        assert db._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION


def test_messages_route_for_lagging_profile(monkeypatch, tmp_path):
    """``GET /api/sessions/<id>/messages?profile=<name>`` was the 503 in the report."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    home = _profile_home(tmp_path)
    _pre_merge_store(home / "state.db")
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name == "scribe")
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda name: home)
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    response = client.get("/api/sessions/kept/messages?profile=scribe")
    assert response.status_code == 200, response.text
    assert [m["content"] for m in response.json()["messages"]] == ["history survives"]


# ── Desktop/dashboard HTTP surfaces on this process's own store (moved from test_web_server.py) ──


@pytest.fixture
def own_store_client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_cli.web_routers import profiles as profiles_routes
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(profiles_routes, "_SIDEBAR_CACHE_TTL_SECONDS", 0.0)
    path = get_hermes_home() / "state.db"
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", path)
    (get_hermes_home() / "config.yaml").write_text("{}\n", encoding="utf-8")
    with SessionDB(db_path=path) as db:
        db.create_session("stale-schema", source="cli")
        db.append_message("stale-schema", role="user", content="hi")
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client, path


def _drop_session_column(path, column):
    with closing(sqlite3.connect(path)) as conn:
        # SQLite refuses DROP COLUMN while an index references it; a pre-column store has neither.
        conn.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
        conn.execute(f"ALTER TABLE sessions DROP COLUMN {column}")
        conn.commit()


def _session_columns(path):
    with closing(sqlite3.connect(path)) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}


def _owned(home):
    from gateway.runtime_ownership import ProfileOwnership
    owner = ProfileOwnership()  # what a running gateway holds on its profile
    owner.reserve([home])
    return owner


@pytest.mark.parametrize("missing_column", ["archived", "pinned", "last_activity_at"])
def test_session_list_reconciles_once_only_when_unowned(own_store_client, missing_column):
    client, path = own_store_client
    _drop_session_column(path, missing_column)

    before = path.read_bytes()
    owner = _owned(path.parent)
    try:
        response = client.get("/api/sessions?limit=50&offset=0")
        sidebar = client.get("/api/profiles/sessions/sidebar").json()
    finally:
        owner.close()
    assert response.status_code == 503
    assert "gateway owns it" in response.json()["detail"]
    # A failed read is a retryable failed load naming the drift, never an empty list.
    assert "behind this Hermes version" in sidebar["errors"][0]["error"]
    assert "sessions" not in sidebar["recents"] and sidebar["recents"]["failed"] is True
    assert path.read_bytes() == before
    assert missing_column not in _session_columns(path)

    response = client.get("/api/sessions?limit=50&offset=0")
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["sessions"]] == ["stale-schema"]
    assert missing_column in _session_columns(path)


def test_startup_eager_reconcile_heals_only_an_unowned_stale_store(own_store_client):
    """After `hermes update` an old-schema store must not fail every poll (#79531/#80037); the
    dashboard is never a second writer while a gateway owns it (#107688)."""
    from hermes_cli import web_server_lifecycle

    _client, path = own_store_client
    _drop_session_column(path, "last_read_at")
    owner = _owned(path.parent)
    try:
        web_server_lifecycle._eager_reconcile_own_session_db()
    finally:
        owner.close()
    assert "last_read_at" not in _session_columns(path)

    web_server_lifecycle._eager_reconcile_own_session_db()
    assert "last_read_at" in _session_columns(path)


# ── Independent review round 1 ──


def _lagging_store_with_history(home, rows=400):
    """A real store with enough history for a multi-page ``messages`` b-tree, one column dropped."""
    path = home / "state.db"
    with SessionDB(db_path=path) as db:
        db.create_session("kept", source="cli")
        for i in range(rows):
            db.append_message("kept", role="user", content=f"history {i} " * 40)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
        conn.execute("ALTER TABLE sessions DROP COLUMN last_read_at")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return path


def _file_state(path):
    """Main file plus WAL content (an empty sidecar a reader creates is not a write)."""
    wal = path.with_name(path.name + "-wal")
    return path.read_bytes(), wal.read_bytes() if wal.exists() else b""


def test_corrupt_non_root_page_plus_drift_is_not_written(tmp_path):
    """P1-1: the drift probe fails before any damaged page is read, so only an integrity check
    under the lock keeps browse from running ALTER TABLE on a corrupt file."""
    path = _lagging_store_with_history(_profile_home(tmp_path))
    with closing(sqlite3.connect(path)) as conn:
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        root = conn.execute("SELECT rootpage FROM sqlite_master WHERE name='messages'").fetchone()[0]
        pages = conn.execute("PRAGMA page_count").fetchone()[0]
    victim = max(root + 5, pages // 2)  # a messages leaf, never page 1 / sqlite_master
    with open(path, "r+b") as fh:
        fh.seek((victim - 1) * page_size + 8)
        fh.write(b"\xde\xad\xbe\xef" * ((page_size - 16) // 4))
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] != "ok"
    before = _file_state(path)

    with pytest.raises(HTTPException) as caught:
        _open_session_db_at_path(path, read_only=True).close()
    assert caught.value.status_code == 503
    assert caught.value.detail["error"] == "state_db_corrupt"
    assert _file_state(path) == before
    assert "last_read_at" not in _session_columns(path)


def test_concurrent_browses_without_a_gateway_all_succeed(tmp_path):
    """P2-1: sibling browses of one lagging, unowned store contend for gateway.lock with each
    other. None may be told a running gateway owns the store; each waits for the sibling."""
    import threading

    path = _lagging_store_with_history(_profile_home(tmp_path), rows=50)
    barrier = threading.Barrier(8)
    results = []

    def browse():
        barrier.wait()
        try:
            _open_session_db_at_path(path, read_only=True).close()
            results.append("200")
        except HTTPException as exc:
            results.append(f"{exc.status_code} {exc.detail}")

    threads = [threading.Thread(target=browse) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == ["200"] * 8, results
    assert "last_read_at" in _session_columns(path)


_PROC_BROWSE = r'''
import sys, time
from pathlib import Path
from fastapi import HTTPException
from hermes_cli.web_server_sessions import _open_session_db_at_path
go = float(sys.argv[2])
while time.time() < go:
    time.sleep(0.0005)
try:
    _open_session_db_at_path(Path(sys.argv[1]), read_only=True).close()
    print("RESULT 200")
except HTTPException as exc:
    print("RESULT", exc.status_code, exc.detail)
'''


@pytest.mark.platforms("linux", "macos")
def test_concurrent_browses_from_processes_all_succeed(tmp_path):
    import os
    import subprocess
    import sys
    import time

    path = _lagging_store_with_history(_profile_home(tmp_path), rows=50)
    go = time.time() + 2.0
    procs = [subprocess.Popen([sys.executable, "-c", _PROC_BROWSE, str(path), str(go)], env=os.environ,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(5)]
    out = [p.communicate(timeout=60)[0] for p in procs]
    results = [line for o in out for line in o.splitlines() if line.startswith("RESULT")]
    assert results == ["RESULT 200"] * 5, out


_PURPOSE_HOLDER = r'''
import sys, time
from pathlib import Path
from gateway.runtime_ownership import ProfileOwnership
purpose = sys.argv[3] or None
owner = ProfileOwnership()
owner.reserve([Path(sys.argv[1])], purpose=purpose)
print("held", flush=True)
time.sleep(float(sys.argv[2]))
'''


def _hold(home, seconds, purpose=""):
    import os
    import subprocess
    import sys
    proc = subprocess.Popen([sys.executable, "-c", _PURPOSE_HOLDER, str(home), str(seconds), purpose],
                            env=os.environ, stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


@pytest.mark.platforms("linux", "macos")
def test_stuck_maintenance_holder_gets_busy_retry_not_a_gateway_remedy(tmp_path):
    """P2-1: when the holder is not a gateway and the store is still behind after the busy budget,
    the refusal says busy/retry and never tells the user to restart a gateway that is not running."""
    from hermes_state_reconcile import BROWSE_RECONCILE_PURPOSE

    home = _profile_home(tmp_path)
    path = _lagging_store_with_history(home, rows=10)
    holder = _hold(home, 30, BROWSE_RECONCILE_PURPOSE)
    try:
        before = _file_state(path)
        with pytest.raises(HTTPException) as caught:
            _open_session_db_at_path(path, read_only=True)
    finally:
        holder.kill()
        holder.wait()
    detail = str(caught.value.detail)
    assert caught.value.status_code == 503
    assert "Retry" in detail
    assert "gateway" not in detail.lower()
    assert _file_state(path) == before


@pytest.mark.platforms("linux", "macos")
def test_gateway_boot_reserve_waits_out_a_browse_reconcile(tmp_path):
    """P3-1: a gateway booting while a browse reconcile holds gateway.lock waits for it (bounded)
    instead of exiting as though another gateway owned the profile. A real owner still refuses."""
    import time

    from gateway.run_bootstrap import _reserve_waiting_out_browse_reconcile
    from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership
    from hermes_state_reconcile import BROWSE_RECONCILE_PURPOSE

    home = _profile_home(tmp_path)
    holder = _hold(home, 1.0, BROWSE_RECONCILE_PURPOSE)
    booting = ProfileOwnership()
    try:
        started = time.monotonic()
        _reserve_waiting_out_browse_reconcile(booting, [home], wait_s=10)
        assert booting.owns(home)
        assert time.monotonic() - started < 9
    finally:
        booting.close()
        holder.kill()
        holder.wait()

    gateway = _hold(home, 30)  # an unmarked holder is a runtime owner: refuse at once
    contender = ProfileOwnership()
    try:
        started = time.monotonic()
        with pytest.raises(OwnershipConflict):
            _reserve_waiting_out_browse_reconcile(contender, [home], wait_s=10)
        assert time.monotonic() - started < 2
    finally:
        contender.close()
        gateway.kill()
        gateway.wait()


# ── Independent review round 2 ──


_RAW_FLOCK_HOLDER = r'''
import fcntl, os, sys, time
from pathlib import Path
home = Path(sys.argv[1]); home.mkdir(parents=True, exist_ok=True)
fh = open(home / "gateway.lock", "a+")
fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
fh.seek(0); fh.truncate(); fh.flush()  # flock held, record not yet published (reserve()'s window)
print("held", flush=True)
time.sleep(float(sys.argv[2]))
'''


@pytest.mark.platforms("linux", "macos")
def test_gateway_boot_waits_while_the_holder_has_not_published_its_record(tmp_path):
    """R2-1: reserve() takes the flock before it writes the purpose record. A boot that loses the
    flock in that window must retry (bounded), not abort as though a gateway owned the profile."""
    import os
    import subprocess
    import sys
    import time

    from gateway.run_bootstrap import _reserve_waiting_out_browse_reconcile
    from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership

    home = _profile_home(tmp_path)
    holder = subprocess.Popen([sys.executable, "-c", _RAW_FLOCK_HOLDER, str(home), "1.0"],
                              env=os.environ, stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    booting = ProfileOwnership()
    try:
        assert (home / "gateway.lock").read_text() == ""
        started = time.monotonic()
        _reserve_waiting_out_browse_reconcile(booting, [home], wait_s=10)
        assert booting.owns(home)
        assert 0.5 < time.monotonic() - started < 9
    finally:
        booting.close()
        holder.kill()
        holder.wait()

    # Still bounded: an unpublished holder that never lets go is refused once the wait expires.
    holder = subprocess.Popen([sys.executable, "-c", _RAW_FLOCK_HOLDER, str(home), "30"],
                              env=os.environ, stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    contender = ProfileOwnership()
    try:
        started = time.monotonic()
        with pytest.raises(OwnershipConflict):
            _reserve_waiting_out_browse_reconcile(contender, [home], wait_s=0.6)
        assert 0.5 < time.monotonic() - started < 3
    finally:
        contender.close()
        holder.kill()
        holder.wait()


_SLOW_BROWSE = r'''
import json, os, sqlite3, sys, time
from pathlib import Path
import hermes_cli.sqlite_safe_read as ssr
from fastapi import HTTPException
from hermes_cli.web_server_sessions import _open_session_db_at_path

phase = sys.argv[2]
original = ssr.connect_tracked

class SlowConnection:
    """A real sqlite connection whose VM runs slowly (stand-in for a cold multi-GB disk). The
    slowdown is installed here, whether or not the code under test sets its own progress handler,
    and chains to that handler when it does."""
    def __init__(self, conn):
        self._conn = conn
        self.set_progress_handler(None, 0)
    def __getattr__(self, name):
        return getattr(self._conn, name)
    def set_progress_handler(self, handler, n):
        def slow():
            time.sleep(0.01)
            return handler() if handler is not None else 0
        return self._conn.set_progress_handler(slow, 5)
    def close(self):
        return self._conn.close()

def holding_lock():
    """Only connections opened while THIS process holds gateway.lock are slowed."""
    lock = Path(sys.argv[1]).with_name("gateway.lock")
    try:
        return json.loads(lock.read_text()).get("pid") == os.getpid()
    except (OSError, ValueError):
        return False

def slowed(path, *a, **kw):
    conn = original(path, *a, **kw)
    read_only = "mode=ro" in str(path)
    if holding_lock() and (phase == "integrity") == read_only:
        print("slowed", flush=True)  # this process holds gateway.lock and the slow phase begins
        return SlowConnection(conn)
    return conn

ssr.connect_tracked = slowed
started = time.monotonic()
print("started", flush=True)
try:
    _open_session_db_at_path(Path(sys.argv[1]), read_only=True).close()
    print("RESULT 200", time.monotonic() - started, flush=True)
except HTTPException as exc:
    print("RESULT", exc.status_code, time.monotonic() - started, exc.detail, flush=True)
'''


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("phase", ["ddl"])
def test_slow_browse_reconcile_gives_up_inside_its_budget_and_boot_waits_it_out(tmp_path, phase):
    """R2-2: the busy timeout bounds lock contention, not SQL execution. A slow DDL must be
    interrupted at the browse budget (rolled back, retryable 503), and gateway boot's wait must
    outlast that budget, so a slow browse never turns into a startup refusal. (The integrity scan
    no longer runs under the lock: see test_gateway_boot_never_waits_on_a_slow_integrity_scan.)"""
    import os
    import subprocess
    import sys
    import time

    import gateway.run_bootstrap as bootstrap
    import hermes_state_reconcile as reconcile
    from gateway.runtime_ownership import ProfileOwnership

    budget_s = getattr(reconcile, "BROWSE_RECONCILE_BUDGET_S", 2.0)
    wait_s = getattr(bootstrap, "_browse_reconcile_reserve_wait_s", lambda: 5.0)()
    assert wait_s > budget_s + 1
    home = _profile_home(tmp_path)
    path = _lagging_store_with_history(home, rows=400)
    before = _file_state(path)
    browse = subprocess.Popen([sys.executable, "-c", _SLOW_BROWSE, str(path), phase], env=os.environ,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert browse.stdout.readline().strip() == "started"
    assert browse.stdout.readline().strip() == "slowed", "the slow phase never ran under the lock"
    booting = ProfileOwnership()
    try:
        bootstrap._reserve_waiting_out_browse_reconcile(booting, [home])  # default wait: must not raise
        assert booting.owns(home)
    finally:
        booting.close()
    out = browse.communicate(timeout=60)[0]
    result = next(line for line in out.splitlines() if line.startswith("RESULT"))
    _, status, elapsed, *detail = result.split(" ", 3)
    assert status == "503", out
    assert float(elapsed) < budget_s + 2, out
    detail = " ".join(detail)
    assert "nothing was changed" in detail
    assert "`hermes -p scribe sessions repair`" in detail and "gateway" not in detail.lower()
    assert _file_state(path) == before
    assert "last_read_at" not in _session_columns(path)


def test_equal_cardinality_corrupt_index_plus_drift_is_not_written(tmp_path):
    """R2-3: quick_check does not compare index keys with the table, so two equal-size indexes
    with swapped root pages pass it. Browse must still refuse as corrupt and write nothing."""
    path = _lagging_store_with_history(_profile_home(tmp_path), rows=10)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE probe_other (v TEXT)")
        conn.execute("CREATE INDEX probe_other_index ON probe_other(v)")
        conn.execute("INSERT INTO probe_other(v) VALUES ('wrong-source')")
        conn.commit()
        target = conn.execute("SELECT rootpage FROM sqlite_master WHERE name='idx_sessions_source'").fetchone()[0]
        other = conn.execute("SELECT rootpage FROM sqlite_master WHERE name='probe_other_index'").fetchone()[0]
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute("UPDATE sqlite_master SET rootpage=? WHERE name='idx_sessions_source'", (other,))
        conn.execute("UPDATE sqlite_master SET rootpage=? WHERE name='probe_other_index'", (target,))
        conn.execute("PRAGMA writable_schema=OFF")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchall() == [("ok",)]
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
    before = _file_state(path)

    with pytest.raises(HTTPException) as caught:
        _open_session_db_at_path(path, read_only=True).close()
    assert caught.value.status_code == 503
    assert caught.value.detail["error"] == "state_db_corrupt"
    assert _file_state(path) == before
    assert "last_read_at" not in _session_columns(path)


def test_failed_column_reconcile_rolls_back_every_addition(tmp_path):
    """R2-4: the shared column reconciler logs and continues on a non-lock error. A populated
    table missing a NOT NULL column cannot be caught up additively; browse must refuse without
    committing the other columns it could add (no partial upgrade without a backup)."""
    path = _lagging_store_with_history(_profile_home(tmp_path), rows=10)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("INSERT INTO system_prompts(hash, prompt) VALUES ('abc', 'retained prompt')")
        conn.execute("ALTER TABLE system_prompts DROP COLUMN prompt")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = _file_state(path)

    with pytest.raises(HTTPException) as caught:
        _open_session_db_at_path(path, read_only=True).close()
    assert caught.value.status_code == 503
    assert "system_prompts.prompt" in str(caught.value.detail)
    assert "nothing was changed" in str(caught.value.detail)
    assert _file_state(path) == before
    assert "last_read_at" not in _session_columns(path)


# ── Maintainer review round 3 ──


def test_integrity_scan_runs_before_the_lock_is_taken(tmp_path, monkeypatch):
    """alt-glitch C1 / pastels m3: the full integrity scan is read-only and scales with file size.
    It must run BEFORE gateway.lock is reserved, so a booting gateway never waits on it."""
    import hermes_state_reconcile as reconcile
    from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership

    home = _profile_home(tmp_path)
    path = _lagging_store_with_history(home, rows=10)
    original = reconcile._integrity_problem
    lock_free_during_scan = []

    def spy(p, budget):
        other = ProfileOwnership()  # its own open file description: conflicts with any holder
        try:
            other.reserve([home])
            lock_free_during_scan.append(True)
        except OwnershipConflict:
            lock_free_during_scan.append(False)
        finally:
            other.close()
        return original(p, budget)

    monkeypatch.setattr(reconcile, "_integrity_problem", spy)
    _open_session_db_at_path(path, read_only=True).close()
    assert lock_free_during_scan == [True]
    assert "last_read_at" in _session_columns(path)


_SLOW_SCAN_BROWSE = r'''
import sys, time
from pathlib import Path
import hermes_cli.sqlite_safe_read as ssr
from fastapi import HTTPException
from hermes_cli.web_server_sessions import _open_session_db_at_path

original = ssr.connect_tracked

class SlowScan:
    """A real read-only connection whose integrity scan runs slowly (a cold multi-GB store)."""
    def __init__(self, conn):
        self._conn = conn
    def __getattr__(self, name):
        return getattr(self._conn, name)
    def execute(self, sql, *a):
        if sql.startswith("PRAGMA integrity_check"):
            time.sleep(float(sys.argv[2]))
        return self._conn.execute(sql, *a)
    def close(self):
        return self._conn.close()

def slowed(path, *a, **kw):
    conn = original(path, *a, **kw)
    return SlowScan(conn) if "mode=ro" in str(path) else conn

ssr.connect_tracked = slowed
print("started", flush=True)
try:
    _open_session_db_at_path(Path(sys.argv[1]), read_only=True).close()
    print("RESULT 200", flush=True)
except HTTPException as exc:
    print("RESULT", exc.status_code, exc.detail, flush=True)
'''


@pytest.mark.platforms("linux", "macos")
def test_gateway_boot_never_waits_on_a_slow_integrity_scan(tmp_path):
    """alt-glitch C1: a browse scanning a large lagging store must not hold gateway.lock. A gateway
    booting mid-scan takes the lock at once; the browse then sees the owner and writes nothing."""
    import os
    import subprocess
    import sys
    import time

    from gateway.runtime_ownership import ProfileOwnership

    home = _profile_home(tmp_path)
    path = _lagging_store_with_history(home, rows=50)
    before = _file_state(path)
    browse = subprocess.Popen([sys.executable, "-c", _SLOW_SCAN_BROWSE, str(path), "4"], env=os.environ,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert browse.stdout.readline().strip() == "started"
    time.sleep(1.5)  # health: allow HX013 -- the child scan is a fixed 4 s stand-in; boot must land mid-scan
    booting = ProfileOwnership()
    try:
        started = time.monotonic()
        booting.reserve([home])  # no boot wait: the lock must simply be free
        took = time.monotonic() - started
        out = browse.communicate(timeout=60)[0]
    finally:
        booting.close()
    assert took < 0.5
    result = next(line for line in out.splitlines() if line.startswith("RESULT"))
    assert result.startswith("RESULT 503"), out
    assert "running gateway owns it" in result
    assert _file_state(path) == before


class _SlowProgress:
    """A real sqlite connection whose VM runs slowly, chaining to the code's own progress handler."""

    def __init__(self, conn):
        self._conn = conn
        self.set_progress_handler(None, 0)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def set_progress_handler(self, handler, n):
        import time

        def slow():
            time.sleep(0.01)  # health: allow HX013 -- the slowed VM IS the behaviour under test (cold disk)
            return handler() if handler is not None else 0
        return self._conn.set_progress_handler(slow, 5)

    def close(self):
        return self._conn.close()


def test_slow_integrity_scan_is_bounded_and_not_repeated_on_every_poll(tmp_path, monkeypatch):
    """alt-glitch C1: the pre-lock scan still has a deadline (retryable 503, nothing written), and a
    sidebar polling the same unchanged store does not rescan the whole file on every request."""
    import hermes_cli.sqlite_safe_read as ssr
    import hermes_state_reconcile as reconcile

    path = _lagging_store_with_history(_profile_home(tmp_path), rows=400)
    before = _file_state(path)
    original = ssr.connect_tracked
    monkeypatch.setattr(ssr, "connect_tracked", lambda p, *a, **kw: (
        _SlowProgress(original(p, *a, **kw)) if "mode=ro" in str(p) else original(p, *a, **kw)))
    monkeypatch.setattr(reconcile, "BROWSE_INTEGRITY_BUDGET_S", 0.3)
    scans = []
    real_scan = reconcile._integrity_problem
    monkeypatch.setattr(reconcile, "_integrity_problem", lambda p, b: scans.append(1) or real_scan(p, b))
    for _ in range(3):
        with pytest.raises(HTTPException) as caught:
            _open_session_db_at_path(path, read_only=True)
        assert caught.value.status_code == 503
        assert "nothing was changed" in caught.value.detail
    assert scans == [1]
    assert _file_state(path) == before


@pytest.mark.platforms("linux", "macos")
def test_unopenable_lock_on_lagging_store_is_a_named_503(tmp_path):
    """pastels m1: a gateway.lock this user cannot open (left by a root-run gateway, #42685) raised
    PermissionError(foreign_stale_lock) straight out of the browse open: an unhandled 500 on
    /api/sessions and /messages?profile= where main answered 503. It must be a named 503 carrying
    the recovery text, with nothing written."""
    import os

    if os.geteuid() == 0:
        pytest.skip("root opens any file")
    home = _profile_home(tmp_path)
    path = _lagging_store_with_history(home, rows=10)
    lock = home / "gateway.lock"
    lock.write_text("")
    lock.chmod(0)
    before = _file_state(path)
    try:
        with pytest.raises(HTTPException) as caught:
            _open_session_db_at_path(path, read_only=True)
    finally:
        lock.chmod(0o600)
    assert caught.value.status_code == 503
    detail = str(caught.value.detail)
    assert "foreign_stale_lock" in detail and "nothing was changed" in detail
    assert _file_state(path) == before


@pytest.mark.platforms("linux", "macos")
def test_boot_refuses_at_once_when_another_home_has_a_real_owner(tmp_path):
    """pastels m2: a multiplexer boot reserving profile A (owned by a standalone gateway) and profile B
    (mid-browse) must refuse at once on A, not wait out B's browse first. Homes the boot did take
    while deciding are released again (all-or-nothing)."""
    import time

    from gateway.run_bootstrap import _reserve_waiting_out_browse_reconcile
    from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership
    from hermes_state_reconcile import BROWSE_RECONCILE_PURPOSE

    owned_home = _profile_home(tmp_path, "alpha")
    browsed_home = _profile_home(tmp_path, "beta")
    free_home = _profile_home(tmp_path, "gamma")
    gateway = _hold(owned_home, 30)
    browse = _hold(browsed_home, 30, BROWSE_RECONCILE_PURPOSE)
    booting = ProfileOwnership()
    try:
        started = time.monotonic()
        with pytest.raises(OwnershipConflict):
            _reserve_waiting_out_browse_reconcile(booting, [free_home, owned_home, browsed_home], wait_s=10)
        assert time.monotonic() - started < 2, "refusal waited for an unrelated browse holder"
        assert booting.homes == [] or not any(booting.owns(h) for h in (free_home, owned_home, browsed_home))
    finally:
        booting.close()
        for proc in (gateway, browse):
            proc.kill()
            proc.wait()


def test_budget_refusal_on_unowned_store_names_repair_not_a_gateway(tmp_path, monkeypatch):
    """Round-2 decision: the 3 s browse budget stays. When it runs out on a store no gateway owns,
    the 503 must be actionable: name `hermes -p NAME sessions repair` (no time limit), say the
    history is untouched, and never tell a user with no gateway to restart one."""
    import hermes_cli.sqlite_safe_read as ssr
    import hermes_state_reconcile as reconcile

    assert reconcile.BROWSE_INTEGRITY_BUDGET_S == reconcile.BROWSE_RECONCILE_BUDGET_S == 3.0
    path = _lagging_store_with_history(_profile_home(tmp_path), rows=400)
    before = _file_state(path)
    original = ssr.connect_tracked
    monkeypatch.setattr(ssr, "connect_tracked", lambda p, *a, **kw: (
        _SlowProgress(original(p, *a, **kw)) if "mode=ro" in str(p) else original(p, *a, **kw)))
    monkeypatch.setattr(reconcile, "BROWSE_INTEGRITY_BUDGET_S", 0.3)
    with pytest.raises(HTTPException) as caught:
        _open_session_db_at_path(path, read_only=True)
    assert caught.value.status_code == 503
    detail = caught.value.detail
    assert "Run `hermes -p scribe sessions repair` to bring it up to date (it has no time limit)" in detail
    assert "Your history is untouched" in detail
    assert "gateway" not in detail.lower()
    assert _file_state(path) == before
    assert "last_read_at" not in _session_columns(path)

    # The named command really closes it: repair has no budget and catches the store up.
    monkeypatch.setattr(ssr, "connect_tracked", original)
    from hermes_state_reconcile import repair_declarative_schema
    import hermes_state_repair
    monkeypatch.setattr(hermes_state_repair, "_live_writer_holds_db", lambda _p: False)
    report = repair_declarative_schema(path, backup=False)
    assert report["repaired"], report
    reconcile._scan_too_slow.clear()
    _open_session_db_at_path(path, read_only=True).close()
    assert "last_read_at" in _session_columns(path)


_LEGACY_GATEWAY = r'''
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1])  # what a pre-v0.22.1 multiplexer does: no per-profile gateway.lock
conn.execute("SELECT count(*) FROM sessions").fetchone()
print("holding", flush=True)
time.sleep(60)
'''


@pytest.mark.platforms("linux")
def test_legacy_gateway_holding_the_store_without_a_lock_is_not_written(tmp_path):
    """alt-glitch W1: a v0.22.0 multiplex gateway keeps a named profile's state.db open without writing
    that profile's gateway.lock/gateway.pid. Browse must still treat it as the owner: no DDL, a named
    503. Plain readers (sibling browses, Desktop) are not owners, so only a holder identified as a
    Hermes gateway process refuses."""
    import os
    import subprocess
    import sys

    home = _profile_home(tmp_path)
    path = _lagging_store_with_history(home, rows=10)
    before = _file_state(path)
    legacy = subprocess.Popen(["hermes", "-c", _LEGACY_GATEWAY, str(path), "gateway", "run"],
                              executable=sys.executable, env=os.environ, stdout=subprocess.PIPE, text=True)
    try:
        assert legacy.stdout.readline().strip() == "holding"
        assert not (home / "gateway.lock").exists() or not (home / "gateway.lock").read_text()
        with pytest.raises(HTTPException) as caught:
            _open_session_db_at_path(path, read_only=True)
    finally:
        legacy.kill()
        legacy.wait()
    assert caught.value.status_code == 503
    assert "running gateway owns it" in caught.value.detail
    assert f"PID {legacy.pid}" in caught.value.detail
    assert _file_state(path) == before
    assert "last_read_at" not in _session_columns(path)


_MULTIPLEXER_LOCK = r'''
import fcntl, json, os, sys, time
fh = open(sys.argv[1], "a+")
fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
fh.seek(0); fh.truncate()
json.dump({"pid": os.getpid(), "kind": "hermes-gateway", "hermes_home": sys.argv[2]}, fh); fh.flush()
print("held", flush=True)
time.sleep(60)
'''


@pytest.mark.platforms("linux", "macos")
@pytest.mark.spawns_gateway_lookalike
def test_owned_refusal_names_the_owning_multiplexers_restart(tmp_path):
    """alt-glitch W2: when the default multiplexer owns a named profile's lock, `hermes -p NAME gateway
    restart` is refused ("Manage the host gateway instead"). Name the owner's own restart command,
    from the lock record's hermes_home."""
    import os
    import subprocess
    import sys

    home = _profile_home(tmp_path)
    path = _lagging_store_with_history(home, rows=10)
    host = subprocess.Popen([sys.executable, "-c", _MULTIPLEXER_LOCK, str(home / "gateway.lock"), str(tmp_path)],
                            env=os.environ, stdout=subprocess.PIPE, text=True)
    try:
        assert host.stdout.readline().strip() == "held"
        with pytest.raises(HTTPException) as caught:
            _open_session_db_at_path(path, read_only=True)
    finally:
        host.kill()
        host.wait()
    detail = caught.value.detail
    assert caught.value.status_code == 503 and "running gateway owns it" in detail
    assert "`hermes gateway restart`" in detail
    assert "-p scribe gateway restart" not in detail


@pytest.mark.platforms("linux", "macos")
def test_read_only_lagging_store_gets_a_permissions_message(tmp_path):
    """alt-glitch S5: a read-only state.db got the generic "restart the gateway" text. A `readonly
    database` error from the catch-up names the permission problem and the path; history untouched."""
    import os

    if os.geteuid() == 0:
        pytest.skip("root writes read-only files")
    path = _lagging_store_with_history(_profile_home(tmp_path), rows=10)
    before = _file_state(path)
    path.chmod(0o444)
    try:
        with pytest.raises(HTTPException) as caught:
            _open_session_db_at_path(path, read_only=True)
    finally:
        path.chmod(0o600)
    detail = str(caught.value.detail)
    assert caught.value.status_code == 503
    assert "not writable" in detail and str(path) in detail
    assert "nothing was changed" in detail
    assert "restart" not in detail.lower()
    assert _file_state(path) == before


def test_every_raw_schema_catch_up_handles_an_old_indexed_column(tmp_path):
    """alt-glitch S1: three hand-ordered catch-ups (reconcile_state_schema, the repair snapshot and the
    browse DDL) each encoded the columns -> SCHEMA_SQL -> columns order. One apply_declared_schema
    now owns it; every raw caller catches an old store (indexed column missing) up the same way."""
    import hermes_state_schema
    from hermes_state_schema import reconcile_state_schema

    path = _lagging_store_with_history(_profile_home(tmp_path), rows=5)
    with closing(sqlite3.connect(path)) as conn:
        reconcile_state_schema(conn)
        assert not conn.in_transaction
    assert "last_read_at" in _session_columns(path)
    assert hasattr(hermes_state_schema, "apply_declared_schema")

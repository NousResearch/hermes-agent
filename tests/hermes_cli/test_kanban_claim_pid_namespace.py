"""Tests: a worker PID is only trusted inside the claimer's PID namespace.

A claim lock is ``<hostname>:<pid>`` (``kanban_db._claimer_id``) and every
reclaim pass used the hostname alone to decide that a PID it could not see was a
dead worker. Hostname and PID namespace are independent in containers: Compose
services joined with ``network_mode: "service:x"`` share one hostname, and
unless ``pid:`` is set too each keeps its own PID namespace, where the same PID
number is an unrelated process.

Measured consequence: a dispatch from a sibling container found four ``running``
claims carrying its own hostname, saw none of the worker PIDs, and closed all
four runs as crashed "pid N not alive" while the workers were alive.

``tasks.claim_pidns`` records the namespace the claim was made in, and
``_claim_pid_checkable`` is the guard every PID probe and every SIGTERM now goes
through. A missing stored namespace fails closed when the local platform can
identify its own (Linux): the PID is unverifiable, so the claim is left to its
TTL. So does a failed lookup of our *own* namespace on Linux. The hostname-only
fallback survives only where the local platform has no namespace identity at
all (macOS/Windows, no ``/proc``). These tests patch
``kanban_db_pidns._local_pid_namespace`` — platform support and id as data —
so both sides are knowable and deterministic on any host.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_pidns as kbp
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    # 0 = no launch-window grace, so a recorded PID is probed on the first pass.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


@pytest.fixture
def local_pidns(monkeypatch):
    """Make this process' PID namespace identity readable and switchable.

    ``ns["id"]`` is what ``_pid_namespace_id()`` reports; assigning a different
    value mid-test is "the same board, read from a container with its own PID
    namespace". ``ns["supported"]`` is whether the platform has a namespace
    identity at all: ``id None`` with it ``True`` is a Linux lookup that
    failed, with it ``False`` it is macOS/Windows.
    """
    ns = {"id": "4026532111", "supported": True}
    monkeypatch.setattr(
        kbp, "_local_pid_namespace",
        lambda: kbp.LocalPidNamespace(ns["id"], ns["supported"]),
    )
    return ns


@pytest.fixture
def signals(monkeypatch):
    """Record every signal any reclaim path would deliver, and deliver none."""
    sent: list[tuple[int, int]] = []

    def _recorder(pid, sig):
        sent.append((int(pid), int(sig)))

    monkeypatch.setattr(kbd, "_kill_fn", lambda signal_fn=None: signal_fn or _recorder)
    return sent


def _host() -> str:
    return kb._claimer_id().split(":", 1)[0]


def _running_claim(conn, *, title: str = "work", pid: int = 4242,
                   max_runtime_seconds=None) -> str:
    """A ``running`` task claimed by this host/namespace with a recorded PID."""
    tid = kb.create_task(
        conn, title=title, assignee="w", max_runtime_seconds=max_runtime_seconds,
    )
    assert kb.claim_task(conn, tid, claimer=f"{_host()}:{title}") is not None
    kbd._set_worker_pid(conn, tid, pid)
    return tid


def _status(conn, tid: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"]


def _pidns_of(conn, tid: str):
    return conn.execute("SELECT claim_pidns FROM tasks WHERE id = ?", (tid,)).fetchone()["claim_pidns"]


def _events(conn, tid: str) -> list[str]:
    return [e.kind for e in kb.list_events(conn, tid)]


def _payload(conn, tid: str, kind: str) -> dict:
    for event in kb.list_events(conn, tid):
        if event.kind == kind:
            return event.payload or {}
    raise AssertionError(f"no {kind!r} event on {tid}: {_events(conn, tid)}")


def _expire_claim(conn, tid: str) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 600, tid),
        )


def _backdate_start(conn, tid: str, seconds: int) -> None:
    then = int(time.time()) - seconds
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (then, tid))
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ? AND ended_at IS NULL",
            (then, tid),
        )


# --- The namespace identity itself ----------------------------------------

def test_parse_pid_namespace_link_extracts_the_inode():
    """``readlink`` text in, inode out — a pure mapping, testable anywhere."""
    assert kbp._parse_pid_namespace_link("pid:[4026532534]") == "4026532534"
    assert kbp._parse_pid_namespace_link("pid:[1]") == "1"
    assert kbp._parse_pid_namespace_link("") is None
    assert kbp._parse_pid_namespace_link("pid:[]") is None
    assert kbp._parse_pid_namespace_link("not a namespace link") is None


@pytest.mark.linux_only
def test_pid_namespace_id_reads_proc_self_ns_pid(monkeypatch):
    monkeypatch.setattr(kbp, "_LOCAL_PID_NS", None)
    local = kbp._local_pid_namespace()
    assert local.supported is True
    assert local.id is not None and local.id.isdigit()
    assert kbp._pid_namespace_id() == local.id
    # Same process, same namespace — and the cached answer is stable.
    assert kbp._local_pid_namespace() is local


@pytest.mark.linux_only
def test_linux_namespace_lookup_failure_is_not_the_unsupported_platform(monkeypatch):
    """``/proc/self/ns/pid`` unreadable (restricted or unmounted ``/proc``).

    Linux *has* a namespace identity; failing to read ours is unknown
    authority, not "no such thing here" — so no hostname-only fallback, and
    the failure is not cached: the next call asks again.
    """
    monkeypatch.setattr(kbp, "_LOCAL_PID_NS", None)
    real_readlink = os.readlink
    proc = {"readable": False}

    def _readlink(path, *args, **kwargs):
        if str(path) == "/proc/self/ns/pid" and not proc["readable"]:
            raise PermissionError(13, "Permission denied", path)
        return real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", _readlink)
    assert kbp._local_pid_namespace() == kbp.LocalPidNamespace(None, True)
    assert kbp._pid_namespace_id() is None
    host_lock = f"{_host()}:1"
    assert kbp._claim_pid_checkable(host_lock, "4026532534") is False
    assert kbp._claim_pid_checkable(host_lock, None) is False

    proc["readable"] = True
    assert kbp._local_pid_namespace().id is not None


def _assert_no_namespace_identity_keeps_hostname_semantics(monkeypatch):
    """Off Linux there is no namespace identity, and that must stay harmless."""
    monkeypatch.setattr(kbp, "_LOCAL_PID_NS", None)
    assert kbp._local_pid_namespace() == kbp.LocalPidNamespace(None, False)
    assert kbp._pid_namespace_id() is None
    # No local identity => today's hostname-only semantics, unchanged.
    assert kbp._claim_pid_checkable(f"{_host()}:1", "4026532534") is True
    assert kbp._claim_pid_checkable(f"{_host()}:1", None) is True
    assert kbp._claim_pid_checkable("other-host:1", None) is False


@pytest.mark.macos_only
def test_pid_namespace_unsupported_on_macos(monkeypatch):
    _assert_no_namespace_identity_keeps_hostname_semantics(monkeypatch)


@pytest.mark.windows_only
def test_pid_namespace_unsupported_on_windows(monkeypatch):
    _assert_no_namespace_identity_keeps_hostname_semantics(monkeypatch)


def test_claim_pid_checkable_matrix(local_pidns):
    host_lock = f"{_host()}:123"
    assert kbp._claim_pid_checkable(host_lock, local_pidns["id"]) is True
    # Local namespace knowable + stored namespace missing (pre-upgrade row or
    # an older writer): fail closed — the PID is unverifiable.
    assert kbp._claim_pid_checkable(host_lock, None) is False       # rollout NULL
    assert kbp._claim_pid_checkable(host_lock, "4026531836") is False  # other namespace
    assert kbp._claim_pid_checkable("other-host:123", local_pidns["id"]) is False
    assert kbp._claim_pid_checkable(None, None) is False


def test_claim_pid_checkable_without_namespace_support_keeps_hostname_semantics(local_pidns):
    """macOS/Windows (no /proc): hostname-only fallback survives intact.

    When the local platform has no namespace identity at all, a NULL stored
    namespace must not freeze recovery — the old semantics are all we have.
    """
    host_lock = f"{_host()}:123"
    local_pidns["id"], local_pidns["supported"] = None, False
    assert kbp._claim_pid_checkable(host_lock, None) is True            # rollout NULL
    assert kbp._claim_pid_checkable(host_lock, "4026532534") is True    # any claim ns
    assert kbp._claim_pid_checkable("other-host:1", None) is False      # host still gates


def test_claim_pid_checkable_fails_closed_when_own_lookup_failed(local_pidns):
    """Linux, but ``/proc/self/ns/pid`` could not be read: unknown authority.

    Absence of our own provenance cannot become provenance. Nothing host-local
    is checkable until the lookup succeeds; the TTL path recovers the claims.
    """
    host_lock = f"{_host()}:123"
    local_pidns["id"] = None  # lookup failed; supported stays True
    assert kbp._claim_pid_checkable(host_lock, "4026532534") is False
    assert kbp._claim_pid_checkable(host_lock, None) is False
    assert kbp._claim_pid_checkable("other-host:1", "4026532534") is False


# --- Recording the namespace on the claim ---------------------------------

def test_claim_records_pid_namespace(conn, local_pidns):
    tid = kb.create_task(conn, title="stamped", assignee="w")
    assert kb.claim_task(conn, tid) is not None

    assert _pidns_of(conn, tid) == local_pidns["id"]
    assert _payload(conn, tid, "claimed")["pidns"] == local_pidns["id"]


# --- detect_crashed_workers ------------------------------------------------

def test_dead_pid_in_another_namespace_is_not_crashed(conn, local_pidns, monkeypatch, signals):
    """The measured incident: four live workers closed as crashed."""
    tid = _running_claim(conn, title="live-elsewhere")
    # Now read the board from a container that shares the hostname only.
    local_pidns["id"] = "4026531836"
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    assert kbd.detect_crashed_workers(conn) == []
    assert _status(conn, tid) == "running"
    assert "crashed" not in _events(conn, tid)
    run = conn.execute(
        "SELECT status, ended_at FROM task_runs WHERE task_id = ?", (tid,)
    ).fetchone()
    assert run["status"] == "running" and run["ended_at"] is None
    assert signals == []


def test_dead_pid_in_the_same_namespace_still_crashes(conn, local_pidns, monkeypatch, signals):
    """Control: the recovery this guard must not break."""
    tid = _running_claim(conn, title="really-dead")
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    assert kbd.detect_crashed_workers(conn) == [tid]
    assert _status(conn, tid) == "ready"
    assert "crashed" in _events(conn, tid)
    # The stamp is claim state: it leaves with the lock.
    assert _pidns_of(conn, tid) is None


def _null_pidns(conn, tid: str) -> None:
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_pidns = NULL WHERE id = ?", (tid,))


def test_migrated_null_pidns_claim_is_not_crashed_and_released_by_ttl(
    conn, local_pidns, monkeypatch, signals,
):
    """Rollout (a): a pre-upgrade row survives with ``claim_pidns = NULL``.

    The migration adds the column without backfilling, so every live claim
    from before the upgrade reads NULL. Read from a sibling container that
    shares our hostname: the crash sweep must neither probe the PID nor
    signal it (``/proc`` there is another world), and the row is released by
    the claim TTL instead.
    """
    tid = _running_claim(conn, title="migrated")
    _null_pidns(conn, tid)
    local_pidns["id"] = "4026531836"  # reader is a sibling container
    probes: list[int] = []
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: probes.append(int(pid)) or False)

    assert kbd.detect_crashed_workers(conn) == []
    assert _status(conn, tid) == "running"
    assert "crashed" not in _events(conn, tid)
    assert probes == [] and signals == []

    # TTL-safe release: once the claim expires, the sweep reclaims without
    # probing or signalling — an unknown namespace cannot certify the PID.
    _expire_claim(conn, tid)
    assert kb.release_stale_claims(conn) == 1
    assert _status(conn, tid) == "ready"
    assert probes == [] and signals == []
    payload = _payload(conn, tid, "reclaimed")
    assert payload["pid_checkable"] is False
    assert payload["termination_attempted"] is False


def _claim_like_an_older_binary(conn, tid: str, *, pid: int) -> None:
    """What a pre-``claim_pidns`` ``_claim_and_open_run`` writes: status,
    lock, expiry, start — and nothing about the namespace column it does not
    know exists."""
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'running', claim_lock = ?, claim_expires = ?, "
            "started_at = COALESCE(started_at, ?), worker_pid = ? "
            "WHERE id = ? AND claim_lock IS NULL",
            (f"{_host()}:older", now + 900, now, pid, tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, f"{_host()}:older", now + 900, pid, now),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, tid))


def test_older_writer_null_pidns_claim_is_never_signalled(
    conn, local_pidns, monkeypatch, signals,
):
    """Rollout (b): an older CLI keeps authoring NULL claims post-migration.

    The real sequence, not a hand-nulled column: this binary claims and
    releases the task (the release must clear ``claim_pidns`` with the lock),
    then an older binary sharing the board claims it without touching the
    column. Same hostname, even the same namespace as the reader — but the
    claim does not say so, and hostname is not authority across the
    mixed-version boundary. The PID must not be probed or signalled; an
    operator reclaim still releases the claim, it just cannot certify the kill.
    """
    probes: list[int] = []
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: probes.append(int(pid)) or False)
    tid = _running_claim(conn, title="older-writer", pid=1111)
    assert _pidns_of(conn, tid) == local_pidns["id"]
    assert kb.reclaim_task(conn, tid, reason="hand back") is True
    # Same namespace, stamped: that kill is legitimate and not what is under test.
    assert signals == [(1111, int(signal.SIGTERM))]
    assert _pidns_of(conn, tid) is None, "release must not leave a stale namespace behind"
    signals.clear()
    probes.clear()

    _claim_like_an_older_binary(conn, tid, pid=4242)
    assert _pidns_of(conn, tid) is None

    assert kbd.detect_crashed_workers(conn) == []
    assert _status(conn, tid) == "running"
    assert probes == [] and signals == []

    # Operator reclaim releases the claim either way, but must not SIGTERM a
    # PID number it cannot verify.
    assert kb.reclaim_task(conn, tid, reason="operator asked") is True
    assert _status(conn, tid) == "ready"
    assert probes == [] and signals == []


def test_failed_local_namespace_lookup_neither_probes_nor_signals(
    conn, local_pidns, monkeypatch, signals,
):
    """A Linux reader whose own ``/proc/self/ns/pid`` lookup failed.

    The claim is fully stamped; it is *our* provenance that is missing. That
    must not re-open hostname-only trust: no probe, no signal, TTL release.
    """
    tid = _running_claim(conn, title="unreadable-proc")
    local_pidns["id"] = None  # supported stays True: Linux, lookup failed
    probes: list[int] = []
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: probes.append(int(pid)) or False)

    assert kbd.detect_crashed_workers(conn) == []
    assert _status(conn, tid) == "running"
    assert probes == [] and signals == []

    _expire_claim(conn, tid)
    assert kb.release_stale_claims(conn) == 1
    assert _status(conn, tid) == "ready"
    assert probes == [] and signals == []
    assert _payload(conn, tid, "reclaimed")["pid_checkable"] is False


# --- release_stale_claims --------------------------------------------------

def test_stale_claim_from_another_namespace_is_released_at_ttl_without_signals(
    conn, local_pidns, monkeypatch, signals,
):
    """Not extended as "pid alive", not signalled — released by the TTL."""
    tid = _running_claim(conn, title="foreign-stale")
    _expire_claim(conn, tid)
    local_pidns["id"] = "4026531836"
    # A PID number that happens to exist in THIS namespace says nothing about
    # the worker, so it must neither extend the claim nor be killed.
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: True)

    assert kb.release_stale_claims(conn) == 1
    assert _status(conn, tid) == "ready"
    assert "claim_extended" not in _events(conn, tid)
    assert signals == []
    payload = _payload(conn, tid, "reclaimed")
    assert payload["host_local"] is True
    assert payload["pid_checkable"] is False


def test_reclaim_payload_reports_the_claim_not_the_kill(
    conn, local_pidns, monkeypatch, signals,
):
    """``host_local`` / ``pid_checkable`` describe the claim, not the signal.

    The termination report carries keys of the same name meaning "did we get far
    enough to signal", so a merge in the wrong order made this payload lie.
    """
    tid = _running_claim(conn, title="same-ns-dead")
    _expire_claim(conn, tid)
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    assert kb.release_stale_claims(conn) == 1
    payload = _payload(conn, tid, "reclaimed")
    assert payload["host_local"] is True
    assert payload["pid_checkable"] is True
    # The termination keys still report the signal that did happen.
    assert payload["termination_attempted"] is True
    assert signals == [(4242, int(signal.SIGTERM))]
    assert _pidns_of(conn, tid) is None


def test_reclaim_payload_without_a_worker_pid_still_reports_the_claim(
    conn, local_pidns, signals,
):
    """No PID to signal must not read as "this claim was not ours"."""
    tid = _running_claim(conn, title="no-pid")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = NULL WHERE id = ?", (tid,))
    _expire_claim(conn, tid)

    assert kb.release_stale_claims(conn) == 1
    payload = _payload(conn, tid, "reclaimed")
    assert payload["host_local"] is True
    assert payload["pid_checkable"] is True
    assert payload["worker_pid"] is None
    assert payload["termination_attempted"] is False
    assert signals == []


def test_stale_claim_in_the_same_namespace_is_extended_while_alive(
    conn, local_pidns, monkeypatch, signals,
):
    """Control: a live local worker still gets its claim extended, not reclaimed."""
    tid = _running_claim(conn, title="local-slow")
    _expire_claim(conn, tid)
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: True)

    assert kb.release_stale_claims(conn) == 0
    assert _status(conn, tid) == "running"
    assert "claim_extended" in _events(conn, tid)
    assert signals == []


# --- enforce_max_runtime / detect_stale_running ---------------------------

def test_max_runtime_does_not_signal_another_namespace(conn, local_pidns, signals):
    tid = _running_claim(conn, title="overrun", max_runtime_seconds=1)
    _backdate_start(conn, tid, 600)
    local_pidns["id"] = "4026531836"

    assert kbd.enforce_max_runtime(conn) == []
    assert _status(conn, tid) == "running"
    assert signals == []


def test_max_runtime_still_kills_in_the_same_namespace(conn, local_pidns, monkeypatch, signals):
    tid = _running_claim(conn, title="overrun-local", max_runtime_seconds=1)
    _backdate_start(conn, tid, 600)
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    assert kbd.enforce_max_runtime(conn) == [tid]
    assert signals == [(4242, int(signal.SIGTERM))]


def test_stale_running_reclaims_but_never_signals_another_namespace(
    conn, local_pidns, signals,
):
    """The heartbeat sweep is namespace-independent; only its kill is withheld."""
    tid = _running_claim(conn, title="no-heartbeat")
    _backdate_start(conn, tid, 7200)
    local_pidns["id"] = "4026531836"

    assert kbd.detect_stale_running(conn, stale_timeout_seconds=60) == [tid]
    assert _status(conn, tid) == "ready"
    assert signals == []


# --- Operator surfaces: hermes kanban reclaim / dashboard -----------------

def test_operator_reclaim_releases_without_signalling_another_namespace(
    conn, local_pidns, signals,
):
    tid = _running_claim(conn, title="operator")
    local_pidns["id"] = "4026531836"

    assert kb.reclaim_task(conn, tid, reason="operator asked") is True
    assert _status(conn, tid) == "ready"
    assert signals == []


def test_operator_reclaim_still_kills_its_own_worker(conn, local_pidns, monkeypatch, signals):
    tid = _running_claim(conn, title="operator-local")
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    assert kb.reclaim_task(conn, tid, reason="operator asked") is True
    assert signals == [(4242, int(signal.SIGTERM))]


def _load_dashboard_plugin():
    """Import plugins/kanban/dashboard/plugin_api.py as a module."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_pidns_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dashboard_status_change_does_not_signal_another_namespace(
    conn, local_pidns, signals,
):
    """Dragging a card out of Running from the dashboard container.

    The dashboard usually runs beside the gateway, not in it (its own PID
    namespace), so the ``worker_pid`` it reads is another process' number here.
    """
    plugin_api = _load_dashboard_plugin()
    tid = _running_claim(conn, title="dragged")
    local_pidns["id"] = "4026531836"

    assert plugin_api._set_status_direct(conn, tid, "ready") is True
    assert _status(conn, tid) == "ready"
    assert signals == []
    assert _pidns_of(conn, tid) is None, "release must not leave a stale namespace behind"


def test_dashboard_status_change_still_kills_its_own_worker(
    conn, local_pidns, monkeypatch, signals,
):
    plugin_api = _load_dashboard_plugin()
    tid = _running_claim(conn, title="dragged-local")
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    assert plugin_api._set_status_direct(conn, tid, "ready") is True
    assert signals == [(4242, int(signal.SIGTERM))]
    assert _pidns_of(conn, tid) is None


# --- Schema migration ------------------------------------------------------

def test_migration_adds_claim_pidns_to_a_legacy_board(tmp_path, kanban_home, local_pidns):
    """A pre-column board gains ``claim_pidns`` and keeps working."""
    db_path = tmp_path / "legacy-kanban.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    legacy.execute(
        """
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
        """
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old board task', 'ready', 1)"
    )
    legacy.commit()
    legacy.close()

    with kbc.connect(db_path) as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(tasks)")}
        assert "claim_pidns" in columns
        # And the migrated row is claimable, stamping the namespace as usual.
        assert kb.claim_task(migrated, "legacy") is not None
        assert migrated.execute(
            "SELECT claim_pidns FROM tasks WHERE id = 'legacy'"
        ).fetchone()["claim_pidns"] == local_pidns["id"]

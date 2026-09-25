"""Worker termination must take out the worker's DESCENDANTS too.

Dispatcher workers are spawned with ``start_new_session=True``, so each
worker leads its own session/process GROUP. Signalling only the worker pid
leaves its children (the observed ``gtimeout`` -> ``pi`` orphan chain)
running against a task that has already been handed to a new worker.

These tests cover both termination paths (reclaim and max-runtime timeout)
with REAL subprocesses, plus the load-bearing safety property: we only ever
group-signal a pid that is ITSELF the group leader — a non-leader pid gets
a single-pid signal, because its group belongs to somebody else.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


pytestmark = [
    pytest.mark.skipif(
        not hasattr(os, "killpg"), reason="POSIX process groups required"
    ),
    # These tests drive REAL worker trees and must deliver REAL signals; the
    # leaked descendants are reparented to init the moment the worker dies,
    # which puts them outside the test subtree the live-system guard allows.
    # Use the supported bypass marker instead of swallowing teardown errors.
    pytest.mark.live_system_guard_bypass,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alive(pid: int) -> bool:
    return kb._pid_alive(pid)


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


class _Tree:
    """A real worker process group: leader shell + a sleeping grandchild."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["/bin/sh", "-c", "sleep 300 & echo $! ; sleep 300"],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert self.proc.stdout is not None
        self.child_pid = int(self.proc.stdout.readline().strip())
        self.pid = self.proc.pid
        # Precondition: the worker really is its own group leader, and the
        # grandchild inherited that group without leading it.
        assert os.getpgid(self.pid) == self.pid
        assert os.getpgid(self.child_pid) == self.pid
        assert _alive(self.pid) and _alive(self.child_pid)

    def cleanup(self) -> None:
        for pid in (self.child_pid, self.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Cleanup must ASSERT the tree is gone rather than swallow failures.
        for pid in (self.pid, self.child_pid):
            assert _wait_gone(pid, timeout=5), f"teardown failed to reap {pid}"


@pytest.fixture
def tree():
    t = _Tree()
    try:
        yield t
    finally:
        t.cleanup()


@pytest.fixture
def bystander():
    """A process spawned OUTSIDE the worker's group. Must always survive."""
    proc = subprocess.Popen(["sleep", "300"])
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=5)


class _JobControlTree:
    """The REAL worker topology: a job-control shell whose ``gtimeout``-
    wrapped grandchild sits in a SEPARATE process group, same session.

    Measured live (production chain shell -> gtimeout -> pi):

        worker 80761 pgid 80761 sid 80761
          desc 80762 ppid 80761 pgid 80762 sid 80761
          desc 80763 ppid 80762 pgid 80762 sid 80761
    """

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [
                "/bin/sh",
                "-c",
                "set -m; sh -c 'sleep 300' & echo $! ; wait",
            ],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert self.proc.stdout is not None
        self.pid = self.proc.pid
        job_pid = int(self.proc.stdout.readline().strip())
        assert os.getsid(self.pid) == self.pid, "worker must lead its session"

        # Collect the live descendants of the worker.
        deadline = time.time() + 5.0
        self.descendants: list[int] = []
        while time.time() < deadline:
            descs = _descendants_of(self.pid)
            if len(descs) >= 2:
                self.descendants = descs
                break
            time.sleep(0.05)
        else:  # pragma: no cover - environment failure, must FAIL not skip
            self.descendants = _descendants_of(self.pid)

        assert job_pid in self.descendants, (
            f"job {job_pid} not among descendants {self.descendants}"
        )
        # PRECONDITION (hard failure, never a skip): at least one descendant
        # is in a DIFFERENT process group but the SAME session.
        separate = [
            d for d in self.descendants
            if os.getpgid(d) != os.getpgid(self.pid)
            and os.getsid(d) == self.pid
        ]
        assert separate, (
            "precondition not established: no descendant in a separate "
            f"process group. worker={self.pid} "
            f"pgid={os.getpgid(self.pid)} descendants="
            + repr([(d, os.getpgid(d), os.getsid(d)) for d in self.descendants])
        )
        self.separate_group_descendants = separate

    def all_pids(self) -> list[int]:
        return [self.pid, *self.descendants]

    def cleanup(self) -> None:
        for pid in reversed(self.all_pids()):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        for pid in self.all_pids():
            assert _wait_gone(pid, timeout=5), f"teardown failed to reap {pid}"


def _descendants_of(pid: int) -> list[int]:
    psutil = pytest.importorskip("psutil")
    try:
        return [p.pid for p in psutil.Process(pid).children(recursive=True)]
    except Exception:
        return []


@pytest.fixture
def jc_tree():
    t = _JobControlTree()
    try:
        yield t
    finally:
        t.cleanup()


@pytest.fixture
def session_bystander():
    """A bystander in its OWN separate session. Must always survive."""
    proc = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# _signal_worker_tree: routing + the safety property (injected fakes)
# ---------------------------------------------------------------------------


def test_group_leader_is_group_signalled_not_pid_signalled() -> None:
    kills: list[tuple[int, int]] = []
    pgkills: list[tuple[int, int]] = []
    kb._signal_worker_tree(
        4242, signal.SIGTERM,
        kill=lambda p, s: kills.append((p, s)),
        killpg=lambda p, s: pgkills.append((p, s)),
        getpgid=lambda p: p,  # leader: pgid == pid
        getsid=lambda _p: -1,  # not a session leader: no session sweep
    )
    assert pgkills == [(4242, signal.SIGTERM)]
    assert kills == []


def test_non_leader_pid_is_signalled_alone_and_its_group_is_spared() -> None:
    """Load-bearing safety property.

    A pid whose group leader is somebody ELSE must never be group-signalled:
    that group is not ours and killpg would take out unrelated processes.
    """
    kills: list[tuple[int, int]] = []
    pgkills: list[tuple[int, int]] = []
    kb._signal_worker_tree(
        4242, signal.SIGTERM,
        kill=lambda p, s: kills.append((p, s)),
        killpg=lambda p, s: pgkills.append((p, s)),
        getpgid=lambda _p: 999,  # foreign group leader
        getsid=lambda _p: -1,
    )
    assert kills == [(4242, signal.SIGTERM)]
    assert pgkills == [], "must not killpg a group we do not own"


def test_pgid_is_derived_not_assumed_equal_to_pid() -> None:
    """getpgid is consulted; ``pgid == pid`` is never assumed."""
    seen: list[int] = []
    kb._signal_worker_tree(
        77, signal.SIGKILL,
        kill=lambda *_a: None,
        killpg=lambda *_a: None,
        getpgid=lambda p: seen.append(p) or p,
        getsid=lambda _p: -1,
    )
    assert seen == [77]


@pytest.mark.parametrize("exc", [ProcessLookupError(), OSError(), AttributeError()])
def test_falls_back_to_pid_signal_when_getpgid_fails(exc) -> None:
    kills: list[tuple[int, int]] = []
    pgkills: list[tuple[int, int]] = []

    def _boom(_p):
        raise exc

    kb._signal_worker_tree(
        11, signal.SIGTERM,
        kill=lambda p, s: kills.append((p, s)),
        killpg=lambda p, s: pgkills.append((p, s)),
        getpgid=_boom,
        getsid=lambda _p: -1,
    )
    assert kills == [(11, signal.SIGTERM)]
    assert pgkills == []


def test_falls_back_to_pid_signal_when_killpg_is_unavailable() -> None:
    """Windows has neither getpgid nor killpg."""
    kills: list[tuple[int, int]] = []
    kb._signal_worker_tree(
        12, signal.SIGTERM,
        kill=lambda p, s: kills.append((p, s)),
        killpg=None,
        getpgid=None,
        getsid=None,
        scan_pids=(),
    )
    assert kills == [(12, signal.SIGTERM)]


# ---------------------------------------------------------------------------
# Session-sweep negative controls (injected fakes)
# ---------------------------------------------------------------------------


def test_non_session_leader_never_triggers_a_session_scan() -> None:
    """We only sweep a session we can prove we created.

    If the worker is not its own session leader, the session belongs to
    somebody else and enumerating/signalling it would be unrelated-process
    slaughter. The scan must not even be consulted.
    """
    consulted: list[str] = []

    def _scan():
        consulted.append("scanned")
        return [1, 2, 3]

    kills: list[tuple[int, int]] = []
    kb._signal_worker_tree(
        4242, signal.SIGTERM,
        kill=lambda p, s: kills.append((p, s)),
        killpg=lambda *_a: None,
        getpgid=lambda _p: 999,
        getsid=lambda _p: 111,  # session led by somebody else
        scan_pids=_scan,
    )
    assert consulted == [], "must not scan a session we do not own"
    assert kills == [(4242, signal.SIGTERM)]


def test_scanned_pid_whose_session_changed_is_skipped() -> None:
    """PID-reuse safety: re-verify the session id right before the kill."""
    worker = 500
    reused = 501
    stable = 502
    calls: dict[int, int] = {}

    def _getsid(p):
        p = int(p)
        calls[p] = calls.get(p, 0) + 1
        if p == worker:
            return worker
        if p == reused:
            # First lookup (scan) says ours; by the second (pre-kill
            # re-verify) the pid has been recycled by another session.
            return worker if calls[p] == 1 else 999
        if p == stable:
            return worker
        return 999

    kills: list[tuple[int, int]] = []
    kb._signal_worker_tree(
        worker, signal.SIGTERM,
        kill=lambda p, s: kills.append((int(p), s)),
        killpg=lambda *_a: None,
        getpgid=lambda p: int(p),
        getsid=_getsid,
        scan_pids=[reused, stable, 777],
    )
    signalled = [p for p, _s in kills]
    assert reused not in signalled, "recycled pid must not be signalled"
    assert stable in signalled
    assert 777 not in signalled


def test_separate_groups_inside_the_owned_session_are_group_signalled() -> None:
    """A `set -m` job leader inside our session gets a group signal too."""
    worker = 600
    job_leader = 601
    job_child = 602
    pgids = {worker: worker, job_leader: job_leader, job_child: job_leader}

    pgkills: list[tuple[int, int]] = []
    kb._signal_worker_tree(
        worker, signal.SIGTERM,
        kill=lambda *_a: None,
        killpg=lambda p, s: pgkills.append((int(p), s)),
        getpgid=lambda p: pgids.get(int(p), int(p)),
        getsid=lambda p: worker if int(p) in pgids else 999,
        scan_pids=[job_leader, job_child],
    )
    assert (job_leader, signal.SIGTERM) in pgkills, (
        "separate group inside the owned session must be group-signalled"
    )
    assert (worker, signal.SIGTERM) in pgkills


# ---------------------------------------------------------------------------
# Real processes: the safety property, live
# ---------------------------------------------------------------------------


def test_real_non_leader_pid_does_not_take_down_its_group(tree: _Tree) -> None:
    """Signalling the grandchild (a non-leader) kills only the grandchild."""
    kb._signal_worker_tree(tree.child_pid, signal.SIGKILL)
    assert _wait_gone(tree.child_pid)
    time.sleep(0.3)
    assert _alive(tree.pid), "the group leader must survive a non-leader signal"


# ---------------------------------------------------------------------------
# Reclaim path (_terminate_reclaimed_worker)
# ---------------------------------------------------------------------------


def test_reclaim_termination_kills_the_whole_worker_tree(
    tree: _Tree, bystander: subprocess.Popen
) -> None:
    info = kb._terminate_reclaimed_worker(tree.pid, kb._claimer_id())
    assert info["host_local"] is True
    assert info["termination_attempted"] is True

    assert _wait_gone(tree.pid), "worker must die"
    assert _wait_gone(tree.child_pid), (
        "descendant leaked: the orphan chain survived the reclaim"
    )
    assert info["terminated"] is True
    assert bystander.poll() is None, "unrelated process must survive a reclaim"


def test_reclaim_termination_ignores_non_host_local_claims(
    tree: _Tree,
) -> None:
    info = kb._terminate_reclaimed_worker(tree.pid, "some-other-host:1234")
    assert info["host_local"] is False
    assert info["termination_attempted"] is False
    time.sleep(0.3)
    assert _alive(tree.pid) and _alive(tree.child_pid)


def test_reclaim_termination_reports_terminated_for_a_dead_pid(
    tree: _Tree,
) -> None:
    tree.cleanup()
    assert _wait_gone(tree.pid)
    info = kb._terminate_reclaimed_worker(tree.pid, kb._claimer_id())
    assert info["termination_attempted"] is True
    assert info["terminated"] is True


# ---------------------------------------------------------------------------
# Timeout path (enforce_max_runtime)
# ---------------------------------------------------------------------------


def test_max_runtime_termination_kills_the_whole_worker_tree(
    kanban_home: Path, tree: _Tree, bystander: subprocess.Popen
) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="overrunning job", assignee="worker",
            max_runtime_seconds=1,
        )
        assert kb.claim_task(conn, tid) is not None
        kb._set_worker_pid(conn, tid, tree.pid)
        old = int(time.time()) - 300
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, tid))
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old, tid),
            )

        assert tid in kb.enforce_max_runtime(conn)

    assert _wait_gone(tree.pid), "worker must die on timeout"
    assert _wait_gone(tree.child_pid), (
        "descendant leaked: the orphan chain survived the timeout reap"
    )
    assert bystander.poll() is None, "unrelated process must survive a timeout reap"


def test_injected_signal_fn_intercepts_the_group_signal(
    kanban_home: Path,
) -> None:
    """The existing ``signal_fn`` test hook still sees every signal.

    Regression guard: if the group path bypassed the hook, a test using a
    live pid would fire a real killpg at the test runner's own group.
    """
    seen: list[tuple[int, int]] = []
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="hooked", assignee="worker", max_runtime_seconds=1,
        )
        assert kb.claim_task(conn, tid) is not None
        kb._set_worker_pid(conn, tid, os.getpid())
        old = int(time.time()) - 300
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, tid))
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old, tid),
            )
        orig_alive = kb._pid_alive
        kb._pid_alive = lambda _pid: False
        try:
            assert tid in kb.enforce_max_runtime(
                conn, signal_fn=lambda p, s: seen.append((p, s)),
            )
        finally:
            kb._pid_alive = orig_alive

    assert seen, "signal_fn must still be called"
    assert all(s == signal.SIGTERM for _p, s in seen)
    assert all(p in (os.getpid(), os.getpgid(os.getpid())) for p, _s in seen)


# ---------------------------------------------------------------------------
# REGRESSION: separate-process-group descendants (the measured leak)
#
# killpg(worker_pgid) does NOT reach a `set -m` job group. The session is the
# ownership invariant; these tests fail on the killpg-only implementation.
# ---------------------------------------------------------------------------


def test_reclaim_kills_separate_process_group_descendants(
    jc_tree: _JobControlTree, session_bystander: subprocess.Popen
) -> None:
    observed = [
        (p, os.getpgid(p), os.getsid(p)) for p in jc_tree.all_pids()
    ]
    print("PID RECEIPTS (reclaim):", observed)

    info = kb._terminate_reclaimed_worker(jc_tree.pid, kb._claimer_id())
    assert info["host_local"] is True
    assert info["termination_attempted"] is True

    assert _wait_gone(jc_tree.pid), "worker must die"
    leaked = [p for p in jc_tree.descendants if not _wait_gone(p)]
    assert leaked == [], (
        f"separate-process-group descendants leaked: {leaked} "
        f"(worker={jc_tree.pid}, separate group="
        f"{jc_tree.separate_group_descendants})"
    )
    assert info["terminated"] is True
    assert session_bystander.poll() is None, (
        "a process in its OWN session must survive the sweep"
    )


def test_max_runtime_kills_separate_process_group_descendants(
    kanban_home: Path,
    jc_tree: _JobControlTree,
    session_bystander: subprocess.Popen,
) -> None:
    observed = [
        (p, os.getpgid(p), os.getsid(p)) for p in jc_tree.all_pids()
    ]
    print("PID RECEIPTS (max_runtime):", observed)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="overrunning job", assignee="worker",
            max_runtime_seconds=1,
        )
        assert kb.claim_task(conn, tid) is not None
        kb._set_worker_pid(conn, tid, jc_tree.pid)
        old = int(time.time()) - 300
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, tid))
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old, tid),
            )
        assert tid in kb.enforce_max_runtime(conn)

    assert _wait_gone(jc_tree.pid), "worker must die on timeout"
    leaked = [p for p in jc_tree.descendants if not _wait_gone(p)]
    assert leaked == [], (
        f"separate-process-group descendants leaked on the timeout path: "
        f"{leaked}"
    )
    assert session_bystander.poll() is None, (
        "a process in its OWN session must survive the timeout reap"
    )


# ---------------------------------------------------------------------------
# REGRESSION: NEW-SESSION descendants (the measured session-sweep leak)
#
# Any descendant may call setsid()/start_new_session and leave the worker's
# session entirely. Node, zsh job control and MCP stdio servers all do this.
# Measured live against the session-only fix:
#
#     worker 90579 pgid 90579 sid 90579
#       desc 90580 ppid 90579 pgid 90580 sid 90580
#     LEAKED after the session-only sweep: [90580]
#
# The third, derived ownership source is process ANCESTRY.
# ---------------------------------------------------------------------------


class _NewSessionTree:
    """A worker whose grandchild lives in a BRAND NEW session."""

    _SCRIPT = (
        "import os,subprocess,time;"
        "subprocess.Popen(['sleep','120'], start_new_session=True);"
        "time.sleep(120)"
    )

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [
                "/bin/sh", "-c",
                f"exec {sys.executable} -c {self._SCRIPT!r}",
            ],
            start_new_session=True,
        )
        self.pid = self.proc.pid
        assert os.getsid(self.pid) == self.pid, "worker must lead its session"

        deadline = time.time() + 10.0
        self.descendants: list[int] = []
        self.new_session_descendants: list[int] = []
        while time.time() < deadline:
            descs = _descendants_of(self.pid)
            new_session = [
                d for d in descs
                if _safe_getsid(d) not in (None, self.pid)
            ]
            if new_session:
                self.descendants = descs
                self.new_session_descendants = new_session
                break
            time.sleep(0.05)
        else:  # pragma: no cover - environment failure, must FAIL not skip
            self.descendants = _descendants_of(self.pid)

        # PRECONDITION (hard failure, never a skip): at least one live
        # descendant is in a DIFFERENT session than the worker.
        assert self.new_session_descendants, (
            "precondition not established: no descendant left the worker's "
            f"session. worker={self.pid} sid={os.getsid(self.pid)} "
            "descendants="
            + repr([
                (d, _safe_getpgid(d), _safe_getsid(d))
                for d in _descendants_of(self.pid)
            ])
        )

    def all_pids(self) -> list[int]:
        return [self.pid, *self.descendants]

    def receipts(self) -> list[tuple]:
        return [
            (p, _safe_getpgid(p), _safe_getsid(p)) for p in self.all_pids()
        ]

    def cleanup(self) -> None:
        for pid in reversed(self.all_pids()):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        for pid in self.all_pids():
            assert _wait_gone(pid, timeout=5), f"teardown failed to reap {pid}"


def _safe_getsid(pid: int):
    try:
        return os.getsid(pid)
    except (ProcessLookupError, OSError):
        return None


def _safe_getpgid(pid: int):
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, OSError):
        return None


@pytest.fixture
def ns_tree():
    t = _NewSessionTree()
    try:
        yield t
    finally:
        t.cleanup()


@pytest.fixture
def new_session_bystander():
    """An unrelated process in its OWN new session. Must always survive."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=True,
    )
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_reclaim_kills_new_session_descendants(
    ns_tree: _NewSessionTree, new_session_bystander: subprocess.Popen
) -> None:
    """A. The reclaim path reaps a descendant that left the session."""
    print("PID RECEIPTS (reclaim, new session):", ns_tree.receipts())
    print("NEW-SESSION descendants:", ns_tree.new_session_descendants)

    info = kb._terminate_reclaimed_worker(ns_tree.pid, kb._claimer_id())
    assert info["host_local"] is True
    assert info["termination_attempted"] is True

    assert _wait_gone(ns_tree.pid), "worker must die"
    leaked = [p for p in ns_tree.new_session_descendants if not _wait_gone(p)]
    assert leaked == [], (
        f"NEW-SESSION descendants leaked: {leaked} (worker={ns_tree.pid})"
    )
    assert [p for p in ns_tree.descendants if not _wait_gone(p)] == []
    assert info["terminated"] is True
    assert new_session_bystander.poll() is None, (
        "an unrelated process in its own session must survive the sweep"
    )


def test_max_runtime_kills_new_session_descendants(
    kanban_home: Path,
    ns_tree: _NewSessionTree,
    new_session_bystander: subprocess.Popen,
) -> None:
    """B. Same regression on the enforce_max_runtime timeout path."""
    print("PID RECEIPTS (max_runtime, new session):", ns_tree.receipts())

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="overrunning job", assignee="worker",
            max_runtime_seconds=1,
        )
        assert kb.claim_task(conn, tid) is not None
        kb._set_worker_pid(conn, tid, ns_tree.pid)
        old = int(time.time()) - 300
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, tid))
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old, tid),
            )
        assert tid in kb.enforce_max_runtime(conn)

    assert _wait_gone(ns_tree.pid), "worker must die on timeout"
    leaked = [p for p in ns_tree.new_session_descendants if not _wait_gone(p)]
    assert leaked == [], (
        f"NEW-SESSION descendants leaked on the timeout path: {leaked}"
    )
    assert new_session_bystander.poll() is None, (
        "an unrelated process in its own session must survive the reap"
    )


def test_parent_first_death_still_reaps_the_reparented_orphan(
    ns_tree: _NewSessionTree, new_session_bystander: subprocess.Popen
) -> None:
    """C. PARENT-FIRST: the ancestry chain is gone, the snapshot is not.

    Once the worker dies its descendants are reparented to init(1), so a
    kill-time ancestry derivation finds NOTHING. The escalation pass must
    reuse the pre-death snapshot.
    """
    orphan = ns_tree.new_session_descendants[0]
    print("PID RECEIPTS (parent-first):", ns_tree.receipts())

    snapshot = kb._capture_worker_tree(ns_tree.pid)
    assert orphan in [p for p, _c, _s in snapshot["pids"]], (
        f"orphan {orphan} missing from the pre-death snapshot {snapshot}"
    )

    # Kill the worker FIRST. The orphan is reparented to init.
    os.kill(ns_tree.pid, signal.SIGKILL)
    assert _wait_gone(ns_tree.pid)
    try:
        ns_tree.proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    assert _alive(orphan), "orphan must still be alive before the sweep"

    # Proof the chain is gone: a fresh derivation finds no descendants.
    fresh = kb._capture_worker_tree(ns_tree.pid)
    assert [p for p, _c, _s in fresh["pids"] if p == orphan] == [], (
        "precondition not established: ancestry to the dead worker survived"
    )

    # The escalation pass reuses the pre-death snapshot. Signalling the dead
    # worker pid itself raises ProcessLookupError (the documented contract
    # callers already handle) — the snapshot sweep runs BEFORE that.
    with pytest.raises(ProcessLookupError):
        kb._signal_worker_tree(
            ns_tree.pid, signal.SIGKILL,
            kill=os.kill, killpg=os.killpg, snapshot=snapshot,
        )
    assert _wait_gone(orphan), (
        f"reparented orphan {orphan} leaked: the snapshot was not reused"
    )
    assert new_session_bystander.poll() is None


def test_snapshotted_pid_whose_create_time_changed_is_skipped() -> None:
    """D. PID-reuse safety on the snapshot (injected, no real pids)."""
    worker, recycled, stable = 700, 701, 702
    before = {
        worker: (1, 10.0), recycled: (worker, 20.0), stable: (worker, 30.0),
    }
    # After the worker's death both are reparented to init; ``recycled`` has
    # been recycled by an unrelated process (different create_time).
    after = {worker: (1, 10.0), recycled: (1, 99.0), stable: (1, 30.0)}

    snapshot = kb._capture_worker_tree(
        worker,
        getpgid=lambda p: int(p),
        getsid=lambda _p: -1,  # not a session leader: ancestry arm only
        scan_pids=(),
        proc_table=before,
    )
    owned = sorted(p for p, _c, _s in snapshot["pids"])
    assert owned == [recycled, stable]

    kills: list[tuple[int, int]] = []
    pgkills: list[tuple[int, int]] = []
    kb._signal_captured_tree(
        snapshot, signal.SIGTERM,
        kill=lambda p, s: kills.append((int(p), s)),
        killpg=lambda p, s: pgkills.append((int(p), s)),
        getsid=lambda _p: -1,
        proc_table=after,
    )
    signalled = [p for p, _s in kills]
    assert recycled not in signalled, "recycled pid must not be signalled"
    assert stable in signalled
    assert recycled not in [p for p, _s in pgkills]


def test_ancestry_sweep_spares_unrelated_processes() -> None:
    """E. Bystander negative control (injected).

    Neither an unrelated process in its own session nor one that merely
    shares a pgid/sid coincidence may be signalled.
    """
    worker, child = 800, 801
    unrelated_session = 900        # ppid 1, own session
    pgid_coincidence = 901         # unrelated, but pgid == worker's pid value
    table = {
        worker: (1, 1.0),
        child: (worker, 2.0),
        unrelated_session: (1, 3.0),
        pgid_coincidence: (1, 4.0),
    }
    pgids = {
        worker: worker, child: child,
        unrelated_session: unrelated_session,
        pgid_coincidence: worker,  # coincidental shared group id
    }
    sids = {
        worker: worker, child: child,
        unrelated_session: unrelated_session,
        pgid_coincidence: worker,  # coincidental shared session id
    }

    kills: list[tuple[int, int]] = []
    pgkills: list[tuple[int, int]] = []
    snapshot = kb._capture_worker_tree(
        worker,
        getpgid=lambda p: pgids.get(int(p), int(p)),
        # Worker does NOT lead its own session here, so the session arm is
        # disabled and the pgid/sid coincidences must not save the bystander.
        getsid=lambda p: sids.get(int(p), -1) if int(p) != worker else -1,
        scan_pids=[unrelated_session, pgid_coincidence],
        proc_table=table,
    )
    kb._signal_captured_tree(
        snapshot, signal.SIGTERM,
        kill=lambda p, s: kills.append((int(p), s)),
        killpg=lambda p, s: pgkills.append((int(p), s)),
        getsid=lambda p: sids.get(int(p), -1),
        proc_table=table,
    )
    signalled = [p for p, _s in kills]
    assert child in signalled, "the real descendant must be signalled"
    assert unrelated_session not in signalled
    assert pgid_coincidence not in signalled
    assert pgid_coincidence not in [p for p, _s in pgkills]
    assert unrelated_session not in [p for p, _s in pgkills]

"""Cross-process update mutual exclusion (``hermes_cli.update_lock``).

Three surfaces can start an update of one install tree: a terminal ``hermes
update``, the dashboard's Update button (which spawns that same command
detached), and the desktop's Update button (Tauri updater → install-mode
bootstrap on its failure screen). Before the shared lock, two of them could run
concurrently and rewrite source under a live interpreter — observed in the wild
as an installer ``git checkout`` rewinding the checkout ~9k commits while a
dashboard-spawned ``hermes update`` was mid-``npm install``, which then failed
against the rewound tree's manifests.

These exercise the real marker file against a temp home — no mocks — because
the contract that matters is what the Rust updater and the Electron gate see on
disk.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.update_lock import (
    HANDOFF_PID_ENV,
    UPDATE_MARKER_MAX_AGE_SECONDS,
    UpdateLock,
    describe_holder,
    read_live_update,
    update_marker_path,
)

# Repo root: the -I -S -B subprocesses insert it on sys.path to import the
# real hermes_cli without site-packages.
REPO_ROOT = Path(__file__).resolve().parents[2]

# A pid no live process owns. os.kill(pid, 0) must report it dead so a crashed
# updater can never wedge every future update. Deliberately larger than any
# platform's pid_t so it also covers the corrupt-marker path (OverflowError).
DEAD_PID = 4294967294


@pytest.fixture
def marker(tmp_path):
    return tmp_path / ".hermes-update-in-progress"


@pytest.fixture
def other_pid():
    """A live process that is not us: the stand-in for another updater (our own pid is ours)."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
    yield proc.pid
    proc.kill()
    proc.wait()


def _claim(marker, pid, started_at=None):
    marker.write_text(f"{pid}\n{int(time.time() if started_at is None else started_at)}\n", encoding="utf-8")


def test_marker_path_follows_process_hermes_home(tmp_path, monkeypatch):
    """The lock must land where the Rust updater and Electron gate look.

    All three resolve the *process* HERMES_HOME; a profile-scoped path would
    put the lock somewhere the other two owners never read.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert update_marker_path() == tmp_path / ".hermes-update-in-progress"


def test_acquire_writes_pid_and_start_time(marker):
    lock = UpdateLock(path=marker)

    assert lock.acquire() is True
    assert lock.acquired is True

    lines = marker.read_text(encoding="utf-8").splitlines()
    assert int(lines[0]) == os.getpid(), (
        "the Electron gate probes this pid for liveness"
    )
    assert int(lines[1]) == pytest.approx(time.time(), abs=5)
    assert len(lines) == 3, "wire format is pid + renewable timestamp + owner fingerprint"
    assert lines[2], "the fingerprint prevents a former owner releasing a successor's claim"


def test_atomic_claim_has_one_winner_and_renews_its_lease(marker, tmp_path):
    """Simultaneous contenders have one owner, whose lease stays fresh during long work."""
    barrier = tmp_path / "go"
    code = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
marker, barrier = Path(sys.argv[2]), Path(sys.argv[3])
while not barrier.exists(): time.sleep(.001)
lock = UpdateLock(path=marker, refresh_interval_seconds=.05)
won = lock.acquire()
if won:
    first = int(marker.read_text().splitlines()[1])
    time.sleep(1.2)
    renewed = int(marker.read_text().splitlines()[1])
    print(f'won:{first}:{renewed}', flush=True)
    lock.release()
else:
    print('lost', flush=True)
"""
    procs = [
        subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", code, str(REPO_ROOT), str(marker), str(barrier)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(2)
    ]
    barrier.touch()
    outputs = [proc.communicate(timeout=10) for proc in procs]

    winners = [out.strip() for out, _ in outputs if out.startswith("won:")]
    assert len(winners) == 1, outputs
    assert [out.strip() for out, _ in outputs].count("lost") == 1, outputs
    _, first, renewed = winners[0].split(":")
    assert int(renewed) > int(first), "the owner must renew before the stale ceiling"


def test_lease_renewal_never_exposes_an_empty_marker(marker):
    """Every reader deletes a marker it cannot parse, so a renewal must swap the whole file in:
    an in-place truncating write showed readers 0 bytes once per tick (#129926 review)."""
    lock = UpdateLock(path=marker, refresh_interval_seconds=0.002)
    assert lock.acquire()
    torn: list[str] = []
    deadline = time.monotonic() + 1.5
    try:
        while time.monotonic() < deadline:
            try:
                text = marker.read_text(encoding="utf-8-sig")
            except FileNotFoundError:
                torn.append("<missing>")
                continue
            lines = text.splitlines()
            if len(lines) < 3 or lines[0] != str(os.getpid()):
                torn.append(text)
        assert torn == [], f"{len(torn)} torn reads, e.g. {torn[:3]!r}"
        assert lock._refresh_thread is not None and lock._refresh_thread.is_alive()
    finally:
        lock.release()
    assert not marker.exists()
    assert not list(marker.parent.glob(f".{marker.name}.*")), "staged claim/renewal files left behind"


def test_a_vanished_marker_is_reclaimed_but_a_successor_is_left_alone(marker, other_pid):
    """A marker deleted under a running owner is claimed again, not abandoned (abandoning it let a
    second updater in); one rewritten by someone else ends the renewals and is never overwritten."""
    lock = UpdateLock(path=marker, refresh_interval_seconds=3600)  # ticks driven by hand below
    assert lock.acquire()
    try:
        fingerprint = marker.read_text().splitlines()[2]
        marker.unlink()
        assert lock._refresh_once() is True
        lines = marker.read_text().splitlines()
        assert lines[0] == str(os.getpid()) and lines[2] == fingerprint
        assert int(lines[1]) == pytest.approx(time.time(), abs=5)

        _claim(marker, other_pid)
        assert lock._refresh_once() is False
        assert marker.read_text().splitlines()[0] == str(other_pid)
    finally:
        lock.release()
    assert marker.read_text().splitlines()[0] == str(other_pid), "release removed a successor's claim"


def test_second_acquire_is_refused_while_the_first_is_live(marker, other_pid):
    """The bug: two updaters mutating one checkout at the same time."""
    _claim(marker, other_pid)

    second = UpdateLock(path=marker)
    assert second.acquire() is False
    assert second.holder is not None
    assert second.holder.pid == other_pid
    assert second.acquired is False


def test_refused_lock_does_not_delete_the_live_owners_marker(marker, other_pid):
    _claim(marker, other_pid)

    second = UpdateLock(path=marker)
    second.acquire()
    second.release()

    assert marker.exists(), "a refused claimant must never clear the live owner's lock"


@pytest.mark.platforms("posix")
def test_marker_owned_by_a_zombie_self_heals(marker):
    """A crashed updater lingering unreaped must not pin the lock for 20 minutes.

    ``os.kill(pid, 0)`` still succeeds for a zombie, so a marker owned by a
    dead-but-unreaped stage used to read as a live update until the age ceiling
    expired (#77259, #120635, #125932).
    """
    pid = os.fork()
    assert pid >= 0
    if pid == 0:
        os._exit(0)  # noqa: P111 — child exits without running pytest teardown

    # Keep the child unreaped (a zombie) and wait until the state probe
    # actually reports 'Z' so the assertion can't race the exit.
    from hermes_cli._early_recovery import _process_state

    became_zombie = False
    for _ in range(40):
        state = _process_state(pid)
        if state is not None and state.upper().startswith("Z"):
            became_zombie = True
            break
        time.sleep(0.05)
    assert became_zombie, "child never reached zombie state on this platform"

    try:
        _claim(marker, pid)
        assert read_live_update(path=marker) is None, "a zombie is not a live update"
        assert not marker.exists(), "the stale marker must self-heal (unlink)"
    finally:
        os.waitpid(pid, 0)  # reap so the test leaks no children


def test_marker_naming_our_own_pid_is_adopted(marker, monkeypatch):
    """A killed update's marker names the pid its retry gets (containers restart pid numbering).

    No other live process can hold our pid, so the claim is ours: take it instead of refusing
    "another update" for up to 20 minutes. It is a new attempt, so it is claimed fresh: a
    nearly-expired claim must still block a second updater for our whole run.
    """
    _claim(marker, os.getpid(), time.time() - UPDATE_MARKER_MAX_AGE_SECONDS + 5)

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    assert lock.acquired is True

    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 60)
    holder = read_live_update(path=marker)
    assert holder is not None and holder.pid == os.getpid(), "a second updater would start mid-run"
    lock.release()
    assert not marker.exists()


def test_release_leaves_a_marker_a_handoff_partner_now_owns(marker):
    """The desktop writes the marker, then the Tauri updater takes ownership.

    Releasing must not delete a marker whose pid is no longer ours — that would
    reopen the gate while the partner is still mid-update.
    """
    lock = UpdateLock(path=marker)
    lock.acquire()

    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")
    lock.release()

    assert marker.exists(), "the partner's marker is not ours to remove"


def test_dead_owner_is_reclaimed_not_honored(marker):
    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getpid()


def test_owner_past_the_age_ceiling_is_reclaimed(marker):
    """A live-but-wedged updater must not hold the lock forever."""
    long_ago = int(time.time()) - UPDATE_MARKER_MAX_AGE_SECONDS - 60
    marker.write_text(f"{os.getpid()}\n{long_ago}\n", encoding="utf-8")

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True


@pytest.mark.parametrize(
    "body",
    ["", "not-a-pid\n123\n", "\n\n", "12345"],
    ids=["empty", "garbage-pid", "blank-lines", "no-start-time"],
)
def test_malformed_markers_never_block_an_update(marker, body):
    marker.write_text(body, encoding="utf-8")

    assert read_live_update(path=marker) is None
    assert UpdateLock(path=marker).acquire() is True


def test_stale_marker_is_removed_on_read(marker):
    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

    assert read_live_update(path=marker) is None
    assert not marker.exists(), "whoever notices a stale marker clears it"


def test_absent_marker_reports_no_live_update(marker):
    assert read_live_update(path=marker) is None


def test_context_manager_releases_even_on_exception(marker):
    with pytest.raises(RuntimeError):
        with UpdateLock(path=marker) as lock:
            assert lock.acquired is True
            raise RuntimeError("update blew up mid-flight")

    assert not marker.exists(), "a crashed update must not strand the lock"


def test_describe_holder_names_the_pid_and_elapsed_time(marker):
    lock = UpdateLock(path=marker)
    lock.acquire()

    holder = read_live_update(path=marker)
    assert holder is not None
    message = describe_holder(holder)

    assert str(os.getpid()) in message, (
        "the user needs the pid to find the other update"
    )
    assert "already running" in message


def test_unwritable_marker_location_does_not_block_the_update(tmp_path):
    """Degrade to pre-lock behavior rather than refusing to update at all.

    An unwritable marker path is a worse reason to block an update than the
    race the lock prevents.
    """
    lock = UpdateLock(path=tmp_path / "nonexistent-file" / "marker")
    (tmp_path / "nonexistent-file").write_text(
        "i am a file, not a dir", encoding="utf-8"
    )

    assert lock.acquire() is True
    assert lock.acquired is False, "nothing was written, so there is nothing to release"


class TestHandoffFromOrchestratingUpdater:
    """The Tauri updater holds the marker, then spawns ``hermes update``.

    The regression: the child saw its own parent's live marker and exited 2,
    so every GUI update failed with "Hermes is still running" and retrying
    just re-ran the same self-deadlock. The parent names its pid in
    HANDOFF_PID_ENV; a live holder matching it is our own orchestrator.
    """

    def test_child_runs_under_the_parents_live_claim(self, marker, monkeypatch, other_pid):
        # other_pid stands in for the live parent updater.
        _claim(marker, other_pid)
        monkeypatch.setenv(HANDOFF_PID_ENV, str(other_pid))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True
        assert lock.acquired is False, "the parent's claim is not ours to own"

        lock.release()
        assert marker.exists(), "the parent still needs its marker after our stage ends"
        assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == other_pid

    def test_handoff_pid_that_is_not_the_live_holder_grants_nothing(
        self, marker, monkeypatch, other_pid
    ):
        """The env var alone must not bypass the lock."""
        _claim(marker, other_pid)
        monkeypatch.setenv(HANDOFF_PID_ENV, str(other_pid + 1))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.holder is not None

    @pytest.mark.parametrize(
        "value",
        ["", "not-a-pid", "-1", "0"],
        ids=["empty", "garbage", "negative", "zero"],
    )
    def test_malformed_handoff_values_fall_back_to_refusal(
        self, marker, monkeypatch, value, other_pid
    ):
        _claim(marker, other_pid)
        monkeypatch.setenv(HANDOFF_PID_ENV, value)

        assert UpdateLock(path=marker).acquire() is False

    def test_handoff_env_with_no_marker_claims_normally(self, marker, monkeypatch):
        """A handoff pid must not stop us writing our own claim when unlocked."""
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid()))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True
        assert lock.acquired is True
        assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getpid()


class TestAncestryHandoff:
    """Staged updaters older than the HANDOFF_PID_ENV export never send it.

    ``hermes-setup`` under ``~/.hermes`` is only refreshed by a full installer
    run, so an updated checkout (new lock) driven by a pre-handoff staged
    updater (old parent) deadlocks on exit 2 forever unless the child also
    recognizes a live holder that is its own process ancestor.

    ``_pid_alive`` is pinned True here because the hermetic conftest guards
    ``os.kill`` probes of pids outside the test subtree (our ppid included);
    liveness has its own coverage above — ancestry is what's under test.
    """

    @pytest.fixture(autouse=True)
    def _liveness_pinned_true(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.update_lock._pid_alive", lambda pid: True)

    def test_marker_owned_by_our_parent_process_is_our_orchestrator(self, marker):
        marker.write_text(f"{os.getppid()}\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True, "a live ancestor's claim is the one we run under"
        assert lock.acquired is False, "the parent's claim is not ours to own"

        lock.release()
        assert marker.exists(), "the parent still needs its marker after our stage ends"

    @pytest.mark.platforms("any")
    def test_grandchild_adopts_orchestrator_marker_without_psutil(self, marker, tmp_path):
        """Regression: the desktop hand-off's grandchild refused its own orchestrator.

        The posix shim (grandparent) holds the marker and spawns ``hermes update``
        (direct child — adopts via getppid). An old-updater update into a PM tree
        then hands off again: ``_old_updater._run_child`` spawns
        ``_update_takeover.py`` as ``python -I -S -B``, where psutil cannot import
        (-S skips site-packages). The psutil-only ancestry walk returned False for
        the two-hops-up shim, and the takeover child refused with exit 2 —
        "Another Hermes update is already running (PID <the shim itself>)" —
        observed live on a macOS rehearsal install, then again on Windows, where
        the stdlib walk had no /proc and no ps. Marked for every lane: the Windows
        lane only imports files carrying a platforms marker, which is how the
        Windows half went unseen. The stdlib fallback walk is what must adopt here.
        """
        import subprocess
        import sys
        from textwrap import dedent

        # Simulate the orchestrator: this test process holds the marker and
        # spawns the -I -B middle, which spawns the -I -S -B takeover-like leaf.
        marker.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")

        leaf = dedent(
            """
            import sys
            from pathlib import Path
            sys.path.insert(0, %(root)r)
            from hermes_cli.update_lock import UpdateLock
            lock = UpdateLock(path=Path(%(marker)r))
            if not lock.acquire():
                print("REFUSED", lock.holder.pid)
                raise SystemExit(2)
            assert lock.acquired is False, "the orchestrator's claim is not ours to own"
            print("ADOPTED")
            """
        ) % {"root": str(REPO_ROOT), "marker": str(marker)}
        middle = dedent(
            """
            import subprocess, sys
            code = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", %(leaf)r],
            ).returncode
            raise SystemExit(code)
            """
        ) % {"leaf": leaf}

        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", middle],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "ADOPTED" in result.stdout
        assert marker.exists(), "the orchestrator still needs its marker after the leaf ends"

    @pytest.mark.platforms("any")
    def test_unrelated_live_holder_is_still_refused_under_stdlib_walk(self, marker, tmp_path):
        """The stdlib fallback must not widen the lock: a foreign pid stays foreign.

        The marker owner is a live *sibling* of the leaf (a sleeper spawned by
        this test), never an ancestor of it — the shape of an unrelated
        concurrent updater. The leaf's ancestry walk dead-ends at pytest and
        the sibling must keep the lock.
        """
        import subprocess
        import sys
        import time as time_mod
        from textwrap import dedent

        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        try:
            marker.write_text(f"{sleeper.pid}\n{int(time_mod.time())}\n", encoding="utf-8")
            assert read_live_update(path=marker) is not None

            leaf = dedent(
                """
                import sys
                from pathlib import Path
                sys.path.insert(0, %(root)r)
                from hermes_cli.update_lock import UpdateLock
                lock = UpdateLock(path=Path(%(marker)r))
                print("ADOPTED" if lock.acquire() else "REFUSED")
                """
            ) % {"root": str(REPO_ROOT), "marker": str(marker)}
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", leaf],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, result.stderr
            assert "REFUSED" in result.stdout
        finally:
            sleeper.kill()
            sleeper.wait()

    def test_live_non_ancestor_holder_is_still_refused(self, marker):
        """Ancestry must not open the lock to unrelated concurrent updaters."""
        marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.holder is not None
        assert lock.holder.pid == DEAD_PID


class _FakeProcess:
    """One link of a stubbed parent chain.

    ``error`` makes this link refuse inspection, standing in for a process the
    sandbox will not let us read (psutil raises ``AccessDenied`` there).
    """

    def __init__(self, pid, parent=None, error=None):
        self.pid = pid
        self._parent = parent
        self._error = error

    def parent(self):
        if self._error is not None:
            raise self._error
        return self._parent


def _pin_ancestry(monkeypatch, leaf):
    """Make ``psutil.Process()`` return the leaf of a stubbed chain."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "Process", lambda *a, **k: leaf)


def _chain(*pids, blocked_above=False):
    """Build us -> pids[0] -> pids[1] ... innermost first.

    ``blocked_above`` caps the chain with a link that raises instead of
    reporting its own parent — the sandboxed ``/proc/1`` case.
    """
    psutil = pytest.importorskip("psutil")
    top = _FakeProcess(1, error=psutil.AccessDenied(pid=1)) if blocked_above else None
    node = top
    for pid in reversed(pids):
        node = _FakeProcess(pid, parent=node)
    return _FakeProcess(os.getpid(), parent=node)


class TestAncestryUnderUnreadableProcesses:
    """Regression: #87514 — an unreadable process ABOVE the orchestrator.

    ``psutil.Process.parents()`` builds the whole chain before returning and
    only tolerates ``NoSuchProcess`` per link, so one ``AccessDenied`` high up
    threw away the ancestors already found. Under firejail with
    ``ptrace_scope=1`` (and in hardened containers) ``/proc/1`` is unreadable,
    so every desktop update refused its own orchestrator's fresh marker and
    exited 2 forever. Ancestry must be decided link by link.
    """

    def test_ancestor_below_an_unreadable_process_is_still_found(self, monkeypatch):
        from hermes_cli.update_lock import _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(2000, blocked_above=True))

        assert _is_ancestor_pid(2000) is True, (
            "the orchestrator is one link up; a process we cannot read above "
            "it must not erase a match already proven"
        )

    def test_deeper_ancestor_below_an_unreadable_process_is_found(self, monkeypatch):
        from hermes_cli.update_lock import _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(2000, 3000, blocked_above=True))

        assert _is_ancestor_pid(3000) is True

    def test_unreadable_process_below_the_match_still_refuses(self, monkeypatch):
        """Failing before a match keeps the conservative refusal."""
        from hermes_cli.update_lock import _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(blocked_above=True))

        assert _is_ancestor_pid(2000) is False

    def test_unrelated_pid_is_refused_on_a_fully_readable_chain(self, monkeypatch):
        from hermes_cli.update_lock import _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(2000, 3000))

        assert _is_ancestor_pid(DEAD_PID) is False

    def test_our_own_pid_is_never_an_ancestor(self, monkeypatch):
        from hermes_cli.update_lock import _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(2000))

        assert _is_ancestor_pid(os.getpid()) is False

    def test_walk_is_bounded(self, monkeypatch):
        """A pathological chain terminates instead of spinning."""
        from hermes_cli.update_lock import _MAX_ANCESTRY_DEPTH, _is_ancestor_pid

        _pin_ancestry(monkeypatch, _chain(*range(2000, 2000 + _MAX_ANCESTRY_DEPTH * 2)))

        assert _is_ancestor_pid(2000 + _MAX_ANCESTRY_DEPTH * 2 - 1) is False

    def test_acquire_accepts_the_orchestrator_behind_an_unreadable_init(
        self, marker, monkeypatch
    ):
        """The reporter's end-to-end symptom: exit 2 on every GUI update."""
        monkeypatch.setattr("hermes_cli.update_lock._pid_alive", lambda pid: True)
        _pin_ancestry(monkeypatch, _chain(2000, blocked_above=True))
        marker.write_text(f"2000\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True, "the hand-off child runs under 2000's claim"
        assert lock.acquired is False, "the orchestrator's claim is not ours to own"

        lock.release()
        assert marker.exists(), "the orchestrator still needs its marker"

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

import pytest

from hermes_cli.update_lock import (
    HANDOFF_PID_ENV,
    UPDATE_MARKER_MAX_AGE_SECONDS,
    UpdateLock,
    UpdateMarkerError,
    describe_holder,
    read_live_update,
    update_marker_path,
)

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
    assert int(lines[0]) == os.getpid(), "the Electron gate probes this pid for liveness"
    assert int(lines[1]) == pytest.approx(time.time(), abs=5)
    assert len(lines) == 2, "wire format is exactly pid + started_at"


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


def test_proven_live_owner_past_the_age_ceiling_is_not_reclaimed(marker, monkeypatch):
    """A slow live updater must never race a second installation mutation."""
    long_ago = int(time.time()) - UPDATE_MARKER_MAX_AGE_SECONDS - 60
    monkeypatch.setattr("hermes_cli.update_lock._pid_create_time", lambda pid: long_ago - 1)
    marker.write_text(f"{os.getpid()}\n{long_ago}\n", encoding="utf-8")

    lock = UpdateLock(path=marker)
    assert lock.acquire() is False
    assert marker.exists()


@pytest.mark.parametrize(
    "body",
    ["", "not-a-pid\n123\n", "\n\n", "12345"],
    ids=["empty", "garbage-pid", "blank-lines", "no-start-time"],
)
def test_marker_that_names_nobody_is_reclaimed_after_a_dwell(marker, body):
    """Upstream main unlinks an unparseable marker (update_lock.py:125). The PR
    made every such body block forever, so a torn write from windows.ps1 broke
    updates until a human deleted a file no message ever named."""
    marker.write_text(body, encoding="utf-8")
    slept = []

    assert read_live_update(path=marker, dwell_seconds=0.01, sleep=slept.append) is None
    assert slept == [0.01], "the dwell must run before reclaiming"
    assert not marker.exists()

    marker.write_text(body, encoding="utf-8")
    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getpid()


@pytest.mark.parametrize(
    "body",
    ["", "not-a-pid\n123\n"],
    ids=["empty", "garbage-pid"],
)
def test_a_body_rewritten_inside_the_dwell_keeps_its_new_claim(marker, body, monkeypatch):
    """The reclaim is an exact-content CAS behind the dwell, so an updater that
    finishes writing mid-dwell never loses the claim it just made."""
    monkeypatch.setattr("hermes_cli.update_lock.pid_alive", lambda pid: True)
    monkeypatch.setattr("hermes_cli.update_lock._pid_create_time", lambda pid: 100.0)
    marker.write_text(body, encoding="utf-8")
    finished = f"{os.getpid()}\n{int(time.time())}\n"

    holder = read_live_update(path=marker, dwell_seconds=0,
                              sleep=lambda _s: marker.write_text(finished, encoding="utf-8"))

    assert holder is not None and holder.pid == os.getpid()
    assert marker.read_text(encoding="utf-8") == finished


def test_an_unreadable_marker_still_fails_closed(marker):
    """Opposite of the case above: unreadable is a genuine unknown, so callers
    must stop rather than mutate an installation someone may be updating."""
    marker.mkdir()

    with pytest.raises(UpdateMarkerError):
        read_live_update(path=marker)
    with pytest.raises(UpdateMarkerError):
        UpdateLock(path=marker).acquire()
    assert marker.is_dir()


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

    assert str(os.getpid()) in message, "the user needs the pid to find the other update"


def test_unwritable_marker_location_blocks_the_update(tmp_path):
    lock = UpdateLock(path=tmp_path / "nonexistent-file" / "marker")
    (tmp_path / "nonexistent-file").write_text("file", encoding="utf-8")
    with pytest.raises(UpdateMarkerError):
        lock.acquire()
    assert lock.acquired is False


class TestHandoffFromOrchestratingUpdater:
    """The Tauri updater holds the marker, then spawns ``hermes update``.

    This channel has a live production writer: ``update_child_env`` in
    ``apps/bootstrap-installer/src-tauri/src/update.rs`` (asserted there by
    ``update_child_env_names_our_pid_for_the_lock_handoff``). Deleting the
    adoption path here dead-ends every GUI update on exit 2.

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

    def test_handoff_pid_that_is_not_the_live_holder_grants_nothing(self, marker, monkeypatch, other_pid):
        """The env var alone must not bypass the lock."""
        _claim(marker, other_pid)
        monkeypatch.setenv(HANDOFF_PID_ENV, str(other_pid + 1))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.holder is not None

    @pytest.mark.parametrize("value", ["", "not-a-pid", "-1", "0"], ids=["empty", "garbage", "negative", "zero"])
    def test_malformed_handoff_values_fall_back_to_refusal(self, marker, monkeypatch, value, other_pid):
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
        monkeypatch.setattr("hermes_cli.update_lock.pid_alive", lambda pid: True)

    def test_marker_owned_by_our_parent_process_is_our_orchestrator(self, marker):
        marker.write_text(f"{os.getppid()}\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True, "a live ancestor's claim is the one we run under"
        assert lock.acquired is False, "the parent's claim is not ours to own"

        lock.release()
        assert marker.exists(), "the parent still needs its marker after our stage ends"
        assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getppid()

    def test_live_non_ancestor_holder_is_still_refused(self, marker):
        """Ancestry must not open the lock to unrelated concurrent updaters."""
        marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.holder is not None
        assert lock.holder.pid == DEAD_PID


def test_release_restores_claim_replaced_during_isolation(marker, monkeypatch):
    from pathlib import Path
    lock = UpdateLock(path=marker)
    assert lock.acquire()
    rename = Path.rename
    successor = f"{DEAD_PID}\n{int(time.time())}\n"
    def replace_then_rename(path, destination):
        marker.write_text(successor)
        return rename(path, destination)
    monkeypatch.setattr(Path, "rename", replace_then_rename)
    lock.release()
    assert marker.read_text() == successor


def test_unknown_owner_blocks_until_the_ceiling_then_expires(marker, monkeypatch):
    """An unprovable owner holds the gate -- but not forever, or one recycled
    PID wedges every update path and the MCP bridge with no way back."""
    monkeypatch.setattr("hermes_cli.update_lock.pid_alive", lambda pid: True)
    monkeypatch.setattr("hermes_cli.update_lock._pid_create_time", lambda pid: None)

    marker.write_text(f"{DEAD_PID}\n{int(time.time()) - 60}\n")
    assert UpdateLock(path=marker).acquire() is False
    assert marker.exists()

    marker.write_text(f"{DEAD_PID}\n{int(time.time()) - UPDATE_MARKER_MAX_AGE_SECONDS - 60}\n")
    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getpid()


def test_unknown_cleanup_artifact_never_expires(marker):
    artifact = marker.with_name(marker.name + ".cas-unknown")
    artifact.write_text("123\n100\n")
    with pytest.raises(UpdateMarkerError):
        UpdateLock(path=marker).acquire()
    assert artifact.exists()


def test_reader_does_not_restore_an_active_release(marker, monkeypatch):
    from pathlib import Path

    lock = UpdateLock(path=marker)
    assert lock.acquire()
    rename = Path.rename

    def read_during_release(path, destination):
        result = rename(path, destination)
        with pytest.raises(UpdateMarkerError, match="cleanup"):
            read_live_update(path=marker)
        return result

    monkeypatch.setattr(Path, "rename", read_during_release)
    lock.release()
    assert not marker.exists()
    assert not list(marker.parent.glob(marker.name + ".cas-*"))


def test_handoff_cannot_authorize_an_unreadable_owner_identity(marker, monkeypatch):
    marker.write_text(f"{os.getpid()}\n{int(time.time())}\n")
    monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid()))
    monkeypatch.setattr("hermes_cli.update_lock._pid_create_time", lambda pid: None)
    assert UpdateLock(path=marker).acquire() is False
    assert marker.exists()


from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.platforms("any")
def test_grandchild_adopts_orchestrator_marker_without_psutil(marker, tmp_path):
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

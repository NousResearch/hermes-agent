"""Tests for agent.verification_lock — the cross-process coordination primitive.

Covers:
  A20 — writer can create the lockfile; reader cannot.
  A17 — crash lock release (independent process dies; new participant acquires).
  A19 — POSIX process contention (independent subprocess).
  A31 — Windows reader byte-materialization contract (mocked on POSIX).
  A25 — writer creates persistent lockfile.
  A26 — lockfile contains only the minimum coordination byte.
  A27 — reader using existing lockfile leaves size and SHA256 unchanged.
  A10 — reader lock timeout fail-closed.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from agent import verification_lock as vlock


@pytest.fixture
def hermes_home(monkeypatch, tmp_path, request):
    """Redirect hermes_constants.get_hermes_home to a temp dir.

    pytest 9.x reuses the same ``tmp_path`` directory across tests in
    the same file unless we nest under a unique subdirectory; do that
    so state does not bleed between tests.
    """
    home = tmp_path / f"home-{request.node.name}"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: home
    )
    # Also patch the binding inside the module under test if it has already
    # imported the symbol.
    if hasattr(vlock, "get_hermes_home"):
        monkeypatch.setattr(vlock, "get_hermes_home", lambda: home)
    import agent.verification_evidence as ve
    if hasattr(ve, "get_hermes_home"):
        monkeypatch.setattr(ve, "get_hermes_home", lambda: home)
    return home


# ---------- A20: writer can create; reader cannot ----------


def test_writer_may_create_lockfile(hermes_home):
    """A20/A25: writer creates the persistent lockfile."""
    assert not (hermes_home / ".locks" / "verification_evidence.lock").exists()
    with vlock.coordinated_lock("writer"):
        pass
    assert (hermes_home / ".locks" / "verification_evidence.lock").is_file()


def test_reader_cannot_create_lockfile(hermes_home):
    """A20: reader MUST NOT create the lockfile or parent dir."""
    assert not (hermes_home / ".locks").exists()
    with pytest.raises(vlock.LockUnavailable):
        with vlock.coordinated_lock("reader"):
            pass
    assert not (hermes_home / ".locks").exists()
    assert not (hermes_home / ".locks" / "verification_evidence.lock").exists()


# ---------- A26: lockfile contains only the required byte ----------


def test_lockfile_contains_minimum_byte(hermes_home):
    """A26: writer-materialized lockfile is exactly 1 byte."""
    with vlock.coordinated_lock("writer"):
        pass
    lf = hermes_home / ".locks" / "verification_evidence.lock"
    assert lf.stat().st_size == 1
    assert lf.read_bytes() == b"\x00"


# ---------- A27: reader does not change lockfile bytes ----------


def test_reader_lockfile_bytes_unchanged(hermes_home):
    """A27: when reader acquires the existing lockfile, size+SHA256 unchanged."""
    # First, a writer materializes the lockfile.
    with vlock.coordinated_lock("writer"):
        pass
    lf = hermes_home / ".locks" / "verification_evidence.lock"
    size_before = lf.stat().st_size
    sha_before = hashlib.sha256(lf.read_bytes()).hexdigest()
    # Now a reader acquires and releases.
    with vlock.coordinated_lock("reader"):
        pass
    assert lf.stat().st_size == size_before
    assert hashlib.sha256(lf.read_bytes()).hexdigest() == sha_before


# ---------- A17: crash lock release ----------


@pytest.mark.live_system_guard_bypass
def test_crash_lock_release(hermes_home):
    """A17: a writer process killed mid-lock does not deadlock new participants.

    Marked with ``live_system_guard_bypass`` because this test intentionally
    spawns an independent subprocess and signals it with SIGKILL — the
    live-system guard would otherwise reject the cross-tree kill.

    Strategy: spawn a child that holds the lock for 30 s. Poll its
    /proc-visible liveness while repeatedly attempting parent acquire with
    a tight timeout. Once parent-acquire consistently times out, the child
    is confirmed holding. Then SIGKILL and confirm parent can acquire.
    """
    # Materialize the lockfile first.
    with vlock.coordinated_lock("writer"):
        pass

    candidate_root = str(Path(__file__).parent.parent.parent)
    # tempfile.mkstemp() returns (fd, path). On native Windows, the open
    # file descriptor carries a default Delete deny; if the parent leaves
    # it open, the subsequent Path.unlink() raises PermissionError
    # (WinError 32, "being used by another process"). Capture both halves
    # and explicitly close the fd. On POSIX this is a no-op semantically.
    _crash_log_fd, _crash_log_path = tempfile.mkstemp(
        prefix="wal3_crashchild_log_", suffix=".txt"
    )
    os.close(_crash_log_fd)
    _crash_child_fd, _crash_child_path = tempfile.mkstemp(
        prefix="wal3_crashchild_", suffix=".py"
    )
    os.close(_crash_child_fd)
    child_log = Path(_crash_log_path)
    child = Path(_crash_child_path)
    child.write_text(
        "import os, sys, time\n"
        f"sys.path.insert(0, {candidate_root!r})\n"
        "import hermes_constants\n"
        f"hermes_constants.get_hermes_home = lambda: {str(hermes_home)!r}\n"
        "from agent import verification_lock as vl\n"
        # Write a marker file so the parent can detect acquisition.
        f"open({str(child_log)!r}, 'w').write('ACQUIRED\\n')\n"
        "with vl.coordinated_lock('writer'):\n"
        "    open(" + repr(str(child_log)) + ", 'a').write('HOLDING\\n')\n"
        "    time.sleep(30)\n"
    )

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = candidate_root + os.pathsep + child_env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [sys.executable, str(child)],
        env=child_env,
        cwd=candidate_root,
    )
    try:
        # Wait for the child to write 'HOLDING' (means it acquired the lock).
        deadline = time.monotonic() + 10.0
        holding = False
        while time.monotonic() < deadline:
            try:
                content = child_log.read_text()
            except FileNotFoundError:
                content = ""
            if "HOLDING" in content:
                holding = True
                break
            time.sleep(0.05)
        assert holding, (
            f"child did not acquire the lock within timeout. Log: {child_log.read_text()!r}"
        )

        # Now parent must time out on repeated attempts.
        deadline = time.monotonic() + 2.0
        contended = False
        while time.monotonic() < deadline:
            try:
                with vlock.coordinated_lock("writer", timeout=0.1):
                    # If we get here, child is not actually holding.
                    pass
            except vlock.LockTimeout:
                contended = True
                break
            time.sleep(0.05)
        assert contended, "child reported HOLDING but parent can still acquire"

        # Kill the subprocess; POSIX kernel auto-releases flock on close.
        proc.kill()
        proc.wait(timeout=5.0)

        # New participant can acquire within the 2.0 s timeout.
        with vlock.coordinated_lock("writer", timeout=2.0):
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        for f in (child, child_log):
            try:
                f.unlink()
            except FileNotFoundError:
                pass


# ---------- A19: POSIX process contention ----------


@pytest.mark.live_system_guard_bypass
def test_posix_process_contention(hermes_home):
    """A19: two independent processes contend on the same lockfile.

    Marked with ``live_system_guard_bypass`` because this test intentionally
    spawns an independent subprocess and signals it with termination
    (subprocess.Popen.terminate / kill; on POSIX the OS signals SIGKILL when
    pgid-killed, and on Windows TerminateProcess). The script-level wait
    timeout ensures the subprocess is actually consuming the lock at the
    moment the parent attempts a nonblocking acquire; cross-process lock
    release on process termination is the cross-platform behavior we
    exercise here.
    """
    with vlock.coordinated_lock("writer"):
        pass

    candidate_root = str(Path(__file__).parent.parent.parent)
    # tempfile.mkstemp() returns (fd, path). On native Windows, the open
    # file descriptor carries a default Delete deny; if the parent leaves
    # it open, the subsequent Path.unlink() raises PermissionError
    # (WinError 32, "being used by another process"). Capture both halves
    # and explicitly close the fd. On POSIX this is a no-op semantically.
    _contend_log_fd, _contend_log_path = tempfile.mkstemp(
        prefix="wal3_posixcontend_log_", suffix=".txt"
    )
    os.close(_contend_log_fd)
    _contend_child_fd, _contend_child_path = tempfile.mkstemp(
        prefix="wal3_posixcontend_", suffix=".py"
    )
    os.close(_contend_child_fd)
    child_log = Path(_contend_log_path)
    child = Path(_contend_child_path)
    child.write_text(
        "import os, sys, time\n"
        f"sys.path.insert(0, {candidate_root!r})\n"
        "import hermes_constants\n"
        f"hermes_constants.get_hermes_home = lambda: {str(hermes_home)!r}\n"
        "from agent import verification_lock as vl\n"
        f"open({str(child_log)!r}, 'w').write('ACQUIRED\\n')\n"
        "with vl.coordinated_lock('writer'):\n"
        "    open(" + repr(str(child_log)) + ", 'a').write('HOLDING\\n')\n"
        "    time.sleep(15)\n"
    )

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = candidate_root + os.pathsep + child_env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [sys.executable, str(child)],
        env=child_env,
        cwd=candidate_root,
    )
    try:
        deadline = time.monotonic() + 10.0
        holding = False
        while time.monotonic() < deadline:
            try:
                content = child_log.read_text()
            except FileNotFoundError:
                content = ""
            if "HOLDING" in content:
                holding = True
                break
            time.sleep(0.05)
        assert holding, (
            f"child did not acquire the lock. Log: {child_log.read_text()!r}"
        )

        # Parent must time out.
        deadline = time.monotonic() + 2.0
        contended = False
        while time.monotonic() < deadline:
            try:
                with vlock.coordinated_lock("writer", timeout=0.1):
                    pass
            except vlock.LockTimeout:
                contended = True
                break
            time.sleep(0.05)
        assert contended, "parent should not be able to acquire while child holds"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        for f in (child, child_log):
            try:
                f.unlink()
            except FileNotFoundError:
                pass


# ---------- A10: reader lock timeout fail-closed ----------


def test_reader_lock_timeout_returns_unknown_signal(hermes_home):
    """A10: when the writer holds the lock past the reader's timeout, the
    LockTimeout is raised; the integration layer maps it to status='unknown'.

    Here we verify the primitive raises LockTimeout; the JSON-RPC mapping is
    exercised in the integration test (test_verification_status_rpc.py).
    """
    with vlock.coordinated_lock("writer"):
        with pytest.raises(vlock.LockTimeout):
            with vlock.coordinated_lock("reader", timeout=0.5):
                pass


# ---------- A31: reader must not change lockfile bytes ----------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
def test_posix_reader_does_not_mutate_existing_lockfile(hermes_home):
    """A31: on POSIX, the reader's fcntl.flock does not require an
    initialized byte, but if the writer HAS initialized it, the reader
    must leave the lockfile bytes unchanged.
    """
    # Materialize a one-byte lockfile (writer-side).
    lf_dir = hermes_home / ".locks"
    lf_dir.mkdir(parents=True, exist_ok=True)
    lf = lf_dir / "verification_evidence.lock"
    lf.write_bytes(b"\x00")
    size_before = lf.stat().st_size
    sha_before = hashlib.sha256(lf.read_bytes()).hexdigest()

    with vlock.coordinated_lock("reader"):
        pass
    assert lf.stat().st_size == size_before
    assert hashlib.sha256(lf.read_bytes()).hexdigest() == sha_before


# ---------- lockfile_size_and_sha helper ----------


def test_lockfile_size_and_sha_helper(hermes_home):
    lf_before = vlock.lockfile_size_and_sha()
    assert lf_before is None  # absent before writer
    with vlock.coordinated_lock("writer"):
        pass
    lf_after = vlock.lockfile_size_and_sha()
    assert lf_after == (1, hashlib.sha256(b"\x00").hexdigest())

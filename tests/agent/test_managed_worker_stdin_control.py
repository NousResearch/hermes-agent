"""Regression for #142271 / #143422: a Windows kanban worker's control reader parked on fd 0
deadlocks the memory provider's first native import (mem0 -> qdrant_client -> numpy).

The worker now reads control frames from a private dup of stdin and leaves fd 0 on NUL on Windows,
so any native DLL init that probes stdin cannot wait on the parked read.
"""

import io
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from agent.managed_worker import WorkerControls, _claim_stdin_control, read_frame

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs in a child interpreter: mutating fd 0 in the test process would break the runner's stdin,
# and the drain must be observed from a process whose stdin really is a pipe.
_PROBE = """
import os, stat, sys
from agent.managed_worker import _claim_stdin_control, read_frame

stream = _claim_stdin_control()
mode = os.fstat(0).st_mode
print("fd0_is_pipe", stat.S_ISFIFO(mode), sep="=")
print("frame", read_frame(stream)["n"], sep="=")
"""


def _probe():
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=REPO_ROOT,
        input='{"n": 7}\n', capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


@pytest.mark.platforms("windows")
def test_stdin_control_disarms_fd0_and_keeps_frames_readable():
    """fd 0 leaves the control pipe (so a parked native import cannot block on it) while the
    frames the owner writes to that pipe still reach the reader."""
    seen = _probe()
    assert seen["fd0_is_pipe"] == "False", "fd 0 still points at the control pipe"
    assert seen["frame"] == "7", "the control channel broke when fd 0 was disarmed"


@pytest.mark.platforms("linux", "macos")
def test_stdin_control_keeps_fd0_on_posix():
    """POSIX has no per-fd loader lock to dodge, so fd 0 stays as the owner passed it."""
    seen = _probe()
    assert seen["fd0_is_pipe"] == "True"
    assert seen["frame"] == "7"


def test_worker_controls_reads_the_given_stream():
    """The reader honours the stream handed to it instead of reaching for sys.stdin itself."""
    class _Channel:
        def send(self, *args, **kwargs):
            raise AssertionError("a finish frame sends nothing")

    controls = WorkerControls(_Channel(), "route", io.BytesIO(b'{"type": "finish"}\n'))
    assert controls.finish.wait(timeout=30), "finish frame was never consumed"


class _FakeStdin:
    """Only ``fileno()`` is used by the claim, so the test can hand it its own pipe."""

    def __init__(self, fd):
        self._fd = fd

    def fileno(self):
        return self._fd


def test_the_posix_branch_leaves_fd0_alone_and_still_reads_frames(monkeypatch):
    """The fd-0 surgery is the Windows workaround; off Windows the owner's stdin stays intact."""
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", _FakeStdin(read_fd))
    monkeypatch.setattr(sys, "platform", "linux")

    stream = _claim_stdin_control()
    try:
        os.fstat(read_fd)  # still the owner's pipe, not replaced or closed
        assert stream.fileno() != read_fd, "the claim re-used the owner's descriptor"
        os.write(write_fd, b'{"n": 11}\n{"n": 12}\n')
        assert read_frame(stream)["n"] == 11
        assert read_frame(stream)["n"] == 12
    finally:
        stream.close()
        os.close(read_fd)
        os.close(write_fd)


def test_each_claim_owns_a_distinct_fd_and_releases_it(monkeypatch):
    """Two claims must not share one fd, and closing a stream must give the fd back (no leak)."""
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", _FakeStdin(read_fd))
    monkeypatch.setattr(sys, "platform", "linux")

    first = _claim_stdin_control()
    second = _claim_stdin_control()
    owned = (first.fileno(), second.fileno())
    try:
        assert len(set(owned)) == 2, "both claims share one descriptor"
        assert read_fd not in owned, "a claim handed out the owner's own descriptor"
    finally:
        first.close()
        second.close()
        os.close(read_fd)
        os.close(write_fd)

    for fd in owned:
        with pytest.raises(OSError):
            os.fstat(fd)


# Runs in a child interpreter: main() dups stderr over this process's stdout and mutates fd 0. The
# child replaces hello/execute so only the wiring under test runs; probes report on stderr.
_WIRING_PROBE = """
import sys
import agent.managed_worker as mw

mw.hello = lambda: {}
mw.validate_bootstrap = lambda frame: frame


def fake_execute(frame, channel, controls_in=None):
    if controls_in is None:
        print("stream=missing", file=sys.stderr, flush=True)
        return
    print("stream=claimed", file=sys.stderr, flush=True)
    print("control=%s" % mw.read_frame(controls_in)["n"], file=sys.stderr, flush=True)


mw.execute = fake_execute
raise SystemExit(mw.main())
"""


_FAILING_CLAIM_PROBE = """
import agent.managed_worker as mw

mw.hello = lambda: {}


def _boom():
    raise OSError('dup2 refused')


mw._claim_stdin_control = _boom
raise SystemExit(mw.main())
"""


def _active_fd_count():
    """Handles this process holds: psutil counts descriptors off Windows; Windows has neither
    ``num_fds`` nor ``/proc/self/fd``, so it asks the kernel for the handle count."""
    if sys.platform == 'win32':
        import ctypes
        import ctypes.wintypes
        kernel32 = ctypes.windll.kernel32
        count = ctypes.wintypes.DWORD()
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE  # int default truncates the pseudo-handle
        kernel32.GetProcessHandleCount.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.wintypes.DWORD)]
        assert kernel32.GetProcessHandleCount(kernel32.GetCurrentProcess(), ctypes.byref(count))
        return count.value
    import psutil
    return psutil.Process().num_fds()


def test_main_hands_the_claimed_stream_to_the_worker_controls():
    """The claim is only wired if execute() receives it: the bootstrap and the first control frame
    both come off that one stream (two streams would lose the frame read ahead by the bootstrap)."""
    proc = subprocess.run(
        [sys.executable, '-c', _WIRING_PROBE], cwd=REPO_ROOT,
        input='{"type": "bootstrap"}\n{"n": 42}\n', capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    seen = dict(line.split('=', 1) for line in proc.stderr.splitlines() if '=' in line)
    assert seen.get('stream') == 'claimed', 'execute() was not handed the claimed stream'
    assert seen.get('control') == '42', 'the bootstrap read swallowed the control frame'


def test_a_failed_claim_still_reports_to_the_owner():
    """The claim runs inside main()'s try: its failure becomes an error frame instead of the silent
    worker death this PR exists to remove."""
    proc = subprocess.run(
        [sys.executable, '-c', _FAILING_CLAIM_PROBE], cwd=REPO_ROOT,
        input=b'', capture_output=True, timeout=180, check=False,
    )
    assert proc.returncode == 1, proc.stdout
    assert b'"type":"hello"' in proc.stdout, 'the owner never saw hello'
    assert b'"type":"error"' in proc.stdout, 'a failed claim died silently'


def test_control_frame_written_while_the_reader_is_parked_is_delivered():
    """The owner may write at any moment, including mid-read: a parked reader must still consume it."""
    parked, release = threading.Event(), threading.Event()

    class _ParkedStream(io.BytesIO):
        """Hangs inside the first readline until the test releases it, so "parked" is observable."""

        def readline(self, *args):
            parked.set()
            release.wait(timeout=30)  # the test releases it; a real pipe would unblock on write
            return b'{"type": "finish"}\n'

    class _Channel:
        def send(self, *args, **kwargs):
            raise AssertionError('the reader only sends while shutting down')

    controls = WorkerControls(_Channel(), 'route', _ParkedStream())
    try:
        assert parked.wait(timeout=30), 'the reader never parked'
        assert controls.reader.is_alive() and not controls.finish.is_set()
        release.set()
        assert controls.finish.wait(timeout=30), 'a frame delivered to a parked read never arrived'
    finally:
        release.set()


@pytest.mark.platforms('linux', 'macos')
def test_claim_reads_control_frames_from_a_tty_stdin(monkeypatch):
    """A worker whose stdin is a terminal has no pipe to dup; the frames must still arrive."""
    import pty
    master_fd, slave_fd = pty.openpty()
    monkeypatch.setattr(sys, 'stdin', _FakeStdin(slave_fd))
    stream = _claim_stdin_control()
    try:
        os.write(master_fd, b'{"n": 9}\n')
        assert read_frame(stream)['n'] == 9
    finally:
        stream.close()
        os.close(slave_fd)
        os.close(master_fd)


def test_a_failed_claim_hands_its_handles_back(monkeypatch):
    """If the fd-0 swap cannot happen the worker still has to raise (the owner gets an error frame),
    so the dup and the NUL handle must be gone by then."""
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, 'stdin', _FakeStdin(read_fd))
    monkeypatch.setattr(sys, 'platform', 'win32')

    real_dup2 = os.dup2

    def _refuse(fd, target):
        if target == read_fd:
            raise OSError('dup2 refused')
        return real_dup2(fd, target)  # pytest's capture restore between phases is not our target

    monkeypatch.setattr(os, 'dup2', _refuse)
    before = _active_fd_count()
    with pytest.raises(OSError):
        _claim_stdin_control()
    assert _active_fd_count() <= before, 'the failed claim leaked a handle'
    os.close(read_fd)
    os.close(write_fd)


# Runs in a child whose stdin really is a pipe the parent never writes to: the end-to-end shape of
# #142271 / #143422, with the production claim in front of the native import.
_NATIVE_IMPORT_PROBE = """
import sys, threading, time
from agent.managed_worker import _claim_stdin_control

stream = _claim_stdin_control()
reader = threading.Thread(target=lambda: stream.read(1), daemon=True)
reader.start()
time.sleep(0.5)
print("parked", reader.is_alive(), flush=True)
import numpy
print("imported", flush=True)
"""


@pytest.mark.platforms("windows")
def test_a_native_import_does_not_deadlock_behind_the_parked_reader(tmp_path):
    """A parked control read must not hold fd 0 while a native DLL init probes stdin.

    The deadlock is host-specific: on some Windows builds the child never returns, on others the
    import completes either way, so this arm is only red where the bug reproduces. It is the only
    arm here that exercises the real failure shape instead of the invariants around it.
    """
    probe = subprocess.run([sys.executable, "-c", "import numpy"], capture_output=True, timeout=180,
                           check=False)
    if probe.returncode != 0:
        pytest.skip("numpy is not importable in this interpreter")

    log = tmp_path / "child.log"
    with log.open("w", encoding="utf-8", errors="replace") as out:
        proc = subprocess.Popen([sys.executable, "-c", _NATIVE_IMPORT_PROBE], cwd=REPO_ROOT,
                                stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT, text=True)
        try:
            # communicate() would close stdin and hand the parked read an EOF, which is the one shape
            # that cannot deadlock; wait instead and leave the pipe open until the child is gone.
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail("the child deadlocked behind the parked control read")
        finally:
            proc.stdin.close()
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "parked True" in text, f"the reader was not parked when the import ran:\n{text}"
    assert "imported" in text, text
    assert proc.returncode == 0, text

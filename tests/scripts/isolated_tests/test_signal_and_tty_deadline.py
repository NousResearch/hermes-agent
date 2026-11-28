"""Behaviour contracts for C2 (signals during cleanup) and F7 (one deadline for the TTY line).

Signals are delivered by calling the installed handler at the fake boundary (no real kill of the
test process); the TTY is an os.pipe and the clock is injected, so nothing sleeps or waits.
"""

import os
import select
import signal
import subprocess
import sys

import pytest

from isolated_tests import executor, main, policy, proof

SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def _deliver(sig):
    signal.getsignal(sig)(sig, None)


@pytest.fixture
def cleanup_calls(monkeypatch):
    calls = []

    def fake_cleanup(c, env, timeout, *, during=None):
        calls.append("cleanup:start")
        if during is not None:
            _deliver(during)
        calls.append("cleanup:end")
        return executor.CLEAN_REMOVED

    def install(during=None):
        monkeypatch.setattr(executor, "cleanup_owned",
                            lambda c, env, timeout: fake_cleanup(c, env, timeout, during=during))
        return calls

    return install


def _run(argv):
    return executor.run_bounded(argv, {}, wall_seconds=60, stdout_cap=100, stderr_cap=100,
                                cleanup=executor.Cleanup(["inspect"], ["rm"], "run1"), cleanup_timeout=10)


@pytest.mark.platforms("posix")
def test_second_signal_during_cleanup_finishes_cleanup_and_restores(cleanup_calls, monkeypatch):
    before = {s: signal.getsignal(s) for s in SIGNALS}
    calls = cleanup_calls(during=signal.SIGTERM)
    real_popen = subprocess.Popen

    def popen_then_sigint(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        _deliver(signal.SIGINT)  # first signal lands as the CLI starts; its group must still die
        return proc

    monkeypatch.setattr(executor.subprocess, "Popen", popen_then_sigint)
    out = _run([sys.executable, "-c", "import time; time.sleep(60)"])
    assert calls == ["cleanup:start", "cleanup:end"]
    assert {s: signal.getsignal(s) for s in SIGNALS} == before
    assert out.interrupted == "SIGINT" and out.returncode is None
    assert main._map_rc(out) == policy.EXIT_INVALID


@pytest.mark.platforms("posix")
def test_signal_only_during_cleanup_still_marks_run_untrusted(cleanup_calls):
    before = {s: signal.getsignal(s) for s in SIGNALS}
    calls = cleanup_calls(during=signal.SIGHUP)
    out = _run([sys.executable, "-c", "pass"])
    assert calls == ["cleanup:start", "cleanup:end"]
    assert {s: signal.getsignal(s) for s in SIGNALS} == before
    assert out.interrupted == "SIGHUP" and out.returncode is None
    assert main._map_rc(out) == policy.EXIT_INVALID


@pytest.mark.platforms("posix")
def test_unsignalled_run_is_unchanged(cleanup_calls):
    calls = cleanup_calls()
    out = _run([sys.executable, "-c", "pass"])
    assert calls == ["cleanup:start", "cleanup:end"]
    assert out.interrupted is None and out.returncode == 0
    assert main._map_rc(out) == policy.EXIT_OK


def _arm_exit_wait(pid):
    """Return a callable that blocks until `pid` has exited, leaving it unreaped."""
    if hasattr(os, "waitid"):  # WNOWAIT leaves the status for the real reaper
        return lambda: os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
    # macOS has no os.waitid; a kqueue NOTE_EXIT watch does not reap either. Armed before
    # the child can exit, so the registration cannot race the exit.
    kq = select.kqueue()
    kq.control([select.kevent(pid, select.KQ_FILTER_PROC, select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                              select.KQ_NOTE_EXIT)], 0)

    def wait():
        try:
            assert kq.control(None, 1, 30), "child did not exit"
        finally:
            kq.close()

    return wait


@pytest.mark.platforms("posix")
def test_kill_group_reaps_a_cli_that_exited_but_is_not_reaped():
    """Regression: on macOS killpg on a zombie-only group raises EPERM, which escaped cleanup
    as an internal error after a wall-clock timeout (prove-human R8)."""
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                            stdin=subprocess.PIPE, start_new_session=True, close_fds=True)
    wait_exited = _arm_exit_wait(proc.pid)
    proc.stdin.close()
    wait_exited()
    assert proc.returncode is None  # exited, still unreaped: the state cleanup meets
    executor._kill_group(proc)
    assert proc.returncode is not None


def test_cleanup_docker_calls_run_outside_the_terminal_process_group(monkeypatch):
    """A terminal Ctrl-C/hangup goes to the foreground group; cleanup children must not be in it."""
    seen = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs))
        if argv == ["rm"]:
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if len(seen) == 1:
            return subprocess.CompletedProcess(argv, 0, b"run1\n", b"")
        return subprocess.CompletedProcess(argv, 1, b"", b"Error: No such container: c")

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    status = executor.cleanup_owned(executor.Cleanup(["inspect"], ["rm"], "run1"), {}, 10)
    assert status == executor.CLEAN_REMOVED
    assert [argv for argv, _ in seen] == [["inspect"], ["rm"], ["inspect"]]
    assert all(kw.get("start_new_session") is True for _, kw in seen)


def test_ctrl_c_outside_the_signal_gate_is_invalid_and_writes_no_result(monkeypatch, tmp_path):
    out = tmp_path / "out"

    def interrupted_bind(_ctx, _cli_home, _suite):
        raise KeyboardInterrupt  # default SIGINT handler: no executor gate is installed here

    monkeypatch.setattr(main, "_resolve", lambda _args: {"out": out})
    monkeypatch.setattr(main, "_bind_runtime", interrupted_bind)
    argv = ["run"]
    for flag in ("runtime", "docker", "socket", "image", "python", "snapshot", "suite",
                 "inputs", "receipt", "attestation"):
        argv += [f"--{flag}", "x"]
    argv += ["--out", str(out)]
    assert main.main(argv) == policy.EXIT_INVALID
    assert out.is_dir()
    assert not (out / "result.json").exists()
    assert not (out / "attestation.json").exists()


class _PipeTty:
    def __init__(self):
        self.r, self.w = os.pipe()
        self.written = []

    def fileno(self):
        return self.r

    def isatty(self):
        return True

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass

    def type(self, data: bytes):
        os.write(self.w, data)

    def close(self):
        os.close(self.r)
        os.close(self.w)


@pytest.fixture
def tty():
    t = _PipeTty()
    yield t
    t.close()


def _clock(*values):
    seq = list(values)
    return lambda: seq.pop(0) if len(seq) > 1 else seq[0]


@pytest.mark.platforms("posix")
def test_partial_line_past_deadline_is_refused(tty):
    tty.type(b"AB12")
    with pytest.raises(proof.ProofRefused, match="no confirmation within 120s"):
        proof._read_line(tty, 120, clock=_clock(0.0, 1.0, 121.0))


@pytest.mark.platforms("posix")
def test_complete_line_returns_only_the_first_line(tty):
    tty.type(b"AB12\nleftover")
    assert proof._read_line(tty, 120, clock=_clock(0.0, 1.0)) == "AB12"


@pytest.mark.platforms("posix")
def test_confirm_accepts_the_typed_code(tty, monkeypatch):
    monkeypatch.setattr(proof, "new_code", lambda: "C0DE42")
    tty.type(b"C0DE42\n")
    assert proof.confirm(tty, "prompt") == "C0DE42"


@pytest.mark.platforms("posix")
def test_confirm_rejects_a_wrong_code(tty, monkeypatch):
    monkeypatch.setattr(proof, "new_code", lambda: "C0DE42")
    tty.type(b"C0DE43\n")
    with pytest.raises(proof.ProofRefused, match="mismatch"):
        proof.confirm(tty, "prompt")


@pytest.mark.platforms("posix")
def test_open_tty_opens_a_device_that_cannot_seek(tmp_path):
    """Regression: open("/dev/tty", "r+") demanded seek, so every real prove-human run refused."""
    fifo = tmp_path / "tty"
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        tty = proof.open_tty(str(fifo))
        assert tty is not None
        assert tty.isatty() is False
        tty.write("code\n")
        tty.flush()
        assert os.read(reader, 64) == b"code\n"
        fd = tty.fileno()
        tty.close()
        assert tty.closed
        with pytest.raises(OSError):
            os.fstat(fd)
    finally:
        os.close(reader)


def test_open_tty_without_a_device_returns_none(tmp_path):
    assert proof.open_tty(str(tmp_path / "missing")) is None

"""Bounded execution of one docker CLI process, with owned-container cleanup.

stdout/stderr are read to the end but only the first N bytes are kept; the
wall clock is enforced by the host. Whatever ends the wait inside the signal
gate (normal exit, timeout, SIGINT/SIGTERM/SIGHUP, an exception) the `finally`
block runs the owned cleanup, kills the CLI's process group and waits for the
reader threads. SIGINT/SIGTERM/SIGHUP never raise here: the handler records
the first signal and kills the CLI group, so a second signal during cleanup
cannot abort it or skip the handler restore; any recorded signal marks the
outcome interrupted. Removal is attempted only after the container's run label
is confirmed, and its absence is then confirmed with Docker's own "No such
container" answer. An inspect error is never treated as "not found". When the
label or the removal cannot be confirmed the status is not removed/absent, the
run maps to exit 8 (INVALID) and the container may be left behind. Cleanup's
docker calls run in their own session, so a terminal Ctrl-C/hangup sent to the
foreground group does not reach them. Nothing else is removed or pruned.

Limit: if the launcher itself is SIGKILLed or the host loses power, no
`finally` runs; the container then stops only by Docker's own handling.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass

_NO_SUCH = (b"no such container", b"no such object")

CLEAN_REMOVED = "removed"
CLEAN_ABSENT = "absent"
CLEAN_LABEL_MISMATCH = "label_mismatch"
CLEAN_UNVERIFIED = "unverified"
CLEAN_NOT_CONFIGURED = "not_configured"
CLEAN_OK = frozenset({CLEAN_REMOVED, CLEAN_ABSENT})


@dataclass(frozen=True)
class Cleanup:
    label_argv: list[str]
    rm_argv: list[str]
    expected_label: str


@dataclass(frozen=True)
class Outcome:
    returncode: int | None
    timed_out: bool
    interrupted: str | None
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    readers_done: bool
    cleanup_status: str


def _drain(stream, cap: int, sink: list) -> None:
    kept = bytearray()
    truncated = False
    for chunk in iter(lambda: stream.read(65536), b""):
        room = cap - len(kept)
        if room > 0:
            kept += chunk[:room]
        if len(chunk) > max(room, 0):
            truncated = True
    sink.append((bytes(kept), truncated))


class _SignalGate:
    """Records SIGINT/SIGTERM/SIGHUP instead of raising, so cleanup always runs to the end."""

    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def __init__(self) -> None:
        self.received: list[str] = []
        self.proc: subprocess.Popen | None = None
        self._old: dict = {}

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in self.SIGNALS:
            self._old[sig] = signal.signal(sig, self._handler)

    def _handler(self, signum, _frame) -> None:
        self.received.append(signal.Signals(signum).name)
        self.kill()

    def kill(self) -> None:
        """Stop the CLI group so `wait` returns; reaping stays with `_finish`."""
        if self.proc is not None and self.proc.returncode is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def restore(self) -> None:
        for sig, prev in self._old.items():
            signal.signal(sig, prev)


def run_bounded(argv: list[str], env: dict[str, str], *, wall_seconds: float,
                stdout_cap: int, stderr_cap: int, cleanup: Cleanup | None,
                cleanup_timeout: float) -> Outcome:
    out: list = []
    err: list = []
    readers: list[threading.Thread] = []
    proc = None
    timed_out = False
    interrupted = None
    gate = _SignalGate()
    gate.install()
    try:
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                shell=False, start_new_session=True, close_fds=True)
        gate.proc = proc
        if gate.received:  # signalled before the group existed
            gate.kill()
        readers = [threading.Thread(target=_drain, args=(proc.stdout, stdout_cap, out), daemon=True),
                   threading.Thread(target=_drain, args=(proc.stderr, stderr_cap, err), daemon=True)]
        for t in readers:
            t.start()
        proc.wait(timeout=wall_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
    except Exception as exc:  # noqa: BLE001 - any launcher error must still clean up
        interrupted = f"exception:{type(exc).__name__}"
    finally:
        status = _finish(proc, readers, env, cleanup, cleanup_timeout)
        gate.restore()
    if gate.received and interrupted is None:
        interrupted = gate.received[0]
    readers_done = all(not t.is_alive() for t in readers)
    so, so_t = out[0] if out else (b"", True)
    se, se_t = err[0] if err else (b"", True)
    rc = None if (timed_out or interrupted or proc is None) else proc.returncode
    return Outcome(rc, timed_out, interrupted, so, se, so_t, se_t, readers_done, status)


def _finish(proc, readers, env, cleanup, timeout) -> str:
    # Container first (kills the in-container tree), then the CLI group.
    status = cleanup_owned(cleanup, env, timeout) if cleanup else CLEAN_NOT_CONFIGURED
    if proc is not None and proc.returncode is None:
        _kill_group(proc)
    if proc is not None and proc.returncode is not None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # stray children of the CLI group
        except (ProcessLookupError, PermissionError):
            pass
    for t in readers:
        t.join(timeout=timeout)
    return status


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin answers EPERM, not ESRCH, when every member of the group is a zombie: the
        # CLI exited (e.g. its container was just removed) but is not reaped yet. The group
        # is our own same-uid child's fresh session, so EPERM has no other plausible cause.
        # Signal the leader directly instead: Popen.kill() reaps a zombie via poll() first,
        # and a live leader cannot survive SIGKILL, so the wait below ends. A PermissionError
        # here is a real anomaly and propagates rather than being swallowed.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    proc.wait()


def _run(argv: list[str], env: dict[str, str], timeout: float):
    try:
        # Own session: a terminal SIGINT/SIGHUP to the foreground group must not kill cleanup.
        done = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                              shell=False, timeout=timeout, check=False, start_new_session=True)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return done


def _absent(done) -> bool:
    return done is not None and done.returncode != 0 and any(
        s in done.stderr.lower() for s in _NO_SUCH)


def cleanup_owned(c: Cleanup, env: dict[str, str], timeout: float) -> str:
    """Remove this run's container only if its label proves ownership."""
    probe = _run(c.label_argv, env, timeout)
    if _absent(probe):
        return CLEAN_ABSENT
    if probe is None or probe.returncode != 0:
        return CLEAN_UNVERIFIED
    if probe.stdout.decode("utf-8", "replace").strip() != c.expected_label:
        return CLEAN_LABEL_MISMATCH
    removed = _run(c.rm_argv, env, timeout)
    if removed is None or removed.returncode != 0:
        return CLEAN_UNVERIFIED
    return CLEAN_REMOVED if _absent(_run(c.label_argv, env, timeout)) else CLEAN_UNVERIFIED

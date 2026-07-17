from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkerLaunchSpec:
    command: tuple[str, ...]
    cwd: str | None
    log_path: str
    handshake_path: str
    task_id: str
    run_id: int
    claim_lock: str
    board: str


@dataclass(frozen=True, slots=True)
class InvalidWorkerLaunchSpecError(Exception):
    field: str
    expected: str

    def __str__(self) -> str:
        return f"worker supervisor {self.field} must be {self.expected}"


def _read_spec(path: Path) -> WorkerLaunchSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    command = raw["command"]
    if not isinstance(command, list) or not all(
        isinstance(part, str) for part in command
    ):
        raise InvalidWorkerLaunchSpecError("command", "a list of strings")
    cwd = raw.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise InvalidWorkerLaunchSpecError("cwd", "a string or null")
    return WorkerLaunchSpec(
        command=tuple(command),
        cwd=cwd,
        log_path=str(raw["log_path"]),
        handshake_path=str(raw["handshake_path"]),
        task_id=str(raw["task_id"]),
        run_id=int(raw["run_id"]),
        claim_lock=str(raw["claim_lock"]),
        board=str(raw["board"]),
    )


def _write_supervisor_pid(path: Path, pid: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(pid))
        handle.flush()


_WORKER_TERMINATE_TIMEOUT_SECONDS = 5.0
_WORKER_TREE_TERMINATE_POLL_SECONDS = 0.05
_IS_WINDOWS = sys.platform == "win32"


def _own_fresh_process_group() -> int | None:
    """Return our session/group only when it is provably the fresh one.

    The dispatcher starts the supervisor in a new session.  Do not turn a
    publication error into a broad process-group signal when that invariant is
    absent (for example, a direct unit invocation in an existing shell group).
    """
    if _IS_WINDOWS:
        return None
    pid = os.getpid()
    try:
        pgid = os.getpgid(pid)
        sid = os.getsid(pid)
    except (AttributeError, OSError):
        return None
    return pgid if pgid == pid == sid else None


def _linux_group_has_live_member(pgid: int, *, exclude_pid: int) -> bool | None:
    """Return executable-member liveness, treating zombies as already dead.

    The supervisor itself is intentionally excluded: it remains alive long
    enough to preserve the original publication exception and reap its direct
    child.  Any inaccessible ``/proc`` state is deliberately unproven.
    """
    if sys.platform != "linux":
        return None
    try:
        entries = list(os.scandir("/proc"))
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdecimal() or int(entry.name) == exclude_pid:
            continue
        try:
            with open(entry.path + "/stat", "r", encoding="utf-8") as handle:
                raw = handle.read()
            # After the final ')' the fields are state, ppid, pgrp, ... .
            close = raw.rfind(")")
            if close < 0:
                return None
            fields = raw[close + 2:].split()
            if len(fields) >= 3 and int(fields[2]) == pgid and fields[0] != "Z":
                return True
        except FileNotFoundError:
            # A process that exited after /proc enumeration is benign.
            continue
        except (PermissionError, OSError, ValueError):
            # A visible unreadable or malformed member makes emptiness
            # unprovable; do not collapse it into a zombie-only group.
            return None
    return False


def _wait_for_owned_group_empty(pgid: int, timeout: float) -> bool:
    """Wait until no executable peer remains in our exact fresh group."""
    deadline = time.monotonic() + timeout
    while True:
        live = _linux_group_has_live_member(pgid, exclude_pid=os.getpid())
        if live is False:
            return True
        if live is None or time.monotonic() >= deadline:
            return False
        time.sleep(_WORKER_TREE_TERMINATE_POLL_SECONDS)


def _kill_owned_group_peers(pgid: int) -> bool | None:
    """Escalate only peers in the supervisor's verified fresh group.

    Returns ``True`` only when the /proc scan was complete; ``None`` means an
    unreadable or malformed visible peer left group emptiness unproven.

    Calling ``killpg(pgid, SIGKILL)`` from the supervisor would kill the
    supervisor before it can reap its direct child or re-raise the *original*
    PID-publication failure.  The group was first proven to be our newly-owned
    session and received a group TERM.  For the required KILL escalation, kill
    every remaining peer in that exact group while excluding the supervisor.
    This has the same tree target without permitting a self-kill race.
    """
    if sys.platform != "linux":
        return None
    try:
        entries = list(os.scandir("/proc"))
    except OSError:
        return None
    me = os.getpid()
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        peer_pid = int(entry.name)
        if peer_pid == me:
            continue
        try:
            with open(entry.path + "/stat", "r", encoding="utf-8") as handle:
                raw = handle.read()
            close = raw.rfind(")")
            if close < 0:
                return None
            fields = raw[close + 2:].split()
            if len(fields) >= 3 and int(fields[2]) == pgid and fields[0] != "Z":
                os.kill(peer_pid, signal.SIGKILL)
        except FileNotFoundError:
            continue
        except ProcessLookupError:
            # A peer that vanished before its signal is a benign exit race.
            continue
        except (PermissionError, OSError, ValueError):
            return None
    return True


def _terminate_and_reap_worker(worker: subprocess.Popen[bytes]) -> None:
    """Stop a worker that cannot be safely registered, preserving the cause."""
    if _IS_WINDOWS:
        try:
            subprocess.run(  # noqa: S603
                ["taskkill", "/PID", str(worker.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            pass
        try:
            worker.wait(timeout=_WORKER_TERMINATE_TIMEOUT_SECONDS)
            return
        except subprocess.TimeoutExpired:
            try:
                worker.kill()
            except OSError:
                pass
        except OSError:
            return
        try:
            worker.wait(timeout=_WORKER_TERMINATE_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return

    pgid = _own_fresh_process_group()
    if pgid is not None:
        try:
            # Do not group-signal an arbitrary/fake direct worker just because
            # this interpreter happens to be a session leader (common in test
            # runners).  The real worker must actually be in our exact group.
            if os.getpgid(worker.pid) != pgid:
                pgid = None
        except (AttributeError, ProcessLookupError, OSError):
            pgid = None
    if pgid is not None:
        # The worker inherits the supervisor's freshly-owned session.  TERM the
        # exact group first so grandchildren see the shutdown request too.
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (AttributeError, ProcessLookupError, OSError):
            pass
        if not _wait_for_owned_group_empty(pgid, _WORKER_TERMINATE_TIMEOUT_SECONDS):
            # See _kill_owned_group_peers: self-exclusion is what lets this
            # function finish the direct-child reap and preserve the original
            # _write_supervisor_pid exception.
            _kill_owned_group_peers(pgid)
            _wait_for_owned_group_empty(pgid, _WORKER_TERMINATE_TIMEOUT_SECONDS)
    try:
        worker.terminate()
    except OSError:
        pass
    try:
        worker.wait(timeout=_WORKER_TERMINATE_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        try:
            worker.kill()
        except OSError:
            pass
    except OSError:
        return
    try:
        worker.wait()
    except OSError:
        pass


def _supervisor_exit_code(returncode: int) -> int:
    if returncode >= 0:
        return min(returncode, 255)
    return min(128 + abs(returncode), 255)


def _install_worker_signal_forwarders(
    worker: subprocess.Popen[bytes], stopped: list[int],
) -> dict[int, object]:
    """Make stop/reclaim signals reach and reap the real worker.

    POSIX dispatch tracks this supervisor PID, not the worker's. Forwarding is
    therefore required for direct ``kill(supervisor_pid, SIGTERM)`` callers;
    parent handshake cleanup additionally signals the entire process group.
    """
    if sys.platform == "win32":
        return {}

    old_handlers: dict[int, object] = {}

    def _forward(signum, _frame):
        # Do not wait here: a signal can interrupt ``worker.wait()`` while its
        # internal waitpid lock is held, so recursively waiting deadlocks. The
        # run loop below owns the bounded terminate -> kill -> reap sequence.
        stopped.append(signum)
        try:
            worker.terminate()
        except OSError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            old_handlers[sig] = signal.signal(sig, _forward)
        except (ValueError, OSError):
            pass
    return old_handlers


def _restore_signal_handlers(old_handlers: dict[int, object]) -> None:
    for sig, handler in old_handlers.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def run(spec_path: Path) -> int:
    spec = _read_spec(spec_path)
    spec_path.unlink(missing_ok=True)

    with Path(spec.log_path).open("ab") as log_handle:
        worker = subprocess.Popen(  # noqa: S603
            list(spec.command),
            cwd=spec.cwd,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
            creationflags=(
                subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            ),
        )
        stopped: list[int] = []
        old_handlers = _install_worker_signal_forwarders(worker, stopped)
        try:
            # The dispatcher intentionally tracks *us*, not this child. We
            # remain alive until authoritative exit classification is durable,
            # preventing generic dead-PID detection from racing rate-limit or
            # clean/protocol semantics.
            _write_supervisor_pid(Path(spec.handshake_path), os.getpid())
            terminate_deadline: float | None = None
            while True:
                if stopped and terminate_deadline is None:
                    terminate_deadline = time.monotonic() + _WORKER_TERMINATE_TIMEOUT_SECONDS
                try:
                    returncode = worker.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    if terminate_deadline is not None and time.monotonic() >= terminate_deadline:
                        try:
                            worker.kill()
                        except OSError:
                            pass
                        returncode = worker.wait()
                        break
            if stopped:
                raise SystemExit(128 + stopped[-1])
        except Exception:
            _terminate_and_reap_worker(worker)
            raise
        finally:
            _restore_signal_handlers(old_handlers)

    from hermes_cli import kanban_db as kb

    kb.finalize_worker_exit(
        kb.WorkerExitObservation(
            task_id=spec.task_id,
            run_id=spec.run_id,
            claim_lock=spec.claim_lock,
            pid=os.getpid(),
            returncode=returncode,
        ),
        board=spec.board,
    )
    return _supervisor_exit_code(returncode)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        code = run(Path(sys.argv[1]))
    except Exception:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
        logging.getLogger(__name__).exception("kanban worker supervisor failed")
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()

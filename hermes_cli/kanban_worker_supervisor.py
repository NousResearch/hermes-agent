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


def _terminate_and_reap_worker(worker: subprocess.Popen[bytes]) -> None:
    """Stop a worker that cannot be safely registered, preserving the cause."""
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

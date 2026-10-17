"""Linux fail-closed gate for the explicit Kanban same-lane resume.

The wrapper performs no Hermes imports.  It installs the parent-death contract,
records/fsyncs its identity, and waits for the controller's post-commit token.
Only then does it verify the committed run, persist ``resume_released``,
acknowledge, clear the pre-publication parent-death signal, and exec Hermes.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import select
import signal
import sqlite3
import sys
import time
from pathlib import Path

_PR_SET_PDEATHSIG = 1


def _process_start(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = raw.rsplit(")", 1)[1].strip().split()
    return int(fields[19])


def _install_parent_death(expected_parent: int, expected_parent_start: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    if os.getppid() != expected_parent or _process_start(expected_parent) != expected_parent_start:
        os._exit(125)


def _clear_parent_death() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, 0, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _atomic_identity(path: Path, payload: dict) -> None:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.child")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _read_line(fd: int, deadline: float) -> str:
    data = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("gate deadline expired")
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            raise TimeoutError("gate deadline expired")
        chunk = os.read(fd, 1)
        if not chunk:
            raise RuntimeError("gate closed before commit")
        if chunk == b"\n":
            return data.decode("utf-8")
        data.extend(chunk)
        if len(data) > 8192:
            raise RuntimeError("gate token too large")


def _persist_release(args, identity: dict, token: dict) -> tuple[int, str]:
    if token != {
        "commit": True,
        "task_id": args.task_id,
        "run_id": int(token.get("run_id", -1)),
        "authorization_id": args.authorization_id,
    }:
        raise RuntimeError("invalid commit token")
    run_id = int(token["run_id"])
    conn = sqlite3.connect(args.db, isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("BEGIN IMMEDIATE")
        task = conn.execute(
            "SELECT status, current_run_id, claim_lock, worker_pid, worker_started_at, session_id "
            "FROM tasks WHERE id = ?",
            (args.task_id,),
        ).fetchone()
        if task is None or task["status"] != "running" or int(task["current_run_id"] or 0) != run_id:
            raise RuntimeError("published task/run identity missing")
        if (
            int(task["worker_pid"] or 0) != identity["pid"]
            or int(task["worker_started_at"] or 0) != identity["process_started_at"]
            or task["session_id"] != args.session_id
        ):
            raise RuntimeError("published child identity mismatch")
        spawned = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
            "AND kind = 'spawned' ORDER BY id DESC LIMIT 1",
            (args.task_id, run_id),
        ).fetchone()
        payload = json.loads(spawned["payload"] or "{}") if spawned else {}
        if payload.get("resume_authorization_id") != args.authorization_id:
            raise RuntimeError("published authorization receipt missing")
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, 'resume_released', ?, ?)",
            (
                args.task_id,
                run_id,
                json.dumps({
                    "same_lane_resume": True,
                    "resume_authorization_id": args.authorization_id,
                    "pid": identity["pid"],
                    "session_id": args.session_id,
                }, separators=(",", ":")),
                int(time.time()),
            ),
        )
        conn.execute("COMMIT")
        return run_id, str(task["claim_lock"] or "")
    except Exception:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gate-fd", type=int, required=True)
    parser.add_argument("--ack-fd", type=int, required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--identity-path", required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-start", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("worker_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.worker_argv and args.worker_argv[0] == "--":
        args.worker_argv = args.worker_argv[1:]
    if not args.worker_argv:
        raise RuntimeError("worker argv is required")

    _install_parent_death(args.parent_pid, args.parent_start)
    identity = {
        "resume_authorization_id": args.authorization_id,
        "launch_nonce": args.launch_nonce,
        "phase": "gated",
        "pid": os.getpid(),
        "process_started_at": _process_start(os.getpid()),
        "process_group_id": os.getpgrp(),
        "session_id": args.session_id,
        "gate_deadline": args.deadline,
        "controller_pid": args.parent_pid,
        "controller_started_at": args.parent_start,
    }
    _atomic_identity(Path(args.identity_path), identity)
    os.write(args.ack_fd, (json.dumps(identity, separators=(",", ":")) + "\n").encode("utf-8"))

    monotonic_deadline = time.monotonic() + max(0.0, args.deadline - time.time())
    token = json.loads(_read_line(args.gate_fd, monotonic_deadline))
    run_id, claim_lock = _persist_release(args, identity, token)
    _clear_parent_death()
    # The release receipt is canonical. If the controller dies after sending
    # the token, its ack pipe may already be closed; that race must not turn a
    # durably released child into a dead RUNNING handle.
    with contextlib.suppress(OSError):
        os.write(args.ack_fd, b'{"released":true}\n')
    env = dict(os.environ)
    env["HERMES_KANBAN_RUN_ID"] = str(run_id)
    env["HERMES_KANBAN_CLAIM_LOCK"] = claim_lock
    env["HERMES_SESSION_ID"] = args.session_id
    os.execvpe(args.worker_argv[0], args.worker_argv, env)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        with contextlib.suppress(Exception):
            # The controller treats any non-ready/ack exit as launch failure.
            os.write(2, f"same-lane gate failed: {exc}\n".encode("utf-8", errors="replace"))
        raise

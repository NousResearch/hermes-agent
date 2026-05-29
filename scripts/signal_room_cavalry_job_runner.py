#!/usr/bin/env python3
"""Durable Signal Room Cavalry job queue and Windows worker bridge."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from signal_room_windows_job_runner import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS,
    RunResult,
    default_runner,
    run_once as _run_once,
    run_worker as _run_worker,
    submit_job as _submit_job,
    write_windows_worker_bundle as _write_windows_worker_bundle,
)


DEFAULT_QUEUE_ROOT = Path("windows") / "cavalry" / "jobs"
DEFAULT_WORKER_DIR = Path("windows") / "cavalry"


def submit_job(
    *,
    queue_root: Path = DEFAULT_QUEUE_ROOT,
    job_id: str | None = None,
    command: Sequence[str],
    cwd: Path | None = None,
    inputs: Sequence[Path] = (),
    outputs: Sequence[Path] = (),
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    return _submit_job(
        lane="cavalry",
        queue_root=queue_root,
        job_id=job_id,
        command=command,
        cwd=cwd,
        inputs=inputs,
        outputs=outputs,
        timeout_seconds=timeout_seconds,
    )


def run_once(
    *,
    queue_root: Path = DEFAULT_QUEUE_ROOT,
    runner: Callable[..., RunResult] = default_runner,
) -> dict[str, Any]:
    return _run_once(queue_root=queue_root, runner=runner)


def run_worker(
    *,
    queue_root: Path = DEFAULT_QUEUE_ROOT,
    poll_seconds: float = 10.0,
    max_jobs: int | None = None,
    runner: Callable[..., RunResult] = default_runner,
) -> dict[str, Any]:
    return _run_worker(queue_root=queue_root, poll_seconds=poll_seconds, max_jobs=max_jobs, runner=runner)


def write_windows_worker_bundle(
    *,
    queue_root: Path = DEFAULT_QUEUE_ROOT,
    out_dir: Path = DEFAULT_WORKER_DIR,
) -> dict[str, str]:
    return _write_windows_worker_bundle(
        lane_title="Cavalry",
        runner_script=Path(__file__),
        queue_root=queue_root,
        out_dir=out_dir,
        powershell_name="run-cavalry-worker.ps1",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    submit.add_argument("--job-id")
    submit.add_argument("--cwd", type=Path)
    submit.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    submit.add_argument("--input", action="append", type=Path, default=[])
    submit.add_argument("--output", action="append", type=Path, default=[])
    submit.add_argument("command", nargs=argparse.REMAINDER)

    run_once_parser = subparsers.add_parser("run-once")
    run_once_parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    worker.add_argument("--poll-seconds", type=float, default=10.0)
    worker.add_argument("--max-jobs", type=int)

    bundle = subparsers.add_parser("write-worker-bundle")
    bundle.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    bundle.add_argument("--out-dir", type=Path, default=DEFAULT_WORKER_DIR)

    args = parser.parse_args()
    if args.command_name == "submit":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        result = submit_job(
            queue_root=args.queue_root,
            job_id=args.job_id,
            command=args.command,
            cwd=args.cwd,
            inputs=args.input,
            outputs=args.output,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command_name == "run-once":
        result = run_once(queue_root=args.queue_root)
    elif args.command_name == "worker":
        result = run_worker(queue_root=args.queue_root, poll_seconds=args.poll_seconds, max_jobs=args.max_jobs)
    else:
        result = write_windows_worker_bundle(queue_root=args.queue_root, out_dir=args.out_dir)

    print(json.dumps(result, indent=2))
    if args.command_name == "run-once" and result.get("processed") and result.get("status") != "done":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

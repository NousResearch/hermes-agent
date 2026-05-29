#!/usr/bin/env python3
"""Durable Signal Room Moho pose-export job queue and Windows worker bridge."""
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
    recover_stale_running_jobs as _recover_stale_running_jobs,
    run_once as _run_once,
    run_worker as _run_worker,
    submit_job as _submit_job,
    write_windows_worker_bundle as _write_windows_worker_bundle,
)


DEFAULT_QUEUE_ROOT = Path("windows") / "moho" / "jobs"
DEFAULT_WORKER_DIR = Path("windows") / "moho"


def submit_pose_export_job(
    *,
    queue_root: Path = DEFAULT_QUEUE_ROOT,
    job_id: str | None = None,
    command: Sequence[str],
    cwd: Path | None = None,
    project: Path,
    output_dir: Path,
    candidate_name: str,
    expected_frame_count: int,
    license_status: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    metadata = {
        "render_tool": "moho",
        "project": str(project),
        "output_dir": str(output_dir),
        "candidate_name": candidate_name,
        "expected_frame_count": int(expected_frame_count),
        "license_status": license_status,
    }
    return _submit_job(
        lane="moho",
        queue_root=queue_root,
        job_id=job_id,
        command=command,
        cwd=cwd,
        inputs=[project],
        outputs=[output_dir],
        timeout_seconds=timeout_seconds,
        require_outputs=True,
        metadata=metadata,
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
    recover_stale_after_seconds: int | None = None,
    runner: Callable[..., RunResult] = default_runner,
) -> dict[str, Any]:
    return _run_worker(
        queue_root=queue_root,
        poll_seconds=poll_seconds,
        max_jobs=max_jobs,
        recover_stale_after_seconds=recover_stale_after_seconds,
        runner=runner,
    )


def recover_stale_running_jobs(
    *,
    queue_root: Path = DEFAULT_QUEUE_ROOT,
    max_age_seconds: int,
    now: str | None = None,
) -> dict[str, Any]:
    return _recover_stale_running_jobs(queue_root=queue_root, max_age_seconds=max_age_seconds, now=now)


def write_windows_worker_bundle(
    *,
    queue_root: Path = DEFAULT_QUEUE_ROOT,
    out_dir: Path = DEFAULT_WORKER_DIR,
) -> dict[str, str]:
    return _write_windows_worker_bundle(
        lane_title="Moho",
        runner_script=Path(__file__),
        queue_root=queue_root,
        out_dir=out_dir,
        powershell_name="run-moho-worker.ps1",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    submit = subparsers.add_parser("submit-pose-export")
    submit.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    submit.add_argument("--job-id")
    submit.add_argument("--cwd", type=Path)
    submit.add_argument("--project", required=True, type=Path)
    submit.add_argument("--output-dir", required=True, type=Path)
    submit.add_argument("--candidate-name", required=True)
    submit.add_argument("--expected-frame-count", required=True, type=int)
    submit.add_argument("--license-status", required=True)
    submit.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    submit.add_argument("command", nargs=argparse.REMAINDER)

    run_once_parser = subparsers.add_parser("run-once")
    run_once_parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    worker.add_argument("--poll-seconds", type=float, default=10.0)
    worker.add_argument("--max-jobs", type=int)
    worker.add_argument("--recover-stale-after-seconds", type=int)

    recover = subparsers.add_parser("recover-stale")
    recover.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    recover.add_argument("--max-age-seconds", required=True, type=int)

    bundle = subparsers.add_parser("write-worker-bundle")
    bundle.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    bundle.add_argument("--out-dir", type=Path, default=DEFAULT_WORKER_DIR)

    args = parser.parse_args()
    try:
        if args.command_name == "submit-pose-export":
            if args.command and args.command[0] == "--":
                args.command = args.command[1:]
            result = submit_pose_export_job(
                queue_root=args.queue_root,
                job_id=args.job_id,
                command=args.command,
                cwd=args.cwd,
                project=args.project,
                output_dir=args.output_dir,
                candidate_name=args.candidate_name,
                expected_frame_count=args.expected_frame_count,
                license_status=args.license_status,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command_name == "run-once":
            result = run_once(queue_root=args.queue_root)
        elif args.command_name == "worker":
            result = run_worker(
                queue_root=args.queue_root,
                poll_seconds=args.poll_seconds,
                max_jobs=args.max_jobs,
                recover_stale_after_seconds=args.recover_stale_after_seconds,
            )
        elif args.command_name == "recover-stale":
            result = recover_stale_running_jobs(queue_root=args.queue_root, max_age_seconds=args.max_age_seconds)
        else:
            result = write_windows_worker_bundle(queue_root=args.queue_root, out_dir=args.out_dir)
    except ValueError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 2

    print(json.dumps(result, indent=2))
    if args.command_name == "run-once" and result.get("processed") and result.get("status") != "done":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

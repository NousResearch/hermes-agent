#!/usr/bin/env python3
"""Shared durable Windows job queue helpers for Signal Room workers."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_TIMEOUT_SECONDS = 60 * 30
QUEUE_STATES = ("queued", "running", "done", "failed")


class RunResult:
    def __init__(self, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_queue(queue_root: Path) -> None:
    for state in QUEUE_STATES:
        (queue_root / state).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def validate_command(command: Sequence[str]) -> list[str]:
    normalized = [str(part) for part in command if str(part)]
    if not normalized:
        raise ValueError("command must contain at least one argument")
    return normalized


def default_job_id(lane: str) -> str:
    return f"{lane}-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def submit_job(
    *,
    lane: str,
    queue_root: Path,
    job_id: str | None,
    command: Sequence[str],
    cwd: Path | None = None,
    inputs: Sequence[Path] = (),
    outputs: Sequence[Path] = (),
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_queue(queue_root)
    normalized_id = job_id or default_job_id(lane)
    job_path = queue_root / "queued" / f"{normalized_id}.json"
    if job_path.exists():
        raise FileExistsError(f"queued job already exists: {job_path}")

    job = {
        "id": normalized_id,
        "job_type": lane,
        "status": "queued",
        "public_release": False,
        "created_at": now_utc(),
        "attempts": 0,
        "timeout_seconds": int(timeout_seconds),
        "cwd": str(cwd) if cwd else None,
        "command": validate_command(command),
        "inputs": [str(path) for path in inputs],
        "outputs": [str(path) for path in outputs],
    }
    if metadata:
        job.update(metadata)
    write_json(job_path, job)
    return job


def default_runner(command: list[str], *, cwd: str | None, timeout_seconds: int) -> RunResult:
    completed = subprocess.run(
        command,
        cwd=cwd or None,
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
        check=False,
    )
    return RunResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def next_queued_job(queue_root: Path) -> Path | None:
    candidates = sorted((queue_root / "queued").glob("*.json"), key=lambda path: (path.stat().st_mtime, path.name))
    return candidates[0] if candidates else None


def claim_job(queue_root: Path, queued_path: Path) -> tuple[Path, dict[str, Any]]:
    job = read_json(queued_path)
    job["status"] = "running"
    job["claimed_at"] = now_utc()
    job["attempts"] = int(job.get("attempts") or 0) + 1
    running_path = queue_root / "running" / queued_path.name
    write_json(queued_path, job)
    queued_path.replace(running_path)
    return running_path, job


def run_once(
    *,
    queue_root: Path,
    runner: Callable[..., RunResult] = default_runner,
) -> dict[str, Any]:
    ensure_queue(queue_root)
    queued_path = next_queued_job(queue_root)
    if queued_path is None:
        return {"processed": False, "queue_root": str(queue_root)}

    running_path, job = claim_job(queue_root, queued_path)
    command = validate_command(job.get("command") or [])
    try:
        result = runner(
            command,
            cwd=job.get("cwd"),
            timeout_seconds=int(job.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS),
        )
    except Exception as exc:
        result = RunResult(returncode=-1, stderr=f"{type(exc).__name__}: {exc}")

    finished_status = "done" if result.returncode == 0 else "failed"
    job.update(
        {
            "status": finished_status,
            "finished_at": now_utc(),
            "returncode": result.returncode,
            "stdout_tail": tail(result.stdout or ""),
            "stderr_tail": tail(result.stderr or ""),
        }
    )
    final_path = queue_root / finished_status / running_path.name
    write_json(running_path, job)
    running_path.replace(final_path)
    return {
        "processed": True,
        "id": job["id"],
        "status": finished_status,
        "returncode": result.returncode,
        "path": str(final_path),
    }


def run_worker(
    *,
    queue_root: Path,
    poll_seconds: float = 10.0,
    max_jobs: int | None = None,
    runner: Callable[..., RunResult] = default_runner,
) -> dict[str, Any]:
    processed = 0
    while max_jobs is None or processed < max_jobs:
        result = run_once(queue_root=queue_root, runner=runner)
        if result.get("processed"):
            processed += 1
            continue
        if max_jobs is not None:
            break
        time.sleep(poll_seconds)
    return {"processed_jobs": processed, "queue_root": str(queue_root)}


def write_windows_worker_bundle(
    *,
    lane_title: str,
    runner_script: Path,
    queue_root: Path,
    out_dir: Path,
    powershell_name: str,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ps1_path = out_dir / powershell_name
    readme_path = out_dir / "README.md"
    ps1_path.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)",
                "$Python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }",
                f"$Runner = Join-Path $RepoRoot '{runner_script.resolve().relative_to(Path.cwd())}'",
                f"$QueueRootSpec = '{queue_root}'",
                "$QueueRoot = if ([System.IO.Path]::IsPathRooted($QueueRootSpec)) { $QueueRootSpec } else { Join-Path $RepoRoot $QueueRootSpec }",
                "Set-Location $RepoRoot",
                "while ($true) {",
                "  & $Python $Runner worker --queue-root $QueueRoot --max-jobs 1",
                "  Start-Sleep -Seconds 10",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    readme_path.write_text(
        "\n".join(
            [
                f"# {lane_title} Windows Worker",
                "",
                f"Run `{powershell_name}` on the Windows laptop to process queued Signal Room {lane_title} jobs.",
                "",
                f"Queue root: `{queue_root}`",
                "",
                "Job files move through `queued`, `running`, `done`, and `failed` directories so laptop restarts leave an auditable trail.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"powershell": str(ps1_path), "readme": str(readme_path)}

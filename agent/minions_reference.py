from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from agent.job_callbacks import deliver_completion
from agent.job_protocol import build_completion_envelope


def _spool_root(spool_dir: str | Path | None = None) -> Path:
    if spool_dir is not None:
        return Path(spool_dir)
    env = os.getenv("HERMES_MINIONS_SPOOL_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".hermes" / "minions-reference"


def _queue_file(spool_root: Path, envelope: dict[str, Any]) -> Path:
    return spool_root / "queue" / str(envelope["kind"]) / f"{envelope['task_id']}.json"


def _completed_file(spool_root: Path, completion: dict[str, Any]) -> Path:
    return spool_root / "completed" / str(completion["kind"]) / f"{completion['task_id']}.json"


def enqueue_job(envelope: dict[str, Any], *, spool_dir: str | Path | None = None) -> dict[str, Any]:
    spool_root = _spool_root(spool_dir)
    queue_file = _queue_file(spool_root, envelope)
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(queue_file, envelope)
    return {
        "task_id": envelope["task_id"],
        "backend": "reference-minions",
        "queue": str(envelope["kind"]),
        "message": f"queued at {queue_file}",
    }


def list_queued_jobs(*, spool_dir: str | Path | None = None) -> list[Path]:
    spool_root = _spool_root(spool_dir)
    queue_root = spool_root / "queue"
    if not queue_root.exists():
        return []
    return sorted(p for p in queue_root.rglob("*.json") if p.is_file())


def process_next_job(
    *,
    spool_dir: str | Path | None = None,
    runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    adapters=None,
    loop=None,
) -> dict[str, Any] | None:
    jobs = list_queued_jobs(spool_dir=spool_dir)
    if not jobs:
        return None
    queue_file = jobs[0]
    envelope = json.loads(queue_file.read_text(encoding="utf-8"))
    queue_file.unlink()

    job_runner = runner or default_runner
    completion = job_runner(envelope)
    completed_file = _completed_file(_spool_root(spool_dir), completion)
    completed_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(completed_file, completion)
    deliver_completion(completion, adapters=adapters, loop=loop)
    return completion


def default_runner(envelope: dict[str, Any]) -> dict[str, Any]:
    payload = envelope.get("payload") or {}
    kind = envelope.get("kind")
    if kind == "background":
        text = f"[reference-minions] Background task complete: {payload.get('prompt', '')}".strip()
    elif kind == "delegation":
        text = f"[reference-minions] Delegated task complete: {payload.get('goal', '')}".strip()
    elif kind == "cron":
        text = f"[reference-minions] Cron job complete: {payload.get('job_name', '')}".strip()
    else:
        text = f"[reference-minions] Completed {kind or 'job'}"
    return build_completion_envelope(
        kind=str(kind),
        task_id=str(envelope.get("task_id")),
        status="succeeded",
        callback=envelope.get("callback") or {"type": "none"},
        summary=text,
        final_output=text,
        metadata={"worker": "reference-minions"},
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

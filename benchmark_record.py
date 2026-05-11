from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from hermes_state import DEFAULT_DB_PATH, SessionDB


def _normalize_artifacts(artifacts: Optional[Sequence[Path]]) -> list[str]:
    return [str(Path(p)) for p in (artifacts or [])]


def record_benchmark_run(
    *,
    benchmark_name: str,
    prompt: str,
    validation_command: str,
    artifacts: Optional[Sequence[Path]] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
    status: str = "running",
) -> str:
    """Create or update a durable task record for a benchmark run."""
    db = SessionDB(db_path or DEFAULT_DB_PATH)
    task_id = task_id or f"benchmark-{benchmark_name}"
    checkpoint_data = {
        "benchmark_name": benchmark_name,
        "prompt": prompt,
        "validation_command": validation_command,
    }

    if session_id:
        existing_session = db.get_session(session_id)
        if not existing_session:
            db.create_session(session_id=session_id, source="benchmark_record", model=None)

    existing = db.get_task(task_id)
    if existing:
        db.update_task(
            task_id,
            session_id=session_id,
            status=status,
            current_step=f"benchmark:{status}",
            checkpoint_data=checkpoint_data,
        )
    else:
        db.create_task(
            task_id=task_id,
            session_id=session_id,
            status=status,
            current_step=f"benchmark:{status}",
            checkpoint_data=checkpoint_data,
        )

    for artifact in _normalize_artifacts(artifacts):
        db.append_task_artifact(task_id, artifact)

    return task_id


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record a Hermes benchmark run in state.db")
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--validation-command", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--session-id")
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--status", default="running")
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args(list(argv) if argv is not None else None)

    task_id = record_benchmark_run(
        benchmark_name=args.benchmark_name,
        prompt=args.prompt,
        validation_command=args.validation_command,
        artifacts=[Path(a) for a in args.artifact],
        task_id=args.task_id,
        session_id=args.session_id,
        db_path=args.db_path,
        status=args.status,
    )
    print(json.dumps({"success": True, "task_id": task_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-run Kanban worker-log paths and readers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def worker_run_log_path(task_id: str, run_id: int, *, board: Optional[str] = None) -> Path:
    from hermes_cli import kanban_db as kb
    return kb.worker_logs_dir(board=board) / f"{task_id}.{int(run_id)}.log"


def read_worker_log(
    task_id: str, *, tail_bytes: Optional[int] = None, board: Optional[str] = None,
    run_id: Optional[int] = None,
) -> Optional[str]:
    from hermes_cli import kanban_db as kb

    log_dir = kb.worker_logs_dir(board=board)
    legacy = kb.worker_log_path(task_id, board=board)
    if run_id is not None:
        run_path = worker_run_log_path(task_id, run_id, board=board)
        paths = [run_path] if run_path.exists() or not legacy.exists() else [legacy]
    else:
        paths = [legacy] if legacy.exists() else sorted(
            log_dir.glob(f"{task_id}.*.log"), key=lambda path: path.name,
        )
    paths = [path for path in paths if path.exists()]
    if not paths:
        return None
    try:
        chunks: list[str] = []
        for path in paths:
            if tail_bytes is None:
                chunks.append(path.read_text(encoding="utf-8-sig", errors="replace"))
                continue
            size = path.stat().st_size
            with open(path, "rb") as f:
                if size > tail_bytes:
                    f.seek(size - tail_bytes)
                    probe = f.tell()
                    if not f.readline().endswith(b"\n") and f.tell() >= size:
                        f.seek(probe)
                chunks.append(f.read().decode("utf-8", errors="replace"))
        return "".join(chunks)
    except OSError:
        return None

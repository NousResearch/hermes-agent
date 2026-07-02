"""Local Kanban worker progress classifier.

The classifier is intentionally read-only: it inspects the board DB, worker log,
workspace mtimes, optional session counters, and host-local process metadata so
operators can decide whether to wait, comment/steer, or reclaim a worker.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_constants import get_hermes_home

PROGRESS_STATES = {
    "productive",
    "backend_wait",
    "external_subprocess_stuck",
    "no_tool_spin",
    "claim_stale",
    "unknown",
}

RECOMMENDED_ACTIONS: dict[str, str] = {
    "productive": "Keep monitoring; visible local progress is still arriving.",
    "backend_wait": "Comment or keep monitoring; avoid stopping solely for missing local output while heartbeat/backend evidence is fresh.",
    "external_subprocess_stuck": "Prompt the worker to report or continue from the partial artifact; inspect terminal media evidence; do not automatically stop terminal-owned subprocesses.",
    "no_tool_spin": "Send one focused steer asking the worker to use tools or block with evidence; reclaim only if the spin repeats.",
    "claim_stale": "Run the dispatcher/reclaim path (for example hermes kanban dispatch --dry-run --explain) and inspect the worker before respawn.",
    "unknown": "Collect more evidence with hermes kanban show, hermes kanban log, and hermes kanban progress --json before killing anything.",
}

DEFAULT_PROGRESS_GRACE_SECONDS = 180
DEFAULT_EXTERNAL_IDLE_SECONDS = 15 * 60
_WORKSPACE_SCAN_LIMIT = 1000
_LOG_TAIL_BYTES = 4096
_RUNTIME_LOG_TAIL_BYTES = 64 * 1024
_EXTERNAL_PROCESS_NAMES = {"yt-dlp", "yt_dlp", "ffmpeg", "ffprobe"}
_RELEVANT_WAIT_MARKERS = (
    "quick move",
    "quick/move",
    "d:/quick/move",
    r"d:\quick\move",
    "backend",
    "model response",
    "waiting for",
    "transcript",
    "transcription",
    "extract",
    "download",
    "yt-dlp",
    "ffmpeg",
)


@dataclass(frozen=True)
class ProgressResult:
    task_id: str
    state: str
    recommended_action: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state,
            "recommended_action": self.recommended_action,
            "evidence": self.evidence,
        }


def classify_task_progress(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    board: Optional[str] = None,
    now: Optional[int] = None,
    progress_grace_seconds: int = DEFAULT_PROGRESS_GRACE_SECONDS,
    external_idle_seconds: int = DEFAULT_EXTERNAL_IDLE_SECONDS,
) -> ProgressResult:
    now_i = int(time.time() if now is None else now)
    task = kb.get_task(conn, task_id)
    if task is None:
        return _result(task_id, "unknown", {"task_exists": False})

    active_run = _active_run(conn, task)
    heartbeat_ts = _coalesce_int(
        task.last_heartbeat_at,
        active_run.last_heartbeat_at if active_run else None,
    )
    heartbeat_age = _age(now_i, heartbeat_ts)
    claim_expires = _coalesce_int(
        task.claim_expires,
        active_run.claim_expires if active_run else None,
    )
    worker_pid = _coalesce_int(
        task.worker_pid,
        active_run.worker_pid if active_run else None,
    )

    process = _process_snapshot(worker_pid, cwd=task.workspace_path, now=now_i)
    worker_log = _worker_log_snapshot(task_id, board=board, now=now_i)
    workspace = _workspace_snapshot(task, board=board, now=now_i)
    db_progress = _db_progress_snapshot(conn, task_id, now=now_i)
    session = _session_snapshot(task.session_id, now=now_i)
    runtime_logs = _runtime_log_snapshot(task.session_id)

    evidence: dict[str, Any] = {
        "task": {
            "status": task.status,
            "assignee": task.assignee,
            "current_run_id": task.current_run_id,
            "session_id": task.session_id,
        },
        "active_run": _run_summary(active_run),
        "heartbeat_at": heartbeat_ts,
        "heartbeat_age_seconds": heartbeat_age,
        "claim_expires": claim_expires,
        "claim_expires_in_seconds": None if claim_expires is None else claim_expires - now_i,
        "worker_pid": worker_pid,
        "process": process,
        "worker_log": worker_log,
        "workspace": workspace,
        "db_progress": db_progress,
        "session": session,
        "runtime_logs": runtime_logs,
    }

    if _claim_is_stale(now_i, heartbeat_age, claim_expires):
        return _result(task_id, "claim_stale", evidence)

    if task.status != "running":
        evidence["reason"] = "task_not_running"
        return _result(task_id, "unknown", evidence)

    newest_progress_age = _newest_visible_progress_age(
        worker_log=worker_log,
        workspace=workspace,
        db_progress=db_progress,
        session=session,
    )
    evidence["newest_visible_progress_age_seconds"] = newest_progress_age
    if newest_progress_age is not None and newest_progress_age <= progress_grace_seconds:
        return _result(task_id, "productive", evidence)

    if _looks_like_external_subprocess_stuck(
        process,
        newest_progress_age,
        external_idle_seconds=external_idle_seconds,
    ):
        return _result(task_id, "external_subprocess_stuck", evidence)

    if _looks_like_no_tool_spin(session, runtime_logs):
        return _result(task_id, "no_tool_spin", evidence)

    if _looks_like_backend_wait(
        task,
        active_run,
        heartbeat_age,
        process,
        worker_log,
        runtime_logs,
    ):
        return _result(task_id, "backend_wait", evidence)

    return _result(task_id, "unknown", evidence)


def _result(task_id: str, state: str, evidence: dict[str, Any]) -> ProgressResult:
    if state not in PROGRESS_STATES:
        state = "unknown"
    return ProgressResult(
        task_id=task_id,
        state=state,
        recommended_action=RECOMMENDED_ACTIONS[state],
        evidence=evidence,
    )


def _active_run(conn: sqlite3.Connection, task: kb.Task) -> Optional[kb.Run]:
    if task.current_run_id is not None:
        run = kb.get_run(conn, int(task.current_run_id))
        if run is not None and run.ended_at is None:
            return run
    row = conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? AND ended_at IS NULL "
        "ORDER BY started_at DESC, id DESC LIMIT 1",
        (task.id,),
    ).fetchone()
    return kb.Run.from_row(row) if row else None


def _run_summary(run: Optional[kb.Run]) -> Optional[dict[str, Any]]:
    if run is None:
        return None
    return {
        "id": run.id,
        "status": run.status,
        "profile": run.profile,
        "started_at": run.started_at,
        "last_heartbeat_at": run.last_heartbeat_at,
        "worker_pid": run.worker_pid,
        "metadata": run.metadata,
    }


def _coalesce_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _age(now: int, ts: Optional[int | float]) -> Optional[int]:
    if ts is None:
        return None
    try:
        return max(0, now - int(ts))
    except (TypeError, ValueError):
        return None


def _claim_is_stale(
    now: int,
    heartbeat_age: Optional[int],
    claim_expires: Optional[int],
) -> bool:
    if heartbeat_age is not None and heartbeat_age > kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS:
        return True
    if claim_expires is not None and claim_expires < now and heartbeat_age is None:
        return True
    if (
        claim_expires is not None
        and claim_expires < now
        and heartbeat_age is not None
        and heartbeat_age > kb.DEFAULT_CLAIM_TTL_SECONDS
    ):
        return True
    return False


def _process_snapshot(pid: Optional[int], *, cwd: Optional[str] = None, now: Optional[int] = None) -> dict[str, Any]:
    if not pid:
        return {"pid": None, "available": False, "alive": None, "children": []}
    snapshot: dict[str, Any] = {
        "pid": int(pid),
        "available": False,
        "alive": None,
        "children": [],
    }
    try:
        import psutil  # type: ignore
    except Exception as exc:
        snapshot["error"] = f"psutil unavailable: {exc}"
        return snapshot
    snapshot["available"] = True
    try:
        proc = psutil.Process(int(pid))
        alive = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        snapshot.update({
            "alive": bool(alive),
            "name": _safe_call(proc.name),
            "status": _safe_call(proc.status),
            "cpu_percent": _safe_call(proc.cpu_percent, 0.0),
        })
        try:
            from tools.process_registry import external_media_process_evidence

            media = external_media_process_evidence(
                int(pid),
                cwd=cwd,
                now=now,
            )
        except Exception:
            media = []
        if media:
            snapshot["external_media_processes"] = media
        children = []
        for child in proc.children(recursive=True):
            children.append({
                "pid": child.pid,
                "name": _safe_call(child.name),
                "status": _safe_call(child.status),
                "cpu_percent": _safe_call(child.cpu_percent, 0.0),
            })
        snapshot["children"] = children
    except Exception as exc:
        snapshot["alive"] = False
        snapshot["error"] = str(exc)
    return snapshot


def _safe_call(fn, default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _worker_log_snapshot(task_id: str, *, board: Optional[str], now: int) -> dict[str, Any]:
    path = kb.worker_log_path(task_id, board=board)
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    try:
        stat = path.stat()
        info.update({
            "mtime": int(stat.st_mtime),
            "mtime_age_seconds": _age(now, stat.st_mtime),
            "size_bytes": int(stat.st_size),
        })
    except OSError as exc:
        info["error"] = str(exc)
        return info
    status = kb.read_worker_log_status(task_id, tail_bytes=_LOG_TAIL_BYTES, board=board)
    if status is not None:
        info.update({
            "encoding": status.encoding,
            "used_fallback": status.used_fallback,
            "had_replacement": status.had_replacement,
            "truncated": status.truncated,
            "tail": status.content[-1200:],
        })
    return info


def _workspace_snapshot(task: kb.Task, *, board: Optional[str], now: int) -> dict[str, Any]:
    path = Path(task.workspace_path) if task.workspace_path else kb.workspaces_root(board=board) / task.id
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "files_scanned": 0,
        "total_size_bytes": 0,
        "latest_mtime": None,
        "latest_mtime_age_seconds": None,
        "latest_path": None,
    }
    if not path.exists() or not path.is_dir():
        return info
    latest_mtime: Optional[float] = None
    latest_path: Optional[Path] = None
    total_size = 0
    scanned = 0
    try:
        for child in path.rglob("*"):
            if scanned >= _WORKSPACE_SCAN_LIMIT:
                info["truncated"] = True
                break
            if not child.is_file():
                continue
            scanned += 1
            try:
                stat = child.stat()
            except OSError:
                continue
            total_size += int(stat.st_size)
            if latest_mtime is None or stat.st_mtime > latest_mtime:
                latest_mtime = stat.st_mtime
                latest_path = child
    except OSError as exc:
        info["error"] = str(exc)
    info.update({
        "files_scanned": scanned,
        "total_size_bytes": total_size,
        "latest_mtime": int(latest_mtime) if latest_mtime is not None else None,
        "latest_mtime_age_seconds": _age(now, latest_mtime),
        "latest_path": str(latest_path) if latest_path else None,
    })
    return info


def _db_progress_snapshot(conn: sqlite3.Connection, task_id: str, *, now: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT kind, created_at FROM task_events WHERE task_id = ? "
        "AND kind NOT IN ("
        "'heartbeat', 'claim_extended', 'claimed', 'created', 'promoted', "
        "'promoted_manual', 'claim_rejected') "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row:
        return {"latest_nonheartbeat_event": None}
    ts = int(row["created_at"])
    return {
        "latest_nonheartbeat_event": row["kind"],
        "latest_nonheartbeat_event_at": ts,
        "latest_nonheartbeat_event_age_seconds": _age(now, ts),
    }


def _session_snapshot(session_id: Optional[str], *, now: int) -> dict[str, Any]:
    if not session_id:
        return {"session_id": None, "available": False}
    db_path = get_hermes_home() / "state.db"
    info: dict[str, Any] = {
        "session_id": session_id,
        "available": db_path.exists(),
        "db_path": str(db_path),
    }
    if not db_path.exists():
        return info
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT message_count, tool_call_count, api_call_count FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row:
                info.update({
                    "message_count": int(row["message_count"] or 0),
                    "tool_call_count": int(row["tool_call_count"] or 0),
                    "api_call_count": int(row["api_call_count"] or 0),
                })
            msg = conn.execute(
                "SELECT role, tool_name, timestamp FROM messages WHERE session_id = ? "
                "ORDER BY timestamp DESC, id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if msg:
                ts = int(float(msg["timestamp"]))
                info["latest_message"] = {
                    "role": msg["role"],
                    "tool_name": msg["tool_name"],
                    "timestamp": ts,
                    "age_seconds": _age(now, ts),
                }
    except sqlite3.Error as exc:
        info["error"] = str(exc)
    return info


def _runtime_log_snapshot(session_id: Optional[str]) -> dict[str, Any]:
    if not session_id:
        return {"session_id": None, "available": False, "matches": []}
    logs_dir = get_hermes_home() / "logs"
    info: dict[str, Any] = {"session_id": session_id, "available": logs_dir.exists(), "matches": []}
    if not logs_dir.exists():
        return info
    matches: list[str] = []
    for path in sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:8]:
        try:
            data = _tail_bytes(path, _RUNTIME_LOG_TAIL_BYTES).decode("utf-8", errors="replace")
        except OSError:
            continue
        for line in data.splitlines():
            if session_id in line:
                matches.append(line[-500:])
    info["matches"] = matches[-5:]
    info["match_count"] = len(matches)
    return info


def _tail_bytes(path: Path, limit: int) -> bytes:
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > limit:
            f.seek(size - limit)
        return f.read()


def _newest_visible_progress_age(
    *,
    worker_log: dict[str, Any],
    workspace: dict[str, Any],
    db_progress: dict[str, Any],
    session: dict[str, Any],
) -> Optional[int]:
    ages: list[int] = []
    for value in (
        worker_log.get("mtime_age_seconds"),
        workspace.get("latest_mtime_age_seconds"),
        db_progress.get("latest_nonheartbeat_event_age_seconds"),
    ):
        if isinstance(value, int):
            ages.append(value)
    latest_msg = session.get("latest_message") if isinstance(session, dict) else None
    if isinstance(latest_msg, dict) and isinstance(latest_msg.get("age_seconds"), int):
        ages.append(latest_msg["age_seconds"])
    return min(ages) if ages else None


def _looks_like_external_subprocess_stuck(
    process: dict[str, Any],
    newest_progress_age: Optional[int],
    *,
    external_idle_seconds: int,
) -> bool:
    media = process.get("external_media_processes") or []
    if any(item.get("is_idle") for item in media):
        return True
    if newest_progress_age is None or newest_progress_age < external_idle_seconds:
        return False
    children = process.get("children") or []
    for child in children:
        name = str(child.get("name") or "").lower()
        if any(marker in name for marker in _EXTERNAL_PROCESS_NAMES):
            cpu = child.get("cpu_percent")
            try:
                if cpu is None or float(cpu) <= 1.0:
                    return True
            except (TypeError, ValueError):
                return True
    return False


def _looks_like_no_tool_spin(session: dict[str, Any], runtime_logs: dict[str, Any]) -> bool:
    try:
        api_calls = int(session.get("api_call_count") or 0)
        tool_calls = int(session.get("tool_call_count") or 0)
    except (TypeError, ValueError):
        return False
    runtime_text = "\n".join(runtime_logs.get("matches") or []).lower()
    return api_calls >= 2 and tool_calls == 0 and "steer" in runtime_text


def _looks_like_backend_wait(
    task: kb.Task,
    active_run: Optional[kb.Run],
    heartbeat_age: Optional[int],
    process: dict[str, Any],
    worker_log: dict[str, Any],
    runtime_logs: dict[str, Any],
) -> bool:
    heartbeat_fresh = heartbeat_age is not None and heartbeat_age <= kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
    process_alive = process.get("alive") is True
    if not (heartbeat_fresh or process_alive):
        return False
    evidence_text = "\n".join(
        str(part or "")
        for part in (
            task.title,
            task.body,
            json.dumps(active_run.metadata or {}, ensure_ascii=False) if active_run else "",
            worker_log.get("tail"),
            "\n".join(runtime_logs.get("matches") or []),
        )
    ).lower()
    return any(marker in evidence_text for marker in _RELEVANT_WAIT_MARKERS)

"""Read-only data adapters for the Hermes AI Office state projection."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any

from hermes_cli.office_redaction import RedactionReport, redact_display_text
from hermes_cli.office_state import OfficeDataSource, SourceStatus, _utc_now_iso
from hermes_constants import get_hermes_home


_KANBAN_STATUSES = ("triage", "todo", "ready", "running", "blocked", "done", "archived")
_ACTIVE_STATUSES = {"triage", "todo", "ready", "running"}
_NEEDS_ATTENTION_STATUSES = {"blocked"}
_MAX_KANBAN_WORK_ITEMS = 200
_MAX_EVENTS_PER_BOARD = 50
_MAX_CRON_JOBS = 100
_MAX_SESSIONS = 50


@dataclass
class OfficeAdapterResult:
    """Normalized read-only result returned by an Office source adapter."""

    source: OfficeDataSource
    rooms: list[dict[str, object]] = field(default_factory=list)
    agents: list[dict[str, object]] = field(default_factory=list)
    work_items: list[dict[str, object]] = field(default_factory=list)
    automations: list[dict[str, object]] = field(default_factory=list)
    topics: list[dict[str, object]] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    provenance: list[dict[str, object]] = field(default_factory=list)
    redactions: RedactionReport = field(default_factory=RedactionReport)

    def to_payload(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "rooms": list(self.rooms),
            "agents": list(self.agents),
            "work_items": list(self.work_items),
            "automations": list(self.automations),
            "topics": list(self.topics),
            "events": list(self.events),
            "provenance": list(self.provenance),
            "redactions": self.redactions.to_dict(),
        }


def _safe_error_summary(exc: BaseException) -> str:
    """Return a compact browser-safe error summary without paths/details."""

    text, _ = redact_display_text(str(exc))
    if not text:
        return exc.__class__.__name__
    # Keep source health useful without exposing filesystem paths or SQL details.
    return f"{exc.__class__.__name__}: {text[:160]}"


def _has_kanban_storage() -> bool:
    from hermes_cli import kanban_db as kb

    default_db = kb.kanban_db_path(board=kb.DEFAULT_BOARD)
    if default_db.exists():
        return True
    root = kb.boards_root()
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / "kanban.db").exists():
            return True
    return False


def _board_has_db(board: str) -> bool:
    from hermes_cli import kanban_db as kb

    return kb.kanban_db_path(board=board).exists()


def _status_counts(tasks: list[Any]) -> dict[str, int]:
    counts = {status: 0 for status in _KANBAN_STATUSES}
    for task in tasks:
        status = str(getattr(task, "status", ""))
        if status in counts:
            counts[status] += 1
    return counts


def _dependency_counts(conn: sqlite3.Connection, task_id: str) -> dict[str, int]:
    parents = conn.execute(
        "SELECT COUNT(*) AS n FROM task_links WHERE child_id = ?", (task_id,)
    ).fetchone()["n"]
    children = conn.execute(
        "SELECT COUNT(*) AS n FROM task_links WHERE parent_id = ?", (task_id,)
    ).fetchone()["n"]
    return {"parents": int(parents or 0), "children": int(children or 0)}


def _safe_display(value: object, report: RedactionReport) -> str:
    text, redactions = redact_display_text(value)
    report.merge(redactions)
    return text


def _kanban_work_item(
    *,
    board_slug: str,
    task: Any,
    conn: sqlite3.Connection,
    report: RedactionReport,
) -> dict[str, object]:
    item: dict[str, object] = {
        "id": f"kanban:{board_slug}:{task.id}",
        "kind": "kanban_task",
        "source": "kanban",
        "source_id": task.id,
        "room_id": f"kanban:{board_slug}",
        "board_id": board_slug,
        "title": _safe_display(task.title, report),
        "status": task.status,
        "assignee": _safe_display(task.assignee, report) if task.assignee else None,
        "priority": task.priority,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "updated_at": getattr(task, "updated_at", None),
        "last_heartbeat_at": task.last_heartbeat_at,
        "dependency_counts": _dependency_counts(conn, task.id),
        "badges": [],
        "provenance": {
            "status": "unknown",
            "missing_reason": "kanban_task_has_no_source_columns",
        },
    }
    badges = item["badges"]
    if isinstance(badges, list):
        if task.status == "blocked":
            badges.append("needs_attention")
        if getattr(task, "consecutive_failures", 0):
            badges.append("failure_history")
        if task.status == "running":
            badges.append("active")
    return item


def _kanban_events(board_slug: str, conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT id, task_id, run_id, kind, created_at
        FROM task_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (_MAX_EVENTS_PER_BOARD,),
    ).fetchall()
    # Oldest first is easier for consumers that draw a timeline.
    rows = list(reversed(rows))
    return [
        {
            "id": f"kanban:{board_slug}:event:{row['id']}",
            "source": "kanban",
            "source_id": int(row["id"]),
            "board_id": board_slug,
            "task_id": row["task_id"],
            "run_id": row["run_id"],
            "kind": row["kind"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def collect_kanban_office_state() -> OfficeAdapterResult:
    """Project Kanban boards/tasks/events into a redacted read-only Office result.

    This function deliberately checks for existing storage before opening a
    Kanban connection because ``kanban_db.connect()`` auto-initializes missing
    databases. AI Office adapters must not create or mutate source stores.
    """

    checked_at = _utc_now_iso()
    if not _has_kanban_storage():
        return OfficeAdapterResult(
            source=OfficeDataSource(id="kanban", status="missing", checked_at=checked_at)
        )

    from hermes_cli import kanban_db as kb

    rooms: list[dict[str, object]] = []
    work_items: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    redactions = RedactionReport()
    warnings: list[str] = []
    board_errors = 0
    readable_boards = 0

    try:
        boards = kb.list_boards(include_archived=False)
    except Exception as exc:
        return OfficeAdapterResult(
            source=OfficeDataSource(
                id="kanban",
                status="error",
                checked_at=checked_at,
                error_summary=_safe_error_summary(exc),
            )
        )

    for board in boards:
        slug = str(board.get("slug") or kb.DEFAULT_BOARD)
        if not _board_has_db(slug):
            warnings.append(f"board_db_missing:{slug}")
            continue
        try:
            with kb.connect(board=slug) as conn:
                tasks = kb.list_tasks(conn, include_archived=False)
                counts = _status_counts(tasks)
                display_name = _safe_display(board.get("name") or slug, redactions)
                rooms.append(
                    {
                        "id": f"kanban:{slug}",
                        "kind": "kanban_board",
                        "source": "kanban",
                        "source_id": slug,
                        "display_name": display_name,
                        "counts": counts,
                        "warnings": [],
                    }
                )
                for task in tasks:
                    if len(work_items) >= _MAX_KANBAN_WORK_ITEMS:
                        if "work_items_truncated" not in warnings:
                            warnings.append("work_items_truncated")
                        continue
                    work_items.append(
                        _kanban_work_item(
                            board_slug=slug,
                            task=task,
                            conn=conn,
                            report=redactions,
                        )
                    )
                events.extend(_kanban_events(slug, conn))
                readable_boards += 1
        except Exception as exc:
            board_errors += 1
            warnings.append(f"board_error:{slug}:{_safe_error_summary(exc)}")

    if readable_boards == 0 and board_errors:
        status: SourceStatus = "error"
    elif board_errors or warnings:
        status = "partial"
    else:
        status = "ok"

    error_summary = None
    if board_errors:
        error_summary = f"{board_errors} kanban board(s) failed to read"
    elif warnings:
        error_summary = "; ".join(warnings[:3])

    return OfficeAdapterResult(
        source=OfficeDataSource(
            id="kanban",
            status=status,
            checked_at=checked_at,
            item_count=len(work_items),
            warning_count=len(warnings),
            error_summary=error_summary,
        ),
        rooms=rooms,
        work_items=work_items,
        events=events,
        redactions=redactions,
    )


def _cron_paths() -> tuple[Path, Path]:
    home = get_hermes_home()
    return home / "cron" / "jobs.json", home / "cron" / "output"


def _read_cron_jobs_file(jobs_file: Path) -> list[dict[str, Any]]:
    raw = jobs_file.read_text(encoding="utf-8")
    data = json.loads(raw)
    jobs = data.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("cron jobs file has non-list jobs field")
    return [job for job in jobs if isinstance(job, dict)]


def _schedule_projection(job: dict[str, Any]) -> dict[str, object]:
    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        kind = schedule.get("kind") or "unknown"
        display = job.get("schedule_display") or schedule.get("display") or schedule.get("expr")
    else:
        kind = "unknown"
        display = job.get("schedule_display") or str(schedule or "")
    return {"kind": str(kind), "display": str(display or "")}


def _parse_delivery_targets(deliver: object) -> list[dict[str, object]]:
    if not deliver:
        return []
    targets: list[dict[str, object]] = []
    for raw_target in str(deliver).split(","):
        target = raw_target.strip()
        if not target:
            continue
        if target in {"local", "origin"}:
            targets.append({"kind": target})
            continue
        parts = target.split(":")
        platform = parts[0] if parts else "unknown"
        targets.append(
            {
                "kind": "explicit",
                "platform": platform or "unknown",
                "has_chat": len(parts) >= 2 and bool(parts[1]),
                "has_thread": len(parts) >= 3 and bool(parts[2]),
            }
        )
    return targets


def _output_artifact_count(output_dir: Path, job_id: str) -> int:
    job_dir = output_dir / job_id
    if not job_dir.is_dir():
        return 0
    return sum(1 for child in job_dir.iterdir() if child.is_file() and not child.name.startswith("."))


def _cron_display_name(job_id: str) -> str:
    """Return a prompt-safe cron display label.

    Cron ``name`` can be auto-derived from prompt/script content at creation
    time, so the Office DTO must not expose it as browser metadata. Keep a
    stable generic label while preserving the real id separately as source_id.
    """

    return f"Cron job {job_id[:8]}" if job_id else "Cron job"


def _cron_error_marker(value: object, marker: str) -> str | None:
    """Return a structured error marker without raw cron/agent output."""

    if value:
        return marker
    return None


def _cron_automation(job: dict[str, Any], output_dir: Path, report: RedactionReport) -> dict[str, object]:
    job_id = str(job.get("id") or "unknown")
    last_error = job.get("last_error")
    last_delivery_error = job.get("last_delivery_error")
    safe_last_error = _cron_error_marker(last_error, "last_error_recorded")
    safe_delivery_error = _cron_error_marker(last_delivery_error, "last_delivery_error_recorded")
    badges: list[str] = []
    if not job.get("enabled", True):
        badges.append("disabled")
    if job.get("last_status") == "error" or safe_last_error or safe_delivery_error:
        badges.append("needs_attention")
    return {
        "id": f"cron:{job_id}",
        "kind": "cron_job",
        "source": "cron",
        "source_id": job_id,
        "name": _cron_display_name(job_id),
        "enabled": bool(job.get("enabled", True)),
        "state": str(job.get("state") or "unknown"),
        "schedule": _schedule_projection(job),
        "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"),
        "last_status": job.get("last_status"),
        "last_error_summary": safe_last_error,
        "last_delivery_error_summary": safe_delivery_error,
        "delivery_targets": _parse_delivery_targets(job.get("deliver")),
        "output_artifact_count": _output_artifact_count(output_dir, job_id),
        "badges": badges,
    }


def collect_cron_office_state() -> OfficeAdapterResult:
    """Project cron jobs into redacted read-only Office automations."""

    checked_at = _utc_now_iso()
    jobs_file, output_dir = _cron_paths()
    if not jobs_file.exists():
        return OfficeAdapterResult(
            source=OfficeDataSource(id="cron", status="missing", checked_at=checked_at)
        )
    redactions = RedactionReport()
    try:
        jobs = _read_cron_jobs_file(jobs_file)
        automations = [
            _cron_automation(job, output_dir, redactions)
            for job in jobs[:_MAX_CRON_JOBS]
        ]
    except Exception as exc:
        return OfficeAdapterResult(
            source=OfficeDataSource(
                id="cron",
                status="error",
                checked_at=checked_at,
                error_summary=_safe_error_summary(exc),
            )
        )
    warnings = 1 if len(jobs) > _MAX_CRON_JOBS else 0
    return OfficeAdapterResult(
        source=OfficeDataSource(
            id="cron",
            status="ok",
            checked_at=checked_at,
            item_count=len(automations),
            warning_count=warnings,
            error_summary="cron job list truncated" if warnings else None,
        ),
        automations=automations,
        redactions=redactions,
    )


def _session_db_path() -> Path:
    return get_hermes_home() / "state.db"


def _session_actor(row: sqlite3.Row, report: RedactionReport) -> dict[str, object]:
    session_id = str(row["id"])
    model = _safe_display(row["model"], report) if row["model"] else None
    return {
        "id": f"session:{session_id[:8]}",
        "kind": "session_actor",
        "source": "sessions",
        "source_id": session_id[:8],
        "session_id_prefix": session_id[:8],
        "source_platform": row["source"] or "unknown",
        "status": "active" if row["ended_at"] is None else "ended",
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "last_active_at": row["last_active_at"] or row["started_at"],
        "end_reason": row["end_reason"],
        "message_count": int(row["message_count"] or 0),
        "tool_call_count": int(row["tool_call_count"] or 0),
        "api_call_count": int(row["api_call_count"] or 0),
        "model": model,
        "title": None,
        "title_policy": "hidden_by_default",
        "topic": {"status": "unknown", "missing_reason": "session_topic_not_normalized"},
    }


def collect_session_office_state() -> OfficeAdapterResult:
    """Project session metadata into redacted Office agents without transcripts."""

    checked_at = _utc_now_iso()
    db_path = _session_db_path()
    if not db_path.exists():
        return OfficeAdapterResult(
            source=OfficeDataSource(id="sessions", status="missing", checked_at=checked_at)
        )
    redactions = RedactionReport()
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT s.id, s.source, s.model, s.started_at, s.ended_at,
                       s.end_reason, s.message_count, s.tool_call_count,
                       s.api_call_count,
                       COALESCE(MAX(m.timestamp), s.started_at) AS last_active_at
                  FROM sessions s
             LEFT JOIN messages m ON m.session_id = s.id
              GROUP BY s.id
              ORDER BY last_active_at DESC
                 LIMIT ?
                """,
                (_MAX_SESSIONS,),
            ).fetchall()
        finally:
            conn.close()
        agents = [_session_actor(row, redactions) for row in rows]
    except Exception as exc:
        return OfficeAdapterResult(
            source=OfficeDataSource(
                id="sessions",
                status="error",
                checked_at=checked_at,
                error_summary=_safe_error_summary(exc),
            )
        )
    return OfficeAdapterResult(
        source=OfficeDataSource(
            id="sessions",
            status="ok",
            checked_at=checked_at,
            item_count=len(agents),
        ),
        agents=agents,
        redactions=redactions,
    )

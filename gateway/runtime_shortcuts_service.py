"""Shared runtime short-status shortcut helpers for the gateway."""

from __future__ import annotations

from typing import Any

from gateway.platforms.base import MessageType


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None or isinstance(value, bool):
            raise TypeError("invalid numeric value")
        return float(value)
    except Exception:
        return default


def _job_rank(job: dict[str, Any]) -> tuple[int, float]:
    status = str(job.get("status") or "").strip().lower()
    priority = 0 if status in {"running", "queued", "cancelling"} else 1
    return priority, -_safe_float(job.get("updated_at"), 0.0)


def format_background_job_short_status(runner: Any, job: dict[str, Any]) -> str:
    status_labels = {
        "queued": "排队中",
        "running": "进行中",
        "cancelling": "停止中",
        "completed": "已完成",
        "failed": "失败",
        "cancelled": "已停止",
    }
    task_id = str(job.get("task_id") or "").strip()
    status = status_labels.get(str(job.get("status") or "").strip().lower(), str(job.get("status") or "unknown"))
    worker_name = str(job.get("worker_name") or "").strip()
    preview = str(job.get("preview") or job.get("prompt") or "").strip()
    age = runner._format_background_job_age(job)
    line = f"后台任务 `{task_id}` 当前{status}"
    if worker_name:
        line += f"，负责人：{worker_name}"
    line += f"，已持续 {age}。"
    if preview:
        line += f"\n内容：{preview}"
    error = str(job.get("error") or "").strip()
    if error:
        line += f"\n错误：{error}"
    session_key = str(job.get("session_key") or "").strip()
    pending_approval_count = 0
    if session_key:
        try:
            pending_approval_count = runner._get_background_job_store().count_pending_approval_requests(
                session_key
            )
        except Exception:
            pending_approval_count = 0
    if pending_approval_count:
        line += f"\n当前卡在授权审批，待处理 {pending_approval_count} 条。"
    return line


def format_running_session_short_status(
    session_key: str,
    agent_ref: Any,
    *,
    detail_builder,
) -> str:
    detail = str(detail_builder(agent_ref, session_key) or "").strip()
    if detail:
        return f"当前前台这轮还在跑：{detail}。"
    return "当前前台这轮还在跑，我还没做完。"


def try_handle_background_job_status_shortcut(runner: Any, event: Any) -> str | None:
    source = getattr(event, "source", None)
    if not source:
        return None
    if event.get_command():
        return None
    if getattr(event, "message_type", None) != MessageType.TEXT:
        return None
    if not runner._looks_like_background_status_query(getattr(event, "text", "")):
        return None

    jobs = runner._background_jobs_for_source(source)
    if not jobs:
        return None

    latest = sorted(jobs, key=_job_rank)[0]
    return runner._format_background_job_short_status(latest)


def try_handle_runtime_status_shortcut(
    runner: Any,
    event: Any,
    *,
    pending_sentinel: Any = None,
) -> str | None:
    source = getattr(event, "source", None)
    if not source:
        return None
    if event.get_command():
        return None
    if getattr(event, "message_type", None) != MessageType.TEXT:
        return None
    if not runner._looks_like_runtime_status_query(getattr(event, "text", "")):
        return None

    session_key = runner._session_key_for_source(source)
    running_agent = runner._running_agents.get(session_key)
    if running_agent and running_agent is not pending_sentinel:
        return runner._format_running_session_short_status(session_key, running_agent)

    jobs = runner._background_jobs_for_source(source)
    if jobs:
        latest = sorted(jobs, key=_job_rank)[0]
        return runner._format_background_job_short_status(latest)
    return None

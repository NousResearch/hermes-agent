"""Shared runtime-status rendering helpers for the gateway."""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None or isinstance(value, bool):
            raise TypeError("invalid numeric value")
        return float(value)
    except Exception:
        return default


def _truncate_status_preview(value: Any, *, limit: int = 120) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _collect_active_sessions(
    runner: Any,
    *,
    now_ts: float,
    pending_sentinel: Any = None,
) -> list[dict[str, Any]]:
    active_sessions: list[dict[str, Any]] = []
    for session_key, agent_ref in getattr(runner, "_running_agents", {}).items():
        if agent_ref is pending_sentinel:
            continue
        session_meta = runner._runtime_session_metadata(session_key)
        started_at = _safe_float(
            getattr(runner, "_running_agents_ts", {}).get(session_key),
            now_ts,
        )
        age_seconds = max(0, int(now_ts - started_at))
        activity: dict[str, Any] = {}
        if hasattr(agent_ref, "get_activity_summary"):
            try:
                raw_activity = agent_ref.get_activity_summary() or {}
            except Exception:
                raw_activity = {}
            if isinstance(raw_activity, dict):
                activity = raw_activity
        active_sessions.append(
            {
                "session_key": session_key,
                "platform": session_meta["platform"],
                "chat_type": session_meta["chat_type"],
                "chat_id": session_meta["chat_id"],
                "age_seconds": age_seconds,
                "current_tool": str(activity.get("current_tool") or "").strip(),
                "last_activity_desc": str(activity.get("last_activity_desc") or "").strip(),
                "api_call_count": int(_safe_float(activity.get("api_call_count"), 0.0)),
                "max_iterations": int(_safe_float(activity.get("max_iterations"), 0.0)),
            }
        )
    active_sessions.sort(key=lambda item: item.get("age_seconds", 0), reverse=True)
    return active_sessions


def _collect_background_jobs(
    runner: Any,
    *,
    now_ts: float,
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs_by_id: dict[str, dict[str, Any]] = {}
    try:
        store = runner._get_background_job_store()
        all_jobs = store.list_jobs()
    except Exception:
        all_jobs = []
    for job in all_jobs:
        if not isinstance(job, dict):
            continue
        task_id = str(job.get("task_id") or "").strip()
        if task_id:
            jobs_by_id[task_id] = job

    background_counts: dict[str, int] = {}
    active_background_jobs: list[dict[str, Any]] = []
    for job in jobs_by_id.values():
        status = str(job.get("status") or "").strip().lower() or "unknown"
        background_counts[status] = background_counts.get(status, 0) + 1
        if status not in {"queued", "running", "cancelling"}:
            continue
        active_background_jobs.append(
            {
                "task_id": str(job.get("task_id") or "").strip(),
                "status": status,
                "worker_name": str(job.get("worker_name") or "").strip(),
                "preview": _truncate_status_preview(job.get("preview") or job.get("prompt") or ""),
                "age_seconds": int(
                    max(
                        0.0,
                        now_ts - _safe_float(job.get("created_at"), now_ts),
                    )
                ),
            }
        )

    active_background_jobs.sort(key=lambda item: item.get("age_seconds", 0), reverse=True)
    return background_counts, active_background_jobs, jobs_by_id


def _collect_auto_vision_summary(runner: Any) -> dict[str, Any]:
    runner._ensure_auto_vision_state()
    runner._prune_auto_vision_state()
    inflight_count = 0
    for task in getattr(runner, "_auto_vision_tasks", {}).values():
        try:
            if task and not task.done():
                inflight_count += 1
        except Exception:
            continue
    cooldown_seconds, cooldown_reason = runner._auto_vision_cooldown_remaining()
    auto_vision_state = "ready"
    if cooldown_seconds > 0:
        auto_vision_state = "cooldown"
    elif inflight_count > 0:
        auto_vision_state = "warming"
    return {
        "state": auto_vision_state,
        "inflight_count": inflight_count,
        "cooldown_seconds": int(max(0.0, cooldown_seconds)),
        "reason": cooldown_reason,
        "cache_entries": len(getattr(runner, "_auto_vision_cache", {})),
    }


def build_runtime_status_summary(
    runner: Any,
    *,
    now_ts: float | None = None,
    pending_sentinel: Any = None,
) -> dict[str, Any]:
    now_ts = time.time() if now_ts is None else float(now_ts)
    active_sessions = _collect_active_sessions(
        runner,
        now_ts=now_ts,
        pending_sentinel=pending_sentinel,
    )
    background_counts, active_background_jobs, jobs_by_id = _collect_background_jobs(
        runner,
        now_ts=now_ts,
    )
    group_archive: dict[str, Any] = {}
    if hasattr(runner, "_build_runtime_group_archive_summary"):
        try:
            group_archive = runner._build_runtime_group_archive_summary()
        except Exception as exc:
            logger.debug("Failed to collect shared group archive runtime stats: %s", exc)
            group_archive = {}

    group_monitoring: dict[str, Any] = {}
    if hasattr(runner, "_build_runtime_group_monitoring_summary"):
        try:
            group_monitoring = runner._build_runtime_group_monitoring_summary()
        except Exception as exc:
            logger.debug("Failed to collect shared group monitoring runtime stats: %s", exc)
            group_monitoring = {}
    direct_shortcuts: dict[str, Any] = {}
    if hasattr(runner, "_build_runtime_direct_shortcut_summary"):
        try:
            direct_shortcuts = runner._build_runtime_direct_shortcut_summary()
        except Exception as exc:
            logger.debug("Failed to collect direct shortcut runtime stats: %s", exc)
            direct_shortcuts = {}

    return {
        "model": runner._build_runtime_model_summary(),
        "approvals": runner._build_runtime_approval_summary(),
        "active_sessions_count": len(active_sessions),
        "active_sessions": active_sessions[:8],
        "background_jobs": {
            "active_count": len(active_background_jobs),
            "total_count": len(jobs_by_id),
            "counts": background_counts,
            "active": active_background_jobs[:8],
        },
        "auto_vision": _collect_auto_vision_summary(runner),
        "group_archive": group_archive,
        "group_monitoring": group_monitoring,
        "direct_shortcuts": direct_shortcuts,
    }


def _runtime_foreground_line(
    runner: Any,
    *,
    session_key: str,
    pending_sentinel: Any = None,
    now_ts: float,
) -> str:
    running_agent = getattr(runner, "_running_agents", {}).get(session_key)
    if not running_agent or running_agent is pending_sentinel:
        return ""
    activity: dict[str, Any] = {}
    if hasattr(running_agent, "get_activity_summary"):
        try:
            raw_activity = running_agent.get_activity_summary() or {}
        except Exception:
            raw_activity = {}
        if isinstance(raw_activity, dict):
            activity = raw_activity
    foreground_bits: list[str] = []
    api_call_count = int(_safe_float(activity.get("api_call_count"), 0.0))
    max_iterations = int(_safe_float(activity.get("max_iterations"), 0.0))
    if api_call_count or max_iterations:
        foreground_bits.append(f"{api_call_count}/{max_iterations}")
    current_tool = str(activity.get("current_tool") or "").strip()
    if current_tool:
        foreground_bits.append(current_tool)
    else:
        last_activity_desc = str(activity.get("last_activity_desc") or "").strip()
        if last_activity_desc:
            foreground_bits.append(last_activity_desc)
    started_at = _safe_float(
        getattr(runner, "_running_agents_ts", {}).get(session_key),
        0.0,
    )
    if started_at > 0:
        foreground_bits.append(f"{max(0, int(now_ts - started_at))}s")
    return " · ".join(bit for bit in foreground_bits if bit) or "running"


def _pending_approval_count(runner: Any, *, session_key: str) -> int:
    pending_approval_count = 0
    try:
        pending_approval_count = runner._get_background_job_store().count_pending_approval_requests(session_key)
    except Exception:
        pending_approval_count = 0
    if session_key in getattr(runner, "_pending_approvals", {}):
        pending_approval_count = max(pending_approval_count, 1)
    try:
        from tools.approval import has_blocking_approval

        if has_blocking_approval(session_key):
            pending_approval_count = max(pending_approval_count, 1)
    except Exception:
        pass
    return pending_approval_count


def _auto_vision_label(runner: Any) -> str:
    runner._ensure_auto_vision_state()
    runner._prune_auto_vision_state()
    inflight_vision = sum(
        1
        for task in getattr(runner, "_auto_vision_tasks", {}).values()
        if task and not task.done()
    )
    cooldown_remaining, cooldown_reason = runner._auto_vision_cooldown_remaining()
    auto_vision_bits: list[str] = []
    if inflight_vision:
        auto_vision_bits.append(f"warming ({inflight_vision} in flight)")
    if cooldown_remaining > 0:
        reason_suffix = f": {cooldown_reason}" if cooldown_reason else ""
        auto_vision_bits.append(f"cooldown ({int(cooldown_remaining)}s{reason_suffix})")
    if not auto_vision_bits:
        auto_vision_bits.append("ready")
    return ", ".join(auto_vision_bits)


def _model_label(runner: Any) -> str:
    model_summary = runner._build_runtime_model_summary()
    model_label = str(model_summary.get("active_model") or model_summary.get("configured_model") or "").strip()
    provider_label = str(model_summary.get("active_provider") or "").strip()
    model_bits = [model_label or "unknown"]
    if provider_label:
        model_bits.append(f"via {provider_label}")
    if bool(model_summary.get("fallback_active")):
        fallback_label = "fallback pinned" if bool(model_summary.get("fallback_pinned")) else "fallback active"
        model_bits.append(f"({fallback_label})")
    return " ".join(bit for bit in model_bits if bit).strip()


async def _await_maybe(value: Any) -> Any:
    """Await coroutine results from AsyncSessionDB wrappers; pass through plain values."""
    if inspect.isawaitable(value):
        return await value
    return value


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


async def render_status_command(
    runner: Any,
    event: Any,
    *,
    now_ts: float | None = None,
    pending_sentinel: Any = None,
    extras_only: bool = False,
) -> str:
    """Render /status.

    When ``extras_only`` is True, return only the QQ/runtime observability
    sections (approvals, auto-vision, group archive, background jobs). The
    caller is expected to prepend the mainline cockpit status first.
    """
    now_ts = time.time() if now_ts is None else float(now_ts)
    source = event.source
    # Prefer async session store when available (mainline path / tests).
    if hasattr(runner, "async_session_store"):
        try:
            session_entry = await runner.async_session_store.get_or_create_session(source)
        except Exception:
            session_entry = runner.session_store.get_or_create_session(source)
    else:
        session_entry = runner.session_store.get_or_create_session(source)
    connected_platforms = [p.value for p in runner.adapters.keys()]
    session_key = session_entry.session_key
    agent = getattr(runner, "_running_agents", {}).get(session_key)
    is_running = agent is not None and (
        pending_sentinel is None or agent is not pending_sentinel
    )

    lines: list[str] = []
    if not extras_only:
        title = None
        db_total_tokens = 0
        session_db = getattr(runner, "_session_db", None)
        if session_db is not None:
            try:
                title = await _await_maybe(session_db.get_session_title(session_entry.session_id))
            except Exception:
                title = None
            # Token totals live in SQLite SessionDB (not SessionEntry).
            try:
                row = await _await_maybe(session_db.get_session(session_entry.session_id))
                if isinstance(row, dict):
                    db_total_tokens = (
                        _int_value(row.get("input_tokens"))
                        + _int_value(row.get("output_tokens"))
                        + _int_value(row.get("cache_read_tokens"))
                        + _int_value(row.get("cache_write_tokens"))
                        + _int_value(row.get("reasoning_tokens"))
                    )
            except Exception:
                db_total_tokens = 0
        if not isinstance(title, str) or not title.strip():
            title = None

        lines = [
            "📊 **Hermes Gateway Status**",
            "",
            f"**Session ID:** `{session_entry.session_id}`",
        ]
        if title:
            lines.append(f"**Title:** {title}")
        lines.extend(
            [
                f"**Created:** {session_entry.created_at.strftime('%Y-%m-%d %H:%M')}",
                f"**Last Activity:** {session_entry.updated_at.strftime('%Y-%m-%d %H:%M')}",
                f"**Cumulative API tokens (re-sent each call):** {db_total_tokens:,}",
                f"**Agent Running:** {'Yes ⚡' if is_running else 'No'}",
            ]
        )

        if is_running:
            foreground = _runtime_foreground_line(
                runner,
                session_key=session_key,
                pending_sentinel=pending_sentinel,
                now_ts=now_ts,
            )
            if foreground:
                lines.append(f"**Foreground:** {foreground}")

        lines.extend(
            [
                "",
                f"**Connected Platforms:** {', '.join(connected_platforms)}",
            ]
        )

    pending_approval_count = _pending_approval_count(runner, session_key=session_key)
    group_archive_summary: dict[str, Any] = {}
    if hasattr(runner, "_build_runtime_group_archive_summary"):
        try:
            group_archive_summary = runner._build_runtime_group_archive_summary()
        except Exception:
            group_archive_summary = {}
    group_monitoring_summary: dict[str, Any] = {}
    if hasattr(runner, "_build_runtime_group_monitoring_summary"):
        try:
            group_monitoring_summary = runner._build_runtime_group_monitoring_summary()
        except Exception:
            group_monitoring_summary = {}
    shared_monitoring_count = int(group_monitoring_summary.get("active_collect_only_groups") or 0)

    lines.extend(
        [
            "",
            f"**Model:** {_model_label(runner)}",
            f"**Pending Approvals:** {pending_approval_count}",
            f"**Auto Vision:** {_auto_vision_label(runner)}",
            "",
            f"**Group Archive:** {int(group_archive_summary.get('raw_message_count') or 0)} raw msg(s)",
            f"**Group Monitoring:** {shared_monitoring_count} collect_only group(s)",
        ]
    )
    direct_shortcut_summary = {}
    if hasattr(runner, "_build_runtime_direct_shortcut_summary"):
        try:
            direct_shortcut_summary = runner._build_runtime_direct_shortcut_summary()
        except Exception:
            direct_shortcut_summary = {}
    direct_shortcut_count = int(direct_shortcut_summary.get("recent_count") or 0)
    if direct_shortcut_count:
        lines.append(f"**Direct Shortcuts:** {direct_shortcut_count} recent hit(s)")
        latest = list(direct_shortcut_summary.get("recent") or [])[:1]
        if latest and isinstance(latest[0], dict):
            trace = latest[0]
            handler_label = str(trace.get("matched_handler") or "unknown").strip()
            text_preview = str(trace.get("text_preview") or "").strip()
            bits = [bit for bit in (handler_label, text_preview) if bit]
            if bits:
                lines.append(f"- {' · '.join(bits)}")
    shared_group_preview = list(group_monitoring_summary.get("groups") or [])[:3]
    for group in shared_group_preview:
        if not isinstance(group, dict):
            continue
        group_target = str(group.get("group_id") or group.get("chat_id") or "").strip()
        group_label = str(group.get("group_name") or group_target or "unknown").strip()
        if group_target and group.get("group_name") and group_target != group["group_name"]:
            group_label = f"{group['group_name']} ({group_target})"
        platform_label = str(group.get("platform_label") or "").strip()
        worker_names = ", ".join(group.get("worker_names") or []) or "无人值守"
        report_label = "日报开" if bool(group.get("daily_report_enabled")) else "日报关"
        preview_bits = [bit for bit in (platform_label, group_label, "collect_only", worker_names, report_label) if bit]
        lines.append(f"- {' · '.join(preview_bits)}")

    jobs = runner._background_jobs_for_source(source)
    if jobs:
        status_labels = {
            "queued": "queued",
            "running": "running",
            "cancelling": "stopping",
            "completed": "done",
            "failed": "failed",
            "cancelled": "stopped",
        }
        lines.extend(["", "**Background Jobs:**"])
        for job in jobs[-5:]:
            label = status_labels.get(job.get("status"), str(job.get("status") or "unknown"))
            worker_name = str(job.get("worker_name") or "").strip()
            worker_suffix = f" · {worker_name}" if worker_name else ""
            lines.append(
                f"- `{job['task_id']}` — {label}{worker_suffix} ({runner._format_background_job_age(job)})"
            )

    return "\n".join(lines)

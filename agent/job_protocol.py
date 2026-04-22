from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.background_jobs import BackgroundTaskRequest, CronTaskRequest, DelegationTaskRequest

PROTOCOL_VERSION = "1.0"


def _platform_callback(platform: str, chat_id: str, thread_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "platform",
        "target": {
            "platform": str(platform),
            "chat_id": str(chat_id),
            "thread_id": str(thread_id) if thread_id not in (None, "") else None,
        },
    }


def _session_callback(session_id: str | None, platform: str | None) -> dict[str, Any]:
    if not session_id:
        return {"type": "none"}
    callback: dict[str, Any] = {"type": "session", "session_id": str(session_id)}
    if platform:
        callback["platform"] = str(platform)
    return callback


def _parse_deliver_target(deliver: str | None) -> dict[str, Any] | None:
    if not deliver or deliver == "origin":
        return None
    parts = str(deliver).split(":")
    if len(parts) < 2:
        return None
    platform = parts[0].strip()
    chat_id = parts[1].strip()
    thread_id = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    if not platform or not chat_id:
        return None
    return {"platform": platform, "chat_id": chat_id, "thread_id": thread_id}


def build_background_job_envelope(request: BackgroundTaskRequest) -> dict[str, Any]:
    callback = {"type": "none"}
    if request.chat_id and request.platform:
        callback = _platform_callback(request.platform, request.chat_id, request.thread_id)
    return {
        "version": PROTOCOL_VERSION,
        "kind": "background",
        "task_id": request.task_id,
        "payload": {
            "prompt": request.prompt,
            "origin": request.origin,
            "platform": request.platform,
            "session_id": request.session_id,
            "user_id": request.user_id,
            "user_name": request.user_name,
            "chat_id": request.chat_id,
            "thread_id": request.thread_id,
        },
        "callback": callback,
    }


def build_delegation_job_envelope(request: DelegationTaskRequest) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "kind": "delegation",
        "task_id": request.task_id,
        "payload": {
            "goal": request.goal,
            "context": request.context,
            "toolsets": request.toolsets,
            "model": request.model,
            "max_iterations": request.max_iterations,
            "task_index": request.task_index,
            "task_count": request.task_count,
        },
        "callback": _session_callback(request.parent_session_id, request.parent_platform),
    }


def build_cron_job_envelope(request: CronTaskRequest) -> dict[str, Any]:
    target = _parse_deliver_target(request.deliver)
    if target is None and request.origin:
        platform = request.origin.get("platform")
        chat_id = request.origin.get("chat_id")
        if platform and chat_id:
            target = {
                "platform": platform,
                "chat_id": chat_id,
                "thread_id": request.origin.get("thread_id"),
            }
    callback = {"type": "none"}
    if target:
        callback = _platform_callback(target["platform"], target["chat_id"], target.get("thread_id"))
    return {
        "version": PROTOCOL_VERSION,
        "kind": "cron",
        "task_id": request.task_id,
        "payload": {
            "job_id": request.job_id,
            "job_name": request.job_name,
            "prompt": request.prompt,
            "schedule_display": request.schedule_display,
            "deliver": request.deliver,
            "origin": request.origin,
            "model": request.model,
            "skills": request.skills,
            "script": request.script,
        },
        "callback": callback,
    }


def build_completion_envelope(
    *,
    kind: str,
    task_id: str,
    status: str,
    callback: dict[str, Any],
    summary: str | None = None,
    final_output: str | None = None,
    error: str | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    envelope = {
        "version": PROTOCOL_VERSION,
        "kind": kind,
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_output": final_output,
        "artifacts": artifacts or [],
        "metadata": metadata or {},
        "callback": callback,
    }
    if error is not None:
        envelope["error"] = error
    return envelope

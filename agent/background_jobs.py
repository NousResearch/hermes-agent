from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from agent.job_protocol import (
    build_background_job_envelope,
    build_cron_job_envelope,
    build_delegation_job_envelope,
)


@dataclass
class QueueSubmission:
    task_id: str
    backend: str
    accepted: bool = True
    queue: str | None = None
    message: str | None = None
    raw_response: str = ""


@dataclass
class QueueRequest:
    task_id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackgroundTaskRequest:
    task_id: str
    prompt: str
    origin: str
    platform: str
    session_id: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    chat_id: str | None = None
    thread_id: str | None = None


@dataclass
class DelegationTaskRequest:
    task_id: str
    goal: str
    context: str | None = None
    toolsets: list[str] | None = None
    model: str | None = None
    max_iterations: int | None = None
    parent_session_id: str | None = None
    parent_platform: str | None = None
    task_index: int = 0
    task_count: int = 1


@dataclass
class CronTaskRequest:
    task_id: str
    job_id: str
    job_name: str
    prompt: str
    schedule_display: str | None = None
    deliver: str | None = None
    origin: dict[str, Any] | None = None
    model: dict[str, Any] | str | None = None
    skills: list[str] | None = None
    script: str | None = None


def preview_prompt(prompt: str, limit: int = 60) -> str:
    prompt = (prompt or "").strip()
    if len(prompt) <= limit:
        return prompt
    return prompt[:limit] + "..."


def _normalize_backend(value: str | None) -> str:
    raw = (value or "local").strip().lower()
    if raw in {"", "local", "thread", "inline"}:
        return "local"
    if raw in {"command", "cmd", "external", "queue", "minions", "gbrain"}:
        return "command"
    return raw


def _backend_key(prefix: str) -> str:
    return f"HERMES_{prefix}_BACKEND"


def _enqueue_key(prefix: str) -> str:
    return f"HERMES_{prefix}_ENQUEUE_CMD"


def _timeout_key(prefix: str) -> str:
    return f"HERMES_{prefix}_ENQUEUE_TIMEOUT"


def get_queue_backend(prefix: str, env: Mapping[str, str] | None = None) -> str:
    env = env or os.environ
    return _normalize_backend(env.get(_backend_key(prefix)))


def queue_backend_enabled(prefix: str, env: Mapping[str, str] | None = None) -> bool:
    return get_queue_backend(prefix, env) != "local"


def _submission_from_response(
    request: QueueRequest,
    prefix: str,
    stdout: str,
    env: Mapping[str, str],
) -> QueueSubmission:
    parsed: dict[str, Any] = {}
    if stdout:
        try:
            maybe = json.loads(stdout)
            if isinstance(maybe, dict):
                parsed = maybe
        except json.JSONDecodeError:
            parsed = {}
    backend = str(parsed.get("backend") or env.get(_backend_key(prefix)) or "command")
    return QueueSubmission(
        task_id=str(parsed.get("task_id") or request.task_id),
        backend=backend,
        accepted=bool(parsed.get("accepted", True)),
        queue=parsed.get("queue"),
        message=parsed.get("message") or (stdout if stdout and not parsed else None),
        raw_response=stdout,
    )


def enqueue_queue_request(
    request: QueueRequest,
    *,
    prefix: str,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> QueueSubmission:
    env = env or os.environ
    backend = get_queue_backend(prefix, env)
    if backend == "local":
        return QueueSubmission(task_id=request.task_id, backend="local")

    command = (env.get(_enqueue_key(prefix)) or "").strip()
    if not command:
        raise ValueError(
            f"{_backend_key(prefix)} is set to an external mode, but "
            f"{_enqueue_key(prefix)} is empty."
        )

    outbound = request.payload
    if not isinstance(outbound, dict) or "version" not in outbound or "kind" not in outbound:
        outbound = asdict(request)
    payload = json.dumps(outbound, ensure_ascii=False)
    run_timeout = timeout if timeout is not None else float(env.get(_timeout_key(prefix), "15"))
    proc = subprocess.run(
        command,
        input=payload,
        text=True,
        shell=True,
        capture_output=True,
        timeout=run_timeout,
        check=False,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = stderr or stdout or f"exit {proc.returncode}"
        raise RuntimeError(f"{prefix.lower()} enqueue command failed: {detail}")
    return _submission_from_response(request, prefix, stdout, env)


def get_background_backend(env: Mapping[str, str] | None = None) -> str:
    return get_queue_backend("BACKGROUND", env)


def background_backend_enabled(env: Mapping[str, str] | None = None) -> bool:
    return queue_backend_enabled("BACKGROUND", env)


def enqueue_background_task(
    request: BackgroundTaskRequest,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> QueueSubmission:
    return enqueue_queue_request(
        QueueRequest(
            task_id=request.task_id,
            kind="background",
            payload=build_background_job_envelope(request),
        ),
        prefix="BACKGROUND",
        env=env,
        timeout=timeout,
    )


def enqueue_background_task_via_command(
    request: BackgroundTaskRequest,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> QueueSubmission:
    return enqueue_queue_request(
        QueueRequest(
            task_id=request.task_id,
            kind="background",
            payload=build_background_job_envelope(request),
        ),
        prefix="BACKGROUND",
        env={**(env or os.environ), _backend_key("BACKGROUND"): "command"},
        timeout=timeout,
    )


def delegation_backend_enabled(env: Mapping[str, str] | None = None) -> bool:
    return queue_backend_enabled("DELEGATION", env)


def enqueue_delegate_task(
    request: DelegationTaskRequest,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    submission = enqueue_queue_request(
        QueueRequest(
            task_id=request.task_id,
            kind="delegation",
            payload=build_delegation_job_envelope(request),
        ),
        prefix="DELEGATION",
        env=env,
        timeout=timeout,
    )
    return asdict(submission)


def cron_backend_enabled(env: Mapping[str, str] | None = None) -> bool:
    return queue_backend_enabled("CRON", env)


def enqueue_cron_task(
    request: CronTaskRequest,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    submission = enqueue_queue_request(
        QueueRequest(
            task_id=request.task_id,
            kind="cron",
            payload=build_cron_job_envelope(request),
        ),
        prefix="CRON",
        env=env,
        timeout=timeout,
    )
    return asdict(submission)

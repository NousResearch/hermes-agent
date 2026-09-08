"""Task-local persisted cron identity for sanitized child-process environments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


CRON_JOB_IDENTITY_VARS = (
    "HERMES_CRON_JOB_ID",
    "HERMES_CRON_JOB_ORIGIN_USER_ID",
    "HERMES_CRON_JOB_ORIGIN_PLATFORM",
)


def cron_job_identity_values(job: dict, job_id: str) -> dict[str, str]:
    """Identity persisted with a cron job, distinct from live sender-session state."""
    raw_origin = job.get("origin")
    origin = raw_origin if isinstance(raw_origin, dict) else {}
    return {
        "HERMES_CRON_JOB_ID": str(job_id),
        "HERMES_CRON_JOB_ORIGIN_USER_ID": str(
            origin.get("user_id") or origin.get("chat_id") or ""
        ),
        "HERMES_CRON_JOB_ORIGIN_PLATFORM": str(origin.get("platform") or ""),
    }


def set_cron_job_identity(job: dict, job_id: str) -> None:
    """Bind one job's persisted identity in the current task context."""
    from gateway.session_context import _VAR_MAP

    for name, value in cron_job_identity_values(job, job_id).items():
        _VAR_MAP[name].set(value)


def clear_cron_job_identity() -> None:
    """Clear cron identity so later work cannot inherit a completed job."""
    from gateway.session_context import _VAR_MAP

    for name in CRON_JOB_IDENTITY_VARS:
        _VAR_MAP[name].set("")


@contextmanager
def cron_job_identity_scope(job: dict, job_id: str) -> Iterator[None]:
    """Bind one job's identity and restore the prior task-local values on exit."""
    from gateway.session_context import _VAR_MAP

    tokens = []
    for name, value in cron_job_identity_values(job, job_id).items():
        var = _VAR_MAP[name]
        tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)

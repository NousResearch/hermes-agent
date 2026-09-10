"""Task-local persisted cron identity for sanitized child-process environments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


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


def set_cron_job_identity(job: dict, job_id: str) -> list[tuple]:
    """Bind one job's persisted identity and return tokens for exact restoration."""
    from gateway.session_context import _VAR_MAP

    tokens = []
    for name, value in cron_job_identity_values(job, job_id).items():
        var = _VAR_MAP[name]
        tokens.append((var, var.set(value)))
    return tokens


def reset_cron_job_identity(tokens: list[tuple]) -> None:
    """Restore the task-local identity that preceded this cron run."""
    for var, token in reversed(tokens):
        var.reset(token)


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

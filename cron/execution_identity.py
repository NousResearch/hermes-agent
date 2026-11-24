"""Which cron execution the current code runs inside, for plugins (``ctx.current_cron_execution()``).

The scheduler binds it once it has won the execution's ``claimed`` → ``running`` transition (or a
restart-safe worker adopted it) and resets it when the fire ends. It is a ContextVar, so it follows
the agent onto the threads the scheduler already runs it on via ``contextvars.copy_context()`` and
never leaks to another concurrently firing job or profile. Model output cannot reach it: tool
subprocesses do not share the interpreter. Delegated children read ``None`` — they are not the
scheduled occurrence.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class CronExecution:
    job_id: str
    job_name: str
    execution_id: str
    source: str  # "builtin" (ticker), "direct" (manual run / dashboard Run now), provider name...
    scheduled_instant: Optional[str]  # the occurrence fired; None for an off-schedule run
    started_at: Optional[str]
    profile: str


@dataclass(frozen=True)
class _BoundCronExecution:
    execution: CronExecution
    delivery_job: dict[str, Any]


_CURRENT: ContextVar[Optional[_BoundCronExecution]] = ContextVar(
    "hermes_cron_execution", default=None,
)


def enter_cron_execution(job: Mapping[str, Any], execution_id: str, record: Mapping[str, Any]) -> Token:
    """Bind this fire's identity from *record*, the ledger row of the running transition."""
    from cron.scheduler_delivery import _snapshot_delivery_job
    from hermes_cli.profiles import current_profile_name

    execution = CronExecution(
        job_id=str(job["id"]),
        job_name=str(job.get("name") or job["id"]),
        execution_id=str(execution_id),
        source=str(record.get("source") or ""),
        scheduled_instant=record.get("scheduled_instant"),
        started_at=record.get("started_at"),
        profile=current_profile_name("default") or "default",
    )
    return _CURRENT.set(_BoundCronExecution(execution, _snapshot_delivery_job(job)))


def exit_cron_execution(token: Optional[Token]) -> None:
    if token is not None:
        _CURRENT.reset(token)


def _current_bound() -> Optional[_BoundCronExecution]:
    from agent.delegation_context import is_delegated_child_context

    return None if is_delegated_child_context() else _CURRENT.get()


def current_cron_execution() -> Optional[CronExecution]:
    bound = _current_bound()
    return bound.execution if bound is not None else None


def _current_cron_execution_job() -> Optional[dict[str, Any]]:
    """Delivery-relevant snapshot of the firing job bound in this context, or ``None`` outside a
    cron run or inside a delegated child."""
    bound = _current_bound()
    return bound.delivery_job if bound is not None else None

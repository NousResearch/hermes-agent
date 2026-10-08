"""User-only OS scheduling for opt-in automatic updates.

The caller owns serialization and durable status. Scheduler identities and paths
are always derived from the current installation, never from persisted paths.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchedulerSpec:
    identity: str
    command: Sequence[str]
    home: Path
    schedule: str
    plan_times: Sequence[str] = ()
    log_directory: Path | None = None

    def __post_init__(self) -> None:
        from hermes_cli.update_auto_schedule_common import validate_spec

        validate_spec(self)


@dataclass
class SchedulerHandle:
    scheduler_type: str
    path: Path
    _rollback_fn: Callable[[], dict[str, Any]] = field(repr=False)
    removed: bool = False
    _receipt: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def rollback(self) -> dict[str, Any]:
        """Return the original verified result on every call, including failures."""
        if self._receipt is None:
            try:
                self._receipt = self._rollback_fn()
            except Exception as exc:
                logger.exception("Unexpected failure while restoring the auto-update scheduler")
                self._receipt = {
                    "ok": False, "scheduler": self.scheduler_type,
                    "errors": [f"rollback raised unexpectedly: {exc}"],
                }
        return self._receipt


class SchedulerRecoveryError(RuntimeError):
    """The mutation failed and its previous state could not be verified."""

    def __init__(self, message: str, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.receipt = receipt


def validate_time(value: str) -> str:
    from hermes_cli.update_auto_schedule_common import parse_time

    return parse_time(value)[2]


def scheduled_action(
    schedule: str, plan_times: Sequence[str], now: datetime | None = None,
) -> str:
    from hermes_cli.update_auto_schedule_common import action_for_time

    return action_for_time(schedule, plan_times, now)


def paths(spec: SchedulerSpec) -> dict[str, Any]:
    from hermes_cli.update_auto_schedule_common import backend

    return backend().paths(spec)


def enable(spec: SchedulerSpec) -> SchedulerHandle:
    from hermes_cli.update_auto_schedule_common import backend

    return backend().enable(spec)


def disable(spec: SchedulerSpec) -> SchedulerHandle:
    from hermes_cli.update_auto_schedule_common import backend

    return backend().disable(spec)

"""Pure capability matching and deterministic runner selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from hermes_cli.fleet_protocol import FleetTask, RunnerCapability, TaskRequirement


@dataclass(frozen=True)
class RunnerStatus:
    capability: RunnerCapability
    last_seen: float
    expires_at: float
    active_load: int = 0


def _requirement(task: FleetTask | TaskRequirement) -> TaskRequirement:
    return task.requirement if isinstance(task, FleetTask) else task


def _matches(requirement: TaskRequirement, runner: RunnerStatus) -> bool:
    capability = runner.capability
    if requirement.models and not set(requirement.models).issubset(capability.models):
        return False
    if requirement.tools and not set(requirement.tools).issubset(capability.tools):
        return False
    if requirement.project and requirement.project not in capability.projects:
        return False
    return True


def eligible_runners(
    task: FleetTask | TaskRequirement,
    runners: Iterable[RunnerStatus],
    *,
    now: float,
) -> list[RunnerStatus]:
    requirement = _requirement(task)
    return [runner for runner in runners if runner.expires_at > now and _matches(requirement, runner)]


def select_runner(
    task: FleetTask | TaskRequirement,
    runners: Iterable[RunnerStatus],
    *,
    now: float,
) -> RunnerStatus | None:
    eligible = eligible_runners(task, runners, now=now)
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda runner: (
            0 if runner.active_load == 0 else 1,
            runner.active_load,
            runner.capability.node_id,
            runner.capability.profile,
        ),
    )


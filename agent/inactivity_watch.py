"""Shared inactivity watchdog helpers for cron and gateway agent runs."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

POLL_INTERVAL_SECONDS = 5.0


@dataclass(frozen=True)
class InactivityDiagnostic:
    last_activity_desc: Any
    seconds_since_activity: Any
    current_tool: Any
    api_call_count: Any
    max_iterations: Any


@dataclass(frozen=True)
class InactivityWaitResult:
    result: Any = None
    timed_out: bool = False


def _activity_tracker_available(agent: Any, *, require_truthy_agent: bool = False) -> bool:
    if require_truthy_agent and not agent:
        return False
    return hasattr(agent, "get_activity_summary")


def get_activity_summary(agent: Any, *, require_truthy_agent: bool = False) -> dict[str, Any]:
    activity = {}
    if _activity_tracker_available(agent, require_truthy_agent=require_truthy_agent):
        try:
            activity = agent.get_activity_summary()
        except Exception:
            pass
    return activity


def seconds_since_activity(agent: Any, *, require_truthy_agent: bool = False) -> Any:
    idle_secs = 0.0
    if _activity_tracker_available(agent, require_truthy_agent=require_truthy_agent):
        try:
            activity = agent.get_activity_summary()
            idle_secs = activity.get("seconds_since_activity", 0.0)
        except Exception:
            pass
    return idle_secs


def is_idle_past_limit(
    agent: Any,
    inactivity_limit: float,
    *,
    require_truthy_agent: bool = False,
) -> bool:
    return seconds_since_activity(
        agent,
        require_truthy_agent=require_truthy_agent,
    ) >= inactivity_limit


def build_activity_diagnostic(
    agent: Any,
    *,
    require_truthy_agent: bool = False,
) -> InactivityDiagnostic:
    activity = get_activity_summary(agent, require_truthy_agent=require_truthy_agent)
    return InactivityDiagnostic(
        last_activity_desc=activity.get("last_activity_desc", "unknown"),
        seconds_since_activity=activity.get("seconds_since_activity", 0),
        current_tool=activity.get("current_tool"),
        api_call_count=activity.get("api_call_count", 0),
        max_iterations=activity.get("max_iterations", 0),
    )


def wait_for_future_or_inactivity(
    future: concurrent.futures.Future,
    *,
    agent: Any,
    inactivity_limit: float | None,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> InactivityWaitResult:
    if inactivity_limit is None:
        return InactivityWaitResult(result=future.result())

    while True:
        done, _ = concurrent.futures.wait({future}, timeout=poll_interval)
        if done:
            return InactivityWaitResult(result=future.result())
        if is_idle_past_limit(agent, inactivity_limit):
            return InactivityWaitResult(timed_out=True)


async def _maybe_await(value: Any) -> None:
    if inspect.isawaitable(value):
        await value


async def wait_for_task_or_inactivity(
    task: asyncio.Future,
    *,
    get_agent: Callable[[], Any],
    inactivity_limit: float | None,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    on_idle_check: Callable[[Any], Awaitable[None] | None] | None = None,
    on_poll: Callable[[], Awaitable[None] | None] | None = None,
    require_truthy_agent: bool = True,
) -> InactivityWaitResult:
    if inactivity_limit is None:
        return InactivityWaitResult(result=await task)

    while True:
        done, _ = await asyncio.wait({task}, timeout=poll_interval)
        if done:
            return InactivityWaitResult(result=task.result())

        agent = get_agent()
        idle_secs = seconds_since_activity(
            agent,
            require_truthy_agent=require_truthy_agent,
        )
        if on_idle_check is not None:
            await _maybe_await(on_idle_check(idle_secs))
        if idle_secs >= inactivity_limit:
            return InactivityWaitResult(timed_out=True)
        if on_poll is not None:
            await _maybe_await(on_poll())

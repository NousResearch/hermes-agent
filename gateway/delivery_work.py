"""Delivery stays active until the adapter and durable ledger have finished."""

import asyncio
from contextlib import contextmanager


def total_work_count(runner):
    """One aggregate for status and shutdown, including final delivery."""
    from gateway.run_runtime import managed_turn_count

    return (
        messaging_work_count(runner)
        + managed_turn_count(runner)
        + runner._active_cron_job_count()
        + runner._active_api_run_count()
        + runner._active_deferred_agent_worker_count()
    )


def messaging_work_count(runner):
    """Bridge runner and adapter task lifetimes without counting a turn twice.

    Adapter-owned tasks precede the ledger write and outlive the agent handler.
    Counting that full lifetime also covers time spent recording the obligation.
    """
    running = getattr(runner, "_running_agents", {})
    pending = set()
    owners = set()
    maps = [getattr(runner, "adapters", {})]
    maps.extend((getattr(runner, "_profile_adapters", {}) or {}).values())
    for adapters in maps:
        for adapter in adapters.values():
            for key, task in getattr(adapter, "_session_tasks", {}).items():
                if not task.done():
                    owners.add(task)
                    if key not in running:
                        pending.add(key)
    replay = getattr(runner, "_delivery_work", {})
    return (
        runner._running_agent_count()
        + len(pending)
        + sum(not task.done() for task in replay if task not in owners)
    )


@contextmanager
def delivery_work(runner):
    """Keep boot/reconnect replay awake through its send and ledger settlement."""
    task = asyncio.current_task()
    work = getattr(runner, "_delivery_work", None)
    if work is None:
        work = runner._delivery_work = {}
    work[task] = work.get(task, 0) + 1
    try:
        yield
    finally:
        work[task] -= 1
        if not work[task]:
            del work[task]

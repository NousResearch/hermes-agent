"""Delivery stays active until the adapter and durable ledger have finished."""

import asyncio
import logging
from contextlib import contextmanager


def _publish_work_count(runner):
    if not getattr(runner, "_running", False):
        return
    persist = getattr(runner, "_persist_active_agents", None)
    if callable(persist):
        try:
            persist()
        except Exception:
            logging.getLogger(__name__).debug(
                "delivery status refresh failed", exc_info=True
            )


def track_session_task(adapter, session_key, task):
    """Track cleanup and publish the owner's completion, including handoffs."""
    adapter._session_tasks[session_key] = task
    try:
        adapter._background_tasks.add(task)
    except TypeError:
        return False
    if hasattr(task, "add_done_callback"):
        task.add_done_callback(adapter._background_tasks.discard)
        task.add_done_callback(adapter._expected_cancelled_tasks.discard)
        task.add_done_callback(
            lambda _: _publish_work_count(getattr(adapter, "gateway_runner", None))
        )
    return True


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
    if work[task] == 1:
        _publish_work_count(runner)
    try:
        yield
    finally:
        work[task] -= 1
        if not work[task]:
            del work[task]
            _publish_work_count(runner)

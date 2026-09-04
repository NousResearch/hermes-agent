"""Keep admitted Group Chat command workers visible to gateway maintenance."""

from __future__ import annotations

import asyncio
from contextvars import copy_context
import threading
from typing import Callable, TypeVar


T = TypeVar("T")


class GroupChatMaintenanceError(RuntimeError):
    """A new send or retry was rejected before any backend work started."""


async def run_group_command_work(runner, action: str, operation: Callable[[], T]) -> T:
    if action not in {"send", "retry", "stop", "approve", "deny"}:
        raise ValueError("Unknown Group Chat work action")
    if action in {"send", "retry"} and (
        getattr(runner, "_external_drain_active", False) is True
        or getattr(runner, "_draining", False) is True
    ):
        raise GroupChatMaintenanceError(
            "New Group Chat work is paused for maintenance. Please try again shortly."
        )

    track = getattr(runner, "_track_deferred_agent_worker", None)
    if not callable(track):
        raise RuntimeError("Group Chat work tracking is unavailable")
    loop = asyncio.get_running_loop()
    completion = loop.create_future()
    try:
        track(completion, None)
    except BaseException:
        completion.set_result(None)
        raise

    context = copy_context()
    lock = threading.Lock()
    started = False
    cancelled = False

    def invoke():
        nonlocal started
        with lock:
            if cancelled:
                return None
            started = True
        return context.run(operation)

    def completed(done):
        if not completion.done():
            completion.set_result(None)
        if not done.cancelled():
            # A disconnected caller may no longer be waiting for this result.
            done.exception()

    try:
        worker = loop.run_in_executor(None, invoke)
    except BaseException:
        completion.set_result(None)
        raise
    worker.add_done_callback(completed)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # Cancellation can skip queued work, but must not hide a running thread.
        with lock:
            if not started:
                cancelled = True
                worker.cancel()
        raise

"""Shared async lifecycle helpers for gateway foreground agent runs."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, MutableMapping, Sequence


@dataclass(slots=True)
class GatewayAgentRuntimeTasks:
    """Async tasks that manage a single foreground gateway agent turn."""

    progress_task: asyncio.Task[Any] | None
    stream_task: asyncio.Task[Any] | None
    tracking_task: asyncio.Task[Any] | None
    interrupt_monitor_task: asyncio.Task[Any] | None
    long_running_notify_task: asyncio.Task[Any] | None


def _read_gateway_agent_activity(agent_ref: Any) -> dict[str, Any]:
    """Best-effort read of the agent activity tracker."""

    if not agent_ref or not hasattr(agent_ref, "get_activity_summary"):
        return {}
    try:
        activity = agent_ref.get_activity_summary() or {}
    except Exception:
        return {}
    return activity if isinstance(activity, dict) else {}


async def _run_stream_consumer_when_ready(
    *,
    stream_consumer_holder: Sequence[Any | None],
    wait_attempts: int = 200,
    wait_interval: float = 0.05,
) -> None:
    """Wait for the sync worker to install a stream consumer, then run it."""

    for _ in range(wait_attempts):
        stream_consumer = stream_consumer_holder[0]
        if stream_consumer is not None:
            await stream_consumer.run()
            return
        await asyncio.sleep(wait_interval)


async def _track_gateway_running_agent(
    *,
    agent_holder: Sequence[Any | None],
    running_agents: MutableMapping[str, Any],
    session_key: str | None,
    wait_interval: float = 0.05,
) -> None:
    """Replace the pending sentinel with the real agent once constructed."""

    while agent_holder[0] is None:
        await asyncio.sleep(wait_interval)
    if session_key:
        running_agents[session_key] = agent_holder[0]


async def _monitor_gateway_interrupts(
    *,
    adapter: Any,
    session_key: str | None,
    agent_holder: Sequence[Any | None],
    logger: Any,
    poll_interval: float = 0.2,
) -> None:
    """Forward adapter pending-message interrupts into the live agent."""

    if not adapter or not session_key:
        return

    while True:
        await asyncio.sleep(poll_interval)
        if hasattr(adapter, "has_pending_interrupt") and adapter.has_pending_interrupt(session_key):
            agent = agent_holder[0]
            if agent:
                pending_event = adapter.get_pending_message(session_key)
                pending_text = pending_event.text if pending_event else None
                logger.debug("Interrupt detected from adapter, signaling agent...")
                agent.interrupt(pending_text)
                return


async def _notify_gateway_long_running(
    *,
    adapter: Any,
    chat_id: str | None,
    metadata: dict[str, Any] | None,
    agent_holder: Sequence[Any | None],
    session_key: str | None,
    detail_builder: Callable[[Any, str], str],
    logger: Any,
    notify_interval: float = 600.0,
    started_at: float | None = None,
) -> None:
    """Send periodic keepalive notices while a foreground turn is still running."""

    if not adapter:
        return

    effective_started_at = time.time() if started_at is None else started_at
    while True:
        await asyncio.sleep(notify_interval)
        elapsed_mins = int((time.time() - effective_started_at) // 60)
        status_detail = detail_builder(agent_holder[0], session_key or "")
        try:
            await adapter.send(
                chat_id,
                f"⏳ Still working... ({elapsed_mins} min elapsed{status_detail})",
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("Long-running notification error: %s", exc)


def start_gateway_agent_runtime_tasks(
    *,
    tool_progress_enabled: bool,
    send_progress_messages: Callable[[], Awaitable[None]],
    stream_consumer_holder: Sequence[Any | None],
    agent_holder: Sequence[Any | None],
    running_agents: MutableMapping[str, Any],
    session_key: str | None,
    adapter: Any,
    chat_id: str | None,
    notify_metadata: dict[str, Any] | None,
    long_running_detail_builder: Callable[[Any, str], str],
    logger: Any,
    notify_interval: float = 600.0,
) -> GatewayAgentRuntimeTasks:
    """Start the async helper tasks that surround the sync worker thread."""

    progress_task = None
    if tool_progress_enabled:
        progress_task = asyncio.create_task(send_progress_messages())

    stream_task = asyncio.create_task(
        _run_stream_consumer_when_ready(
            stream_consumer_holder=stream_consumer_holder,
        )
    )
    tracking_task = asyncio.create_task(
        _track_gateway_running_agent(
            agent_holder=agent_holder,
            running_agents=running_agents,
            session_key=session_key,
        )
    )
    interrupt_monitor_task = asyncio.create_task(
        _monitor_gateway_interrupts(
            adapter=adapter,
            session_key=session_key,
            agent_holder=agent_holder,
            logger=logger,
        )
    )
    long_running_notify_task = asyncio.create_task(
        _notify_gateway_long_running(
            adapter=adapter,
            chat_id=chat_id,
            metadata=notify_metadata,
            agent_holder=agent_holder,
            session_key=session_key,
            detail_builder=long_running_detail_builder,
            logger=logger,
            notify_interval=notify_interval,
        )
    )
    return GatewayAgentRuntimeTasks(
        progress_task=progress_task,
        stream_task=stream_task,
        tracking_task=tracking_task,
        interrupt_monitor_task=interrupt_monitor_task,
        long_running_notify_task=long_running_notify_task,
    )


def build_gateway_inactivity_timeout_response(
    *,
    agent_holder: Sequence[Any | None],
    result_holder: Sequence[dict[str, Any] | None],
    tools_holder: Sequence[list[Any] | None],
    session_key: str | None,
    agent_timeout: float,
    logger: Any,
) -> dict[str, Any]:
    """Construct the user-facing timeout payload for an idle foreground turn."""

    timed_out_agent = agent_holder[0]
    activity = _read_gateway_agent_activity(timed_out_agent)
    last_desc = activity.get("last_activity_desc", "unknown")
    secs_ago = activity.get("seconds_since_activity", 0)
    current_tool = activity.get("current_tool")
    iteration_num = activity.get("api_call_count", 0)
    iteration_max = activity.get("max_iterations", 0)

    logger.error(
        "Agent idle for %.0fs (timeout %.0fs) in session %s "
        "| last_activity=%s | iteration=%s/%s | tool=%s",
        secs_ago,
        agent_timeout,
        session_key,
        last_desc,
        iteration_num,
        iteration_max,
        current_tool or "none",
    )

    if timed_out_agent and hasattr(timed_out_agent, "interrupt"):
        timed_out_agent.interrupt("Execution timed out (inactivity)")

    timeout_mins = int(agent_timeout // 60) or 1
    diag_lines = [
        f"⏱️ Agent inactive for {timeout_mins} min — no tool calls or API responses."
    ]
    if current_tool:
        diag_lines.append(
            f"The agent appears stuck on tool `{current_tool}` "
            f"({secs_ago:.0f}s since last activity, iteration {iteration_num}/{iteration_max})."
        )
    else:
        diag_lines.append(
            f"Last activity: {last_desc} ({secs_ago:.0f}s ago, iteration {iteration_num}/{iteration_max}). "
            "The agent may have been waiting on an API response."
        )
    diag_lines.append(
        "To increase the limit, set agent.gateway_timeout in config.yaml "
        "(value in seconds, 0 = no limit) and restart the gateway.\n"
        "Try again, or use /reset to start fresh."
    )

    return {
        "final_response": "\n".join(diag_lines),
        "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
        "api_calls": iteration_num,
        "tools": tools_holder[0] or [],
        "history_offset": 0,
        "failed": True,
    }


async def wait_for_gateway_agent_result(
    *,
    run_sync: Callable[[], dict[str, Any]],
    agent_holder: Sequence[Any | None],
    result_holder: Sequence[dict[str, Any] | None],
    tools_holder: Sequence[list[Any] | None],
    session_key: str | None,
    logger: Any,
    poll_interval: float = 5.0,
) -> dict[str, Any]:
    """Run the sync agent worker with inactivity-based timeout polling."""

    agent_timeout_raw = float(os.getenv("HERMES_AGENT_TIMEOUT", 1800))
    agent_timeout = agent_timeout_raw if agent_timeout_raw > 0 else None
    loop = asyncio.get_running_loop()
    executor_task = asyncio.ensure_future(loop.run_in_executor(None, run_sync))

    if agent_timeout is None:
        return await executor_task

    while True:
        done, _ = await asyncio.wait({executor_task}, timeout=poll_interval)
        if done:
            return executor_task.result()

        activity = _read_gateway_agent_activity(agent_holder[0])
        idle_secs = activity.get("seconds_since_activity", 0.0)
        if idle_secs >= agent_timeout:
            return build_gateway_inactivity_timeout_response(
                agent_holder=agent_holder,
                result_holder=result_holder,
                tools_holder=tools_holder,
                session_key=session_key,
                agent_timeout=agent_timeout,
                logger=logger,
            )


async def cleanup_gateway_agent_runtime_tasks(
    *,
    tasks: GatewayAgentRuntimeTasks,
    session_key: str | None,
    running_agents: MutableMapping[str, Any],
    running_agents_ts: MutableMapping[str, Any],
    stream_wait_timeout: float = 5.0,
) -> None:
    """Cancel helper tasks and clear session tracking for a foreground turn."""

    if tasks.progress_task:
        tasks.progress_task.cancel()
    tasks.interrupt_monitor_task.cancel()
    tasks.long_running_notify_task.cancel()

    if tasks.stream_task:
        try:
            await asyncio.wait_for(tasks.stream_task, timeout=stream_wait_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            tasks.stream_task.cancel()
            try:
                await tasks.stream_task
            except asyncio.CancelledError:
                pass

    tasks.tracking_task.cancel()
    if session_key and session_key in running_agents:
        del running_agents[session_key]
    if session_key:
        running_agents_ts.pop(session_key, None)

    for task in (
        tasks.progress_task,
        tasks.interrupt_monitor_task,
        tasks.tracking_task,
        tasks.long_running_notify_task,
    ):
        if task:
            try:
                await task
            except asyncio.CancelledError:
                pass


def mark_gateway_streaming_delivery_state(
    *,
    response: dict[str, Any],
    stream_consumer: Any,
) -> dict[str, Any]:
    """Mark the response as already delivered when streaming handled it."""

    if stream_consumer and getattr(stream_consumer, "already_sent", False) and isinstance(response, dict):
        response["already_sent"] = True
    return response

"""Physical gateway worker lifetime, independent of cancellation of its asyncio wrapper."""
from __future__ import annotations

import asyncio
import logging
import threading

logger = logging.getLogger("gateway.run")


def start_turn_worker(self, turn_ctx, run_sync):
    """Schedule ``run_sync`` on the executor plus the inactivity watchdog thread.

    *Inactivity* timeout (agent.gateway_timeout / HERMES_AGENT_TIMEOUT, env wins; 0 = unlimited),
    not wall-clock. The daemon watchdog is independent of asyncio: cgroup memory reclaim can
    starve the loop that runs the normal timeout poll."""
    from gateway.run import _float_env, _watch_gateway_turn_inactivity
    from tools.process_registry import process_registry
    agent_holder, session_key, run_generation = turn_ctx.agent_holder, turn_ctx.session_key, turn_ctx.run_generation
    _agent_timeout, _agent_warning = (
        v if v > 0 else None
        for v in (_float_env("HERMES_AGENT_TIMEOUT", 1800), _float_env("HERMES_AGENT_TIMEOUT_WARNING", 900))
    )

    # background=true processes survive a turn: reap only children created by THIS turn on timeout.
    _turn_task_id = turn_ctx.session_id or ""
    # The daemon watchdog is independent of asyncio: cgroup memory reclaim may starve the event loop
    # that runs the normal timeout poll, but it need not also postpone cleanup until the loop recovers
    # (#76115).
    _turn_process_baseline = process_registry.snapshot_running_ids(_turn_task_id)
    turn_ctx.process_task_id = _turn_task_id
    turn_ctx.process_baseline = _turn_process_baseline
    # task_id is session-scoped: gate the reap on this claim still being current so a replacement
    # turn's fresh process isn't killed by this turn's stale baseline.
    worker = self._RunAgentWorker(
        agent_timeout=_agent_timeout, agent_warning=_agent_warning, task_id=_turn_task_id,
        process_baseline=_turn_process_baseline, worker_done=threading.Event(),
        timeout_fired=threading.Event(), cleanup_lock=threading.Lock(),
        is_current=(
            (lambda: self._is_session_run_current(session_key, run_generation))
            if run_generation is not None
            else (lambda: True)
        ),
    )
    if session_key and run_generation is not None and self._is_session_run_current(session_key, run_generation):
        self._session_state(session_key).turn.worker_done = worker.worker_done
    if session_key and run_generation is not None:
        self._track_chat_stop_worker(session_key, run_generation, worker.worker_done)

    reservation = getattr(turn_ctx, "chat_execution_reservation", None)
    actual_session_id = turn_ctx.session_id

    def _run_sync_with_timeout_lifecycle():
        store = None
        started = False
        execution_scope = None
        try:
            if reservation is not None:
                from gateway.chat_execution_recovery import ChatExecutionStore
                home, reserved_key, reserved_generation = reservation
                if (reserved_key, reserved_generation) != (session_key, run_generation) or not worker.is_current():
                    raise RuntimeError("Durable worker admission is stale")
                store = ChatExecutionStore(home)
                if not store.bind_worker_started(reserved_key, reserved_generation, actual_session_id):
                    raise RuntimeError("Durable worker admission refused")
                started = True
                from agent.chat_execution_context import bind_chat_execution
                execution_scope = bind_chat_execution(home, reserved_key, reserved_generation)
                execution_scope.__enter__()
            return run_sync()
        finally:
            if execution_scope is not None:
                execution_scope.__exit__(None, None, None)
            if started:
                try:
                    if not store.worker_finished(reserved_key, reserved_generation):
                        logger.error("Durable worker completion refused for %s/%s", reserved_key, reserved_generation)
                except Exception:
                    # An unrecorded completion remains pending/unknown, never recovered by inference.
                    logger.exception("Failed durable worker completion for %s/%s", reserved_key, reserved_generation)
            worker.worker_done.set()
            # `.turn.agent` stays reachable until the *next* turn is claimed; clearing the
            # ownership markers now means a /stop on the finished turn no longer reaps background
            # work it left running.
            # `.turn.agent` on the session state is only reset to _AGENT_PENDING_SENTINEL when the
            # *next* turn is claimed (see _session_state(...).turn.agent = ... at claim time), so a
            # stale reference to this exact agent instance stays reachable from
            # _interrupt_and_clear_session() until then. See #76115.
            _finished_agent = agent_holder[0] if agent_holder else None
            if _finished_agent is not None:
                _finished_agent._gateway_turn_process_task_id = ""
                _finished_agent._gateway_turn_process_baseline = frozenset()

    if _agent_timeout is not None:
        threading.Thread(
            target=_watch_gateway_turn_inactivity,
            kwargs={
                "agent_holder": agent_holder, "timeout": _agent_timeout, "poll_interval": 5.0,
                **self._reaper_kwargs(worker),
            },
            name=f"gateway-turn-watchdog-{_turn_task_id[:12]}",
            daemon=True,
        ).start()
    worker.executor_task = asyncio.ensure_future(
        self._run_in_executor_with_context(_run_sync_with_timeout_lifecycle)
    )
    execution_id = getattr(turn_ctx, "internal_plugin_execution_id", None)
    if execution_id is not None:
        self._track_internal_plugin_execution_worker(execution_id, worker.worker_done)
    return worker

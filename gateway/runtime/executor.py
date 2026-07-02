"""Optional runtime executor that processes queued runs from RunManager.

The ``RuntimeExecutor`` bridges ``RunManager`` (status/events) with an
external AIAgent via a configurable ``AgentFactory``.  It is deliberately
separate from the HTTP route layer — routes remain thin and the executor
can be enabled/disabled or swapped per-deployment.

Design
------
- Executor is **optional** — existing runtime routes work without it.
- ``AgentFactory`` is a pluggable callable/class injected at construction
  time so unit tests can use a deterministic fake.
- ``SessionKeyFactory`` generates gateway session keys for executor-owned
  runs (no real gateway context required).
- Execution is async and single-run (``execute_run``); a background loop
  (``run_once`` / ``start`` / ``stop``) is provided for polling but
  callers can also hand-pick runs via ``execute_run`` directly.
- Status transitions, events, bindings, and cleanup are driven through
  ``RunManager`` and ``RuntimeControlBridge`` — the executor never
  bypasses either.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional, Protocol

from gateway.runtime.control_bridge import RuntimeControlBridge
from gateway.runtime.run_manager import RunManager

logger = logging.getLogger(__name__)


class AgentFactory(Protocol):
    """Protocol for AIAgent factories consumed by ``RuntimeExecutor``.

    Implementations receive the run context and must return an object
    with an ``async def run_conversation(message: str) -> dict`` method
    (matching ``AIAgent.run_conversation``).
    """

    async def create_agent(
        self,
        run_id: str,
        session_id: str,
        message: str,
        session_key: str,
        **kwargs: Any,
    ) -> Any:
        ...


class SessionKeyFactory:
    """Default session-key factory for executor-owned runs.

    Produces a deterministic ``exec-<run_id>`` string so the key is
    stable and traceable.
    """

    def create_session_key(self, run_id: str, session_id: str) -> str:
        return f"exec-{run_id}-{session_id}"


class _FakeAgentResult:
    """Minimal fake agent result for deterministic testing."""

    def __init__(self, result: Optional[Dict[str, Any]] = None):
        self._result = result or {
            "final_response": "fake agent response",
            "completed": True,
        }


class FakeAgentFactory:
    """Deterministic fake ``AgentFactory`` for unit tests.

    Each call to ``create_agent`` returns a mock whose
    ``run_conversation`` returns a fixed dict or raises a
    configurable exception for failure-path testing.

    Parameters
    ----------
    result:
        The dict returned by ``run_conversation``.  Defaults to
        ``{"final_response": "fake agent response", "completed": True}``.
    fail:
        When ``True``, ``run_conversation`` raises ``RuntimeError``
        with *fail_message*.
    fail_message:
        Exception message used when *fail* is ``True``.
    request_approval:
        When not ``None``, the fake agent will call this function
        before returning to simulate an approval-request lifecycle.
        Receives ``(run_id, run_manager, control_bridge)``.
    request_clarify:
        When not ``None``, the fake agent will call this function
        before returning to simulate a clarify-request lifecycle.
        Receives ``(run_id, run_manager, control_bridge)``.
    delay:
        Simulated execution delay in seconds (default 0).
    """

    def __init__(
        self,
        result: Optional[Dict[str, Any]] = None,
        fail: bool = False,
        fail_message: str = "fake agent error",
        request_approval: Optional[Callable] = None,
        request_clarify: Optional[Callable] = None,
        delay_seconds: float = 0.0,
        delay: float = 0,
    ):
        self._result = result or {
            "final_response": "fake agent response",
            "completed": True,
        }
        self._fail = fail
        self._fail_message = fail_message
        self._request_approval = request_approval
        self._request_clarify = request_clarify
        self._delay_seconds = max(0.0, float(delay_seconds or 0.0))
        self._delay = delay
        self.created_agents: list[Any] = []

    async def create_agent(
        self,
        run_id: str,
        session_id: str,
        message: str,
        session_key: str,
        **kwargs: Any,
    ) -> Any:
        agent = _FakeAgent(
            run_id=run_id,
            result=self._result,
            fail=self._fail,
            fail_message=self._fail_message,
            request_approval=self._request_approval,
            request_clarify=self._request_clarify,
            delay_seconds=self._delay_seconds,
            delay=self._delay,
        )
        self.created_agents.append(agent)
        return agent


class _FakeAgent:
    """Minimal agent-like object returned by FakeAgentFactory."""

    def __init__(
        self,
        run_id: str,
        result: Dict[str, Any],
        fail: bool = False,
        fail_message: str = "fake agent error",
        request_approval: Optional[Callable] = None,
        request_clarify: Optional[Callable] = None,
        delay_seconds: float = 0.0,
        delay: float = 0,
    ):
        self._run_id = run_id
        self._result = result
        self._fail = fail
        self._fail_message = fail_message
        self._request_approval = request_approval
        self._request_clarify = request_clarify
        self._delay_seconds = max(0.0, float(delay_seconds or 0.0))
        self._delay = delay
        self.interrupted = False

    def interrupt(self, reason: str = "run_stop") -> None:
        self.interrupted = True

    async def run_conversation(
        self,
        user_message: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self._delay:
            await asyncio.sleep(self._delay)

        if self._request_approval:
            self._request_approval(self._run_id)

        if self._request_clarify:
            self._request_clarify(self._run_id)

        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)

        if self._interrupt_requested():
            return {"failed": True, "error": "interrupted"}

        if self._fail:
            raise RuntimeError(self._fail_message)

        return dict(self._result)

    def _interrupt_requested(self) -> bool:
        return self.interrupted


def _redact_error(error: str) -> str:
    """Redact secrets and trim stack traces for safe logging/event payloads."""
    try:
        from agent.redact import redact_sensitive_text
    except ImportError:
        return str(error)
    return redact_sensitive_text(str(error))


class RuntimeExecutor:
    """Processes queued runs through a pluggable AIAgent factory.

    Parameters
    ----------
    run_manager:
        The ``RunManager`` instance.  Must be the same one used by the
        runtime routes so the executor can claim and update runs.
    control_bridge:
        Optional ``RuntimeControlBridge`` for live binding (stop/cancel).
        Without a bridge, the executor still manages status/events but
        stop signals are RunManager-only.
    agent_factory:
        Optional ``AgentFactory`` instance.  Defaults to ``None`` —
        ``execute_run`` will raise ``RuntimeError`` if no factory is
        configured.  Tests inject ``FakeAgentFactory`` here.
    session_factory:
        Optional ``SessionKeyFactory`` for generating session keys.
        Defaults to ``SessionKeyFactory()``.
    """

    def __init__(
        self,
        run_manager: RunManager,
        *,
        control_bridge: Optional[RuntimeControlBridge] = None,
        agent_factory: Optional[Any] = None,
        session_factory: Optional[SessionKeyFactory] = None,
    ):
        self._run_manager = run_manager
        self._control_bridge = control_bridge
        self._agent_factory = agent_factory
        self._session_factory = session_factory or SessionKeyFactory()
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._active_tasks: Dict[str, asyncio.Task] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    # -- public execution API ------------------------------------------------

    async def execute_run(self, run_id: str) -> Dict[str, Any]:
        """Execute one run identified by *run_id*.

        Steps
        -----
        1. Claim the run (``queued`` → ``running``).
        2. Generate a session key.
        3. Create an agent via the configured factory.
        4. Bind via ``RuntimeControlBridge`` if available.
        5. Call ``agent.run_conversation()``.
        6. On success: mark ``completed``, append ``done`` event.
        7. On failure: mark ``failed``, redact error, append ``error`` + ``done`` events.
        8. On cancellation: mark ``cancelled``.
        9. Unbind via ``RuntimeControlBridge`` in ``finally``.

        Returns a dict with keys ``run_id``, ``status``, and optionally
        ``error`` on failure.
        """
        # 1. Claim
        claimed = self._run_manager.claim_queued_run(run_id)
        if claimed is None:
            existing = self._run_manager.get_status(run_id)
            if existing is None:
                return {"error": "not_found", "message": f"Run not found: {run_id}"}
            return {
                "error": "conflict",
                "message": f"Run {run_id} is already claimed or terminal (status={existing['status']})",
            }

        session_key = self._session_factory.create_session_key(
            run_id, claimed["session_id"],
        )

        if self._agent_factory is None:
            self._run_manager.fail_run(
                run_id,
                error="RuntimeExecutor has no agent_factory configured",
            )
            return {
                "error": "not_supported",
                "message": "RuntimeExecutor has no agent_factory configured",
            }

        # 2-3. Create agent
        try:
            agent = await self._agent_factory.create_agent(
                run_id=run_id,
                session_id=claimed["session_id"],
                message=claimed.get("message", ""),
                session_key=session_key,
                model=claimed.get("model"),
            )
        except Exception as exc:
            error = _redact_error(str(exc))
            self._run_manager.fail_run(run_id, error=error)
            return {"error": "agent_creation_failed", "message": error}

        # 4. Bind
        if self._control_bridge is not None:
            try:
                self._control_bridge.bind_run(run_id, session_key, agent)
            except Exception:
                logger.debug("bind_run failed for run=%s", run_id, exc_info=True)

        task = asyncio.current_task()
        if task is not None:
            self._active_tasks[run_id] = task

        try:
            # 5. Execute — handle both async (FakeAgentFactory) and
            #    sync (real AIAgent.run_conversation) agents.
            try:
                user_msg = claimed.get("message", "") or ""
                if asyncio.iscoroutinefunction(agent.run_conversation):
                    result = await agent.run_conversation(
                        user_message=user_msg,
                    )
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None, agent.run_conversation, user_msg,
                    )
            except asyncio.CancelledError:
                self._run_manager.stop_run(run_id)
                return {"run_id": run_id, "status": "cancelled"}
            except Exception as exc:
                error = _redact_error(str(exc))
                self._run_manager.fail_run(run_id, error=error)
                return {"run_id": run_id, "status": "failed", "error": error}

            # 6. Check if run was already terminated (e.g. by concurrent stop)
            current_status = self._run_manager.get_status(run_id)
            if current_status and current_status.get("terminal"):
                return {"run_id": run_id, "status": current_status["status"]}

            # 7. Check result for structured failure
            if isinstance(result, dict) and result.get("failed"):
                error = _redact_error(str(result.get("error", "agent run failed")))
                self._run_manager.fail_run(run_id, error=error)
                return {"run_id": run_id, "status": "failed", "error": error}

            # 8. Complete
            final_response = (
                result.get("final_response", "") if isinstance(result, dict) else str(result)
            )
            self._run_manager.complete_run(run_id, result=final_response)
            return {"run_id": run_id, "status": "completed", "result": final_response}

        finally:
            # 9. Unbind
            if self._control_bridge is not None:
                try:
                    self._control_bridge.unbind_run(run_id)
                except Exception:
                    logger.debug("unbind_run failed for run=%s", run_id, exc_info=True)
            self._active_tasks.pop(run_id, None)

    async def run_once(self) -> Optional[str]:
        """Claim and execute the first available queued run.

        Returns the ``run_id`` if one was executed, or ``None`` if
        no queued run was available.
        """
        queued_ids = self._list_queued_run_ids()
        if not queued_ids:
            return None
        run_id = queued_ids[0]
        await self.execute_run(run_id)
        return run_id

    async def cancel_run(self, run_id: str) -> Dict[str, Any]:
        """Request cancellation of an active executor-owned run.

        If the run has an active task, it is cancelled and the bridge
        is signalled.  Terminal runs are no-ops.
        """
        if self._control_bridge is not None:
            return self._control_bridge.stop_run(run_id)
        return self._run_manager.stop_run(run_id)

    # -- background loop ----------------------------------------------------

    async def start(self, *, poll_interval: float = 1.0) -> None:
        """Start the background polling loop.

        The loop calls ``run_once`` every *poll_interval* seconds.
        Idempotent — subsequent calls are no-ops while running.
        """
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop(poll_interval))

    async def stop(self) -> None:
        """Stop the background polling loop and cancel active tasks."""
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            self._poll_task = None
        for run_id, task in list(self._active_tasks.items()):
            task.cancel()
        self._active_tasks.clear()

    async def _poll_loop(self, interval: float) -> None:
        while self._running:
            try:
                await self.run_once()
            except Exception:
                logger.debug("RuntimeExecutor poll cycle failed", exc_info=True)
            await asyncio.sleep(interval)

    # -- helpers ------------------------------------------------------------

    def _list_queued_run_ids(self) -> list[str]:
        """Return run_ids of all runs in 'queued' status.

        Uses the internal RunManager data directly since there is no
        dedicated list endpoint.
        """
        queued = []
        for run_id, status_obj in self._run_manager.list_runs().items():
            if status_obj.status == "queued":
                queued.append(run_id)
        return sorted(queued)

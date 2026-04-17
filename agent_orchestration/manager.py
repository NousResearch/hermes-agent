"""
AgentManager — Sub-Agent Lifecycle Management
===============================================

Ported from Claude Code's ``InProcessBackend + spawnInProcess`` pattern.
Uses Hermes' existing ``delegate_tool._build_child_agent()`` construction
pattern for thread-safe agent creation.

Key design:
  - Child agents built on main thread (like delegate_tool)
  - Execution via ThreadPoolExecutor
  - Global variable save/restore (model_tools._last_resolved_tool_names)
  - Delegation depth limit: max 2 levels
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Status constants
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"


@dataclass
class AgentRecord:
    """Tracks a spawned sub-agent's lifecycle."""
    agent_id: str
    goal: str
    status: str = STATUS_PENDING
    created_at: float = 0.0
    finished_at: Optional[float] = None
    future: Optional[Future] = None
    child_agent: Any = None  # AIAgent instance
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    iterations: int = 0
    model: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()


class AgentManager:
    """Manages the lifecycle of spawned sub-agents.

    Singleton per session.  Created by ``on_session_start`` hook,
    cleaned up by ``on_session_end`` hook.
    """

    def __init__(self, max_agents: int = 5):
        self._agents: Dict[str, AgentRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_agents)
        self._max_agents = max_agents
        self._parent_agent = None  # Set when session starts

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(
                1 for r in self._agents.values()
                if r.status in (STATUS_PENDING, STATUS_RUNNING)
            )

    def set_parent_agent(self, parent_agent) -> None:
        """Store reference to the parent agent for child construction."""
        self._parent_agent = parent_agent

    def spawn(
        self,
        goal: str,
        context: Optional[str] = None,
        toolsets: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_iterations: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        """Spawn a background sub-agent and return its agent_id.

        The child is built on the calling thread (thread-safe per
        delegate_tool pattern), then submitted to the executor pool.

        Returns:
            agent_id string
        Raises:
            RuntimeError if max_agents limit reached or no parent agent set
        """
        if self._parent_agent is None:
            raise RuntimeError("No parent agent set — call set_parent_agent() first")

        with self._lock:
            if self.active_count >= self._max_agents:
                raise RuntimeError(
                    f"Max concurrent agents ({self._max_agents}) reached. "
                    "Wait for an agent to finish or increase orchestration.max_agents."
                )

            agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
            record = AgentRecord(agent_id=agent_id, goal=goal)
            self._agents[agent_id] = record

        # Build child agent on the calling (main) thread
        child = self._build_child(
            goal=goal,
            context=context,
            toolsets=toolsets,
            model=model,
            max_iterations=max_iterations or 50,
        )
        record.child_agent = child
        record.model = getattr(child, "model", None)

        # Submit to executor
        record.status = STATUS_RUNNING
        record.future = self._executor.submit(
            self._run_agent, agent_id, goal, child
        )

        return agent_id

    def get_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a specific agent or all agents."""
        if agent_id:
            with self._lock:
                rec = self._agents.get(agent_id)
                if not rec:
                    return {"error": f"Agent '{agent_id}' not found"}
                return self._record_to_dict(rec)
        else:
            with self._lock:
                return {
                    "agents": {
                        aid: self._record_to_dict(rec)
                        for aid, rec in self._agents.items()
                    },
                    "active_count": self.active_count,
                    "max_agents": self._max_agents,
                }

    def send_message(self, agent_id: str, content: str) -> bool:
        """Send a message to a running agent via clarify_callback injection.

        Uses the same inject_message pattern as PluginContext.
        """
        with self._lock:
            rec = self._agents.get(agent_id)
            if not rec or rec.status != STATUS_RUNNING:
                return False

        child = rec.child_agent
        if child is None:
            return False

        # Try clarify_callback channel (set by _build_child if available)
        clarify_cb = getattr(child, "clarify_callback", None)
        if clarify_cb and callable(clarify_cb):
            try:
                clarify_cb(content)
                return True
            except Exception as exc:
                logger.debug("clarify_callback injection failed: %s", exc)

        # Fallback: try interrupt queue
        interrupt_q = getattr(child, "_interrupt_queue", None)
        if interrupt_q:
            try:
                interrupt_q.put(content)
                return True
            except Exception as exc:
                logger.debug("interrupt_queue injection failed: %s", exc)

        logger.warning("No message injection channel for agent %s", agent_id)
        return False

    def terminate(self, agent_id: str) -> bool:
        """Request termination of a running agent."""
        with self._lock:
            rec = self._agents.get(agent_id)
            if not rec or rec.status not in (STATUS_PENDING, STATUS_RUNNING):
                return False

        child = rec.child_agent
        if child:
            # Set interrupt flag — agent checks this each iteration
            if hasattr(child, "_interrupt_requested"):
                child._interrupt_requested = True
            # Also try cancel the future
            if rec.future and not rec.future.done():
                rec.future.cancel()

        rec.status = STATUS_INTERRUPTED
        rec.finished_at = time.time()
        return True

    def cleanup_all(self) -> None:
        """Terminate all running agents and shut down executor."""
        with self._lock:
            for rec in self._agents.values():
                if rec.status in (STATUS_PENDING, STATUS_RUNNING):
                    child = rec.child_agent
                    if child and hasattr(child, "_interrupt_requested"):
                        child._interrupt_requested = True
                    if rec.future and not rec.future.done():
                        rec.future.cancel()
                    rec.status = STATUS_INTERRUPTED
                    rec.finished_at = time.time()
                # Close child resources
                if rec.child_agent and hasattr(rec.child_agent, "close"):
                    try:
                        rec.child_agent.close()
                    except Exception:
                        pass

        self._executor.shutdown(wait=False, cancel_futures=True)
        self._agents.clear()

    def _build_child(
        self,
        goal: str,
        context: Optional[str],
        toolsets: Optional[List[str]],
        model: Optional[str],
        max_iterations: int,
    ):
        """Build a child AIAgent using delegate_tool's construction pattern."""
        from tools.delegate_tool import (
            _build_child_agent,
            _build_child_system_prompt,
        )

        parent = self._parent_agent

        # Save parent tool names before child construction mutates the global
        import model_tools
        parent_tool_names = list(model_tools._last_resolved_tool_names)

        try:
            child = _build_child_agent(
                task_index=0,
                goal=goal,
                context=context,
                toolsets=toolsets,
                model=model,
                max_iterations=max_iterations,
                parent_agent=parent,
            )
            # Save parent tool names for restoration after child runs
            child._delegate_saved_tool_names = parent_tool_names
        finally:
            # Restore global immediately after construction
            model_tools._last_resolved_tool_names = parent_tool_names

        return child

    def _run_agent(
        self, agent_id: str, goal: str, child
    ) -> Dict[str, Any]:
        """Run a child agent to completion (called from executor thread)."""
        start = time.monotonic()
        try:
            result = child.run_conversation(user_message=goal)
            duration = round(time.monotonic() - start, 2)

            summary = result.get("final_response") or ""
            completed = result.get("completed", False)
            interrupted = result.get("interrupted", False)

            if interrupted:
                status = STATUS_INTERRUPTED
            elif summary:
                status = STATUS_COMPLETED
            else:
                status = STATUS_FAILED

            entry = {
                "agent_id": agent_id,
                "status": status,
                "summary": summary,
                "api_calls": result.get("api_calls", 0),
                "duration_seconds": duration,
                "model": getattr(child, "model", None),
            }

            # Update record
            with self._lock:
                rec = self._agents.get(agent_id)
                if rec:
                    rec.status = status
                    rec.result = entry
                    rec.finished_at = time.time()
                    rec.iterations = result.get("api_calls", 0)
                    if status == STATUS_FAILED:
                        rec.error = result.get("error", "No response produced")

            return entry

        except Exception as exc:
            duration = round(time.monotonic() - start, 2)
            logger.exception("Agent %s failed", agent_id)

            with self._lock:
                rec = self._agents.get(agent_id)
                if rec:
                    rec.status = STATUS_FAILED
                    rec.error = str(exc)
                    rec.finished_at = time.time()

            return {
                "agent_id": agent_id,
                "status": STATUS_FAILED,
                "error": str(exc),
                "duration_seconds": duration,
            }

        finally:
            # Restore parent tool names
            import model_tools
            saved = getattr(child, "_delegate_saved_tool_names", None)
            if isinstance(saved, list):
                model_tools._last_resolved_tool_names = list(saved)

            # Close child resources
            try:
                if hasattr(child, "close"):
                    child.close()
            except Exception:
                pass

    @staticmethod
    def _record_to_dict(rec: AgentRecord) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "agent_id": rec.agent_id,
            "goal": rec.goal[:100],
            "status": rec.status,
            "model": rec.model,
            "iterations": rec.iterations,
        }
        if rec.error:
            d["error"] = rec.error
        if rec.result:
            d["summary"] = rec.result.get("summary", "")[:200]
        return d


# Module-level singleton — one AgentManager per process
_manager: Optional[AgentManager] = None
_manager_lock = threading.Lock()


def get_manager() -> AgentManager:
    """Get or create the global AgentManager."""
    global _manager
    with _manager_lock:
        if _manager is None:
            from .config import get_max_agents
            _manager = AgentManager(max_agents=get_max_agents())
        return _manager


def reset_manager() -> None:
    """Reset the global manager (used by on_session_end hook)."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.cleanup_all()
            _manager = None

"""
SubagentCoordinator — Unified Subagent Lifecycle Management.

Replaces the simple _active_children list with a structured state machine
and event system per parent agent.

Design principles (Module Independence MI):
  - Does NOT mix event logic into delegate_tool or AIAgent
  - Integrates with EventBus via optional subscribe/emit (no hard dependency)
  - Backward compatible: _active_children list interface is preserved
  - Thread-safe: all operations use the coordinator's own lock

Lifecycle states (AgentLifecycle):
    started   → subagent has been constructed and registered
    running   → subagent is executing (run_conversation in progress)
    completed → subagent finished normally (completed=True)
    error     → subagent raised an exception
    timeout   → subagent exceeded its time budget

Event types emitted via optional EventBus:
    agent.started   → subagent registered, about to run
    agent.completed → subagent finished normally
    agent.error     → subagent raised an exception
    agent.timeout   → subagent exceeded time budget
    agent.interrupt → interrupt propagated to subagent
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.hermes.analytics import EventBus

logger = logging.getLogger(__name__)

# OrphanCleanupManager (optional — gracefully absent when psutil unavailable)
try:
    from agent.hermes.orphan_process_manager import OrphanCleanupManager
    _ORPHAN_MANAGER_AVAILABLE = True
except ImportError:
    _ORPHAN_MANAGER_AVAILABLE = False


# ─── Lifecycle State Machine ────────────────────────────────────────────────────


class AgentLifecycle(Enum):
    """
    Lifecycle states for a registered subagent.

    State diagram:
        started → running → completed
                       ↘ error
                       ↘ timeout
        started → running → (interrupt) → (no explicit state, runs to completion/error)
    """
    STARTED = auto()    # Registered, about to run
    RUNNING = auto()    # Actively executing
    COMPLETED = auto()  # Finished normally
    ERROR = auto()      # Raised an exception
    TIMEOUT = auto()    # Exceeded time budget


# Valid transitions: current_state → set of allowed next states
_LIFECYCLE_TRANSITIONS: Dict[AgentLifecycle, frozenset] = {
    AgentLifecycle.STARTED: frozenset([AgentLifecycle.RUNNING, AgentLifecycle.COMPLETED, AgentLifecycle.ERROR, AgentLifecycle.TIMEOUT]),
    AgentLifecycle.RUNNING: frozenset([AgentLifecycle.COMPLETED, AgentLifecycle.ERROR, AgentLifecycle.TIMEOUT]),
    AgentLifecycle.COMPLETED: frozenset(),  # Terminal
    AgentLifecycle.ERROR: frozenset(),       # Terminal
    AgentLifecycle.TIMEOUT: frozenset(),     # Terminal
}


def _transition(state: AgentLifecycle, next_state: AgentLifecycle) -> AgentLifecycle:
    """Validate and return the next lifecycle state."""
    if next_state not in _LIFECYCLE_TRANSITIONS.get(state, frozenset()):
        logger.warning(
            "Invalid lifecycle transition %s → %s (current: %s); keeping %s",
            state.name, next_state.name, state.name, state.name,
        )
        return state
    return next_state


# ─── Subagent Record ───────────────────────────────────────────────────────────


@dataclass
class SubagentRecord:
    """
    Structured record for a registered subagent.

    Tracks lifecycle state, timing, result metadata, and error details
    for one subagent invocation.
    """
    agent: Any  # The AIAgent instance (kept weak-ref friendly)
    task_index: int = 0
    goal: str = ""
    lifecycle: AgentLifecycle = AgentLifecycle.STARTED
    started_at: float = field(default_factory=time.monotonic)
    ended_at: Optional[float] = None
    duration_seconds: float = 0.0
    exit_reason: str = "unknown"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    depth: int = 0  # Delegation depth when this subagent was spawned

    @property
    def is_alive(self) -> bool:
        """True while the subagent is in a non-terminal state."""
        return self.lifecycle in (AgentLifecycle.STARTED, AgentLifecycle.RUNNING)

    @property
    def is_terminal(self) -> bool:
        """True once the subagent has reached a final state."""
        return self.lifecycle in (AgentLifecycle.COMPLETED, AgentLifecycle.ERROR, AgentLifecycle.TIMEOUT)


# ─── SubagentCoordinator ──────────────────────────────────────────────────────


class SubagentCoordinator:
    """
    Unified lifecycle manager for subagents spawned by a single parent agent.

    Replaces the simple _active_children list with:
      - Per-subagent lifecycle state machine
      - Event subscription (agent.started / completed / error / timeout)
      - Optional EventBus integration (fire-and-forget, no hard dependency)
      - Thread-safe registration / unregistration
      - Backward-compatible _active_children list interface

    One instance per parent AIAgent, stored in InterruptState.
    """

    # Optional EventBus event types for subagent lifecycle
    _EVENT_TYPES = {
        "started",
        "completed",
        "error",
        "timeout",
        "interrupt",
    }

    def __init__(
        self,
        parent_agent: Any = None,
        event_bus: Optional["EventBus"] = None,
        load_router: Any = None,
    ):
        self._parent_agent = parent_agent
        self._records: List[SubagentRecord] = []
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable[["SubagentEvent"], None]]] = {}
        self._sub_lock = threading.Lock()
        # EventBus injected via constructor (avoids runtime circular import)
        self._event_bus: Optional["EventBus"] = event_bus
        # Optional LoadAwareRouter for load-based routing (GL design: decoupled)
        self._load_router: Any = load_router

    # ── Load Router Integration ──────────────────────────────────────────────

    @property
    def load_router(self) -> Any:
        """Return the attached LoadAwareRouter, if any."""
        return self._load_router

    def attach_load_router(self, router: Any) -> None:
        """
        Attach a LoadAwareRouter to sync subagent lifecycle with load metrics.

        The router's on_task_start / on_task_complete / on_task_error
        methods are called on each lifecycle transition.
        """
        self._load_router = router
        if router is not None:
            # Sync initial state from coordinator records
            router.sync_with_coordinator()
            logger.debug("SubagentCoordinator attached LoadAwareRouter")

    def detach_load_router(self) -> None:
        """Detach the LoadAwareRouter."""
        self._load_router = None

    # ── SharedStateStore Integration ─────────────────────────────────────────

    @property
    def shared_state(self) -> Any:
        """
        Return the SharedStateStore attached to this coordinator.

        Returns None if no SharedStateStore has been attached yet.
        Lazy-created on first access if the coordinator has a parent agent
        with an EventBus.
        """
        if hasattr(self, "_shared_state") and self._shared_state is not None:
            return self._shared_state

        # Lazy creation: attach a new SharedStateStore if we have a parent EventBus
        parent = self._parent_agent
        if parent is None:
            return None
        event_bus = getattr(parent, "_event_bus", None)
        if event_bus is None:
            return None

        from agent.shared_state import SharedStateStore
        session_id = getattr(parent, "session_id", "") or ""
        self._shared_state = SharedStateStore(session_id=session_id)
        self._shared_state.attach_to_event_bus(event_bus)
        logger.debug("SubagentCoordinator lazy-created SharedStateStore (session_id=%s)", session_id)
        return self._shared_state

    def attach_shared_state(self, store: Any) -> None:
        """
        Attach a SharedStateStore to this coordinator.

        The store's EventBus attachment is managed independently —
        this just holds a reference for access via .shared_state.
        """
        self._shared_state = store
        logger.debug("SubagentCoordinator attached SharedStateStore")

    def detach_shared_state(self) -> None:
        """Detach and shut down the SharedStateStore."""
        store = getattr(self, "_shared_state", None)
        if store is not None:
            store.detach_from_event_bus()
            self._shared_state = None
            logger.debug("SubagentCoordinator detached SharedStateStore")

    def _notify_load_router(self, event: str, record: SubagentRecord) -> None:
        """Notify the load router of a lifecycle event (fire-and-forget)."""
        router = self._load_router
        if router is None:
            return
        try:
            agent_id = id(record.agent)
            if event == "running":
                # Task transition STARTED → RUNNING: mark task start
                router.on_task_start(agent_id)
            elif event == "completed":
                router.on_task_complete(agent_id, record.duration_seconds, success=True)
            elif event == "error":
                router.on_task_complete(agent_id, record.duration_seconds, success=False)
            elif event == "timeout":
                router.on_task_complete(agent_id, record.duration_seconds, success=False)
        except Exception as e:
            logger.debug("Load router notification failed for %s: %s", event, e)

    # ── Orphan Cleanup Integration ─────────────────────────────────────────────

    def initialize_orphan_cleanup(self) -> None:
        """
        Initialize orphan cleanup manager with this coordinator's EventBus.

        Called by Bootstrap after the EventBus is set up, so that orphan
        cleanup events are observable via the telemetry pipeline.
        """
        if not _ORPHAN_MANAGER_AVAILABLE:
            return
        try:
            manager = OrphanCleanupManager.get_instance()
            manager.set_event_bus(self._get_event_bus())
            manager.register_with_shutdown_manager()
            manager.start_periodic_scan()
            logger.debug("OrphanCleanupManager initialized for SubagentCoordinator")
        except Exception as e:
            logger.debug("Failed to initialize OrphanCleanupManager: %s", e)

    # ── EventBus Integration ─────────────────────────────────────────────────

    def _get_event_bus(self) -> Optional["EventBus"]:
        """Return the injected EventBus (constructor-injected)."""
        return self._event_bus

    def _emit_event_bus(
        self,
        event_type: str,
        record: SubagentRecord,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a lifecycle event to the optional EventBus (fire-and-forget)."""
        bus = self._get_event_bus()
        if bus is None:
            return
        try:
            payload: Dict[str, Any] = {
                "subagent_id": id(record.agent),
                "task_index": record.task_index,
                "goal": record.goal,
                "lifecycle": record.lifecycle.name,
                "duration_seconds": record.duration_seconds,
                "exit_reason": record.exit_reason,
            }
            if record.error:
                payload["error"] = record.error
            if record.result:
                payload["summary"] = record.result.get("summary", "")
                payload["status"] = record.result.get("status", "")
            if extra:
                payload.update(extra)
            bus.emit_event(f"agent.{event_type}", payload)
        except Exception as e:
            logger.debug("EventBus emit failed for agent.%s: %s", event_type, e)

    # ── Lifecycle State Machine ──────────────────────────────────────────────

    def _transition_record(
        self,
        record: SubagentRecord,
        next_state: AgentLifecycle,
        **kwargs,
    ) -> SubagentRecord:
        """Apply a validated state transition to a record and emit events."""
        prev = record.lifecycle
        record.lifecycle = _transition(prev, next_state)

        if prev != record.lifecycle:
            event_type = record.lifecycle.name.lower()
            self._emit_internal_event(event_type, record)

            # Also emit to EventBus for analytics consumers
            self._emit_event_bus(event_type, record)

            # Notify LoadAwareRouter of lifecycle transition
            self._notify_load_router(event_type, record)

        return record

    # ── Internal Event Subscription ──────────────────────────────────────────

    def subscribe(
        self,
        event_type: str,
        handler: Callable[["SubagentEvent"], None],
    ) -> None:
        """
        Subscribe to subagent lifecycle events.

        Args:
            event_type: One of "started", "running", "completed", "error", "timeout", "interrupt"
            handler: Callback receiving a SubagentEvent
        """
        if event_type not in self._EVENT_TYPES:
            logger.warning("Unknown subagent event type: %s", event_type)
            return
        with self._sub_lock:
            self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[["SubagentEvent"], None],
    ) -> None:
        """Unsubscribe a previously registered handler."""
        with self._sub_lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    h for h in self._subscribers[event_type]
                    if h != handler
                ]

    def _emit_internal_event(
        self,
        event_type: str,
        record: SubagentRecord,
    ) -> None:
        """Deliver an event to all internal subscribers (fire-and-forget)."""
        event = SubagentEvent(
            event_type=event_type,
            record=record,
            coordinator=self,
        )
        with self._sub_lock:
            handlers = list(self._subscribers.get(event_type, []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(
                    "SubagentEvent handler failed for %s: %s",
                    event_type, e,
                )

    # ── Registration / Unregistration ─────────────────────────────────────────

    def register(
        self,
        agent: Any,
        task_index: int = 0,
        goal: str = "",
        depth: int = 0,
    ) -> SubagentRecord:
        """
        Register a new subagent and transition it to STARTED state.

        Thread-safe. Emits "agent.started" event.

        Args:
            agent: The AIAgent instance
            task_index: Index of this task in the batch
            goal: Description of the delegated task
            depth: The delegation depth at which this subagent was spawned.
                   Used for depth limit detection and flat-mode enforcement.
        """
        record = SubagentRecord(
            agent=agent,
            task_index=task_index,
            goal=goal,
            lifecycle=AgentLifecycle.STARTED,
            started_at=time.monotonic(),
            depth=depth,
        )
        with self._lock:
            self._records.append(record)

        logger.debug(
            "Subagent registered: task_index=%d, goal=%r",
            task_index, goal[:60] if goal else "",
        )
        return record

    def mark_running(self, agent: Any) -> None:
        """
        Transition a registered subagent to RUNNING state.

        Called when the subagent's run_conversation begins.
        Thread-safe.
        """
        record = self._find_record(agent)
        if record is None:
            logger.warning("mark_running: agent not found in coordinator")
            return
        self._transition_record(record, AgentLifecycle.RUNNING)
        # Register thread with OrphanCleanupManager for GL resource cleanup
        if _ORPHAN_MANAGER_AVAILABLE:
            try:
                manager = OrphanCleanupManager.get_instance()
                manager.register_thread(
                    thread_id=threading.current_thread().ident,
                    session_id=getattr(self._parent_agent, 'session_id', '') or '',
                    goal=record.goal,
                )
            except Exception as e:
                logger.debug("Orphan cleanup registration failed: %s", e)

    def mark_completed(
        self,
        agent: Any,
        result: Optional[Dict[str, Any]] = None,
        exit_reason: str = "completed",
    ) -> None:
        """
        Transition a subagent to COMPLETED state.

        Called on normal subagent exit. Thread-safe.
        """
        record = self._find_record(agent)
        if record is None:
            logger.warning("mark_completed: agent not found in coordinator")
            return
        record.result = result
        record.exit_reason = exit_reason
        record.ended_at = time.monotonic()
        record.duration_seconds = round(record.ended_at - record.started_at, 3)
        self._transition_record(record, AgentLifecycle.COMPLETED)
        logger.debug(
            "Subagent completed: task_index=%d, reason=%s, duration=%.2fs",
            record.task_index, exit_reason, record.duration_seconds,
        )

    def mark_error(
        self,
        agent: Any,
        error: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Transition a subagent to ERROR state.

        Called when a subagent raises an exception. Thread-safe.
        """
        record = self._find_record(agent)
        if record is None:
            logger.warning("mark_error: agent not found in coordinator")
            return
        record.error = error
        record.result = result
        record.ended_at = time.monotonic()
        record.duration_seconds = round(record.ended_at - record.started_at, 3)
        record.exit_reason = "error"
        self._transition_record(record, AgentLifecycle.ERROR)
        # Trigger orphan cleanup for GL (Generative Loop) resource cleanup
        if _ORPHAN_MANAGER_AVAILABLE:
            try:
                manager = OrphanCleanupManager.get_instance()
                manager.cleanup_thread(
                    thread_id=threading.current_thread().ident,
                    reason="error",
                )
            except Exception as e:
                logger.debug("Orphan cleanup on error failed: %s", e)
        logger.debug(
            "Subagent error: task_index=%d, error=%s",
            record.task_index, str(error)[:120],
        )

    def mark_timeout(self, agent: Any) -> None:
        """
        Transition a subagent to TIMEOUT state.

        Called when a subagent exceeds its time budget. Thread-safe.
        """
        record = self._find_record(agent)
        if record is None:
            logger.warning("mark_timeout: agent not found in coordinator")
            return
        record.ended_at = time.monotonic()
        record.duration_seconds = round(record.ended_at - record.started_at, 3)
        record.exit_reason = "timeout"
        self._transition_record(record, AgentLifecycle.TIMEOUT)
        # Trigger orphan cleanup for GL (Generative Loop) resource cleanup
        if _ORPHAN_MANAGER_AVAILABLE:
            try:
                manager = OrphanCleanupManager.get_instance()
                manager.cleanup_thread(
                    thread_id=threading.current_thread().ident,
                    reason="timeout",
                )
            except Exception as e:
                logger.debug("Orphan cleanup on timeout failed: %s", e)
        logger.debug("Subagent timeout: task_index=%d", record.task_index)

    def unregister(self, agent: Any) -> None:
        """
        Remove a subagent from the active registry.

        This is called after the subagent thread finishes, regardless of
        outcome. The lifecycle state is already set by mark_* before this call.
        Thread-safe.
        """
        with self._lock:
            self._records = [r for r in self._records if r.agent is not agent]
        # Unregister thread from OrphanCleanupManager (called on normal exit)
        if _ORPHAN_MANAGER_AVAILABLE:
            try:
                OrphanCleanupManager.get_instance().unregister_thread(threading.current_thread().ident)
            except Exception:
                pass

    def _find_record(self, agent: Any) -> Optional[SubagentRecord]:
        """Find the record for a given agent (caller holds lock or accepts race)."""
        for record in self._records:
            if record.agent is agent:
                return record
        return None

    # ── Interrupt Propagation ─────────────────────────────────────────────────

    def interrupt_all(self, message: str = "") -> int:
        """
        Propagate an interrupt to all currently-running subagents.

        Returns the number of subagents that were interrupted.
        Thread-safe.
        """
        count = 0
        with self._lock:
            for record in self._records:
                if record.is_alive:
                    try:
                        agent = record.agent
                        if hasattr(agent, "_interrupt_requested"):
                            agent._interrupt_requested = True
                        if hasattr(agent, "_interrupt_message"):
                            agent._interrupt_message = message
                        count += 1
                    except Exception as e:
                        logger.debug("interrupt_all: failed to interrupt agent: %s", e)

        if count > 0:
            self._emit_interrupt_event(count, message)
            logger.info("Interrupted %d subagent(s)", count)
        return count

    def _emit_interrupt_event(self, count: int, message: str) -> None:
        """Emit interrupt events to internal subscribers."""
        event_type = "interrupt"
        # Build a synthetic record for the interrupt event
        synthetic = SubagentRecord(
            agent=self._parent_agent,
            goal=f"interrupt:{count} agents",
            lifecycle=AgentLifecycle.RUNNING,
        )
        with self._sub_lock:
            handlers = list(self._subscribers.get(event_type, []))
        for handler in handlers:
            try:
                handler(
                    SubagentEvent(
                        event_type=event_type,
                        record=synthetic,
                        coordinator=self,
                        extra={"interrupted_count": count, "message": message},
                    )
                )
            except Exception as e:
                logger.warning("Interrupt event handler failed: %s", e)

    # ── Backward-Compatible _active_children List Interface ───────────────────

    @property
    def _active_children(self) -> List[Any]:
        """
        Return the list of live agent instances.

        Provided for backward compatibility with code that accesses
        ``parent_agent._active_children`` directly.
        """
        with self._lock:
            return [r.agent for r in self._records if r.is_alive]

    @property
    def _all_children(self) -> List[Any]:
        """Return all registered agents (including terminated)."""
        with self._lock:
            return [r.agent for r in self._records]

    @property
    def _active_children_lock(self) -> threading.Lock:
        """Return the coordinator's lock for code that accesses it directly."""
        return self._lock

    # ── Queries ─────────────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        """Number of subagents currently in a non-terminal state."""
        with self._lock:
            return sum(1 for r in self._records if r.is_alive)

    @property
    def total_count(self) -> int:
        """Total number of subagents ever registered."""
        with self._lock:
            return len(self._records)

    def get_records(self) -> List[SubagentRecord]:
        """Return a snapshot of all subagent records."""
        with self._lock:
            return list(self._records)

    def get_record(self, agent: Any) -> Optional[SubagentRecord]:
        """Return the record for a specific agent, or None."""
        with self._lock:
            return self._find_record(agent)

    def get_summary(self) -> Dict[str, Any]:
        """Return an aggregate summary of all subagent activity."""
        with self._lock:
            counts = {st.name.lower(): 0 for st in AgentLifecycle}
            total_duration = 0.0
            for r in self._records:
                counts[r.lifecycle.name.lower()] += 1
                total_duration += r.duration_seconds

            return {
                "total": len(self._records),
                "active": sum(1 for r in self._records if r.is_alive),
                "by_state": counts,
                "total_duration_seconds": round(total_duration, 3),
            }


# ─── SubagentEvent ─────────────────────────────────────────────────────────────


@dataclass
class SubagentEvent:
    """
    Event delivered to subscribers of SubagentCoordinator.

    Attributes:
        event_type:   "started" | "running" | "completed" | "error" | "timeout" | "interrupt"
        record:       SubagentRecord for the subagent that triggered the event
        coordinator:  The SubagentCoordinator instance that emitted this event
        extra:        Additional context (used for interrupt events)
        timestamp:    When the event was created
    """
    event_type: str
    record: SubagentRecord
    coordinator: SubagentCoordinator
    extra: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

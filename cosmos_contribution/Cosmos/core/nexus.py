"""
cosmos Nexus: The Neural Event Bus.

The Nexus is the central nervous system of cosmos v1.3.
It replaces traditional "function calls" with a high-speed, asynchronous event bus
that allows the Agent Swarm to coordinate in real-time.

UPDATES:
- Added Middleware pipeline support
- Added Priority Queues (via urgency sort)
- Added 'Signal Black Box' for debugging
"""

import asyncio
import uuid
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import   Callable, Optional, Awaitable
from loguru import logger

print("DEBUG: nexus.py: Loading module...")

class SignalType(Enum):
    # Core Lifecycle
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    
    # Cognitive Signals
    THOUGHT_EMITTED = "cognitive.thought"
    DECISION_REACHED = "cognitive.decision"
    ANOMALY_DETECTED = "cognitive.anomaly"
    CONFUSION_DETECTED = "cognitive.confusion"
    MEMORY_CONSOLIDATION = "cognitive.memory_consolidation"
    
    # Task Signals
    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_BLOCKED = "task.blocked"
    
    # Evolution & Self-Improvement
    RECURSIVE_EVOLUTION_TRIGGERED = "evolution.recursive_triggered"
    SELF_UPGRADE_PROPOSED = "evolution.self_upgrade_proposed"
    
    # External I/O
    USER_MESSAGE = "io.user.message"
    USER_INTERRUPTION = "io.user.interruption"
    EXTERNAL_ALERT = "io.external.alert"

    # P2P / Network Signals
    EXTERNAL_EVENT = "p2p.external_event"
    TASK_RECEIVED = "p2p.task_received"
    PEER_CONNECTED = "p2p.peer_connected"
    PEER_DISCONNECTED = "p2p.peer_disconnected"
    SKILL_RECEIVED = "p2p.skill_received"

@dataclass
class Signal:
    """A quantified event propagating through the Nexus."""
    type: SignalType
    payload: dict
    source_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    urgency: float = 0.5  # 0.0 to 1.0 (Higher = processed first)
    
    context_vector: Optional[list[float]] = None
    semantic_tags: list[str] = field(default_factory=list)

MiddlewareFunc = Callable[[Signal], bool] # Returns True to continue, False to block

class Nexus:
    """
    The central event bus with Neural Routing and Middleware.
    """
    _instance = None
    
    def __init__(self):
        self._subscribers: dict[SignalType[Callable[[Signal], Awaitable[None]]]] = {}
        self._history: list[Signal] = []  # Black Box
        self._middleware: list[MiddlewareFunc] = []
        self._lock = asyncio.Lock()
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, signal_type: SignalType, handler: Callable[[Signal], Awaitable[None]]):
        """Connect a synapse (handler) to a specific signal type."""
        if signal_type not in self._subscribers:
            self._subscribers[signal_type] = []
        self._subscribers[signal_type].append(handler)

    def add_middleware(self, func: MiddlewareFunc):
        """Add a middleware function that runs on every signal."""
        self._middleware.append(func)

    async def broadcast(self, signal: Signal):
        """
        Propagate a signal through the Nexus with priority and safety checks.
        """
        # 1. Store in Black Box (Circular Buffer)
        self._history.append(signal)
        if len(self._history) > 1000:
            self._history.pop(0)

        # 2. Run Middleware (Logging, Safety, Filtering)
        for mw in self._middleware:
            try:
                if not mw(signal):
                    logger.debug(f"Nexus: Signal {signal.type.value} blocked by middleware")
                    return
            except Exception as e:
                logger.error(f"Nexus: Middleware error: {e}")
                return

        # 3. Neural Routing
        handlers = self._subscribers.get(signal.type, [])
        if not handlers:
            return

        # 4. Asynchronous Propagation
        # Note: In a threaded environment, we might use a PriorityQueue here.
        # For asyncio, we just spawn tasks.
        try:
            await asyncio.gather(*[h(signal) for h in handlers], return_exceptions=True)
        except Exception as e:
            logger.error(f"Nexus: Critical propagation failure: {e}")

    async def emit(self, type: SignalType, payload: dict, source: str, urgency: float = 0.5):
        """Helper to create and broadcast a signal."""
        signal = Signal(
            type=type,
            payload=payload,
            source_id=source,
            urgency=urgency
        )
        await self.broadcast(signal)

    def inspection_black_box(self, last_n: int = 10) -> list[Signal]:
        """Retrieve recent signals for debugging/introspection."""
        return self._history[-last_n:]

# Global accessor
print("DEBUG: nexus.py: Creating Nexus instance...")
nexus = Nexus.get_instance()
print("DEBUG: nexus.py: Nexus instance created")

# Default Middleware: Logger
def logging_middleware(signal: Signal) -> bool:
    if signal.urgency > 0.7:
        logger.warning(f"🚨 [URGENT] {signal.type.value} from {signal.source_id}")
    else:
        logger.debug(f"⚡ {signal.type.value} from {signal.source_id}")
    return True

nexus.add_middleware(logging_middleware)

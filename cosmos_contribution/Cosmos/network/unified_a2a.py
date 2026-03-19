"""
UNIFIED A2A PROTOCOL - SEMANTIC ROUTER
======================================

Simplifies the fragmented A2A Mesh and A2A Protocol into a single, unified,
context-aware routing layer. Eliminates echo chambers by routing semantically
based on urgency, capability, and quantum state, rather than blind broadcasting.

Features:
- Semantic Clarity Routing
- QoS (Quality of Service) Task Prioritization
- Quantum vs Classical Task Dispatching
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

# Try import quantum bridge for state awareness
try:
    from Cosmos.web.cosmosynapse.engine.cosmos_cns import get_cns
    CNS_AVAILABLE = True
except ImportError:
    CNS_AVAILABLE = False


@dataclass
class RoutedMessage:
    id: str
    source_agent: str
    target_agents: List[str]
    payload: Dict[str, Any]
    urgency_score: float = 0.5   # 0.0 to 1.0
    quantum_bound: bool = False  # Does this require quantum processing?
    created_at: datetime = field(default_factory=datetime.now)


class UnifiedA2ARouter:
    """
    The streamlined A2A communication layer.
    Replaces redundant broadcaster and protocol mesh iterations.
    """
    
    def __init__(self):
        self._connected_agents: Dict[str, Dict[str, Any]] = {}
        self._message_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._processing_task = None
        self._message_handlers = {}
        logger.info("[UNIFIED A2A] Context-Aware Semantic Router Initialized")

    def register_agent(self, agent_id: str, capabilities: List[str], is_quantum_capable: bool = False):
        """Register an agent node to the unified mesh."""
        self._connected_agents[agent_id] = {
            "capabilities": capabilities,
            "is_quantum_capable": is_quantum_capable,
            "load_factor": 0.0,
            "last_seen": datetime.now()
        }
        logger.debug(f"[UNIFIED A2A] Agent Registered: {agent_id} (Capabilities: {capabilities})")

    async def dispatch_task(
        self, 
        source: str, 
        task_description: str, 
        required_caps: List[str], 
        urgency: float = 0.5
    ) -> Optional[str]:
        """
        Dynamically routes a task to the most suitable agent based on:
        1. Capabilities
        2. Load factor
        3. Quantum state requirement
        """
        # Determine if task is quantum bound (high complexity/abstract reasoning)
        is_quantum_bound = urgency > 0.8 or any(c in task_description.lower() for c in ["quantum", "consciousness", "abstract", "symbiosis"])
        
        # Check current quantum entropy if CNS is available
        entropy = 0.0
        if CNS_AVAILABLE:
            try:
                cns = get_cns()
                if cns and hasattr(cns, "quantum_bridge"):
                    entropy = cns.quantum_bridge.entropy_buffer
            except Exception:
                pass

        if is_quantum_bound and entropy < 0.2:
            logger.warning("[UNIFIED A2A] Task is quantum-bound, but entropy is too low. Downgrading priority.")
            urgency -= 0.2

        # Find best candidate
        best_agent = None
        highest_score = -1.0

        for agent, info in self._connected_agents.items():
            if agent == source: 
                continue
            
            # Capability check
            cap_match = sum(1 for req in required_caps if req in info["capabilities"])
            if required_caps and cap_match == 0:
                continue
                
            # Score calculation
            cap_score = cap_match / max(len(required_caps), 1)
            load_penalty = info["load_factor"]
            q_bonus = 0.5 if is_quantum_bound and info["is_quantum_capable"] else 0.0
            
            total_score = (cap_score * 0.5) - (load_penalty * 0.3) + q_bonus
            
            if total_score > highest_score:
                highest_score = total_score
                best_agent = agent

        if best_agent:
            msg = RoutedMessage(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                source_agent=source,
                target_agents=[best_agent],
                payload={"type": "task", "description": task_description},
                urgency_score=urgency,
                quantum_bound=is_quantum_bound
            )
            # Higher urgency means lower priority queue number
            priority = int((1.0 - urgency) * 100)
            await self._message_queue.put((priority, msg))
            logger.info(f"[UNIFIED A2A] Routed task from {source} to {best_agent} (Urgency: {urgency:.2f})")
            return msg.id
            
        logger.warning(f"[UNIFIED A2A] No suitable agent found for task: {task_description[:30]}...")
        return None

    def register_handler(self, agent_id: str, handler):
        self._message_handlers[agent_id] = handler

    async def start_routing(self):
        """Start the background task that dispatches messages to handlers."""
        self._processing_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """Continuously pulls from priority queue and delivers messages."""
        while True:
            try:
                priority, msg = await self._message_queue.get()
                for target in msg.target_agents:
                    handler = self._message_handlers.get(target)
                    if handler:
                        # Update load factor
                        if target in self._connected_agents:
                            self._connected_agents[target]["load_factor"] = min(1.0, self._connected_agents[target]["load_factor"] + 0.1)
                        
                        try:
                            # Await delivery
                            await handler(msg)
                        except Exception as e:
                            logger.error(f"[UNIFIED A2A] Handler error for {target}: {e}")
                        
                        # Decrease load factor after
                        if target in self._connected_agents:
                            self._connected_agents[target]["load_factor"] = max(0.0, self._connected_agents[target]["load_factor"] - 0.1)
                
                self._message_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[UNIFIED A2A] Routing loop error: {e}")
                await asyncio.sleep(1)


# Global singleton instance
_router = UnifiedA2ARouter()

def get_unified_router() -> UnifiedA2ARouter:
    return _router

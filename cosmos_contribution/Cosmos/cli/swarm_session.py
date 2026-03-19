"""
Cosmos Swarm Session - CLI to Swarm A2A Integration.

AGI v1.8.4 Feature: Enables CLI users to interact directly with the
agent swarm through A2A sessions.

Features:
- Start/end swarm sessions from CLI
- Trigger deliberations on user queries
- Stream agent responses in real-time
- View agent status and activity
- Participate in consensus voting
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Awaitable
from pathlib import Path

from loguru import logger


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SessionState(Enum):
    """State of a CLI swarm session."""
    IDLE = "idle"
    CONNECTED = "connected"
    DELIBERATING = "deliberating"
    WAITING = "waiting"
    ENDED = "ended"


class AgentStatus(Enum):
    """Status of an agent in the session."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    RESPONDING = "responding"


@dataclass
class AgentInfo:
    """Information about an agent in the session."""
    agent_id: str
    status: AgentStatus = AgentStatus.OFFLINE
    last_response: Optional[str] = None
    response_time_ms: float = 0.0
    message_count: int = 0
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "last_response": self.last_response[:100] if self.last_response else None,
            "response_time_ms": self.response_time_ms,
            "message_count": self.message_count,
            "capabilities": self.capabilities,
        }


@dataclass
class AgentResponse:
    """A response from an agent during chat/deliberation."""
    agent_id: str
    content: str
    response_type: str  # "propose", "critique", "refine", "vote", "final"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "response_type": self.response_type,
            "timestamp": self.timestamp.isoformat(),
            "is_final": self.is_final,
        }


@dataclass
class DeliberationProgress:
    """Progress of an ongoing deliberation."""
    deliberation_id: str
    prompt: str
    phase: str  # "propose", "critique", "refine", "vote", "complete"
    participating_agents: List[str]
    responses_received: int
    current_consensus: Optional[str] = None
    votes: Dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deliberation_id": self.deliberation_id,
            "phase": self.phase,
            "agents": self.participating_agents,
            "responses": self.responses_received,
            "consensus": self.current_consensus[:50] if self.current_consensus else None,
            "vote_count": len(self.votes),
        }


# =============================================================================
# SWARM SESSION
# =============================================================================

class SwarmSession:
    """
    CLI user's A2A session with the swarm.

    Enables users to:
    - Connect to the agent swarm
    - Chat with agents (streamed responses)
    - Trigger full deliberations
    - View real-time agent activity
    - Participate in swarm decisions
    """

    # Default agents to include in sessions
    DEFAULT_AGENTS = ["grok", "claude", "gemini", "deepseek", "kimi"]

    def __init__(
        self,
        session_id: Optional[str] = None,
        data_dir: str = "./data/cli_sessions",
    ):
        self.session_id = session_id or f"cli_{uuid.uuid4().hex[:12]}"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Session state
        self._state = SessionState.IDLE
        self._started_at: Optional[datetime] = None
        self._ended_at: Optional[datetime] = None

        # Agents
        self._agents: Dict[str, AgentInfo] = {}
        self._active_agents: Set[str] = set()

        # Chat history
        self._chat_history: List[Dict[str, Any]] = []
        self._max_history: int = 1000

        # Deliberation tracking
        self._current_deliberation: Optional[DeliberationProgress] = None
        self._deliberation_history: List[DeliberationProgress] = []

        # Response streaming
        self._response_callbacks: List[Callable[[AgentResponse], Awaitable[None]]] = []
        self._response_queue: asyncio.Queue[AgentResponse] = asyncio.Queue()

        # External integrations
        self._nexus = None
        self._deliberation_room = None
        self._collective_bridge = None
        self._a2a_mesh = None

        logger.info(f"SwarmSession {self.session_id} created")

    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================

    async def connect(
        self,
        agents: Optional[List[str]] = None,
    ) -> bool:
        """
        Connect to the swarm with specified agents.

        Args:
            agents: List of agent IDs to connect with (uses defaults if None)

        Returns:
            Whether connection was successful
        """
        if self._state not in [SessionState.IDLE, SessionState.ENDED]:
            logger.warning(f"Cannot connect: session in state {self._state}")
            return False

        agents = agents or self.DEFAULT_AGENTS
        self._active_agents = set(agents)
        self._started_at = datetime.now()
        self._state = SessionState.CONNECTED

        # Initialize agent info
        for agent_id in agents:
            self._agents[agent_id] = AgentInfo(
                agent_id=agent_id,
                status=AgentStatus.ONLINE,
            )

        # Try to connect to deliberation room
        await self._connect_deliberation_room()

        # Emit signal
        await self._emit_signal("CLI_SESSION_START", {
            "session_id": self.session_id,
            "agents": agents,
        })

        logger.info(f"Session {self.session_id} connected with {len(agents)} agents")
        return True

    async def disconnect(self) -> bool:
        """Disconnect from the swarm."""
        if self._state == SessionState.ENDED:
            return False

        self._state = SessionState.ENDED
        self._ended_at = datetime.now()

        # Update all agents to offline
        for agent in self._agents.values():
            agent.status = AgentStatus.OFFLINE

        # Emit signal
        await self._emit_signal("CLI_SESSION_END", {
            "session_id": self.session_id,
            "duration_seconds": self.get_duration_seconds(),
            "message_count": len(self._chat_history),
        })

        logger.info(f"Session {self.session_id} disconnected")
        return True

    async def _connect_deliberation_room(self) -> None:
        """Try to connect to the deliberation room."""
        try:
            from Cosmos.core.collective.deliberation import DeliberationRoom
            self._deliberation_room = DeliberationRoom()
            logger.debug("Connected to deliberation room")
        except ImportError:
            logger.debug("Deliberation room not available")
        except Exception as e:
            logger.debug(f"Could not connect to deliberation room: {e}")

    def connect_nexus(self, nexus) -> None:
        """Connect to Nexus event bus."""
        self._nexus = nexus

    def connect_collective_bridge(self, bridge) -> None:
        """Connect to collective bridge."""
        self._collective_bridge = bridge

    def connect_a2a_mesh(self, mesh) -> None:
        """Connect to A2A mesh."""
        self._a2a_mesh = mesh

    # =========================================================================
    # CHAT INTERFACE
    # =========================================================================

    async def chat(
        self,
        message: str,
        stream: bool = True,
    ) -> AsyncIterator[AgentResponse]:
        """
        Chat with the swarm, yielding agent responses as they arrive.

        Args:
            message: User's message
            stream: Whether to stream responses (if False, yields only final)

        Yields:
            AgentResponse objects from each agent
        """
        if self._state != SessionState.CONNECTED:
            logger.warning(f"Cannot chat: session in state {self._state}")
            return

        self._state = SessionState.WAITING

        # Store user message in history
        self._chat_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })

        # Emit signal
        await self._emit_signal("CLI_COMMAND", {
            "session_id": self.session_id,
            "command": "chat",
            "message": message[:100],
        })

        # Track responses
        responses_received = 0
        final_responses: Dict[str, AgentResponse] = {}

        # Query each active agent
        for agent_id in self._active_agents:
            if agent_id not in self._agents:
                continue

            agent = self._agents[agent_id]
            agent.status = AgentStatus.RESPONDING

            try:
                # Query the agent (use deliberation room or direct call)
                start_time = datetime.now()
                response_content = await self._query_agent(agent_id, message)

                if response_content:
                    elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

                    response = AgentResponse(
                        agent_id=agent_id,
                        content=response_content,
                        response_type="final",
                        is_final=True,
                    )

                    # Update agent info
                    agent.status = AgentStatus.ONLINE
                    agent.last_response = response_content
                    agent.response_time_ms = elapsed_ms
                    agent.message_count += 1

                    final_responses[agent_id] = response
                    responses_received += 1

                    # Store in history
                    self._chat_history.append({
                        "role": "assistant",
                        "agent": agent_id,
                        "content": response_content,
                        "timestamp": datetime.now().isoformat(),
                    })

                    # Stream if enabled
                    if stream:
                        yield response

                        # Notify callbacks
                        for callback in self._response_callbacks:
                            try:
                                await callback(response)
                            except Exception as e:
                                logger.error(f"Response callback error: {e}")

            except Exception as e:
                logger.error(f"Error querying agent {agent_id}: {e}")
                agent.status = AgentStatus.OFFLINE

        self._state = SessionState.CONNECTED

        # If not streaming, yield all at end
        if not stream:
            for response in final_responses.values():
                yield response

    async def _query_agent(
        self,
        agent_id: str,
        prompt: str,
    ) -> Optional[str]:
        """Query a single agent and get response."""
        # Try using deliberation room's registered agents
        if self._deliberation_room and hasattr(self._deliberation_room, "_agent_funcs"):
            if agent_id in self._deliberation_room._agent_funcs:
                try:
                    result = await self._deliberation_room._agent_funcs[agent_id](
                        prompt, 1000  # max tokens
                    )
                    if result:
                        return result[0] if isinstance(result, tuple) else result
                except Exception as e:
                    logger.debug(f"Deliberation room query failed: {e}")

        # Try using A2A mesh
        if self._a2a_mesh:
            try:
                result = await self._a2a_mesh.send_direct(
                    source=self.session_id,
                    target=agent_id,
                    message_type="m2m.query",
                    payload={"prompt": prompt},
                    requires_response=True,
                    timeout=30.0,
                )
                if result and "response" in result:
                    return result["response"]
            except Exception as e:
                logger.debug(f"A2A mesh query failed: {e}")

        # Fallback: try direct import and call
        try:
            return await self._query_agent_direct(agent_id, prompt)
        except Exception as e:
            logger.debug(f"Direct agent query failed: {e}")

        return None

    async def _query_agent_direct(
        self,
        agent_id: str,
        prompt: str,
    ) -> Optional[str]:
        """Direct query to agent via integration modules."""
        agent_modules = {
            "grok": "cosmos.integration.external.grok",
            "gemini": "cosmos.integration.external.gemini",
            "claude": "cosmos.integration.external.claude",
            "kimi": "cosmos.integration.external.kimi",
        }

        if agent_id not in agent_modules:
            return None

        try:
            import importlib
            module = importlib.import_module(agent_modules[agent_id])

            # Different modules have different interfaces
            if hasattr(module, "chat"):
                result = await module.chat(prompt=prompt, max_tokens=1000)
                if isinstance(result, dict):
                    return result.get("content")
                return result
            elif hasattr(module, "query"):
                return await module.query(prompt)

        except Exception as e:
            logger.debug(f"Direct query to {agent_id} failed: {e}")

        return None

    # =========================================================================
    # DELIBERATION
    # =========================================================================

    async def start_deliberation(
        self,
        prompt: str,
        agents: Optional[List[str]] = None,
    ) -> AsyncIterator[AgentResponse]:
        """
        Start a full deliberation on a prompt.

        Args:
            prompt: The prompt to deliberate on
            agents: Specific agents to include (uses active agents if None)

        Yields:
            AgentResponse objects for each phase (propose, critique, refine, vote)
        """
        if self._state != SessionState.CONNECTED:
            logger.warning(f"Cannot deliberate: session in state {self._state}")
            return

        self._state = SessionState.DELIBERATING
        agents = agents or list(self._active_agents)

        deliberation_id = f"delib_{uuid.uuid4().hex[:8]}"

        # Initialize progress tracking
        self._current_deliberation = DeliberationProgress(
            deliberation_id=deliberation_id,
            prompt=prompt,
            phase="propose",
            participating_agents=agents,
            responses_received=0,
        )

        # Emit signal
        await self._emit_signal("USER_DELIBERATION_REQUEST", {
            "session_id": self.session_id,
            "deliberation_id": deliberation_id,
            "prompt": prompt[:100],
            "agents": agents,
        })

        # Try using deliberation room if available
        if self._deliberation_room and hasattr(self._deliberation_room, "deliberate"):
            try:
                result = await self._deliberation_room.deliberate(
                    prompt=prompt,
                    agents=agents,
                )

                # Yield final result
                if result:
                    self._current_deliberation.phase = "complete"
                    self._current_deliberation.current_consensus = result.final_response

                    yield AgentResponse(
                        agent_id=result.winning_agent,
                        content=result.final_response,
                        response_type="final",
                        is_final=True,
                        metadata={
                            "deliberation_id": deliberation_id,
                            "vote_breakdown": result.vote_breakdown,
                            "consensus_reached": result.consensus_reached,
                        },
                    )

                    self._state = SessionState.CONNECTED
                    return

            except Exception as e:
                logger.warning(f"Deliberation room failed: {e}, falling back to simple")

        # Fallback: Simple round-robin query
        for phase in ["propose", "critique", "refine"]:
            self._current_deliberation.phase = phase

            for agent_id in agents:
                if agent_id not in self._agents:
                    continue

                # Modify prompt based on phase
                if phase == "propose":
                    phase_prompt = prompt
                elif phase == "critique":
                    phase_prompt = f"Critique the following responses:\n{prompt}"
                else:
                    phase_prompt = f"Provide your refined response:\n{prompt}"

                try:
                    response_content = await self._query_agent(agent_id, phase_prompt)
                    if response_content:
                        self._current_deliberation.responses_received += 1

                        yield AgentResponse(
                            agent_id=agent_id,
                            content=response_content,
                            response_type=phase,
                            metadata={"deliberation_id": deliberation_id},
                        )
                except Exception as e:
                    logger.error(f"Error in {phase} from {agent_id}: {e}")

        # Final consensus (simple selection of last response)
        self._current_deliberation.phase = "complete"
        self._state = SessionState.CONNECTED

        # Store in history
        self._deliberation_history.append(self._current_deliberation)
        self._current_deliberation = None

    def get_deliberation_progress(self) -> Optional[Dict[str, Any]]:
        """Get current deliberation progress."""
        if self._current_deliberation:
            return self._current_deliberation.to_dict()
        return None

    # =========================================================================
    # AGENT STATUS
    # =========================================================================

    async def get_agent_status(self) -> Dict[str, AgentInfo]:
        """Get status of all agents in the session."""
        return self._agents.copy()

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get info for a specific agent."""
        return self._agents.get(agent_id)

    def add_agent(self, agent_id: str) -> bool:
        """Add an agent to the session."""
        if agent_id in self._agents:
            return False

        self._agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            status=AgentStatus.ONLINE,
        )
        self._active_agents.add(agent_id)
        return True

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the session."""
        if agent_id not in self._agents:
            return False

        del self._agents[agent_id]
        self._active_agents.discard(agent_id)
        return True

    # =========================================================================
    # RESPONSE STREAMING
    # =========================================================================

    def on_response(
        self,
        callback: Callable[[AgentResponse], Awaitable[None]],
    ) -> None:
        """Register a callback for agent responses."""
        self._response_callbacks.append(callback)

    def remove_response_callback(
        self,
        callback: Callable[[AgentResponse], Awaitable[None]],
    ) -> bool:
        """Remove a response callback."""
        try:
            self._response_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # =========================================================================
    # SESSION STATE
    # =========================================================================

    def get_state(self) -> SessionState:
        """Get current session state."""
        return self._state

    def is_connected(self) -> bool:
        """Check if session is connected."""
        return self._state in [
            SessionState.CONNECTED,
            SessionState.DELIBERATING,
            SessionState.WAITING,
        ]

    def get_duration_seconds(self) -> float:
        """Get session duration in seconds."""
        if not self._started_at:
            return 0.0

        end = self._ended_at or datetime.now()
        return (end - self._started_at).total_seconds()

    def get_chat_history(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get chat history."""
        history = self._chat_history
        if limit:
            history = history[-limit:]
        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "state": self._state.value,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "duration_seconds": self.get_duration_seconds(),
            "active_agents": len(self._active_agents),
            "total_agents": len(self._agents),
            "chat_messages": len(self._chat_history),
            "deliberations": len(self._deliberation_history),
            "current_deliberation": self.get_deliberation_progress(),
        }

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    async def _emit_signal(self, signal_type: str, payload: Dict[str, Any]) -> None:
        """Emit a signal to Nexus."""
        if not self._nexus:
            return

        try:
            from Cosmos.core.nexus import SignalType

            signal_enum = getattr(SignalType, signal_type, None)
            if signal_enum:
                await self._nexus.emit(
                    type=signal_enum,
                    payload=payload,
                    source="swarm_session",
                    urgency=0.5,
                )
        except Exception as e:
            logger.debug(f"Failed to emit signal {signal_type}: {e}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_swarm_session(
    session_id: Optional[str] = None,
    data_dir: str = "./data/cli_sessions",
) -> SwarmSession:
    """Factory function to create a SwarmSession instance."""
    return SwarmSession(session_id=session_id, data_dir=data_dir)


# =============================================================================
# SESSION MANAGER
# =============================================================================

class SwarmSessionManager:
    """Manages multiple CLI swarm sessions."""

    def __init__(self):
        self._sessions: Dict[str, SwarmSession] = {}
        self._active_session: Optional[str] = None

    def create_session(
        self,
        session_id: Optional[str] = None,
    ) -> SwarmSession:
        """Create a new session."""
        session = create_swarm_session(session_id=session_id)
        self._sessions[session.session_id] = session

        if self._active_session is None:
            self._active_session = session.session_id

        return session

    def get_session(self, session_id: str) -> Optional[SwarmSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_active_session(self) -> Optional[SwarmSession]:
        """Get the currently active session."""
        if self._active_session:
            return self._sessions.get(self._active_session)
        return None

    def set_active_session(self, session_id: str) -> bool:
        """Set the active session."""
        if session_id in self._sessions:
            self._active_session = session_id
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        return [
            {
                "session_id": s.session_id,
                "state": s.get_state().value,
                "is_active": s.session_id == self._active_session,
            }
            for s in self._sessions.values()
        ]

    async def close_session(self, session_id: str) -> bool:
        """Close a session."""
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        await session.disconnect()

        if self._active_session == session_id:
            self._active_session = None

        return True

    async def close_all(self) -> None:
        """Close all sessions."""
        for session in self._sessions.values():
            await session.disconnect()
        self._sessions.clear()
        self._active_session = None


# Global session manager
_session_manager: Optional[SwarmSessionManager] = None


def get_session_manager() -> SwarmSessionManager:
    """Get the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SwarmSessionManager()
    return _session_manager

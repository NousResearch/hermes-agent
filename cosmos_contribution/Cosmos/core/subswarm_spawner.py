"""
Cosmos Sub-Swarm Spawner.

"Doomsday device? No no, it's a SWARM spawner! Much more terrifying!"

This module enables APIs and integrations to dynamically spin up specialized
sub-swarms for complex operations like market analysis, trading execution,
or multi-source research.

AGI Feature Set (v1.7):
- API-triggered sub-swarm creation
- Specialized agent groups for domains (trading, research, coding)
- Automatic result aggregation and merging
- Sub-swarm lifecycle management
- Integration with Nexus for event coordination
- Resource-aware spawning with limits

Use Cases:
- DexScreener API → Trading sub-swarm for token analysis
- Polymarket API → Prediction sub-swarm for odds evaluation
- Research query → Multi-source research sub-swarm
- Code task → Code review sub-swarm with specialist agents
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import   Optional, Callable, Tuple
from enum import Enum
from collections import defaultdict

from loguru import logger


# =============================================================================
# SUB-SWARM TYPES AND CONFIG
# =============================================================================

class SubSwarmType(Enum):
    """Types of specialized sub-swarms."""
    TRADING = "trading"           # Market analysis and execution
    RESEARCH = "research"         # Multi-source information gathering
    CODING = "coding"             # Code generation and review
    ANALYSIS = "analysis"         # Data analysis and synthesis
    PREDICTION = "prediction"     # Prediction market evaluation
    CREATIVE = "creative"         # Brainstorming and ideation
    CUSTOM = "custom"             # User-defined


class SubSwarmState(Enum):
    """States of a sub-swarm."""
    SPAWNING = "spawning"
    ACTIVE = "active"
    DELIBERATING = "deliberating"
    MERGING = "merging"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubSwarmConfig:
    """Configuration for a sub-swarm."""
    swarm_type: SubSwarmType
    min_agents: int = 2
    max_agents: int = 5
    timeout_seconds: float = 300.0      # 5 minute default
    require_consensus: bool = True       # Require agent agreement
    consensus_threshold: float = 0.7     # 70% agreement needed
    parallel_execution: bool = True      # Run agents in parallel
    merge_strategy: str = "vote"         # "vote", "aggregate", "best", "chain"
    agent_types: list[str] = field(default_factory=list)  # Specific agent types


@dataclass
class SubSwarmAgent:
    """An agent participating in a sub-swarm."""
    agent_id: str
    agent_type: str
    role: str  # "leader", "worker", "critic", "synthesizer"

    # Results
    result: Optional[dict] = None
    confidence: float = 0.0
    execution_time_ms: float = 0.0

    # State
    state: str = "pending"  # "pending", "running", "complete", "failed"
    error: Optional[str] = None


@dataclass
class SubSwarm:
    """A specialized sub-swarm for complex operations."""
    swarm_id: str
    swarm_type: SubSwarmType
    config: SubSwarmConfig

    # Task info
    task: str
    context: dict = field(default_factory=dict)
    trigger_source: str = ""  # What triggered this (API, agent, user)

    # Agents
    agents: list[SubSwarmAgent] = field(default_factory=list)

    # State
    state: SubSwarmState = SubSwarmState.SPAWNING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Results
    merged_result: Optional[dict] = None
    consensus_reached: bool = False
    consensus_score: float = 0.0

    # Metrics
    total_tokens_used: int = 0
    total_execution_time_ms: float = 0.0

    def is_active(self) -> bool:
        return self.state in [SubSwarmState.SPAWNING, SubSwarmState.ACTIVE, SubSwarmState.DELIBERATING]

    def agent_completion_rate(self) -> float:
        if not self.agents:
            return 0.0
        completed = sum(1 for a in self.agents if a.state == "complete")
        return completed / len(self.agents)


@dataclass
class SubSwarmResult:
    """Result of a sub-swarm operation."""
    swarm_id: str
    swarm_type: SubSwarmType
    success: bool

    # Results
    merged_result: dict
    individual_results: list[Tuple[str, float]]  # [(agent_id, result, confidence), ...]

    # Consensus
    consensus_reached: bool
    consensus_score: float

    # Timing
    total_time_ms: float
    agent_count: int

    # Metadata
    trigger_source: str
    context: dict = field(default_factory=dict)


# =============================================================================
# SUB-SWARM SPAWNER
# =============================================================================

class SubSwarmSpawner:
    """
    Spawns and manages specialized sub-swarms for complex operations.

    Features:
    - API-triggered spawning (DexScreener, Polymarket, etc.)
    - Specialized agent composition per swarm type
    - Parallel execution with result aggregation
    - Consensus-based decision making
    - Automatic lifecycle management
    """

    def __init__(
        self,
        max_concurrent_swarms: int = 5,
        max_agents_per_swarm: int = 10,
        default_timeout: float = 300.0,
    ):
        self.max_concurrent = max_concurrent_swarms
        self.max_agents = max_agents_per_swarm
        self.default_timeout = default_timeout

        # Active swarms
        self._swarms: dict[str, SubSwarm] = {}
        self._swarm_history: list[SubSwarmResult] = []

        # Agent factory (set externally)
        self._agent_factory: Optional[Callable] = None
        self._agent_executor: Optional[Callable] = None

        # Callbacks
        self._on_swarm_complete: list[Callable] = []
        self._on_agent_result: list[Callable] = []

        # Lock
        self._lock = asyncio.Lock()

        # Type-specific agent configurations
        self._type_configs = self._init_type_configs()

        logger.info("SubSwarmSpawner initialized")

    def _init_type_configs(self) -> dict[SubSwarmType[str]]:
        """Initialize default configurations for swarm types."""
        return {
            SubSwarmType.TRADING: {
                "agent_types": ["ResearchAgent", "CriticAgent", "PlannerAgent"],
                "roles": ["analyst", "risk_assessor", "executor"],
                "merge_strategy": "vote",
                "min_agents": 3,
                "require_consensus": True,
            },
            SubSwarmType.RESEARCH: {
                "agent_types": ["ResearchAgent", "ResearchAgent", "CriticAgent"],
                "roles": ["primary_researcher", "secondary_researcher", "synthesizer"],
                "merge_strategy": "aggregate",
                "min_agents": 2,
                "require_consensus": False,
            },
            SubSwarmType.CODING: {
                "agent_types": ["CodeAgent", "CriticAgent", "CodeAgent"],
                "roles": ["implementer", "reviewer", "tester"],
                "merge_strategy": "chain",
                "min_agents": 2,
                "require_consensus": True,
            },
            SubSwarmType.ANALYSIS: {
                "agent_types": ["ResearchAgent", "ReasoningAgent", "CriticAgent"],
                "roles": ["data_gatherer", "analyst", "validator"],
                "merge_strategy": "aggregate",
                "min_agents": 3,
                "require_consensus": False,
            },
            SubSwarmType.PREDICTION: {
                "agent_types": ["ResearchAgent", "ReasoningAgent", "CriticAgent", "ReasoningAgent"],
                "roles": ["researcher", "predictor", "devil_advocate", "synthesizer"],
                "merge_strategy": "vote",
                "min_agents": 3,
                "require_consensus": True,
            },
            SubSwarmType.CREATIVE: {
                "agent_types": ["CreativeAgent", "CreativeAgent", "CriticAgent"],
                "roles": ["ideator_1", "ideator_2", "curator"],
                "merge_strategy": "aggregate",
                "min_agents": 2,
                "require_consensus": False,
            },
        }

    def set_agent_factory(self, factory: Callable):
        """Set the factory function for creating agents."""
        self._agent_factory = factory

    def set_agent_executor(self, executor: Callable):
        """Set the executor function for running agent tasks."""
        self._agent_executor = executor

    async def spawn_subswarm(
        self,
        swarm_type: SubSwarmType,
        task: str,
        context: Optional[dict] = None,
        trigger_source: str = "api",
        config_override: Optional[SubSwarmConfig] = None,
    ) -> SubSwarm:
        """
        Spawn a new sub-swarm for a task.

        Args:
            swarm_type: Type of sub-swarm to create
            task: The task/query for the swarm
            context: Additional context (e.g., API data, market info)
            trigger_source: What triggered this swarm
            config_override: Optional custom configuration

        Returns:
            SubSwarm object (starts running immediately)
        """
        async with self._lock:
            active_count = sum(1 for s in self._swarms.values() if s.is_active())
            if active_count >= self.max_concurrent:
                raise RuntimeError(f"Maximum concurrent sub-swarms ({self.max_concurrent}) reached")

        # Get type-specific config
        type_config = self._type_configs.get(swarm_type, {})

        # Build configuration
        if config_override:
            config = config_override
        else:
            config = SubSwarmConfig(
                swarm_type=swarm_type,
                min_agents=type_config.get("min_agents", 2),
                max_agents=min(type_config.get("max_agents", 5), self.max_agents),
                timeout_seconds=self.default_timeout,
                require_consensus=type_config.get("require_consensus", True),
                merge_strategy=type_config.get("merge_strategy", "vote"),
                agent_types=type_config.get("agent_types", ["ResearchAgent"]),
            )

        # Create swarm
        swarm_id = f"subswarm_{swarm_type.value}_{uuid.uuid4().hex[:8]}"

        swarm = SubSwarm(
            swarm_id=swarm_id,
            swarm_type=swarm_type,
            config=config,
            task=task,
            context=context or {},
            trigger_source=trigger_source,
        )

        # Create agents
        roles = type_config.get("roles", ["worker"] * len(config.agent_types))

        for i, agent_type in enumerate(config.agent_types[:config.max_agents]):
            role = roles[i] if i < len(roles) else "worker"
            agent = SubSwarmAgent(
                agent_id=f"{swarm_id}_agent_{i}",
                agent_type=agent_type,
                role=role,
            )
            swarm.agents.append(agent)

        async with self._lock:
            self._swarms[swarm_id] = swarm

        logger.info(f"Spawned sub-swarm {swarm_id}: {swarm_type.value} with {len(swarm.agents)} agents")

        # Emit spawn signal
        await self._emit_swarm_signal("spawn", swarm)

        # Start execution
        asyncio.create_task(self._execute_swarm(swarm))

        return swarm

    async def _execute_swarm(self, swarm: SubSwarm):
        """Execute the sub-swarm task."""
        swarm.state = SubSwarmState.ACTIVE

        try:
            # Execute agents
            if swarm.config.parallel_execution:
                await self._execute_parallel(swarm)
            else:
                await self._execute_sequential(swarm)

            # Deliberate if needed
            if swarm.config.require_consensus:
                swarm.state = SubSwarmState.DELIBERATING
                await self._deliberate(swarm)

            # Merge results
            swarm.state = SubSwarmState.MERGING
            await self._merge_results(swarm)

            swarm.state = SubSwarmState.COMPLETE
            swarm.completed_at = datetime.now()

        except asyncio.TimeoutError:
            swarm.state = SubSwarmState.FAILED
            logger.error(f"Sub-swarm {swarm.swarm_id} timed out")

        except asyncio.CancelledError:
            swarm.state = SubSwarmState.CANCELLED
            logger.info(f"Sub-swarm {swarm.swarm_id} cancelled")

        except Exception as e:
            swarm.state = SubSwarmState.FAILED
            logger.error(f"Sub-swarm {swarm.swarm_id} failed: {e}")

        finally:
            # Create result record
            result = SubSwarmResult(
                swarm_id=swarm.swarm_id,
                swarm_type=swarm.swarm_type,
                success=swarm.state == SubSwarmState.COMPLETE,
                merged_result=swarm.merged_result,
                individual_results=[
                    (a.agent_id, a.result, a.confidence) for a in swarm.agents
                ],
                consensus_reached=swarm.consensus_reached,
                consensus_score=swarm.consensus_score,
                total_time_ms=swarm.total_execution_time_ms,
                agent_count=len(swarm.agents),
                trigger_source=swarm.trigger_source,
                context=swarm.context,
            )

            self._swarm_history.append(result)

            # Emit complete signal
            await self._emit_swarm_signal("complete", swarm)

            # Notify callbacks
            for callback in self._on_swarm_complete:
                try:
                    await callback(result)
                except Exception:
                    pass

    async def _execute_parallel(self, swarm: SubSwarm):
        """Execute all agents in parallel."""
        import time

        async def execute_agent(agent: SubSwarmAgent):
            start = time.time()
            agent.state = "running"

            try:
                if self._agent_executor:
                    result = await asyncio.wait_for(
                        self._agent_executor(
                            agent.agent_type,
                            swarm.task,
                            swarm.context,
                            agent.role,
                        ),
                        timeout=swarm.config.timeout_seconds
                    )

                    agent.result = result.get("output")
                    agent.confidence = result.get("confidence", 0.5)
                    agent.state = "complete"
                else:
                    # Simulated execution
                    await asyncio.sleep(0.5)
                    agent.result = f"[{agent.agent_type}] Analysis of: {swarm.task[:50]}..."
                    agent.confidence = 0.7
                    agent.state = "complete"

            except asyncio.TimeoutError:
                agent.state = "failed"
                agent.error = "Timeout"

            except Exception as e:
                agent.state = "failed"
                agent.error = str(e)

            finally:
                agent.execution_time_ms = (time.time() - start) * 1000
                swarm.total_execution_time_ms += agent.execution_time_ms

        await asyncio.gather(*[execute_agent(a) for a in swarm.agents])

    async def _execute_sequential(self, swarm: SubSwarm):
        """Execute agents sequentially (for chain strategy)."""
        import time

        previous_result = None

        for agent in swarm.agents:
            start = time.time()
            agent.state = "running"

            try:
                # Build context with previous result
                context = {**swarm.context}
                if previous_result:
                    context["previous_result"] = previous_result

                if self._agent_executor:
                    result = await asyncio.wait_for(
                        self._agent_executor(
                            agent.agent_type,
                            swarm.task,
                            context,
                            agent.role,
                        ),
                        timeout=swarm.config.timeout_seconds / len(swarm.agents)
                    )

                    agent.result = result.get("output")
                    agent.confidence = result.get("confidence", 0.5)
                    previous_result = agent.result
                    agent.state = "complete"
                else:
                    await asyncio.sleep(0.3)
                    agent.result = f"[{agent.agent_type}] Step result for: {swarm.task[:30]}..."
                    agent.confidence = 0.7
                    previous_result = agent.result
                    agent.state = "complete"

            except Exception as e:
                agent.state = "failed"
                agent.error = str(e)

            finally:
                agent.execution_time_ms = (time.time() - start) * 1000
                swarm.total_execution_time_ms += agent.execution_time_ms

    async def _deliberate(self, swarm: SubSwarm):
        """Have agents deliberate on results."""
        # Collect successful results
        results = [(a.agent_id, a.result, a.confidence) for a in swarm.agents if a.state == "complete"]

        if len(results) < 2:
            swarm.consensus_reached = True
            swarm.consensus_score = 1.0
            return

        # Simple voting: each agent "votes" for their answer
        # Higher confidence = higher weight
        total_confidence = sum(r[2] for r in results)

        if total_confidence > 0:
            # Normalize weights
            weights = {r[0]: r[2] / total_confidence for r in results}

            # For now, mark consensus if dict agent has majority confidence
            max_weight = max(weights.values())
            swarm.consensus_reached = max_weight >= swarm.config.consensus_threshold
            swarm.consensus_score = max_weight

    async def _merge_results(self, swarm: SubSwarm):
        """Merge agent results based on strategy."""
        successful_agents = [a for a in swarm.agents if a.state == "complete"]

        if not successful_agents:
            swarm.merged_result = None
            return

        strategy = swarm.config.merge_strategy

        if strategy == "best":
            # Take highest confidence result
            best = max(successful_agents, key=lambda a: a.confidence)
            swarm.merged_result = best.result

        elif strategy == "vote":
            # Weighted vote (already done in deliberate)
            # Take result with highest confidence
            best = max(successful_agents, key=lambda a: a.confidence)
            swarm.merged_result = {
                "primary_result": best.result,
                "primary_confidence": best.confidence,
                "all_results": [
                    {"agent": a.agent_id, "result": a.result, "confidence": a.confidence}
                    for a in successful_agents
                ],
            }

        elif strategy == "aggregate":
            # Combine all results
            swarm.merged_result = {
                "aggregated_results": [
                    {
                        "agent": a.agent_id,
                        "type": a.agent_type,
                        "role": a.role,
                        "result": a.result,
                        "confidence": a.confidence,
                    }
                    for a in successful_agents
                ],
                "total_agents": len(successful_agents),
                "average_confidence": sum(a.confidence for a in successful_agents) / len(successful_agents),
            }

        elif strategy == "chain":
            # Take final result in chain
            swarm.merged_result = successful_agents[-1].result if successful_agents else None

        else:
            swarm.merged_result = successful_agents[0].result if successful_agents else None

    async def _emit_swarm_signal(self, event_type: str, swarm: SubSwarm):
        """Emit a Nexus signal for swarm events."""
        try:
            from Cosmos.core.nexus import nexus, SignalType

            signal_map = {
                "spawn": SignalType.SUBSWARM_SPAWN,
                "complete": SignalType.SUBSWARM_COMPLETE,
                "merge": SignalType.SUBSWARM_MERGE,
            }

            signal_type = signal_map.get(event_type, SignalType.EXTERNAL_EVENT)

            await nexus.emit(
                signal_type,
                {
                    "swarm_id": swarm.swarm_id,
                    "swarm_type": swarm.swarm_type.value,
                    "state": swarm.state.value,
                    "agent_count": len(swarm.agents),
                    "trigger_source": swarm.trigger_source,
                    "task": swarm.task[:100],
                },
                source="subswarm_spawner",
                urgency=0.6,
            )

        except Exception as e:
            logger.debug(f"Failed to emit swarm signal: {e}")

    async def cancel_swarm(self, swarm_id: str) -> bool:
        """Cancel an active sub-swarm."""
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            return False

        if not swarm.is_active():
            return False

        swarm.state = SubSwarmState.CANCELLED
        logger.info(f"Cancelled sub-swarm: {swarm_id}")
        return True

    def get_swarm(self, swarm_id: str) -> Optional[SubSwarm]:
        """Get a sub-swarm by ID."""
        return self._swarms.get(swarm_id)

    def get_active_swarms(self) -> list[SubSwarm]:
        """Get all active sub-swarms."""
        return [s for s in self._swarms.values() if s.is_active()]

    def get_stats(self) -> dict:
        """Get spawner statistics."""
        return {
            "total_swarms": len(self._swarms),
            "active_swarms": len(self.get_active_swarms()),
            "history_count": len(self._swarm_history),
            "swarms_by_type": {
                t.value: sum(1 for s in self._swarms.values() if s.swarm_type == t)
                for t in SubSwarmType
            },
            "recent_success_rate": (
                sum(1 for r in self._swarm_history[-20:] if r.success) / max(1, len(self._swarm_history[-20:]))
            ),
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

subswarm_spawner = SubSwarmSpawner()


# =============================================================================
# API TRIGGER FUNCTIONS
# =============================================================================

async def spawn_trading_swarm(
    task: str,
    market_data: Optional[dict] = None,
    trigger: str = "api",
) -> SubSwarm:
    """Spawn a trading analysis sub-swarm."""
    return await subswarm_spawner.spawn_subswarm(
        SubSwarmType.TRADING,
        task,
        context={"market_data": market_data} if market_data else {},
        trigger_source=trigger,
    )


async def spawn_research_swarm(
    query: str,
    sources: Optional[list[str]] = None,
    trigger: str = "api",
) -> SubSwarm:
    """Spawn a research sub-swarm."""
    return await subswarm_spawner.spawn_subswarm(
        SubSwarmType.RESEARCH,
        query,
        context={"sources": sources} if sources else {},
        trigger_source=trigger,
    )


async def spawn_prediction_swarm(
    question: str,
    odds_data: Optional[dict] = None,
    trigger: str = "polymarket",
) -> SubSwarm:
    """Spawn a prediction analysis sub-swarm (for Polymarket etc)."""
    return await subswarm_spawner.spawn_subswarm(
        SubSwarmType.PREDICTION,
        question,
        context={"odds": odds_data} if odds_data else {},
        trigger_source=trigger,
    )


async def spawn_coding_swarm(
    task: str,
    code_context: Optional[str] = None,
    trigger: str = "api",
) -> SubSwarm:
    """Spawn a coding sub-swarm."""
    return await subswarm_spawner.spawn_subswarm(
        SubSwarmType.CODING,
        task,
        context={"code": code_context} if code_context else {},
        trigger_source=trigger,
    )

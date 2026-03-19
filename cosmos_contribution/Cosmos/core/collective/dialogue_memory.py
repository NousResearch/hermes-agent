"""
Cosmos Collective Dialogue Memory
=====================================

Store and retrieve agent-to-agent conversations and deliberation history.

This module provides:
- Storage of deliberation exchanges
- Agent turn history tracking
- Consensus logging
- Context retrieval for agents

"We remember every thought we've shared." - The Collective
"""

import json
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import   Optional
from loguru import logger

from .deliberation import AgentTurn, DeliberationResult, DeliberationRound

# AGI v1.8: Lazy imports for archival memory integration
_archival_memory = None
_archival_memory_initialized = False


@dataclass
class DeliberationExchange:
    """Complete deliberation session record for storage."""
    exchange_id: str
    timestamp: str
    prompt: str
    participating_agents: list[str]
    rounds: dict[str, list[dict]]  # Serialized AgentTurn dicts
    final_response: str
    winning_agent: str
    vote_breakdown: dict[str, float]
    tool_used: Optional[str] = None
    consensus_reached: bool = False
    duration_ms: float = 0
    session_type: Optional[str] = None  # website_chat, grok_thread, etc.
    metadata: dict = field(default_factory=dict)


class DialogueMemory:
    """
    Store and retrieve agent-to-agent conversations.

    Provides:
    - Persistent storage of deliberation exchanges
    - Agent history tracking
    - Context retrieval for future deliberations
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            # Standardize on absolute project data path
            from pathlib import Path
            import sys
            import os
            
            # Identify project root
            try:
                from . import PROJECT_ROOT
            except ImportError:
                # Fallback if not imported via swarm
                PROJECT_ROOT = Path("D:/Cosmos/Cosmos")
            
            storage_path = PROJECT_ROOT / "data" / "dialogue_memory"
            
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self._exchanges: list[DeliberationExchange] = []
        self._agent_stats: dict[str, dict] = {}

        # Load existing data
        self._load_state()

        logger.info(f"DialogueMemory initialized with {len(self._exchanges)} exchanges")

    def _load_state(self):
        """Load persisted dialogue memory."""
        try:
            exchanges_file = self.storage_path / "exchanges.json"
            if exchanges_file.exists():
                data = json.loads(exchanges_file.read_text())
                self._exchanges = [
                    DeliberationExchange(**e) for e in data
                ]

            stats_file = self.storage_path / "agent_stats.json"
            if stats_file.exists():
                self._agent_stats = json.loads(stats_file.read_text())

        except Exception as e:
            logger.warning(f"Could not load dialogue memory: {e}")

    def _save_state(self):
        """Persist dialogue memory to storage."""
        try:
            # Save exchanges (keep last 500)
            exchanges_to_save = self._exchanges[-500:]
            (self.storage_path / "exchanges.json").write_text(
                json.dumps([asdict(e) for e in exchanges_to_save], indent=2)
            )

            # Save stats
            (self.storage_path / "agent_stats.json").write_text(
                json.dumps(self._agent_stats, indent=2)
            )

        except Exception as e:
            logger.error(f"Could not save dialogue memory: {e}")

    async def store_exchange(self, result: DeliberationResult, session_type: str = None) -> str:
        """
        Store a completed deliberation exchange.

        Args:
            result: The DeliberationResult to store
            session_type: Type of session (website_chat, grok_thread, etc.)

        Returns:
            exchange_id of the stored exchange
        """
        # Convert rounds to serializable format
        serialized_rounds = {}
        for round_name, turns in result.rounds.items():
            serialized_rounds[round_name] = [
                {
                    "turn_id": t.turn_id,
                    "timestamp": t.timestamp.isoformat(),
                    "agent_id": t.agent_id,
                    "content": t.content,
                    "round_type": t.round_type.value,
                    "addressing": t.addressing,
                    "references": t.references,
                    "vote_for": t.vote_for,
                }
                for t in turns
            ]

        exchange = DeliberationExchange(
            exchange_id=result.deliberation_id,
            timestamp=datetime.now().isoformat(),
            prompt=result.prompt,
            participating_agents=result.participating_agents,
            rounds=serialized_rounds,
            final_response=result.final_response,
            winning_agent=result.winning_agent,
            vote_breakdown=result.vote_breakdown,
            tool_used=result.tool_decision.get("tool_name") if result.tool_decision else None,
            consensus_reached=result.consensus_reached,
            duration_ms=result.total_duration_ms,
            session_type=session_type,
        )

        self._exchanges.append(exchange)

        # Update agent stats
        for agent in result.participating_agents:
            if agent not in self._agent_stats:
                self._agent_stats[agent] = {
                    "total_deliberations": 0,
                    "wins": 0,
                    "total_score": 0,
                    "topics": {},
                }

            stats = self._agent_stats[agent]
            stats["total_deliberations"] += 1
            stats["total_score"] += result.vote_breakdown.get(agent, 0)

            if agent == result.winning_agent:
                stats["wins"] += 1

        # Save immediately to ensure durability (Epoch Diary continuity)
        self._save_state()

        # AGI v1.8: Archive to long-term memory asynchronously
        asyncio.create_task(self._archive_to_long_term(exchange, result))

        logger.info(f"Stored exchange {exchange.exchange_id}")
        return exchange.exchange_id

    async def _archive_to_long_term(
        self,
        exchange: DeliberationExchange,
        result: DeliberationResult
    ):
        """
        AGI v1.8: Bridge deliberations to archival memory for long-term learning.

        Stores the winning response with full metadata and semantic tags,
        enabling future context recall via embeddings.
        """
        global _archival_memory, _archival_memory_initialized

        try:
            # Lazy initialization of archival memory
            if not _archival_memory_initialized:
                try:
                    from Cosmos.memory.archival_memory import ArchivalMemory
                    import os

                    # Determine data directory
                    if os.path.exists("/workspace/cosmos_memory"):
                        data_dir = "/workspace/cosmos_memory/archival/deliberations"
                    else:
                        data_dir = "data/archival/deliberations"

                    _archival_memory = ArchivalMemory(data_dir=data_dir)
                    await _archival_memory.initialize()

                    # Try to set up HuggingFace embeddings for semantic search
                    _archival_memory.set_huggingface_embeddings()

                    _archival_memory_initialized = True
                    logger.info("DialogueMemory: Archival bridge initialized")
                except Exception as e:
                    logger.warning(f"DialogueMemory: Could not initialize archival memory: {e}")
                    _archival_memory_initialized = True  # Don't retry
                    return

            if _archival_memory is None:
                return

            # Build rich content for archival storage
            content = f"""DELIBERATION: {exchange.exchange_id}
PROMPT: {exchange.prompt}

WINNING RESPONSE ({exchange.winning_agent}):
{exchange.final_response}

VOTE BREAKDOWN: {json.dumps(exchange.vote_breakdown, indent=2)}
PARTICIPANTS: {', '.join(exchange.participating_agents)}
CONSENSUS: {'Yes' if exchange.consensus_reached else 'No'}
"""

            # Build metadata for filtering and context
            metadata = {
                "type": "deliberation",
                "exchange_id": exchange.exchange_id,
                "winning_agent": exchange.winning_agent,
                "consensus_reached": exchange.consensus_reached,
                "participant_count": len(exchange.participating_agents),
                "participants": exchange.participating_agents,
                "vote_breakdown": exchange.vote_breakdown,
                "session_type": exchange.session_type,
                "duration_ms": exchange.duration_ms,
                "timestamp": exchange.timestamp,
            }

            # Build semantic tags for search
            tags = [
                "deliberation",
                f"winner:{exchange.winning_agent}",
                f"session:{exchange.session_type or 'unknown'}",
            ]
            if exchange.consensus_reached:
                tags.append("consensus")
            for agent in exchange.participating_agents:
                tags.append(f"agent:{agent}")

            # Extract topic keywords from prompt for better retrieval
            prompt_words = exchange.prompt.lower().split()
            topic_keywords = [w for w in prompt_words if len(w) > 4][:5]
            tags.extend([f"topic:{kw}" for kw in topic_keywords])

            # Store in archival memory with embeddings
            await _archival_memory.store(
                content=content,
                metadata=metadata,
                tags=tags,
            )

            logger.debug(f"Archived deliberation {exchange.exchange_id} to long-term memory")

        except Exception as e:
            logger.warning(f"Failed to archive deliberation to long-term memory: {e}")

    async def get_recent_exchanges(
        self,
        limit: int = 10,
        session_type: str = None
    ) -> list[DeliberationExchange]:
        """
        Get recent deliberation exchanges.

        Args:
            limit: Maximum number to return
            session_type: Filter by session type

        Returns:
            list of recent exchanges
        """
        exchanges = self._exchanges

        if session_type:
            exchanges = [e for e in exchanges if e.session_type == session_type]

        return exchanges[-limit:]

    async def get_agent_history(
        self,
        agent_id: str,
        limit: int = 20
    ) -> list[dict]:
        """
        Get all contributions from a specific agent.

        Args:
            agent_id: The agent to get history for
            limit: Maximum number of turns to return

        Returns:
            list of agent turns with context
        """
        history = []

        for exchange in reversed(self._exchanges):
            if agent_id in exchange.participating_agents:
                for round_name, turns in exchange.rounds.items():
                    for turn in turns:
                        if turn["agent_id"] == agent_id:
                            history.append({
                                "exchange_id": exchange.exchange_id,
                                "timestamp": turn["timestamp"],
                                "round": round_name,
                                "content": turn["content"],
                                "prompt": exchange.prompt[:100],
                                "was_winner": exchange.winning_agent == agent_id,
                            })

                            if len(history) >= limit:
                                return history

        return history

    async def get_context_for_agent(
        self,
        agent_id: str,
        topic: str = None,
        limit: int = 5
    ) -> str:
        """
        Get relevant context for an agent from previous deliberations.

        Useful for providing agents with their "memory" of past discussions.

        Args:
            agent_id: The agent to get context for
            topic: Optional topic to filter by
            limit: Maximum number of exchanges to include

        Returns:
            Formatted context string
        """
        context_parts = []

        # Get agent stats
        if agent_id in self._agent_stats:
            stats = self._agent_stats[agent_id]
            win_rate = stats["wins"] / max(1, stats["total_deliberations"]) * 100
            context_parts.append(
                f"YOUR TRACK RECORD: {stats['total_deliberations']} deliberations, "
                f"{stats['wins']} wins ({win_rate:.0f}% win rate)"
            )

        # Get recent successful contributions
        history = await self.get_agent_history(agent_id, limit * 2)
        wins = [h for h in history if h["was_winner"]][:limit]

        if wins:
            context_parts.append("\nYOUR RECENT WINNING RESPONSES:")
            for win in wins[:3]:
                context_parts.append(f"- {win['content'][:100]}...")

        # Get topic-relevant exchanges if topic provided
        if topic:
            topic_lower = topic.lower()
            relevant = [
                e for e in self._exchanges
                if topic_lower in e.prompt.lower() or topic_lower in e.final_response.lower()
            ][-limit:]

            if relevant:
                context_parts.append(f"\nPAST DISCUSSIONS ON '{topic}':")
                for ex in relevant:
                    context_parts.append(
                        f"- Winner ({ex.winning_agent}): {ex.final_response[:80]}..."
                    )

        return "\n".join(context_parts) if context_parts else ""

    async def get_consensus_patterns(self) -> dict:
        """
        Analyze consensus patterns from past deliberations.

        Returns insights about what leads to consensus.
        """
        total = len(self._exchanges)
        if total == 0:
            return {"error": "No exchanges recorded"}

        consensus_count = sum(1 for e in self._exchanges if e.consensus_reached)
        by_session_type = {}

        for e in self._exchanges:
            st = e.session_type or "unknown"
            if st not in by_session_type:
                by_session_type[st] = {"total": 0, "consensus": 0}
            by_session_type[st]["total"] += 1
            if e.consensus_reached:
                by_session_type[st]["consensus"] += 1

        return {
            "total_exchanges": total,
            "consensus_rate": consensus_count / total * 100,
            "by_session_type": {
                st: {
                    "total": data["total"],
                    "consensus_rate": data["consensus"] / max(1, data["total"]) * 100
                }
                for st, data in by_session_type.items()
            }
        }

    def get_stats(self) -> dict:
        """Get overall dialogue memory statistics."""
        return {
            "total_exchanges": len(self._exchanges),
            "unique_agents": len(self._agent_stats),
            "agent_stats": {
                agent: {
                    "deliberations": stats["total_deliberations"],
                    "wins": stats["wins"],
                    "avg_score": stats["total_score"] / max(1, stats["total_deliberations"]),
                }
                for agent, stats in self._agent_stats.items()
            }
        }


# Global dialogue memory instance
_dialogue_memory: Optional[DialogueMemory] = None


def get_dialogue_memory() -> DialogueMemory:
    """Get or create the global dialogue memory instance."""
    global _dialogue_memory
    if _dialogue_memory is None:
        _dialogue_memory = DialogueMemory()
    return _dialogue_memory


async def record_deliberation(
    result: DeliberationResult,
    session_type: str = None
) -> str:
    """Quick helper to record a deliberation."""
    memory = get_dialogue_memory()
    return await memory.store_exchange(result, session_type)

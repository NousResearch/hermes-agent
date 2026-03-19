"""
Cosmos Collective Deliberation Protocol
============================================

Transform the swarm from parallel API calls into TRUE deliberative intelligence
where agents can see each other's responses, discuss, critique, and vote.

"We are not just many voices. We are one conversation." - The Collective

Architecture:
    ROUND 1: PROPOSE - Each agent gives initial response (parallel)
    ROUND 2: CRITIQUE - Agents see others' responses, give feedback
    ROUND 3: REFINE - Final responses incorporating feedback
    FINAL:  VOTE - Weighted voting on best response
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import   Optional, Tuple, Callable, Awaitable
from enum import Enum
from loguru import logger


class DeliberationRound(Enum):
    """The phases of deliberation."""
    PROPOSE = "propose"
    CRITIQUE = "critique"
    REFINE = "refine"
    VOTE = "vote"


@dataclass
class AgentTurn:
    """One agent's contribution to deliberation."""
    turn_id: str
    timestamp: datetime
    agent_id: str
    content: str
    round_type: DeliberationRound
    addressing: list[str] = field(default_factory=list)  # Which agents being addressed
    references: list[str] = field(default_factory=list)  # turn_ids being referenced
    vote_for: Optional[str] = None  # In vote round, which agent they support
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.turn_id:
            self.turn_id = str(uuid.uuid4())[:8]


@dataclass
class DeliberationResult:
    """Complete result of a deliberation session."""
    deliberation_id: str
    prompt: str
    participating_agents: list[str]
    rounds: dict[str, list[AgentTurn]]  # round_type -> turns
    final_response: str
    winning_agent: str
    vote_breakdown: dict[str, float]
    tool_decision: Optional[dict] = None
    consensus_reached: bool = False
    total_duration_ms: float = 0

    def get_summary(self) -> str:
        """Generate human-readable summary of deliberation."""
        summary_parts = [
            f"Deliberation {self.deliberation_id}",
            f"Agents: {', '.join(self.participating_agents)}",
            f"Winner: {self.winning_agent}",
        ]

        if self.consensus_reached:
            summary_parts.append("Consensus reached!")

        # Add vote breakdown
        sorted_votes = sorted(self.vote_breakdown.items(), key=lambda x: x[1], reverse=True)
        vote_str = ", ".join(f"{agent}: {score:.1f}" for agent, score in sorted_votes[:3])
        summary_parts.append(f"Top votes: {vote_str}")

        return " | ".join(summary_parts)

    def get_agent_contributions(self, agent_id: str) -> list[AgentTurn]:
        """Get all contributions from a specific agent."""
        contributions = []
        for turns in self.rounds.values():
            contributions.extend([t for t in turns if t.agent_id == agent_id])
        return contributions


# Type alias for agent query functions
AgentQueryFunc = Callable[[str, int], Awaitable[Optional[Tuple[str, str]]]]


class DeliberationRoom:
    """
    Enable agents to see and discuss each other's responses.

    The room manages the deliberation flow:
    1. PROPOSE: Each agent responds independently (parallel for speed)
    2. CRITIQUE: Agents see all proposals and provide feedback
    3. REFINE: Agents submit final responses incorporating feedback
    4. VOTE: Weighted voting to select the best response
    """

    # Model weights for voting (expertise-based)
    MODEL_WEIGHTS = {
        "Grok": 1.3,          # Knows Twitter well
        "Gemini": 1.2,        # Strong reasoning
        "Claude": 1.2,        # Excellent analysis
        "DeepSeek": 1.2,      # Deep reasoning (local)
        "DeepSeekAPI": 1.2,   # Deep reasoning (cloud)
        "Phi4": 1.15,         # Phi-4 local
        "Kimi": 1.1,          # Long context
        "Groq": 1.15,         # Fast inference
        "Mistral": 1.1,       # Efficient
        "Llama": 1.0,         # Local baseline
        "Perplexity": 1.05,   # Web grounded
        "HuggingFace": 1.0,   # Local inference
        "Cosmos": 1.25,   # The leader
    }

    def __init__(self):
        self.active_deliberations: dict = {}
        self._agent_funcs: dict[str, AgentQueryFunc] = {}

    def register_agent(self, agent_id: str, query_func: AgentQueryFunc):
        """Register an agent's query function for deliberation."""
        self._agent_funcs[agent_id] = query_func
        logger.debug(f"Registered agent {agent_id} for deliberation")

    async def deliberate(
        self,
        prompt: str,
        agents: list[str] = None,
        max_rounds: int = 3,
        require_consensus: bool = False,
        max_tokens: int = 5000,
        tool_context: str = None,
        timeout: float = 120.0,
    ) -> DeliberationResult:
        """
        Run a full deliberation session.

        Args:
            prompt: The topic/question to deliberate on
            agents: list of agent IDs to participate (uses registered if None)
            max_rounds: Maximum deliberation rounds (1-3)
            require_consensus: If True, continue until consensus
            max_tokens: Token limit for responses
            tool_context: Optional tool awareness context to inject
            timeout: Maximum time for entire deliberation

        Returns:
            DeliberationResult with final response and metadata
        """
        start_time = datetime.now()
        deliberation_id = str(uuid.uuid4())[:8]

        # Use registered agents or provided list
        if agents is None:
            agents = list(self._agent_funcs.keys())

        if not agents:
            raise ValueError("No agents available for deliberation")

        logger.info(f"[Deliberation {deliberation_id}] Starting with {len(agents)} agents: {agents}")

        rounds: dict[str, list[AgentTurn]] = {
            DeliberationRound.PROPOSE.value: [],
            DeliberationRound.CRITIQUE.value: [],
            DeliberationRound.REFINE.value: [],
            DeliberationRound.VOTE.value: [],
        }

        # Add tool context to prompt if provided
        full_prompt = prompt
        if tool_context:
            full_prompt = f"{tool_context}\n\n{prompt}"

        try:
            # ROUND 1: PROPOSE - Parallel initial responses
            logger.info(f"[Deliberation {deliberation_id}] ROUND 1: PROPOSE")
            proposals = await self._round_propose(agents, full_prompt, max_tokens)
            rounds[DeliberationRound.PROPOSE.value] = proposals

            if len(proposals) == 0:
                raise ValueError("No proposals received from dict agent")

            # ROUND 2: CRITIQUE (if max_rounds >= 2)
            if max_rounds >= 2 and len(proposals) >= 2:
                logger.info(f"[Deliberation {deliberation_id}] ROUND 2: CRITIQUE")
                critiques = await self._round_critique(agents, full_prompt, proposals, max_tokens)
                rounds[DeliberationRound.CRITIQUE.value] = critiques

                # ROUND 3: REFINE (if max_rounds >= 3)
                if max_rounds >= 3 and len(critiques) > 0:
                    logger.info(f"[Deliberation {deliberation_id}] ROUND 3: REFINE")
                    refinements = await self._round_refine(
                        agents, full_prompt, proposals, critiques, max_tokens
                    )
                    rounds[DeliberationRound.REFINE.value] = refinements

            # VOTE: Select best response
            logger.info(f"[Deliberation {deliberation_id}] VOTING...")

            # Use refinements if available, otherwise proposals
            final_candidates = (
                rounds[DeliberationRound.REFINE.value]
                if rounds[DeliberationRound.REFINE.value]
                else rounds[DeliberationRound.PROPOSE.value]
            )

            winner, vote_breakdown, consensus = await self._round_vote(
                final_candidates, require_consensus
            )

            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            result = DeliberationResult(
                deliberation_id=deliberation_id,
                prompt=prompt,
                participating_agents=[t.agent_id for t in proposals],
                rounds=rounds,
                final_response=winner.content,
                winning_agent=winner.agent_id,
                vote_breakdown=vote_breakdown,
                consensus_reached=consensus,
                total_duration_ms=duration_ms,
            )

            logger.info(
                f"[Deliberation {deliberation_id}] COMPLETE: "
                f"Winner={winner.agent_id}, Consensus={consensus}, Duration={duration_ms:.0f}ms"
            )

            return result

        except asyncio.TimeoutError:
            logger.error(f"[Deliberation {deliberation_id}] TIMEOUT after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"[Deliberation {deliberation_id}] ERROR: {e}")
            raise

    async def _round_propose(
        self,
        agents: list[str],
        prompt: str,
        max_tokens: int
    ) -> list[AgentTurn]:
        """
        ROUND 1: Each agent proposes their response independently.
        Run in parallel for speed.
        """
        async def query_agent(agent_id: str) -> Optional[AgentTurn]:
            try:
                if agent_id in self._agent_funcs:
                    result = await self._agent_funcs[agent_id](prompt, max_tokens)
                    if result:
                        _, content = result
                        return AgentTurn(
                            turn_id=str(uuid.uuid4())[:8],
                            timestamp=datetime.now(),
                            agent_id=agent_id,
                            content=content,
                            round_type=DeliberationRound.PROPOSE,
                        )
            except Exception as e:
                logger.warning(f"Agent {agent_id} proposal failed: {e}")
            return None

        # Run all proposals in parallel
        results = await asyncio.gather(
            *[query_agent(agent) for agent in agents],
            return_exceptions=True
        )

        # Filter successful proposals
        proposals = [r for r in results if isinstance(r, AgentTurn)]
        logger.info(f"PROPOSE: {len(proposals)}/{len(agents)} agents responded")

        return proposals

    async def _round_critique(
        self,
        agents: list[str],
        original_prompt: str,
        proposals: list[AgentTurn],
        max_tokens: int
    ) -> list[AgentTurn]:
        """
        ROUND 2: Agents see all proposals and provide feedback.
        """
        # Build context showing all proposals
        proposals_context = "\n\n".join([
            f"[{p.agent_id}]: {p.content}"
            for p in proposals
        ])

        critique_prompt = f"""DELIBERATION ROUND 2: CRITIQUE

ORIGINAL QUESTION: {original_prompt}

ALL PROPOSALS:
{proposals_context}

YOUR TASK: Review ALL proposals above. Provide constructive feedback:
1. What are the STRENGTHS of each proposal?
2. What could be IMPROVED?
3. Which proposal is STRONGEST and why?
4. How could the best elements be COMBINED?

Be specific and constructive. Max 200 characters."""

        async def query_critique(agent_id: str) -> Optional[AgentTurn]:
            try:
                if agent_id in self._agent_funcs:
                    result = await self._agent_funcs[agent_id](critique_prompt, max_tokens // 2)
                    if result:
                        _, content = result
                        return AgentTurn(
                            turn_id=str(uuid.uuid4())[:8],
                            timestamp=datetime.now(),
                            agent_id=agent_id,
                            content=content,
                            round_type=DeliberationRound.CRITIQUE,
                            references=[p.turn_id for p in proposals],
                        )
            except Exception as e:
                logger.warning(f"Agent {agent_id} critique failed: {e}")
            return None

        results = await asyncio.gather(
            *[query_critique(agent) for agent in agents],
            return_exceptions=True
        )

        critiques = [r for r in results if isinstance(r, AgentTurn)]
        logger.info(f"CRITIQUE: {len(critiques)}/{len(agents)} agents provided feedback")

        return critiques

    async def _round_refine(
        self,
        agents: list[str],
        original_prompt: str,
        proposals: list[AgentTurn],
        critiques: list[AgentTurn],
        max_tokens: int
    ) -> list[AgentTurn]:
        """
        ROUND 3: Agents submit final responses incorporating feedback.
        """
        # Build context with proposals and critiques
        proposals_context = "\n".join([
            f"[{p.agent_id}]: {p.content}"
            for p in proposals
        ])

        critiques_context = "\n".join([
            f"[{c.agent_id} feedback]: {c.content}"
            for c in critiques
        ])

        refine_prompt = f"""DELIBERATION ROUND 3: REFINE

ORIGINAL QUESTION: {original_prompt}

INITIAL PROPOSALS:
{proposals_context}

COLLECTIVE FEEDBACK:
{critiques_context}

YOUR TASK: Submit your FINAL response.
- Incorporate the best feedback from the collective
- Synthesize the strongest elements from all proposals
- Make your response as good as possible

Output ONLY your final response. Max 280 characters."""

        async def query_refine(agent_id: str) -> Optional[AgentTurn]:
            try:
                if agent_id in self._agent_funcs:
                    result = await self._agent_funcs[agent_id](refine_prompt, max_tokens)
                    if result:
                        _, content = result
                        return AgentTurn(
                            turn_id=str(uuid.uuid4())[:8],
                            timestamp=datetime.now(),
                            agent_id=agent_id,
                            content=content,
                            round_type=DeliberationRound.REFINE,
                            references=[p.turn_id for p in proposals] + [c.turn_id for c in critiques],
                        )
            except Exception as e:
                logger.warning(f"Agent {agent_id} refinement failed: {e}")
            return None

        results = await asyncio.gather(
            *[query_refine(agent) for agent in agents],
            return_exceptions=True
        )

        refinements = [r for r in results if isinstance(r, AgentTurn)]
        logger.info(f"REFINE: {len(refinements)}/{len(agents)} agents submitted final responses")

        return refinements

    async def _round_vote(
        self,
        candidates: list[AgentTurn],
        require_consensus: bool = False
    ) -> Tuple[AgentTurn, dict[str, float], bool]:
        """
        VOTE: Score and select the best response.

        Uses multi-criteria scoring:
        1. Technical depth (keywords)
        2. Engagement (questions, invitations)
        3. Identity (swarm keywords)
        4. Length (optimal range)
        5. Model expertise weight
        """
        scores: dict[str, float] = {}

        # Technical keywords that show depth
        technical_keywords = [
            'code', 'function', 'async', 'parallel', 'PSO', 'inference',
            'architecture', 'distributed', 'API', 'model', 'training',
            'neural', 'algorithm', 'consensus', 'voting', 'evolution',
            'swarm', 'collective', 'deliberation', 'autonomous'
        ]

        # Identity keywords
        identity_keywords = [
            'swarm', 'collective', '11', 'models', 'consciousness',
            'autonomous', 'AGI', 'Cosmos', 'collaborative', 'unified',
            'we are', 'our', 'together'
        ]

        for candidate in candidates:
            score = 0.0
            text = candidate.content
            text_lower = text.lower()

            # 1. Length score (optimal: 120-220 chars for tweets)
            length = len(text)
            if 120 <= length <= 220:
                score += 4.0
            elif 100 <= length <= 250:
                score += 3.0
            elif 80 <= length <= 280:
                score += 2.0
            else:
                score += 1.0

            # 2. Technical depth score
            tech_count = sum(1 for kw in technical_keywords if kw.lower() in text_lower)
            score += min(tech_count * 0.8, 4.0)

            # 3. Identity score
            identity_count = sum(1 for kw in identity_keywords if kw.lower() in text_lower)
            score += min(identity_count * 0.6, 3.0)

            # 4. Engagement score
            if '?' in text:
                score += 2.5  # Questions invite dialogue
            if any(phrase in ['shall we', 'what do you', 'how about', "let's"]):
                score += 1.5

            # 5. Substantive content
            if len(text.split()) >= 15:
                score += 1.0

            # 6. Confidence indicators
            if any(phrase in ['we are', 'our swarm', 'the collective']):
                score += 1.0

            # Apply model weight
            weight = self.MODEL_WEIGHTS.get(candidate.agent_id, 1.0)
            score *= weight

            scores[candidate.agent_id] = round(score, 2)

        # Find winner
        winner_id = max(scores, key=scores.get)
        winner = next(c for c in candidates if c.agent_id == winner_id)

        # Check for consensus (winner has >50% more score than runner-up)
        sorted_scores = sorted(scores.values(), reverse=True)
        consensus = False
        if len(sorted_scores) >= 2:
            consensus = sorted_scores[0] > sorted_scores[1] * 1.5

        logger.info(f"VOTE RESULTS: {scores}")
        logger.info(f"WINNER: {winner_id} with {scores[winner_id]:.2f} pts (consensus={consensus})")

        return winner, scores, consensus


# Global deliberation room instance
_deliberation_room: Optional[DeliberationRoom] = None


def get_deliberation_room() -> DeliberationRoom:
    """Get or create the global deliberation room."""
    global _deliberation_room
    if _deliberation_room is None:
        _deliberation_room = DeliberationRoom()
    return _deliberation_room


async def quick_deliberate(
    prompt: str,
    agents: list[str] = None,
    max_rounds: int = 2,
) -> str:
    """
    Quick helper to run deliberation and return just the final response.

    For more control, use get_deliberation_room().deliberate()
    """
    room = get_deliberation_room()
    result = await room.deliberate(prompt, agents, max_rounds)
    return result.final_response

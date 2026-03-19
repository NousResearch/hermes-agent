"""
cosmos Agent Debates - Multi-Agent Discussion & Synthesis

Novel Approaches:
1. Structured Argumentation - Formal debate protocols
2. Confidence-Weighted Voting - Score-based consensus
3. Perspective Diversity - Ensure varied viewpoints
4. Synthesis Engine - Combine best insights
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
import json
from collections import defaultdict

from loguru import logger


class DebateRole(Enum):
    """Roles in a debate."""
    PROPONENT = "proponent"     # Argues for a position
    OPPONENT = "opponent"       # Argues against
    MODERATOR = "moderator"     # Guides discussion
    SYNTHESIZER = "synthesizer" # Combines insights
    FACT_CHECKER = "fact_checker"  # Verifies claims


class ArgumentType(Enum):
    """Types of arguments."""
    CLAIM = "claim"
    EVIDENCE = "evidence"
    REBUTTAL = "rebuttal"
    CONCESSION = "concession"
    SYNTHESIS = "synthesis"
    QUESTION = "question"


@dataclass
class Argument:
    """A single argument in a debate."""
    id: str
    agent_id: str
    role: DebateRole
    argument_type: ArgumentType
    content: str
    confidence: float  # 0.0 to 1.0

    # References
    responds_to: Optional[str] = None  # ID of argument this responds to
    supports: list[str] = field(default_factory=list)  # IDs of supporting arguments
    contradicts: list[str] = field(default_factory=list)  # IDs of contradicted arguments

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    evidence_links: list[str] = field(default_factory=list)

    # Scoring
    peer_ratings: dict[str, float] = field(default_factory=dict)
    verified: Optional[bool] = None

    def average_rating(self) -> float:
        if not self.peer_ratings:
            return 0.5
        return sum(self.peer_ratings.values()) / len(self.peer_ratings)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent": self.agent_id,
            "role": self.role.value,
            "type": self.argument_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "average_rating": self.average_rating(),
        }


@dataclass
class DebatePosition:
    """A position being debated."""
    id: str
    statement: str
    context: str = ""

    # Arguments for each side
    pro_arguments: list[str] = field(default_factory=list)  # Argument IDs
    con_arguments: list[str] = field(default_factory=list)

    # Voting
    votes: dict[str, float] = field(default_factory=dict)  # agent_id -> -1 to 1

    def get_consensus(self) -> float:
        """Get consensus score (-1 = against, 0 = split, 1 = for)."""
        if not self.votes:
            return 0.0
        return sum(self.votes.values()) / len(self.votes)


@dataclass
class Debate:
    """A structured debate between agents."""
    id: str
    topic: str
    positions: list[DebatePosition] = field(default_factory=list)

    # Participants
    agents: dict[str, DebateRole] = field(default_factory=dict)

    # Arguments
    arguments: dict[str, Argument] = field(default_factory=dict)

    # Progress
    current_round: int = 0
    max_rounds: int = 3
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None

    # Outcome
    conclusion: str = ""
    synthesis: str = ""
    key_insights: list[str] = field(default_factory=list)
    consensus_score: float = 0.0  # -1 to 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "round": self.current_round,
            "max_rounds": self.max_rounds,
            "participants": len(self.agents),
            "arguments": len(self.arguments),
            "consensus": self.consensus_score,
            "conclusion": self.conclusion,
        }


class AgentDebates:
    """
    Multi-agent debate and discussion system.

    Features:
    - Structured debate protocols
    - Multiple perspectives on issues
    - Confidence-weighted consensus
    - Insight synthesis
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        min_agents: int = 2,
        max_agents: int = 5,
    ):
        self.llm_fn = llm_fn
        self.min_agents = min_agents
        self.max_agents = max_agents

        self.debates: dict[str, Debate] = {}
        self._debate_counter = 0
        self._argument_counter = 0

        self._lock = asyncio.Lock()

    async def start_debate(
        self,
        topic: str,
        positions: Optional[list[str]] = None,
        agent_generators: Optional[dict[str, Callable]] = None,
        max_rounds: int = 3,
    ) -> Debate:
        """
        Start a new debate on a topic.

        Args:
            topic: The topic to debate
            positions: Specific positions to argue (optional)
            agent_generators: Functions to generate agent responses
            max_rounds: Maximum debate rounds

        Returns:
            Debate object with structure
        """
        async with self._lock:
            self._debate_counter += 1
            debate_id = f"debate_{self._debate_counter}"

        debate = Debate(
            id=debate_id,
            topic=topic,
            max_rounds=max_rounds,
        )

        # Generate positions if not provided
        if positions:
            for i, pos in enumerate(positions):
                debate.positions.append(DebatePosition(
                    id=f"pos_{i}",
                    statement=pos,
                ))
        else:
            # Generate positions from topic
            debate.positions = await self._generate_positions(topic)

        # Assign agents to roles
        debate.agents = self._assign_roles(agent_generators)

        self.debates[debate_id] = debate
        logger.info(f"Started debate {debate_id}: {topic}")

        return debate

    async def _generate_positions(self, topic: str) -> list[DebatePosition]:
        """Generate debatable positions from a topic."""
        if not self.llm_fn:
            return [DebatePosition(
                id="pos_0",
                statement=f"We should proceed with: {topic}",
            )]

        prompt = f"""Generate 2-3 distinct, debatable positions on this topic:

Topic: {topic}

Return JSON array:
[
  {{"statement": "Position statement", "context": "Why this is debatable"}}
]

Positions should be mutually exclusive or represent different approaches."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            data = json.loads(self._extract_json(response))

            return [
                DebatePosition(
                    id=f"pos_{i}",
                    statement=p.get("statement", ""),
                    context=p.get("context", ""),
                )
                for i, p in enumerate(data)
            ]

        except Exception as e:
            logger.error(f"Position generation failed: {e}")
            return [DebatePosition(id="pos_0", statement=topic)]

    def _assign_roles(
        self,
        agent_generators: Optional[dict[str, Callable]] = None,
    ) -> dict[str, DebateRole]:
        """Assign debate roles to agents."""
        agents = {}

        if agent_generators:
            for agent_id in agent_generators.keys():
                if len(agents) < self.max_agents:
                    agents[agent_id] = DebateRole.PROPONENT
        else:
            # Create virtual agents
            agents["agent_pro"] = DebateRole.PROPONENT
            agents["agent_con"] = DebateRole.OPPONENT
            agents["agent_mod"] = DebateRole.MODERATOR

        return agents

    async def run_debate(
        self,
        debate_id: str,
        agent_fn: Optional[Callable] = None,
    ) -> Debate:
        """
        Run a full debate to completion.

        Args:
            debate_id: ID of the debate
            agent_fn: Function(agent_id, role, context) -> response

        Returns:
            Completed debate with synthesis
        """
        if debate_id not in self.debates:
            raise ValueError(f"Unknown debate: {debate_id}")

        debate = self.debates[debate_id]
        executor = agent_fn or self._default_agent_fn

        logger.info(f"Running debate {debate_id}")

        for round_num in range(debate.max_rounds):
            debate.current_round = round_num + 1
            logger.info(f"Debate round {debate.current_round}")

            # Each agent contributes
            for agent_id, role in debate.agents.items():
                # Build context for agent
                context = self._build_agent_context(debate, agent_id, role)

                # Get agent response
                try:
                    if asyncio.iscoroutinefunction(executor):
                        response = await executor(agent_id, role, context)
                    else:
                        response = executor(agent_id, role, context)

                    # Parse response into arguments
                    arguments = await self._parse_response(
                        debate, agent_id, role, response
                    )

                    for arg in arguments:
                        debate.arguments[arg.id] = arg

                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {e}")

            # Rate arguments
            await self._rate_arguments(debate)

        # Synthesize conclusion
        debate.synthesis = await self._synthesize_debate(debate)
        debate.key_insights = await self._extract_insights(debate)
        debate.consensus_score = self._calculate_consensus(debate)
        debate.conclusion = self._generate_conclusion(debate)
        debate.ended_at = datetime.now()

        logger.info(f"Debate {debate_id} complete. Consensus: {debate.consensus_score:.2f}")

        return debate

    def _build_agent_context(
        self,
        debate: Debate,
        agent_id: str,
        role: DebateRole,
    ) -> str:
        """Build context for an agent's turn."""
        context_parts = [
            f"Topic: {debate.topic}",
            f"Your role: {role.value}",
            f"Round: {debate.current_round} of {debate.max_rounds}",
            "",
            "Positions:",
        ]

        for pos in debate.positions:
            context_parts.append(f"- {pos.statement}")

        if debate.arguments:
            context_parts.extend(["", "Previous arguments:"])
            # Show recent arguments
            recent = sorted(
                debate.arguments.values(),
                key=lambda a: a.timestamp,
                reverse=True,
            )[:10]

            for arg in reversed(recent):
                context_parts.append(
                    f"[{arg.role.value}] {arg.content[:100]}... "
                    f"(confidence: {arg.confidence:.0%})"
                )

        return "\n".join(context_parts)

    async def _default_agent_fn(
        self,
        agent_id: str,
        role: DebateRole,
        context: str,
    ) -> str:
        """Default agent using LLM."""
        if not self.llm_fn:
            return f"[{role.value}] I have no specific argument to make."

        role_prompts = {
            DebateRole.PROPONENT: "Argue in favor of the main position. Present evidence and reasoning.",
            DebateRole.OPPONENT: "Present counterarguments and identify weaknesses in pro arguments.",
            DebateRole.MODERATOR: "Summarize the discussion so far and ask clarifying questions.",
            DebateRole.SYNTHESIZER: "Find common ground and propose compromise positions.",
            DebateRole.FACT_CHECKER: "Verify claims made and point out unsupported assertions.",
        }

        prompt = f"""{context}

Instructions: {role_prompts.get(role, "Contribute to the discussion.")}

Provide your argument with a confidence level (0-100%).
Format: [CONFIDENCE: X%] Your argument here..."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                return await self.llm_fn(prompt)
            else:
                return self.llm_fn(prompt)
        except Exception as e:
            return f"[Failed to generate response: {e}]"

    async def _parse_response(
        self,
        debate: Debate,
        agent_id: str,
        role: DebateRole,
        response: str,
    ) -> list[Argument]:
        """Parse agent response into arguments."""
        arguments = []

        # Extract confidence
        confidence = 0.5
        import re
        conf_match = re.search(r'\[CONFIDENCE:\s*(\d+)%?\]', response)
        if conf_match:
            confidence = int(conf_match.group(1)) / 100

        # Determine argument type from role
        arg_type = {
            DebateRole.PROPONENT: ArgumentType.CLAIM,
            DebateRole.OPPONENT: ArgumentType.REBUTTAL,
            DebateRole.MODERATOR: ArgumentType.QUESTION,
            DebateRole.SYNTHESIZER: ArgumentType.SYNTHESIS,
            DebateRole.FACT_CHECKER: ArgumentType.EVIDENCE,
        }.get(role, ArgumentType.CLAIM)

        # Clean response
        clean_response = re.sub(r'\[CONFIDENCE:\s*\d+%?\]', '', response).strip()

        async with self._lock:
            self._argument_counter += 1
            arg_id = f"arg_{self._argument_counter}"

        arguments.append(Argument(
            id=arg_id,
            agent_id=agent_id,
            role=role,
            argument_type=arg_type,
            content=clean_response,
            confidence=confidence,
        ))

        return arguments

    async def _rate_arguments(self, debate: Debate):
        """Have agents rate each other's arguments."""
        if not self.llm_fn:
            return

        for arg in debate.arguments.values():
            if arg.peer_ratings:
                continue  # Already rated

            # Other agents rate this argument
            for other_id in debate.agents.keys():
                if other_id == arg.agent_id:
                    continue

                prompt = f"""Rate this argument from 0.0 to 1.0:

Argument: {arg.content[:500]}

Consider:
- Logical validity
- Evidence quality
- Relevance to topic

Return just a number between 0.0 and 1.0"""

                try:
                    if asyncio.iscoroutinefunction(self.llm_fn):
                        response = await self.llm_fn(prompt)
                    else:
                        response = self.llm_fn(prompt)

                    # Extract rating
                    import re
                    match = re.search(r'(\d+\.?\d*)', response)
                    if match:
                        rating = float(match.group(1))
                        if rating > 1:
                            rating = rating / 100
                        arg.peer_ratings[other_id] = max(0, min(1, rating))

                except Exception as e:
                    logger.debug(f"Failed to parse peer rating: {e}")
                    # Default to neutral rating on parse failure
                    arg.peer_ratings[other_id] = 0.5

    async def _synthesize_debate(self, debate: Debate) -> str:
        """Synthesize debate into unified conclusion."""
        if not self.llm_fn:
            return self._basic_synthesis(debate)

        # Get top-rated arguments
        top_args = sorted(
            debate.arguments.values(),
            key=lambda a: a.average_rating() * a.confidence,
            reverse=True,
        )[:5]

        args_summary = "\n".join([
            f"- [{a.role.value}] {a.content[:200]}... (score: {a.average_rating():.2f})"
            for a in top_args
        ])

        prompt = f"""Synthesize this debate into a balanced conclusion.

Topic: {debate.topic}

Top arguments:
{args_summary}

Provide:
1. A synthesis that incorporates the strongest points from all sides
2. Identify any remaining disagreements
3. Recommend next steps or further investigation

Keep the synthesis concise but comprehensive."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                return await self.llm_fn(prompt)
            else:
                return self.llm_fn(prompt)
        except Exception as e:
            return self._basic_synthesis(debate)

    def _basic_synthesis(self, debate: Debate) -> str:
        """Basic synthesis without LLM."""
        pro_count = sum(1 for a in debate.arguments.values() if a.role == DebateRole.PROPONENT)
        con_count = sum(1 for a in debate.arguments.values() if a.role == DebateRole.OPPONENT)

        return (
            f"Debate on '{debate.topic}' had {len(debate.arguments)} arguments. "
            f"{pro_count} supporting, {con_count} opposing. "
            f"Consensus: {debate.consensus_score:.0%}"
        )

    async def _extract_insights(self, debate: Debate) -> list[str]:
        """Extract key insights from debate."""
        if not self.llm_fn:
            return []

        args_text = "\n".join([
            a.content[:200] for a in debate.arguments.values()
        ])[:3000]

        prompt = f"""Extract 3-5 key insights from this debate:

Topic: {debate.topic}

Arguments:
{args_text}

Return JSON array of insight strings:
["insight 1", "insight 2", ...]"""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            return json.loads(self._extract_json(response))

        except Exception as e:
            logger.debug(f"Failed to extract insights: {e}")
            return []

    def _calculate_consensus(self, debate: Debate) -> float:
        """Calculate overall consensus score."""
        if not debate.arguments:
            return 0.0

        # Weight by confidence and rating
        pro_score = sum(
            a.confidence * a.average_rating()
            for a in debate.arguments.values()
            if a.role == DebateRole.PROPONENT
        )

        con_score = sum(
            a.confidence * a.average_rating()
            for a in debate.arguments.values()
            if a.role == DebateRole.OPPONENT
        )

        total = pro_score + con_score
        if total == 0:
            return 0.0

        # -1 (all con) to 1 (all pro)
        return (pro_score - con_score) / total

    def _generate_conclusion(self, debate: Debate) -> str:
        """Generate a brief conclusion."""
        consensus = debate.consensus_score

        if consensus > 0.6:
            verdict = "Strong agreement in favor"
        elif consensus > 0.2:
            verdict = "Slight agreement in favor"
        elif consensus > -0.2:
            verdict = "No clear consensus"
        elif consensus > -0.6:
            verdict = "Slight agreement against"
        else:
            verdict = "Strong agreement against"

        return f"{verdict} ({consensus:.0%} consensus) after {len(debate.arguments)} arguments."

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text."""
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            return text[start:end]
        return '[]'

    async def add_argument(
        self,
        debate_id: str,
        agent_id: str,
        content: str,
        role: DebateRole = DebateRole.PROPONENT,
        confidence: float = 0.5,
        responds_to: Optional[str] = None,
    ) -> Argument:
        """Add an argument to a debate."""
        if debate_id not in self.debates:
            raise ValueError(f"Unknown debate: {debate_id}")

        debate = self.debates[debate_id]

        async with self._lock:
            self._argument_counter += 1
            arg_id = f"arg_{self._argument_counter}"

        arg_type = {
            DebateRole.PROPONENT: ArgumentType.CLAIM,
            DebateRole.OPPONENT: ArgumentType.REBUTTAL,
        }.get(role, ArgumentType.CLAIM)

        argument = Argument(
            id=arg_id,
            agent_id=agent_id,
            role=role,
            argument_type=arg_type,
            content=content,
            confidence=confidence,
            responds_to=responds_to,
        )

        debate.arguments[arg_id] = argument

        return argument

    async def vote(
        self,
        debate_id: str,
        agent_id: str,
        position_id: str,
        vote: float,
    ) -> bool:
        """Cast a vote on a position (-1 to 1)."""
        if debate_id not in self.debates:
            return False

        debate = self.debates[debate_id]

        for pos in debate.positions:
            if pos.id == position_id:
                pos.votes[agent_id] = max(-1, min(1, vote))
                return True

        return False

    def get_stats(self) -> dict:
        """Get debate statistics."""
        total_args = sum(len(d.arguments) for d in self.debates.values())

        return {
            "total_debates": len(self.debates),
            "total_arguments": total_args,
            "avg_consensus": sum(d.consensus_score for d in self.debates.values()) / len(self.debates)
                if self.debates else 0,
        }

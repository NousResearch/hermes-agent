"""
Cosmos Dynamic Token Orchestrator
======================================

Manages token budgets across the entire agent collective:
- Per-agent budget allocation based on tier (LOCAL / API_STANDARD / API_PREMIUM)
- Grok+Kimi tandem sessions with intelligent handoff
- Automatic rebalancing based on usage patterns
- Background orchestration loop with periodic snapshots

"Why let one brain hog all the neurons?" - The Collective
"""

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from typing import   Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Guarded imports -- these may not be available in every environment
# ---------------------------------------------------------------------------

try:
    from Cosmos.core.dynamic_limits import get_limits, get_max_tokens, ModelTier
except ImportError:
    get_limits = None
    get_max_tokens = None
    ModelTier = None

try:
    from Cosmos.core.token_saver import ContextCompressor
except ImportError:
    ContextCompressor = None

try:
    from Cosmos.core.nexus import nexus, SignalType
except ImportError:
    nexus = None
    SignalType = None


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AgentTier(Enum):
    LOCAL = "local"              # DeepSeek, Phi, HuggingFace, Llama -- unlimited
    API_STANDARD = "api_standard"  # Groq, Perplexity, Mistral
    API_PREMIUM = "api_premium"    # Grok, Gemini, Claude, Kimi, ClaudeOpus


AGENT_TIER_MAP: dict[str, AgentTier] = {
    "deepseek": AgentTier.LOCAL,
    "phi": AgentTier.LOCAL,
    "huggingface": AgentTier.LOCAL,
    "llama": AgentTier.LOCAL,
    "groq": AgentTier.API_STANDARD,
    "perplexity": AgentTier.API_STANDARD,
    "mistral": AgentTier.API_STANDARD,
    "grok": AgentTier.API_PREMIUM,
    "gemini": AgentTier.API_PREMIUM,
    "claude": AgentTier.API_PREMIUM,
    "kimi": AgentTier.API_PREMIUM,
    "claudeopus": AgentTier.API_PREMIUM,
    "cosmos": AgentTier.LOCAL,
    "swarm-mind": AgentTier.LOCAL,
}


@dataclass
class AgentBudget:
    agent_id: str
    tier: AgentTier
    allocated_tokens: int
    used_tokens: int = 0
    used_cost: float = 0.0
    requests_count: int = 0
    last_request: Optional[datetime] = None
    efficiency_score: float = 1.0
    is_tandem: bool = False


@dataclass
class TandemSession:
    session_id: str
    primary: str
    secondary: str
    shared_context: str
    primary_budget: int
    secondary_budget: int
    handoff_count: int = 0
    total_tokens_used: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    task_type: str = "chat"


@dataclass
class AllocationResult:
    approved: bool
    allocated_tokens: int
    suggested_alternative: Optional[str] = None
    reason: str = ""


@dataclass
class OrchestratorSnapshot:
    timestamp: datetime
    total_budget_remaining: int
    agent_budgets: dict  # serialized AgentBudget dicts
    active_tandem_sessions: int
    efficiency_rating: float
    top_performer: str


# =============================================================================
# TOKEN ORCHESTRATOR
# =============================================================================

class TokenOrchestrator:
    """Central token budget manager for the Cosmos collective."""

    def __init__(
        self,
        daily_api_budget: int = 500_000,
        rebalance_interval_seconds: float = 300.0,
    ):
        self._daily_budget = daily_api_budget
        self._rebalance_interval = rebalance_interval_seconds
        self._agent_budgets: dict[str, AgentBudget] = {}
        self._tandem_sessions: dict[str, TandemSession] = {}
        self._snapshots: deque = deque(maxlen=288)  # 24h of 5-min snapshots
        self._running = False
        self._budget_reset_date: Optional[date] = None
        self._initialize_budgets()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _initialize_budgets(self):
        """Set up initial per-agent budgets based on tier."""
        local_agents = [
            aid for aid, tier in AGENT_TIER_MAP.items()
            if tier == AgentTier.LOCAL
        ]
        standard_agents = [
            aid for aid, tier in AGENT_TIER_MAP.items()
            if tier == AgentTier.API_STANDARD
        ]
        premium_agents = [
            aid for aid, tier in AGENT_TIER_MAP.items()
            if tier == AgentTier.API_PREMIUM
        ]

        standard_share = int(self._daily_budget * 0.15)
        premium_share = int(self._daily_budget * 0.85)

        per_standard = (
            standard_share // len(standard_agents) if standard_agents else 0
        )
        per_premium = (
            premium_share // len(premium_agents) if premium_agents else 0
        )

        for aid in local_agents:
            self._agent_budgets[aid] = AgentBudget(
                agent_id=aid,
                tier=AgentTier.LOCAL,
                allocated_tokens=999_999_999,
            )

        for aid in standard_agents:
            self._agent_budgets[aid] = AgentBudget(
                agent_id=aid,
                tier=AgentTier.API_STANDARD,
                allocated_tokens=per_standard,
            )

        for aid in premium_agents:
            self._agent_budgets[aid] = AgentBudget(
                agent_id=aid,
                tier=AgentTier.API_PREMIUM,
                allocated_tokens=per_premium,
            )

        self._budget_reset_date = date.today()
        logger.info(
            f"TokenOrchestrator initialised: {len(local_agents)} local, "
            f"{len(standard_agents)} standard ({per_standard} tok each), "
            f"{len(premium_agents)} premium ({per_premium} tok each)"
        )

    # ------------------------------------------------------------------
    # Core allocation
    # ------------------------------------------------------------------

    async def allocate(
        self,
        agent_id: str,
        task_type: str,
        estimated_tokens: int,
    ) -> AllocationResult:
        """Request token allocation for an agent."""
        budget = self._agent_budgets.get(agent_id)

        # Unknown agent -- treat as standard
        if budget is None:
            tier = AGENT_TIER_MAP.get(agent_id, AgentTier.API_STANDARD)
            budget = AgentBudget(
                agent_id=agent_id,
                tier=tier,
                allocated_tokens=0,
            )
            self._agent_budgets[agent_id] = budget

        # LOCAL agents are always approved
        if budget.tier == AgentTier.LOCAL:
            result = AllocationResult(
                approved=True,
                allocated_tokens=estimated_tokens,
                reason="local_unlimited",
            )
            await self._try_emit_signal(
                "ORCHESTRATOR_ALLOCATION_APPROVED",
                {"agent_id": agent_id, "tokens": estimated_tokens, "task_type": task_type},
            )
            return result

        remaining = budget.allocated_tokens - budget.used_tokens
        if estimated_tokens <= remaining:
            result = AllocationResult(
                approved=True,
                allocated_tokens=estimated_tokens,
                reason="within_budget",
            )
            await self._try_emit_signal(
                "ORCHESTRATOR_ALLOCATION_APPROVED",
                {"agent_id": agent_id, "tokens": estimated_tokens, "task_type": task_type},
            )
            return result

        # Budget tight -- suggest a cheaper alternative
        alternative = await self.get_cheapest_adequate(task_type)
        if alternative and alternative != agent_id:
            result = AllocationResult(
                approved=False,
                allocated_tokens=0,
                suggested_alternative=alternative,
                reason=f"budget_exceeded: {remaining} remaining, need {estimated_tokens}",
            )
        else:
            # Approve partial allocation with whatever is left
            result = AllocationResult(
                approved=remaining > 0,
                allocated_tokens=max(remaining, 0),
                reason="partial_budget" if remaining > 0 else "budget_exhausted",
            )

        await self._try_emit_signal(
            "ORCHESTRATOR_ALLOCATION_REQUESTED",
            {
                "agent_id": agent_id,
                "tokens_requested": estimated_tokens,
                "approved": result.approved,
                "reason": result.reason,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Usage reporting
    # ------------------------------------------------------------------

    async def report_usage(
        self,
        agent_id: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str = "chat",
        quality_score: float = 0.0,
    ):
        """Record actual token usage after completion."""
        budget = self._agent_budgets.get(agent_id)
        if budget is None:
            return

        total = input_tokens + output_tokens
        budget.used_tokens += total
        budget.requests_count += 1
        budget.last_request = datetime.utcnow()

        # Update efficiency score (rolling average blending quality and cost)
        if quality_score > 0:
            n = budget.requests_count
            budget.efficiency_score = (
                (budget.efficiency_score * (n - 1) + quality_score) / n
            )

        await self._try_emit_signal(
            "ORCHESTRATOR_USAGE_REPORTED",
            {
                "agent_id": agent_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "task_type": task_type,
            },
        )

    # ------------------------------------------------------------------
    # Tandem sessions (Grok + Kimi)
    # ------------------------------------------------------------------

    _GROK_KEYWORDS = frozenset([
        "research", "real-time", "twitter", "x", "humor",
        "controversy", "search", "trending",
    ])
    _KIMI_KEYWORDS = frozenset([
        "document", "reasoning", "synthesis", "planning",
        "analysis", "long", "complex", "architecture",
    ])

    async def start_tandem(
        self,
        task: str,
        task_type: str = "chat",
    ) -> TandemSession:
        """Create a Grok+Kimi tandem session."""
        task_lower = task.lower()

        grok_score = sum(1 for kw in self._GROK_KEYWORDS if kw in task_lower)
        kimi_score = sum(1 for kw in self._KIMI_KEYWORDS if kw in task_lower)

        if grok_score >= kimi_score:
            primary, secondary = "grok", "kimi"
        else:
            primary, secondary = "kimi", "grok"

        # 60/40 budget split
        total_budget = 50_000  # sensible default per-session
        primary_budget = int(total_budget * 0.6)
        secondary_budget = total_budget - primary_budget

        session = TandemSession(
            session_id=str(uuid.uuid4()),
            primary=primary,
            secondary=secondary,
            shared_context=task,
            primary_budget=primary_budget,
            secondary_budget=secondary_budget,
            task_type=task_type,
        )
        self._tandem_sessions[session.session_id] = session

        await self._try_emit_signal(
            "ORCHESTRATOR_TANDEM_STARTED",
            {
                "session_id": session.session_id,
                "primary": primary,
                "secondary": secondary,
                "task": task[:200],
            },
        )

        logger.info(
            f"Tandem session {session.session_id[:8]} started: "
            f"{primary} (lead) + {secondary}"
        )
        return session

    async def tandem_handoff(
        self,
        session_id: str,
        context_summary: str,
    ) -> str:
        """Hand off from primary to secondary in a tandem session."""
        session = self._tandem_sessions.get(session_id)
        if session is None:
            return "[error] tandem session not found"

        # Compress context if ContextCompressor is available
        compressed = context_summary
        if ContextCompressor is not None:
            try:
                compressor = ContextCompressor()
                messages = [{"role": "assistant", "content": context_summary}]
                result = await compressor.compress(
                    messages,
                    target_tokens=2000,
                    strategy="extractive",
                )
                compressed = result.summary
            except Exception as exc:
                logger.warning(f"Tandem context compression failed: {exc}")

        # Try to call the secondary agent's provider
        secondary_response = await self._invoke_tandem_agent(
            session.secondary,
            compressed,
            session.task_type,
        )

        session.handoff_count += 1
        session.shared_context = compressed

        await self._try_emit_signal(
            "ORCHESTRATOR_TANDEM_HANDOFF",
            {
                "session_id": session_id,
                "from": session.primary,
                "to": session.secondary,
                "handoff_count": session.handoff_count,
            },
        )

        return secondary_response

    async def _invoke_tandem_agent(
        self,
        agent_id: str,
        context: str,
        task_type: str,
    ) -> str:
        """Attempt to call a provider's swarm_respond for tandem handoff."""
        try:
            if agent_id == "grok":
                from Cosmos.integration.external.grok import GrokProvider
                provider = GrokProvider()
                return await provider.swarm_respond(context, task_type=task_type)
            elif agent_id == "kimi":
                from Cosmos.integration.external.kimi import KimiProvider
                provider = KimiProvider()
                return await provider.swarm_respond(context, task_type=task_type)
        except ImportError:
            logger.warning(f"Provider for {agent_id} not available for tandem handoff")
        except Exception as exc:
            logger.error(f"Tandem handoff to {agent_id} failed: {exc}")

        return f"[tandem handoff to {agent_id} unavailable]"

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    async def rebalance(self):
        """Redistribute budgets based on usage patterns."""
        today = date.today()

        # Day rollover -- reset daily budgets
        if self._budget_reset_date != today:
            logger.info("TokenOrchestrator: daily budget reset")
            self._budget_reset_date = today
            for budget in self._agent_budgets.values():
                budget.used_tokens = 0
                budget.used_cost = 0.0
                budget.requests_count = 0
            self._initialize_budgets()
            return

        # Identify API agents only
        api_budgets = [
            b for b in self._agent_budgets.values()
            if b.tier != AgentTier.LOCAL
        ]
        if not api_budgets:
            return

        # Find under-utilised (<25% used) and over-utilised (>90% used) agents
        under_used: list[AgentBudget] = []
        over_used: list[AgentBudget] = []

        for b in api_budgets:
            if b.allocated_tokens <= 0:
                continue
            usage_ratio = b.used_tokens / b.allocated_tokens
            if usage_ratio < 0.25 and b.requests_count > 0:
                under_used.append(b)
            elif usage_ratio > 0.90:
                over_used.append(b)

        if not under_used or not over_used:
            return

        # Transfer 25% of each under-used agent's remaining tokens to over-used
        total_transferable = 0
        for b in under_used:
            transfer = int((b.allocated_tokens - b.used_tokens) * 0.25)
            b.allocated_tokens -= transfer
            total_transferable += transfer

        if total_transferable > 0 and over_used:
            per_agent = total_transferable // len(over_used)
            for b in over_used:
                b.allocated_tokens += per_agent

        await self._try_emit_signal(
            "ORCHESTRATOR_REBALANCED",
            {
                "transferred_tokens": total_transferable,
                "from_agents": [b.agent_id for b in under_used],
                "to_agents": [b.agent_id for b in over_used],
            },
        )

        logger.info(
            f"TokenOrchestrator rebalanced: {total_transferable} tokens "
            f"from {[b.agent_id for b in under_used]} "
            f"to {[b.agent_id for b in over_used]}"
        )

    # ------------------------------------------------------------------
    # Cheapest adequate agent selection
    # ------------------------------------------------------------------

    async def get_cheapest_adequate(
        self,
        task_type: str,
        min_quality: float = 0.6,
    ) -> str:
        """Find the cheapest agent that meets a quality threshold."""
        # Prefer LOCAL tier first, then STANDARD, then PREMIUM
        tier_order = [AgentTier.LOCAL, AgentTier.API_STANDARD, AgentTier.API_PREMIUM]

        for tier in tier_order:
            candidates = [
                b for b in self._agent_budgets.values()
                if b.tier == tier and b.efficiency_score >= min_quality
            ]
            if not candidates:
                continue

            # Among candidates in this tier, pick the one with the best
            # efficiency score (most bang for buck)
            best = max(candidates, key=lambda b: b.efficiency_score)

            # For API tiers, make sure they still have budget
            if tier != AgentTier.LOCAL:
                remaining = best.allocated_tokens - best.used_tokens
                if remaining <= 0:
                    continue

            return best.agent_id

        # Absolute fallback
        return "deepseek"

    # ------------------------------------------------------------------
    # Dashboard / status
    # ------------------------------------------------------------------

    def get_dashboard(self) -> dict:
        """Return current orchestrator state for the web UI."""
        api_used = sum(
            b.used_tokens
            for b in self._agent_budgets.values()
            if b.tier != AgentTier.LOCAL
        )
        return {
            "daily_budget": self._daily_budget,
            "total_used": sum(b.used_tokens for b in self._agent_budgets.values()),
            "total_remaining": self._daily_budget - api_used,
            "agents": {
                aid: {
                    "tier": b.tier.value,
                    "allocated": b.allocated_tokens,
                    "used": b.used_tokens,
                    "remaining": b.allocated_tokens - b.used_tokens,
                    "requests": b.requests_count,
                    "efficiency": round(b.efficiency_score, 3),
                    "is_tandem": b.is_tandem,
                }
                for aid, b in self._agent_budgets.items()
            },
            "active_tandems": len(self._tandem_sessions),
            "tandem_sessions": [
                {
                    "session_id": s.session_id,
                    "primary": s.primary,
                    "secondary": s.secondary,
                    "handoffs": s.handoff_count,
                    "tokens_used": s.total_tokens_used,
                    "task_type": s.task_type,
                }
                for s in self._tandem_sessions.values()
            ],
            "efficiency_rating": self._calculate_efficiency(),
            "top_performer": self._get_top_performer(),
            "snapshots_count": len(self._snapshots),
        }

    def _calculate_efficiency(self) -> float:
        """Calculate overall collective efficiency."""
        scores = [
            b.efficiency_score
            for b in self._agent_budgets.values()
            if b.requests_count > 0
        ]
        return sum(scores) / len(scores) if scores else 1.0

    def _get_top_performer(self) -> str:
        """Get the most efficient agent."""
        best = max(
            self._agent_budgets.values(),
            key=lambda b: b.efficiency_score if b.requests_count > 0 else 0,
            default=None,
        )
        return best.agent_id if best else "none"

    # ------------------------------------------------------------------
    # Background orchestration loop
    # ------------------------------------------------------------------

    async def start_background_orchestration(self):
        """Background loop: rebalance periodically, emit snapshots."""
        self._running = True
        logger.info("TokenOrchestrator background loop started")
        while self._running:
            try:
                await self.rebalance()

                # Create and store snapshot
                snapshot = OrchestratorSnapshot(
                    timestamp=datetime.utcnow(),
                    total_budget_remaining=self._daily_budget - sum(
                        b.used_tokens
                        for b in self._agent_budgets.values()
                        if b.tier != AgentTier.LOCAL
                    ),
                    agent_budgets={
                        aid: asdict(b)
                        for aid, b in self._agent_budgets.items()
                    },
                    active_tandem_sessions=len(self._tandem_sessions),
                    efficiency_rating=self._calculate_efficiency(),
                    top_performer=self._get_top_performer(),
                )
                self._snapshots.append(snapshot)

            except Exception as exc:
                logger.error(f"Orchestrator rebalance error: {exc}")

            await asyncio.sleep(self._rebalance_interval)

    async def stop(self):
        """Stop the background orchestration loop."""
        self._running = False
        logger.info("TokenOrchestrator stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _try_emit_signal(self, signal_name: str, payload: dict):
        """Emit a nexus signal if the bus and signal type are available."""
        if nexus is None or SignalType is None:
            return
        signal_type = getattr(SignalType, signal_name, None)
        if signal_type is None:
            return
        try:
            await nexus.emit(
                type=signal_type,
                payload=payload,
                source="token_orchestrator",
            )
        except Exception as exc:
            logger.debug(f"Nexus emit ({signal_name}) failed: {exc}")


# =============================================================================
# SINGLETON
# =============================================================================

_token_orchestrator: Optional[TokenOrchestrator] = None


def get_token_orchestrator() -> TokenOrchestrator:
    """Return the global TokenOrchestrator instance (lazy-init)."""
    global _token_orchestrator
    if _token_orchestrator is None:
        _token_orchestrator = TokenOrchestrator()
    return _token_orchestrator

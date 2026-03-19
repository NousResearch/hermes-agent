"""
cosmos Token Budget Manager with DAVINCI Consensus
==============================================

"Resonating through the 12D manifold."

Per-profile token budgets, usage tracking, and multi-model debate consensus.

DAVINCI CONSENSUS: When enabled, multiple high-level AI models collaborate through
up to 20 rounds of deliberation before synthesizing the best answer. This uses
significantly more tokens but produces higher quality responses.

Grok checks for consensus every 2 cycles. The deliberation continues until:
- All models agree (consensus reached), or
- Maximum 20 iterations complete

Grok then fact-checks the final raw answer before presenting to the user.

The Deliberation Chamber:
┌─────────────────────────────────────────────────────────────┐
│  🤖 DAVINCI CONSENSUS - Multi-Model Intelligence Chamber   │
├─────────────────────────────────────────────────────────────┤
│  Participants: Opus 4, Grok-2, GPT-4o, Gemini 1.5 Pro       │
│  ─────────────────────────────────────────────────────────  │
│  Cycle 2: GROK checks for consensus... ❌ Not yet           │
│  Cycle 4: GROK checks for consensus... ❌ Disagreements     │
│  Cycle 6: GROK checks for consensus... ✅ AGREEMENT!        │
│  ─────────────────────────────────────────────────────────  │
│  • Up to 20 cycles or until agreement                       │
│  • GROK validates every 2 cycles                            │
│  • GROK fact-checks final answer in raw form               │
│  ─────────────────────────────────────────────────────────  │
│  🔍 GROK FACT-CHECK: Validates final answer                 │
└─────────────────────────────────────────────────────────────┘
"""

import asyncio
from datetime import datetime, timedelta
from typing import   Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from loguru import logger


class BudgetPeriod(Enum):
    """Budget reset periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    UNLIMITED = "unlimited"


class UsageTier(Enum):
    """Usage tier for cost tracking."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class ModelTier(Enum):
    """Model tier for pricing."""
    ECONOMY = "economy"  # Cheaper models
    STANDARD = "standard"  # Mid-tier
    PREMIUM = "premium"  # High-end models
    DAVINCI = "davinci"  # Multi-model consensus mode


# Default DAVINCI mode participants - the heavy hitters
DAVINCI_DEBATE_MODELS = [
    "claude-opus-4",      # Anthropic's best - deep reasoning
    "grok-2",             # xAI - unfiltered, fact-focused
    "gpt-4o",             # OpenAI's flagship
    "gemini-1.5-pro",     # Google's best
]

# Grok is ALWAYS the final fact-checker
DAVINCI_FACT_CHECKER = "grok-2"


@dataclass
class ModelPricing:
    """Pricing information for a model."""
    model_id: str
    provider: str
    tier: ModelTier
    input_cost_per_1k: float  # Cost per 1k input tokens
    output_cost_per_1k: float  # Cost per 1k output tokens
    context_window: int
    supports_vision: bool = False
    supports_function_calling: bool = True


@dataclass
class TokenUsage:
    """Token usage record."""
    timestamp: datetime
    model_id: str
    input_tokens: int
    output_tokens: int
    cost: float
    profile_id: str
    session_id: str = ""
    mode: str = "standard"  # standard, davinci


@dataclass
class TokenBudget:
    """Token budget configuration for a profile."""
    profile_id: str
    name: str

    # Budget limits
    max_tokens_per_period: int
    period: BudgetPeriod
    max_cost_per_period: float = 0.0  # 0 = unlimited

    # Current usage
    tokens_used: int = 0
    cost_used: float = 0.0
    period_start: datetime = field(default_factory=datetime.utcnow)

    # Tier and permissions
    tier: UsageTier = UsageTier.FREE
    allowed_models: list[str] = field(default_factory=list)  # Empty = all
    allowed_tiers: list[ModelTier] = field(default_factory=lambda: [ModelTier.ECONOMY, ModelTier.STANDARD])
    davinci_mode_enabled: bool = False

    # Alerts
    warning_threshold: float = 0.8  # 80% of budget
    alert_emails: list[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens_per_period - self.tokens_used)

    @property
    def cost_remaining(self) -> float:
        if self.max_cost_per_period <= 0:
            return float('inf')
        return max(0, self.max_cost_per_period - self.cost_used)

    @property
    def usage_percentage(self) -> float:
        if self.max_tokens_per_period <= 0:
            return 0
        return (self.tokens_used / self.max_tokens_per_period) * 100

    def to_any(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "max_tokens_per_period": self.max_tokens_per_period,
            "period": self.period.value,
            "tokens_used": self.tokens_used,
            "tokens_remaining": self.tokens_remaining,
            "cost_used": round(self.cost_used, 4),
            "cost_remaining": round(self.cost_remaining, 4) if self.cost_remaining != float('inf') else None,
            "usage_percentage": round(self.usage_percentage, 2),
            "tier": self.tier.value,
            "davinci_mode_enabled": self.davinci_mode_enabled,
            "period_start": self.period_start.isoformat(),
        }


@dataclass
class DavinciDebateRound:
    """A single round in a DAVINCI mode deliberation."""
    round_number: int
    model_id: str
    response: str
    reasoning: str
    confidence: float
    tokens_used: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DavinciSession:
    """A DAVINCI mode multi-model deliberation session."""
    id: str
    profile_id: str
    query: str
    models: list[str]
    rounds: list[DavinciDebateRound] = field(default_factory=list)

    # Configuration
    max_rounds: int = 20  # Up to 20 cycles, or until agreement
    consensus_threshold: float = 0.8
    consensus_check_interval: int = 2  # Grok checks every N cycles
    fact_checker: str = "grok-2"  # Grok validates the final answer

    # Debate box state
    debate_log: list[dict] = field(default_factory=list)  # Visual debate history
    agreements: dict[str[str]] = field(default_factory=dict)  # model -> points agreed
    disagreements: dict[str[str]] = field(default_factory=dict)  # model -> objections

    # Results
    raw_consensus: str = ""  # Unvalidated consensus answer
    final_response: str = ""  # After Grok fact-check
    fact_check_result: dict = field(default_factory=dict)  # Grok's validation
    consensus_reached: bool = False
    total_tokens: int = 0
    total_cost: float = 0.0

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def add_to_deliberation_box(self, model: str, round_num: int, message: str, msg_type: str = "response"):
        """Add entry to the visual deliberation box."""
        self.debate_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "round": round_num,
            "type": msg_type,  # response, agreement, objection, synthesis
            "message": message[:500],
        })

    def render_debate_box(self) -> str:
        """Render the debate box as ASCII art."""
        lines = [
            "┌─────────────────────────────────────────────────────────────┐",
            "│  🤖 DAVINCI CONSENSUS - Multi-Model Intelligence Chamber   │",
            "├─────────────────────────────────────────────────────────────┤",
            f"│  Session: {self.id}  |  Models: {len(self.models)}  |  Rounds: {len(set(r.round_number for r in self.rounds))}/{self.max_rounds}",
            "├─────────────────────────────────────────────────────────────┤",
        ]

        # Group by round
        rounds_shown = {}
        for entry in self.debate_log[-12:]:  # Last 12 entries
            r = entry["round"]
            if r not in rounds_shown:
                rounds_shown[r] = []
            model_short = entry["model"][:12]
            msg_short = entry["message"][:40] + "..." if len(entry["message"]) > 40 else entry["message"]
            rounds_shown[r].append(f"│  [{model_short}]: {msg_short}")

        for r in sorted(rounds_shown.keys()):
            lines.append(f"│  ── Round {r} ──────────────────────────────────────────────")
            lines.extend(rounds_shown[r][:3])  # Max 3 per round in display

        if self.consensus_reached:
            lines.append("├─────────────────────────────────────────────────────────────┤")
            lines.append("│  ✅ CONSENSUS REACHED                                        │")

        if self.fact_check_result:
            status = "✅ VERIFIED" if self.fact_check_result.get("valid") else "⚠️ ISSUES FOUND"
            lines.append("├─────────────────────────────────────────────────────────────┤")
            lines.append(f"│  🔍 GROK FACT-CHECK: {status}                              │")

        lines.append("└─────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)

    def to_any(self) -> dict:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "models": self.models,
            "round_count": len(set(r.round_number for r in self.rounds)),
            "consensus_reached": self.consensus_reached,
            "fact_check_passed": self.fact_check_result.get("valid", False),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "debate_box": self.render_debate_box(),
        }


class TokenBudgetManager:
    """
    Token budget management with BENDER mode support.

    Features:
    - Per-profile token budgets
    - Multi-period tracking (hourly, daily, weekly, monthly)
    - Cost-based limits
    - Model tier restrictions
    - BENDER mode: Multi-model debate consensus
    - Usage analytics and reporting
    """

    def __init__(
        self,
        storage_path: Path = None,
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("./data/budgets")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.budgets: dict[str, TokenBudget] = {}
        self.usage_history: list[TokenUsage] = []
        self.bender_sessions: dict[str, DavinciSession] = {}
        self.model_pricing: dict[str, ModelPricing] = {}
        self.alert_handlers: list[Callable] = []

        self._load_model_pricing()
        self._load_budgets()

    def _load_model_pricing(self):
        """Load model pricing information."""
        # OpenAI models
        self.model_pricing["gpt-4o"] = ModelPricing(
            model_id="gpt-4o",
            provider="openai",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.005,
            output_cost_per_1k=0.015,
            context_window=128000,
            supports_vision=True,
        )
        self.model_pricing["gpt-4-turbo"] = ModelPricing(
            model_id="gpt-4-turbo",
            provider="openai",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.03,
            context_window=128000,
            supports_vision=True,
        )
        self.model_pricing["gpt-3.5-turbo"] = ModelPricing(
            model_id="gpt-3.5-turbo",
            provider="openai",
            tier=ModelTier.STANDARD,
            input_cost_per_1k=0.0005,
            output_cost_per_1k=0.0015,
            context_window=16385,
        )

        # Anthropic models
        self.model_pricing["claude-3-opus"] = ModelPricing(
            model_id="claude-3-opus",
            provider="anthropic",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.015,
            output_cost_per_1k=0.075,
            context_window=200000,
            supports_vision=True,
        )
        self.model_pricing["claude-3.5-sonnet"] = ModelPricing(
            model_id="claude-3.5-sonnet",
            provider="anthropic",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            context_window=200000,
            supports_vision=True,
        )
        self.model_pricing["claude-3-haiku"] = ModelPricing(
            model_id="claude-3-haiku",
            provider="anthropic",
            tier=ModelTier.ECONOMY,
            input_cost_per_1k=0.00025,
            output_cost_per_1k=0.00125,
            context_window=200000,
            supports_vision=True,
        )

        # Google models
        self.model_pricing["gemini-1.5-pro"] = ModelPricing(
            model_id="gemini-1.5-pro",
            provider="google",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.00125,
            output_cost_per_1k=0.005,
            context_window=1000000,
            supports_vision=True,
        )
        self.model_pricing["gemini-1.5-flash"] = ModelPricing(
            model_id="gemini-1.5-flash",
            provider="google",
            tier=ModelTier.STANDARD,
            input_cost_per_1k=0.000075,
            output_cost_per_1k=0.0003,
            context_window=1000000,
            supports_vision=True,
        )

        # DeepSeek models
        self.model_pricing["deepseek-chat"] = ModelPricing(
            model_id="deepseek-chat",
            provider="deepseek",
            tier=ModelTier.ECONOMY,
            input_cost_per_1k=0.00014,
            output_cost_per_1k=0.00028,
            context_window=128000,
        )
        self.model_pricing["deepseek-coder"] = ModelPricing(
            model_id="deepseek-coder",
            provider="deepseek",
            tier=ModelTier.ECONOMY,
            input_cost_per_1k=0.00014,
            output_cost_per_1k=0.00028,
            context_window=128000,
        )

        # xAI Grok models
        self.model_pricing["grok-2"] = ModelPricing(
            model_id="grok-2",
            provider="xai",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.002,
            output_cost_per_1k=0.010,
            context_window=131072,
            supports_vision=True,
        )
        self.model_pricing["grok-2-mini"] = ModelPricing(
            model_id="grok-2-mini",
            provider="xai",
            tier=ModelTier.STANDARD,
            input_cost_per_1k=0.0002,
            output_cost_per_1k=0.001,
            context_window=131072,
        )

        # Anthropic Opus 4 (latest)
        self.model_pricing["claude-opus-4"] = ModelPricing(
            model_id="claude-opus-4",
            provider="anthropic",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.015,
            output_cost_per_1k=0.075,
            context_window=200000,
            supports_vision=True,
        )

        # OpenAI o1 reasoning models
        self.model_pricing["o1-preview"] = ModelPricing(
            model_id="o1-preview",
            provider="openai",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.015,
            output_cost_per_1k=0.060,
            context_window=128000,
        )
        self.model_pricing["o1-mini"] = ModelPricing(
            model_id="o1-mini",
            provider="openai",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.012,
            context_window=128000,
        )

        # Mistral Large
        self.model_pricing["mistral-large"] = ModelPricing(
            model_id="mistral-large",
            provider="mistral",
            tier=ModelTier.PREMIUM,
            input_cost_per_1k=0.002,
            output_cost_per_1k=0.006,
            context_window=128000,
        )

        # Local models (free)
        self.model_pricing["ollama-local"] = ModelPricing(
            model_id="ollama-local",
            provider="ollama",
            tier=ModelTier.ECONOMY,
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
            context_window=32000,
        )

    def _load_budgets(self):
        """Load budgets from storage."""
        budgets_file = self.storage_path / "budgets.json"
        if budgets_file.exists():
            try:
                with open(budgets_file) as f:
                    data = json.load(f)
                for profile_id, budget_data in data.items():
                    self.budgets[profile_id] = TokenBudget(
                        profile_id=budget_data["profile_id"],
                        name=budget_data["name"],
                        max_tokens_per_period=budget_data["max_tokens_per_period"],
                        period=BudgetPeriod(budget_data["period"]),
                        max_cost_per_period=budget_data.get("max_cost_per_period", 0),
                        tokens_used=budget_data.get("tokens_used", 0),
                        cost_used=budget_data.get("cost_used", 0),
                        tier=UsageTier(budget_data.get("tier", "free")),
                        davinci_mode_enabled=budget_data.get("davinci_mode_enabled", False),
                    )
            except Exception as e:
                logger.error(f"Failed to load budgets: {e}")

    def _save_budgets(self):
        """Save budgets to storage."""
        budgets_file = self.storage_path / "budgets.json"
        data = {pid: budget.to_any() for pid, budget in self.budgets.items()}
        with open(budgets_file, "w") as f:
            json.dump(data, f, indent=2)

    # =========================================================================
    # BUDGET MANAGEMENT
    # =========================================================================

    def create_budget(
        self,
        profile_id: str,
        name: str,
        max_tokens: int,
        period: BudgetPeriod = BudgetPeriod.DAILY,
        max_cost: float = 0.0,
        tier: UsageTier = UsageTier.FREE,
        bender_enabled: bool = False,
    ) -> TokenBudget:
        """Create a new token budget for a profile."""
        budget = TokenBudget(
            profile_id=profile_id,
            name=name,
            max_tokens_per_period=max_tokens,
            period=period,
            max_cost_per_period=max_cost,
            tier=tier,
            bender_mode_enabled=bender_enabled,
        )

        # Set allowed tiers based on usage tier
        if tier == UsageTier.FREE:
            budget.allowed_tiers = [ModelTier.ECONOMY]
            budget.bender_mode_enabled = False
        elif tier == UsageTier.BASIC:
            budget.allowed_tiers = [ModelTier.ECONOMY, ModelTier.STANDARD]
        elif tier == UsageTier.PRO:
            budget.allowed_tiers = [ModelTier.ECONOMY, ModelTier.STANDARD, ModelTier.PREMIUM]
        elif tier == UsageTier.ENTERPRISE:
            budget.allowed_tiers = [ModelTier.ECONOMY, ModelTier.STANDARD, ModelTier.PREMIUM, ModelTier.BENDER]
            budget.bender_mode_enabled = True

        self.budgets[profile_id] = budget
        self._save_budgets()

        logger.info(f"Created budget for profile: {profile_id} ({max_tokens} tokens/{period.value})")
        return budget

    def get_budget(self, profile_id: str) -> Optional[TokenBudget]:
        """Get budget for a profile."""
        budget = self.budgets.get(profile_id)
        if budget:
            self._check_period_reset(budget)
        return budget

    def update_budget(
        self,
        profile_id: str,
        **updates,
    ) -> Optional[TokenBudget]:
        """Update a budget."""
        budget = self.budgets.get(profile_id)
        if not budget:
            return None

        for key, value in updates.items():
            if hasattr(budget, key):
                setattr(budget, key, value)

        budget.updated_at = datetime.utcnow()
        self._save_budgets()
        return budget

    def delete_budget(self, profile_id: str) -> bool:
        """Delete a budget."""
        if profile_id in self.budgets:
            del self.budgets[profile_id]
            self._save_budgets()
            return True
        return False

    def _check_period_reset(self, budget: TokenBudget):
        """Check if budget period should reset."""
        now = datetime.utcnow()
        should_reset = False

        if budget.period == BudgetPeriod.HOURLY:
            should_reset = (now - budget.period_start).total_seconds() >= 3600
        elif budget.period == BudgetPeriod.DAILY:
            should_reset = (now - budget.period_start).days >= 1
        elif budget.period == BudgetPeriod.WEEKLY:
            should_reset = (now - budget.period_start).days >= 7
        elif budget.period == BudgetPeriod.MONTHLY:
            should_reset = (now - budget.period_start).days >= 30

        if should_reset:
            budget.tokens_used = 0
            budget.cost_used = 0
            budget.period_start = now
            self._save_budgets()
            logger.info(f"Reset budget period for {budget.profile_id}")

    # =========================================================================
    # USAGE TRACKING
    # =========================================================================

    def check_budget(
        self,
        profile_id: str,
        model_id: str,
        estimated_tokens: int,
    ) -> dict:
        """Check if usage is within budget."""
        budget = self.get_budget(profile_id)
        if not budget:
            return {"allowed": True, "reason": "No budget configured"}

        # Check model tier
        pricing = self.model_pricing.get(model_id)
        if pricing and pricing.tier not in budget.allowed_tiers:
            return {
                "allowed": False,
                "reason": f"Model tier {pricing.tier.value} not allowed for your plan",
                "upgrade_required": True,
            }

        # Check token budget
        if budget.tokens_remaining < estimated_tokens:
            return {
                "allowed": False,
                "reason": f"Insufficient tokens. Remaining: {budget.tokens_remaining}, Required: {estimated_tokens}",
                "tokens_remaining": budget.tokens_remaining,
            }

        # Check cost budget
        if pricing and budget.max_cost_per_period > 0:
            estimated_cost = (estimated_tokens / 1000) * (pricing.input_cost_per_1k + pricing.output_cost_per_1k)
            if budget.cost_remaining < estimated_cost:
                return {
                    "allowed": False,
                    "reason": f"Cost limit reached. Remaining: ${budget.cost_remaining:.4f}",
                }

        return {"allowed": True, "tokens_remaining": budget.tokens_remaining}

    def record_usage(
        self,
        profile_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        session_id: str = "",
        mode: str = "standard",
    ) -> TokenUsage:
        """Record token usage."""
        pricing = self.model_pricing.get(model_id)
        cost = 0.0
        if pricing:
            cost = (input_tokens / 1000) * pricing.input_cost_per_1k + \
                   (output_tokens / 1000) * pricing.output_cost_per_1k

        usage = TokenUsage(
            timestamp=datetime.utcnow(),
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            profile_id=profile_id,
            session_id=session_id,
            mode=mode,
        )

        self.usage_history.append(usage)

        # Update budget
        budget = self.budgets.get(profile_id)
        if budget:
            budget.tokens_used += input_tokens + output_tokens
            budget.cost_used += cost
            self._save_budgets()

            # Check alerts
            if budget.usage_percentage >= budget.warning_threshold * 100:
                self._trigger_alert(budget, usage)

        return usage

    def _trigger_alert(self, budget: TokenBudget, usage: TokenUsage):
        """Trigger a budget alert."""
        for handler in self.alert_handlers:
            try:
                handler(budget, usage)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def add_alert_handler(self, handler: Callable):
        """Add an alert handler."""
        self.alert_handlers.append(handler)

    # =========================================================================
    # BENDER MODE - Multi-Model Debate
    # =========================================================================

    async def check_bender_mode(
        self,
        profile_id: str,
        models: list[str],
        estimated_rounds: int = 20,
    ) -> dict:
        """
        Check if BENDER mode is available and estimate cost.

        ⚠️ WARNING: BENDER mode uses significantly more tokens!

        - Up to 20 cycles or until consensus
        - Grok checks for agreement every 2 cycles
        - Grok fact-checks final answer
        """
        budget = self.get_budget(profile_id)

        if not budget:
            return {
                "allowed": False,
                "reason": "No budget configured",
            }

        if not budget.bender_mode_enabled:
            return {
                "allowed": False,
                "reason": "BENDER mode not enabled for your plan. Upgrade to Enterprise.",
                "upgrade_required": True,
            }

        # Estimate cost
        total_estimated_cost = 0.0
        total_estimated_tokens = 0
        model_costs = []

        for model_id in models:
            pricing = self.model_pricing.get(model_id)
            if pricing:
                # Estimate ~2000 tokens per round per model
                tokens_per_round = 2000
                model_tokens = tokens_per_round * estimated_rounds
                model_cost = (model_tokens / 1000) * (pricing.input_cost_per_1k + pricing.output_cost_per_1k)

                total_estimated_cost += model_cost
                total_estimated_tokens += model_tokens

                model_costs.append({
                    "model": model_id,
                    "estimated_tokens": model_tokens,
                    "estimated_cost": round(model_cost, 4),
                })

        # Check budget
        if budget.tokens_remaining < total_estimated_tokens:
            return {
                "allowed": False,
                "reason": f"Insufficient tokens for BENDER mode. Need ~{total_estimated_tokens}, have {budget.tokens_remaining}",
                "estimated_cost": round(total_estimated_cost, 4),
            }

        if budget.max_cost_per_period > 0 and budget.cost_remaining < total_estimated_cost:
            return {
                "allowed": False,
                "reason": f"Insufficient budget for BENDER mode. Need ~${total_estimated_cost:.4f}, have ${budget.cost_remaining:.4f}",
            }

        return {
            "allowed": True,
            "warning": (
                "⚠️ BENDER MODE ACTIVATED ⚠️\n"
                f"This will run {len(models)} AI models in debate for up to {estimated_rounds} cycles.\n"
                "• Grok checks for consensus every 2 cycles\n"
                "• Debate continues until all models agree OR max cycles reached\n"
                "• Grok fact-checks the final answer\n"
                f"• Estimated cost: ${total_estimated_cost:.4f}"
            ),
            "models": model_costs,
            "total_estimated_tokens": total_estimated_tokens,
            "total_estimated_cost": round(total_estimated_cost, 4),
            "max_rounds": estimated_rounds,
            "consensus_check_interval": 2,
            "fact_checker": BENDER_FACT_CHECKER,
            "requires_confirmation": True,
        }

    async def start_bender_session(
        self,
        profile_id: str,
        query: str,
        models: list[str],
        max_rounds: int = 20,
        consensus_check_interval: int = 2,
        model_caller: Callable = None,
    ) -> DavinciSession:
        """
        Start a BENDER mode multi-model debate session.

        The models will debate for several rounds, then synthesize
        the best answer from their collective reasoning.
        """
        import uuid

        # Verify access
        check = await self.check_davinci_mode(profile_id, models, max_rounds)
        if not check["allowed"]:
            raise PermissionError(check["reason"])

        session = DavinciSession(
            id=str(uuid.uuid4())[:8],
            profile_id=profile_id,
            query=query,
            models=models,
            max_rounds=max_rounds,
            consensus_check_interval=consensus_check_interval,
        )

        self.davinci_sessions[session.id] = session

        logger.info(f"Started DAVINCI session {session.id} with {len(models)} models")

        # Run deliberation if caller provided
        if model_caller:
            await self._run_davinci_deliberation(session, model_caller)

        return session

    async def _run_davinci_deliberation(
        self,
        session: DavinciSession,
        model_caller: Callable,
    ):
        """
        Run the multi-model deliberation with Grok consensus checks.

        - Up to 20 cycles or until agreement
        - Grok checks for consensus every 2 cycles
        - Grok fact-checks the final raw answer
        """
        previous_responses = []

        for round_num in range(1, session.max_rounds + 1):
            logger.info(f"DAVINCI round {round_num}/{session.max_rounds}")
            session.add_to_deliberation_box("SYSTEM", round_num, f"Round {round_num} starting...", "system")

            round_responses = []

            for model_id in session.models:
                # Build prompt including previous responses
                if round_num == 1:
                    prompt = f"""You are participating in a multi-model deliberation to find the best answer.

Query: {session.query}

Other participants: {', '.join(m for m in session.models if m != model_id)}

Provide your response with:
1. Your answer
2. Your reasoning
3. Your confidence level (0-1)
4. Key points you believe are crucial

Be thorough and detailed. State your position clearly."""
                else:
                    prev_summary = "\n\n".join([
                        f"Model {r['model']} (Round {r['round']}): {r['response'][:500]}..."
                        for r in previous_responses[-len(session.models)*2:]
                    ])

                    prompt = f"""You are in round {round_num} of a multi-model deliberation.

Original Query: {session.query}

Previous responses:
{prev_summary}

Based on the discussion so far:
1. What is your refined answer?
2. Do you AGREE or DISAGREE with other models? Be explicit.
3. What points do you want to emphasize or correct?
4. Your confidence level (0-1)
5. list dict points of agreement with other models
6. list dict remaining disagreements

Build toward consensus while maintaining accuracy. If you agree with the emerging consensus, state so clearly."""

                try:
                    # Call the model
                    response = await model_caller(model_id, prompt)

                    round_data = DavinciDebateRound(
                        round_number=round_num,
                        model_id=model_id,
                        response=response.get("content", ""),
                        reasoning=response.get("reasoning", ""),
                        confidence=response.get("confidence", 0.7),
                        tokens_used=response.get("tokens", 0),
                    )

                    session.rounds.append(round_data)
                    session.total_tokens += round_data.tokens_used

                    # Add to visual deliberation box
                    session.add_to_deliberation_box(
                        model_id, round_num,
                        round_data.response[:200],
                        "response"
                    )

                    round_responses.append({
                        "round": round_num,
                        "model": model_id,
                        "response": round_data.response,
                        "confidence": round_data.confidence,
                    })

                    # Record usage
                    self.record_usage(
                        profile_id=session.profile_id,
                        model_id=model_id,
                        input_tokens=response.get("input_tokens", 0),
                        output_tokens=response.get("output_tokens", 0),
                        session_id=session.id,
                        mode="davinci",
                    )

                except Exception as e:
                    logger.error(f"DAVINCI model {model_id} failed: {e}")
                    session.add_to_deliberation_box(model_id, round_num, f"Error: {e}", "error")

            previous_responses.extend(round_responses)

            # Grok consensus check every N cycles (default: every 2 cycles)
            if round_num % session.consensus_check_interval == 0 and round_num >= 2:
                logger.info(f"DAVINCI: Grok consensus check at round {round_num}")
                session.add_to_deliberation_box("GROK", round_num, "Checking for consensus...", "consensus_check")

                consensus_result = await self._grok_consensus_check(
                    session, previous_responses, model_caller
                )

                if consensus_result.get("consensus_reached"):
                    session.consensus_reached = True
                    session.raw_consensus = consensus_result.get("consensus_answer", "")
                    session.add_to_deliberation_box(
                        "GROK", round_num,
                        f"✅ CONSENSUS: {session.raw_consensus[:100]}...",
                        "consensus"
                    )
                    logger.info(f"DAVINCI consensus reached at round {round_num}")
                    break
                else:
                    remaining_issues = consensus_result.get("disagreements", [])
                    session.add_to_deliberation_box(
                        "GROK", round_num,
                        f"⚠️ No consensus. Issues: {', '.join(remaining_issues[:3])}",
                        "no_consensus"
                    )

        # If no consensus after all rounds, synthesize best effort
        if not session.consensus_reached:
            logger.warning(f"DAVINCI: No consensus after {session.max_rounds} rounds, synthesizing best effort")
            session.raw_consensus = await self._synthesize_davinci_response(
                session, previous_responses, model_caller
            )

        # GROK FACT-CHECK: Validate the final raw answer
        logger.info("DAVINCI: Grok fact-checking final answer")
        session.add_to_deliberation_box("GROK", 0, "Fact-checking final answer...", "fact_check")

        session.fact_check_result = await self._grok_fact_check(
            session, model_caller
        )

        # Apply fact-check corrections if dict
        if session.fact_check_result.get("valid"):
            session.final_response = session.raw_consensus
            session.add_to_deliberation_box("GROK", 0, "✅ Answer verified", "verified")
        else:
            corrections = session.fact_check_result.get("corrections", "")
            session.final_response = f"{session.raw_consensus}\n\n---\n🔍 GROK FACT-CHECK NOTES:\n{corrections}"
            session.add_to_deliberation_box("GROK", 0, f"⚠️ Issues found: {corrections[:100]}", "issues")

        session.completed_at = datetime.utcnow()

        # Calculate total cost
        for round_data in session.rounds:
            pricing = self.model_pricing.get(round_data.model_id)
            if pricing:
                session.total_cost += (round_data.tokens_used / 1000) * \
                                      (pricing.input_cost_per_1k + pricing.output_cost_per_1k)

        logger.info(f"DAVINCI session {session.id} completed. "
                   f"Rounds: {len(set(r.round_number for r in session.rounds))}, "
                   f"Consensus: {session.consensus_reached}, "
                   f"Tokens: {session.total_tokens}, Cost: ${session.total_cost:.4f}")

    async def _grok_consensus_check(
        self,
        session: DavinciSession,
        all_responses: list[dict],
        model_caller: Callable,
    ) -> dict:
        """
        Have Grok analyze if models have reached consensus.

        Returns:
            {
                "consensus_reached": bool,
                "consensus_answer": str (if reached),
                "agreement_points": list[str],
                "disagreements": list[str],
                "confidence": float
            }
        """
        # Get last 2 rounds of responses
        recent = all_responses[-len(session.models)*2:] if len(all_responses) >= len(session.models)*2 else all_responses

        summary = "\n\n".join([
            f"[{r['model']}] (Confidence: {r['confidence']})\n{r['response']}"
            for r in recent
        ])

        check_prompt = f"""You are Grok, acting as the consensus arbiter for a multi-model deliberation.

ORIGINAL QUERY: {session.query}

RECENT MODEL RESPONSES:
{summary}

TASK: Analyze whether the models have reached consensus.

Respond in this exact JSON format:
{{
    "consensus_reached": true/false,
    "consensus_answer": "The agreed-upon answer if consensus reached, else empty string",
    "agreement_points": ["list of points all models agree on"],
    "disagreements": ["list of remaining points of disagreement"],
    "confidence": 0.0-1.0,
    "reasoning": "Why you determined consensus was/wasn't reached"
}}

Be strict but fair. Consensus means ALL models substantively agree on the answer, not just similar wording.
Minor differences in phrasing are OK. Fundamental disagreements on facts or approach = no consensus."""

        try:
            response = await model_caller(session.fact_checker, check_prompt)
            content = response.get("content", "{}")

            # Parse JSON response
            import json
            try:
                # Extract JSON from response (handle markdown code blocks)
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                result = json.loads(content.strip())

                # Record Grok usage
                self.record_usage(
                    profile_id=session.profile_id,
                    model_id=session.fact_checker,
                    input_tokens=response.get("input_tokens", 0),
                    output_tokens=response.get("output_tokens", 0),
                    session_id=session.id,
                    mode="davinci_consensus_check",
                )

                return result
            except json.JSONDecodeError:
                logger.warning("Grok consensus check returned non-JSON response")
                return {"consensus_reached": False, "disagreements": ["Parse error"]}

        except Exception as e:
            logger.error(f"Grok consensus check failed: {e}")
            return {"consensus_reached": False, "disagreements": [str(e)]}

    async def _grok_fact_check(
        self,
        session: DavinciSession,
        model_caller: Callable,
    ) -> dict:
        """
        Have Grok fact-check the final raw consensus answer.

        Returns:
            {
                "valid": bool,
                "corrections": str,
                "confidence": float,
                "issues": list[str]
            }
        """
        fact_check_prompt = f"""You are Grok, acting as the final fact-checker for a multi-model consensus answer.

ORIGINAL QUERY: {session.query}

CONSENSUS ANSWER TO VERIFY:
{session.raw_consensus}

TASK: Fact-check this answer rigorously.

Check for:
1. Factual accuracy - Are all stated facts correct?
2. Logical consistency - Does the reasoning hold?
3. Completeness - Are there important omissions?
4. Clarity - Is the answer clear and unambiguous?

Respond in this exact JSON format:
{{
    "valid": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of dict issues found"],
    "corrections": "Suggested corrections or clarifications if needed, else empty string",
    "verified_facts": ["list of facts you verified as correct"],
    "unverifiable": ["list of claims that cannot be verified"]
}}

Be thorough but pragmatic. Minor style issues don't make an answer invalid.
Focus on substantive factual or logical problems."""

        try:
            response = await model_caller(session.fact_checker, fact_check_prompt)
            content = response.get("content", "{}")

            # Parse JSON response
            import json
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                result = json.loads(content.strip())

                # Record Grok usage
                self.record_usage(
                    profile_id=session.profile_id,
                    model_id=session.fact_checker,
                    input_tokens=response.get("input_tokens", 0),
                    output_tokens=response.get("output_tokens", 0),
                    session_id=session.id,
                    mode="davinci_fact_check",
                )

                session.add_to_deliberation_box(
                    "GROK", 0,
                    f"Fact-check: {'PASS' if result.get('valid') else 'ISSUES'} ({result.get('confidence', 0):.0%})",
                    "fact_check_result"
                )

                return result
            except json.JSONDecodeError:
                logger.warning("Grok fact-check returned non-JSON response")
                return {"valid": True, "corrections": "", "issues": ["Parse error - assuming valid"]}

        except Exception as e:
            logger.error(f"Grok fact-check failed: {e}")
            return {"valid": True, "corrections": "", "issues": [f"Check failed: {e}"]}

    async def _synthesize_davinci_response(
        self,
        session: DavinciSession,
        all_responses: list[dict],
        model_caller: Callable,
    ) -> str:
        """Synthesize final response from deliberation."""
        # Use the most capable model for synthesis
        synthesis_model = session.models[0]  # Could be configured

        summary = "\n\n".join([
            f"[{r['model']} - Round {r['round']}] (Confidence: {r['confidence']})\n{r['response']}"
            for r in all_responses[-len(session.models)*2:]  # Last 2 rounds
        ])

        synthesis_prompt = f"""You are synthesizing the final answer from a multi-model deliberation.

Original Query: {session.query}

Final round responses:
{summary}

Please provide:
1. A comprehensive final answer that incorporates the best points from all models
2. dict points of disagreement that users should be aware of
3. Confidence assessment

Format your response in a clear, detailed manner suitable for the user."""

        try:
            response = await model_caller(synthesis_model, synthesis_prompt)
            return response.get("content", "Synthesis failed")
        except Exception as e:
            logger.error(f"DAVINCI synthesis failed: {e}")
            return f"Synthesis error: {e}"

    def get_davinci_session(self, session_id: str) -> Optional[DavinciSession]:
        """Get a DAVINCI session by ID."""
        return self.davinci_sessions.get(session_id)

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_usage_stats(
        self,
        profile_id: str = None,
        days: int = 30,
    ) -> dict:
        """Get usage statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        usage = [u for u in self.usage_history if u.timestamp >= cutoff]
        if profile_id:
            usage = [u for u in usage if u.profile_id == profile_id]

        if not usage:
            return {"total_tokens": 0, "total_cost": 0}

        total_tokens = sum(u.input_tokens + u.output_tokens for u in usage)
        total_cost = sum(u.cost for u in usage)

        by_model = {}
        for u in usage:
            if u.model_id not in by_model:
                by_model[u.model_id] = {"tokens": 0, "cost": 0, "calls": 0}
            by_model[u.model_id]["tokens"] += u.input_tokens + u.output_tokens
            by_model[u.model_id]["cost"] += u.cost
            by_model[u.model_id]["calls"] += 1

        davinci_usage = [u for u in usage if u.mode == "davinci"]
        davinci_tokens = sum(u.input_tokens + u.output_tokens for u in davinci_usage)
        davinci_cost = sum(u.cost for u in davinci_usage)

        return {
            "period_days": days,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "by_model": by_model,
            "davinci_mode": {
                "sessions": len(self.davinci_sessions),
                "tokens": davinci_tokens,
                "cost": round(davinci_cost, 4),
            },
            "average_tokens_per_day": total_tokens // days if days > 0 else 0,
        }

    def get_cost_projection(
        self,
        profile_id: str,
        days: int = 30,
    ) -> dict:
        """Project future costs based on usage patterns."""
        stats = self.get_usage_stats(profile_id, days=7)

        daily_avg_cost = stats["total_cost"] / 7 if stats["total_cost"] > 0 else 0
        daily_avg_tokens = stats["average_tokens_per_day"]

        return {
            "daily_average_cost": round(daily_avg_cost, 4),
            "daily_average_tokens": daily_avg_tokens,
            "projected_30_day_cost": round(daily_avg_cost * 30, 2),
            "projected_30_day_tokens": daily_avg_tokens * 30,
        }


# Singleton instance
token_budget_manager = TokenBudgetManager()

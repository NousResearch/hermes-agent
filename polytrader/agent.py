from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .execution import ClobV2ExecutionClient, build_clob_v2_client
from .models import ExecutionReceipt, MarketMetadata, OrderBookQuote, TradeDecision
from .risk import RiskLimits, check_risk
from .strategy import evaluate_buy


@dataclass(frozen=True)
class CycleInputs:
    market: MarketMetadata
    quote: OrderBookQuote
    model_probability: float
    collateral_balance: float
    open_positions: int
    strategy: str = "FORECAST"


class PolyTraderAgent:
    """One-cycle orchestration core; data fetching stays outside for testability."""

    def __init__(self, settings: Settings, executor: ClobV2ExecutionClient | None = None) -> None:
        self.settings = settings
        self.executor = executor

    def evaluate(self, inputs: CycleInputs) -> TradeDecision:
        requested = min(self.settings.max_collateral_per_trade, max(0.0, self.settings.max_collateral_per_trade))
        decision = evaluate_buy(
            inputs.strategy,
            inputs.market,
            inputs.quote,
            model_probability=inputs.model_probability,
            collateral_size=requested,
            min_edge=self.settings.min_edge,
        )
        if decision.action != "BUY":
            return decision
        risk = check_risk(
            inputs.collateral_balance,
            decision.collateral_size,
            open_positions=inputs.open_positions,
            limits=RiskLimits(
                max_collateral_per_trade=self.settings.max_collateral_per_trade,
                min_collateral_balance=self.settings.min_collateral_balance,
                max_open_positions=self.settings.max_open_positions,
            ),
        )
        if not risk.allowed:
            return TradeDecision(
                strategy=decision.strategy,
                action="SKIP",
                token_id=decision.token_id,
                side=decision.side,
                price=decision.price,
                collateral_size=0.0,
                edge_after_fees=decision.edge_after_fees,
                reason=risk.reason,
                metadata=decision.metadata,
            )
        return decision

    def execute(self, decision: TradeDecision, market: MarketMetadata) -> ExecutionReceipt:
        executor = self.executor
        if executor is None:
            client = None if self.settings.dry_run else build_clob_v2_client(self.settings)
            executor = ClobV2ExecutionClient(
                client=client,
                dry_run=self.settings.dry_run,
                order_type=self.settings.order_type,
                post_only=self.settings.post_only,
            )
        return executor.place_order(decision, market)

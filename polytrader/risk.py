from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_collateral_per_trade: float
    min_collateral_balance: float
    max_open_positions: int


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str


def check_risk(
    collateral_balance: float,
    requested_collateral: float,
    *,
    open_positions: int,
    limits: RiskLimits,
) -> RiskDecision:
    if collateral_balance < limits.min_collateral_balance:
        return RiskDecision(False, "collateral balance floor not met")
    if requested_collateral <= 0:
        return RiskDecision(False, "requested collateral must be positive")
    if requested_collateral > limits.max_collateral_per_trade:
        return RiskDecision(False, "requested collateral exceeds MAX_COLLATERAL_PER_TRADE")
    if collateral_balance - requested_collateral < limits.min_collateral_balance:
        return RiskDecision(False, "trade would breach collateral balance floor")
    if open_positions >= limits.max_open_positions:
        return RiskDecision(False, "open positions at MAX_OPEN_POSITIONS")
    return RiskDecision(True, "risk checks passed")

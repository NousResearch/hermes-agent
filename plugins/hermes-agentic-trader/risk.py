"""Fail-closed deterministic live-risk evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .mandate import Mandate, validate_mandate
from .policy import TraderPolicy
from .quote_cache import QuoteRecord


class RiskReason(str, Enum):
    PAPER_MODE = "PAPER_MODE"
    INVALID_QUOTE = "INVALID_QUOTE"
    CHAIN_DENIED = "CHAIN_DENIED"
    SLIPPAGE = "SLIPPAGE"
    MANDATE_INVALID = "MANDATE_INVALID"
    ROLLOUT_CAP = "ROLLOUT_CAP"
    NOTIONAL_UNAVAILABLE = "NOTIONAL_UNAVAILABLE"
    PORTFOLIO_UNAVAILABLE = "PORTFOLIO_UNAVAILABLE"
    DAILY_PNL_UNAVAILABLE = "DAILY_PNL_UNAVAILABLE"
    LIQUIDITY_UNAVAILABLE = "LIQUIDITY_UNAVAILABLE"
    POSITION_LIMIT = "POSITION_LIMIT"
    DAILY_LIMIT = "DAILY_LIMIT"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"


@dataclass(frozen=True)
class QuoteRiskContext:
    chain_id: int
    sell_token: str
    buy_token: str
    sell_amount_base_units: int
    slippage_bps: int


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: Optional[RiskReason]
    message: str


def _is_evm_address(value: object) -> bool:
    text = str(value or "")
    if len(text) != 42 or not text.startswith("0x"):
        return False
    try:
        int(text[2:], 16)
    except ValueError:
        return False
    return True


def quote_context(record: QuoteRecord) -> QuoteRiskContext:
    args = record.request_args

    try:
        chain_id = int(args["chainId"])
        sell_amount = int(str(args["sellAmount"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("quote request missing valid chainId/sellAmount") from exc

    sell_token = str(args.get("sellToken") or "").lower()
    buy_token = str(args.get("buyToken") or "").lower()
    if chain_id <= 0:
        raise ValueError("quote chainId must be positive")
    if sell_amount <= 0:
        raise ValueError("quote sellAmount must be positive")
    if not _is_evm_address(sell_token) or not _is_evm_address(buy_token):
        raise ValueError("quote buyToken/sellToken must be EVM addresses")

    raw_slippage = args.get("slippageBps", 100)
    try:
        slippage = int(raw_slippage)
    except (TypeError, ValueError) as exc:
        raise ValueError("quote slippageBps must be an integer") from exc
    if slippage <= 0:
        raise ValueError("quote slippageBps must be positive")

    return QuoteRiskContext(
        chain_id=chain_id,
        sell_token=sell_token,
        buy_token=buy_token,
        sell_amount_base_units=sell_amount,
        slippage_bps=slippage,
    )


def evaluate_bound_quote(
    *,
    policy: TraderPolicy,
    record: QuoteRecord,
    mandate: Optional[Mandate],
    notional_usd: Optional[float],
    portfolio_value_usd: Optional[float],
    daily_loss_pct: Optional[float],
    pool_liquidity_usd: Optional[float],
    min_pool_liquidity_usd: float = 100_000.0,
) -> RiskDecision:
    if policy.mode != "live":
        return RiskDecision(False, RiskReason.PAPER_MODE, "paper mode blocks execution")

    try:
        quote = quote_context(record)
    except ValueError as exc:
        return RiskDecision(False, RiskReason.INVALID_QUOTE, str(exc))

    if quote.chain_id not in policy.allowed_chain_ids:
        return RiskDecision(
            False,
            RiskReason.CHAIN_DENIED,
            f"chainId {quote.chain_id} is not allowed by trader policy",
        )

    if quote.slippage_bps > policy.max_slippage_bps:
        return RiskDecision(
            False,
            RiskReason.SLIPPAGE,
            f"quote slippage {quote.slippage_bps} bps exceeds policy maximum",
        )

    if mandate is None:
        return RiskDecision(
            False,
            RiskReason.MANDATE_INVALID,
            "signed live mandate is missing",
        )

    ok, error = validate_mandate(mandate)
    if not ok:
        return RiskDecision(False, RiskReason.MANDATE_INVALID, error)

    if quote.chain_id not in mandate.allowed_chain_ids:
        return RiskDecision(
            False,
            RiskReason.CHAIN_DENIED,
            f"chainId {quote.chain_id} is not allowed by mandate",
        )

    effective_slippage = min(
        policy.max_slippage_bps,
        mandate.max_slippage_bps,
    )
    if quote.slippage_bps > effective_slippage:
        return RiskDecision(
            False,
            RiskReason.SLIPPAGE,
            f"quote slippage {quote.slippage_bps} bps exceeds signed mandate",
        )

    caps = [mandate.max_trade_usd]
    if policy.rollout_cap_usd is not None:
        caps.append(policy.rollout_cap_usd)
    effective_cap = min(caps)
    if effective_cap <= 0:
        return RiskDecision(
            False,
            RiskReason.ROLLOUT_CAP,
            "no positive live rollout cap is available",
        )

    if notional_usd is None:
        return RiskDecision(
            False,
            RiskReason.NOTIONAL_UNAVAILABLE,
            "USD notional cannot be established from the bound quote yet",
        )
    if notional_usd <= 0:
        return RiskDecision(
            False,
            RiskReason.INVALID_QUOTE,
            "USD notional must be positive",
        )
    if notional_usd > effective_cap:
        return RiskDecision(
            False,
            RiskReason.ROLLOUT_CAP,
            f"trade notional exceeds effective live cap USD {effective_cap:,.2f}",
        )

    if portfolio_value_usd is None or portfolio_value_usd <= 0:
        return RiskDecision(
            False,
            RiskReason.PORTFOLIO_UNAVAILABLE,
            "fresh portfolio value is required for live execution",
        )
    position_cap = portfolio_value_usd * (policy.max_position_pct / 100.0)
    if notional_usd > position_cap:
        return RiskDecision(
            False,
            RiskReason.POSITION_LIMIT,
            f"trade exceeds {policy.max_position_pct:.2f}% portfolio limit",
        )

    if daily_loss_pct is None:
        return RiskDecision(
            False,
            RiskReason.DAILY_PNL_UNAVAILABLE,
            "fresh daily P&L is required for live execution",
        )
    if daily_loss_pct >= policy.max_daily_loss_pct:
        return RiskDecision(
            False,
            RiskReason.DAILY_LIMIT,
            "daily loss limit reached",
        )

    if pool_liquidity_usd is None:
        return RiskDecision(
            False,
            RiskReason.LIQUIDITY_UNAVAILABLE,
            "fresh pool liquidity is required for live execution",
        )
    if pool_liquidity_usd < min_pool_liquidity_usd:
        return RiskDecision(
            False,
            RiskReason.LOW_LIQUIDITY,
            "pool liquidity is below configured minimum",
        )

    return RiskDecision(True, None, "APPROVE")

"""Deterministic pre-trade risk gate — not prompt-injectable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from hermes_trader.config import TraderConfig, TRADER_HOME_SUBDIR
from hermes_trader.market_state import MarketState
from hermes_trader.risk.mandate import Mandate, default_mandate_path, load_mandate, validate_mandate
from hermes_trader.risk.rollout import chain_allowed_by_rollout, trade_within_rollout_cap
from hermes_trader.risk.size_modifier import apply_size_multiplier, compute_size_multiplier

Action = Literal["buy", "sell", "hold", "rebalance", "watch"]
EXECUTABLE_ACTIONS = frozenset({"buy", "sell", "rebalance"})


class RejectReason(str, Enum):
    KILL_SWITCH = "KILL_SWITCH"
    PAPER_MODE = "PAPER_MODE"
    CHAIN_DENIED = "CHAIN_DENIED"
    OVERSIZE = "OVERSIZE"
    DAILY_LIMIT = "DAILY_LIMIT"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    SLIPPAGE = "SLIPPAGE"
    NO_MANDATE = "NO_MANDATE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    HOLD = "HOLD"
    INVALID_INTENT = "INVALID_INTENT"
    ROLLOUT_CHAIN = "ROLLOUT_CHAIN"
    ROLLOUT_CAP = "ROLLOUT_CAP"


@dataclass
class TradeIntent:
    action: str
    chain: str
    token_address: str
    size_usd: float
    confidence: float
    reasoning: str = ""
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    strategy_tag: Optional[str] = None
    pool_liquidity_usd: Optional[float] = None
    slippage_bps: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TradeIntent":
        return cls(
            action=str(data.get("action", "hold")).strip().lower(),
            chain=str(data.get("chain", "")).strip().lower(),
            token_address=str(data.get("token_address", "")).strip(),
            size_usd=float(data.get("size_usd", 0) or 0),
            confidence=float(data.get("confidence", 0) or 0),
            reasoning=str(data.get("reasoning", "")),
            stop_loss_pct=_optional_float(data.get("stop_loss_pct")),
            take_profit_pct=_optional_float(data.get("take_profit_pct")),
            strategy_tag=_optional_str(data.get("strategy_tag")),
            pool_liquidity_usd=_optional_float(data.get("pool_liquidity_usd")),
            slippage_bps=_optional_int(data.get("slippage_bps")),
        )


@dataclass(frozen=True)
class OrderRequest:
    chain: str
    token_address: str
    action: str
    size_usd: float
    max_slippage_bps: int
    tool: str
    reasoning: str
    strategy_tag: Optional[str] = None


@dataclass(frozen=True)
class GateDecision:
    approved: bool
    reason_code: Optional[RejectReason]
    message: str
    order: Optional[OrderRequest] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reason_code": self.reason_code.value if self.reason_code else None,
            "message": self.message,
            "order": (
                {
                    "chain": self.order.chain,
                    "token_address": self.order.token_address,
                    "action": self.order.action,
                    "size_usd": self.order.size_usd,
                    "max_slippage_bps": self.order.max_slippage_bps,
                    "tool": self.order.tool,
                    "reasoning": self.order.reasoning,
                    "strategy_tag": self.order.strategy_tag,
                }
                if self.order
                else None
            ),
        }


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def is_kill_switch_active() -> bool:
    """Env var or file sentinel halts all write tools."""
    env = os.environ.get("HERMES_TRADER_KILL_SWITCH", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    sentinel = _hermes_home() / TRADER_HOME_SUBDIR / "KILL_SWITCH"
    return sentinel.is_file()


def portfolio_value_usd(state: MarketState) -> float:
    total = 0.0
    for token in state.portfolio_tokens:
        if token.balance_usd is not None:
            total += token.balance_usd
    return total


@dataclass
class RiskGate:
    """Evaluate TradeIntent against deterministic rules."""

    config: TraderConfig = field(default_factory=TraderConfig)

    def evaluate(
        self,
        intent: TradeIntent,
        *,
        market_state: Optional[MarketState] = None,
        mandate: Optional[Mandate] = None,
        mandate_path: Optional[Path] = None,
        daily_loss_pct: float = 0.0,
        portfolio_value_usd_override: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> GateDecision:
        if not intent.chain:
            return self._reject(RejectReason.INVALID_INTENT, "TradeIntent missing chain")
        if not intent.token_address and intent.action in EXECUTABLE_ACTIONS:
            return self._reject(RejectReason.INVALID_INTENT, "TradeIntent missing token_address")

        action = intent.action
        if action not in EXECUTABLE_ACTIONS:
            return self._reject(
                RejectReason.HOLD,
                f"Non-executable action '{action}' — no order placed",
            )

        if is_kill_switch_active():
            return self._reject(
                RejectReason.KILL_SWITCH,
                "HERMES_TRADER_KILL_SWITCH is active — all execution halted",
            )

        if self.config.mode == "paper":
            return self._reject(
                RejectReason.PAPER_MODE,
                "Paper mode blocks live execution — set mode: live and sign mandate.json",
            )

        if intent.chain not in self.config.allowed_chains:
            return self._reject(
                RejectReason.CHAIN_DENIED,
                f"Chain '{intent.chain}' not in allowed_chains {self.config.allowed_chains}",
            )

        if not chain_allowed_by_rollout(intent.chain, self.config):
            return self._reject(
                RejectReason.ROLLOUT_CHAIN,
                (
                    f"Chain '{intent.chain}' not allowed in rollout stage "
                    f"'{self.config.rollout_stage}'"
                ),
            )

        if intent.confidence < self.config.min_confidence:
            return self._reject(
                RejectReason.LOW_CONFIDENCE,
                (
                    f"Confidence {intent.confidence:.2f} below "
                    f"min_confidence {self.config.min_confidence:.2f}"
                ),
            )

        if daily_loss_pct >= self.config.max_daily_loss_pct:
            return self._reject(
                RejectReason.DAILY_LIMIT,
                (
                    f"Daily loss {daily_loss_pct:.2f}% exceeds "
                    f"max_daily_loss_pct {self.config.max_daily_loss_pct:.2f}%"
                ),
            )

        if intent.pool_liquidity_usd is not None:
            if intent.pool_liquidity_usd < self.config.min_pool_liquidity_usd:
                return self._reject(
                    RejectReason.LOW_LIQUIDITY,
                    (
                        f"Pool liquidity ${intent.pool_liquidity_usd:,.0f} below "
                        f"min_pool_liquidity_usd ${self.config.min_pool_liquidity_usd:,.0f}"
                    ),
                )

        if intent.slippage_bps is not None:
            if intent.slippage_bps > self.config.max_slippage_bps:
                return self._reject(
                    RejectReason.SLIPPAGE,
                    (
                        f"Quoted slippage {intent.slippage_bps} bps exceeds "
                        f"max_slippage_bps {self.config.max_slippage_bps}"
                    ),
                )

        effective_size = intent.size_usd
        if self.config.enable_size_modifier and effective_size > 0:
            multiplier = compute_size_multiplier(intent, market_state=market_state)
            effective_size = apply_size_multiplier(effective_size, multiplier)

        ok_cap, cap_msg = trade_within_rollout_cap(effective_size, self.config)
        if not ok_cap:
            return self._reject(RejectReason.ROLLOUT_CAP, cap_msg)

        pv = portfolio_value_usd_override
        if pv is None and market_state is not None:
            pv = portfolio_value_usd(market_state)
        if pv is not None and pv > 0 and effective_size > 0:
            cap = (self.config.max_position_pct / 100.0) * pv
            if effective_size > cap:
                return self._reject(
                    RejectReason.OVERSIZE,
                    (
                        f"size_usd ${effective_size:,.2f} exceeds "
                        f"{self.config.max_position_pct:.1f}% of portfolio "
                        f"(${cap:,.2f})"
                    ),
                )

        mandate_obj = mandate
        if mandate_obj is None:
            mandate_obj = load_mandate(mandate_path or default_mandate_path())
        if mandate_obj is None:
            return self._reject(
                RejectReason.NO_MANDATE,
                "mandate.json missing — sign mandate before live execution",
            )
        ok, err = validate_mandate(mandate_obj, now=now)
        if not ok:
            return self._reject(RejectReason.NO_MANDATE, f"mandate invalid: {err}")

        tool = "submit_gasless_swap" if intent.chain == "base" else "execute_swap"
        order = OrderRequest(
            chain=intent.chain,
            token_address=intent.token_address,
            action=action,
            size_usd=effective_size,
            max_slippage_bps=self.config.max_slippage_bps,
            tool=tool,
            reasoning=intent.reasoning,
            strategy_tag=intent.strategy_tag,
        )
        return GateDecision(
            approved=True,
            reason_code=None,
            message="APPROVE",
            order=order,
        )

    @staticmethod
    def _reject(reason: RejectReason, message: str) -> GateDecision:
        return GateDecision(approved=False, reason_code=reason, message=message)
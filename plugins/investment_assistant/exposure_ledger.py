"""Normalize complex brokerage holdings into an exposure ledger."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .drift import extract_target_positions, selected_map_id
from .storage import new_id, utc_now


OPTION_CODE_RE = re.compile(
    r"^(?P<market>[A-Z]+)\.(?P<underlying>[A-Z0-9]+)"
    r"(?P<expiry>\d{6})(?P<option_type>[CP])(?P<strike>\d+)$"
)
OPTION_MULTIPLIER = 100


DEFAULT_LEVERAGED_ETF_MAP: dict[str, tuple[str, float]] = {
    "US.SNXX": ("US.SNDK", 2.0),
    "US.MVLL": ("US.MRVL", 2.0),
    "US.LITX": ("US.LITE", 2.0),
    "US.COHX": ("US.COHR", 2.0),
    "US.GEVX": ("US.GEV", 2.0),
    "US.BEX": ("US.BE", 2.0),
    "US.ARMG": ("US.ARM", 2.0),
    "US.KORU": ("KR.EQUITY_BASKET", 3.0),
}

DEFAULT_ETF_SYMBOLS = {
    "US.QQQ",
    "US.SOXX",
    "US.SMH",
    "US.DRAM",
    "US.KORU",
}

DEFAULT_SYMBOL_SLEEVE_HINTS = {
    "US.NVDA": "compute_accelerator",
    "US.AMD": "compute_accelerator",
    "US.AVGO": "custom_silicon_networking",
    "US.MRVL": "custom_silicon_networking",
    "US.MU": "memory_storage",
    "US.SNDK": "memory_storage",
    "US.WDC": "memory_storage",
    "US.DRAM": "memory_storage",
    "US.LITE": "optical_networking",
    "US.COHR": "optical_networking",
    "US.CIEN": "optical_networking",
    "US.ANET": "networking",
    "US.TSM": "foundry_equipment",
    "US.ASML": "foundry_equipment",
    "US.ETN": "power_cooling",
    "US.VRT": "power_cooling",
    "US.GEV": "power_generation",
    "US.VST": "power_generation",
    "US.QQQ": "core_beta",
    "US.SOXX": "semiconductor_etf",
    "US.SMH": "semiconductor_etf",
}


InstrumentType = Literal["stock", "etf", "leveraged_etf", "option_leg", "unknown"]
Direction = Literal["long", "short", "flat"]
OptionType = Literal["call", "put"]
OptionSide = Literal["long", "short"]
IntentConfidence = Literal["low", "medium", "high"]


class ExposureLedgerPreferences(BaseModel):
    """Configuration for deterministic exposure normalization."""

    leveraged_etf_map: dict[str, tuple[str, float]] = Field(
        default_factory=lambda: dict(DEFAULT_LEVERAGED_ETF_MAP)
    )
    etf_symbols: set[str] = Field(default_factory=lambda: set(DEFAULT_ETF_SYMBOLS))
    symbol_sleeve_hints: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_SYMBOL_SLEEVE_HINTS))
    include_default_sleeve_hints: bool = True


class ExposureIntentPolicy(BaseModel):
    """User-confirmed intent policy applied after deterministic structure parsing."""

    defined_risk_credit_spread_intent: Literal[
        "ask",
        "independent_high_iv_premium",
        "assignment_transition_plan",
        "tactical_option_trade",
    ] = "ask"
    long_call_intent: Literal[
        "ask",
        "long_term_capital_efficiency",
        "tactical_directional_trade",
        "hedge_or_combo_leg",
    ] = "ask"
    short_call_intent: Literal[
        "ask",
        "covered_call_income",
        "premium_income",
        "tactical_bearish_trade",
    ] = "ask"
    short_put_intent: Literal[
        "ask",
        "premium_income_willing_assignment",
        "premium_income_no_assignment",
        "tactical_bullish_trade",
    ] = "ask"
    leveraged_etf_intent: Literal[
        "ask",
        "long_term_leveraged_exposure_limited_rebalance",
        "tactical_trade",
        "separate_overlay_budget",
    ] = "ask"
    negative_cash_policy: Literal[
        "ask",
        "temporary_allowed_strict_risk",
        "reduce_margin_first",
        "active_margin_efficiency",
    ] = "ask"
    ignored_markets: set[str] = Field(default_factory=set)


class OptionLeg(BaseModel):
    raw_code: str
    name: str = ""
    underlying: str
    expiry: str
    option_type: OptionType
    side: OptionSide
    strike: float
    quantity: float
    contract_count: float = Field(ge=0)
    multiplier: int = OPTION_MULTIPLIER
    cost_price: float | None = None
    market_value: float = 0
    option_last_price: float | None = None
    underlying_last_price: float | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    rho: float | None = None
    opening_cashflow: float | None = None
    unrealized_pnl: float | None = None


class OptionStrategyGroup(BaseModel):
    strategy_id: str = Field(default_factory=lambda: new_id("osg"))
    underlying: str
    expiry: str
    option_type: OptionType | Literal["mixed"]
    strategy_type: str
    direction: Literal["bullish", "bearish", "neutral", "mixed", "unknown"] = "unknown"
    legs: list[OptionLeg] = Field(default_factory=list)
    contract_count: float = Field(ge=0)
    market_value: float = 0
    net_opening_credit: float | None = None
    spread_width: float | None = None
    defined_risk: bool = False
    max_profit: float | None = None
    max_loss: float | None = None
    short_assignment_notional: float = 0
    effective_long_exposure: float = 0
    effective_short_exposure: float = 0
    coverage_status: Literal["covered", "partially_covered", "uncovered", "not_applicable"] = "not_applicable"
    underlying_share_quantity: float = 0
    required_underlying_share_quantity: float = 0
    intent_guess: str = ""
    intent_confidence: IntentConfidence = "low"
    needs_human_clarification: bool = True
    risk_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class NormalizedExposurePosition(BaseModel):
    raw_code: str
    name: str = ""
    instrument_type: InstrumentType
    underlying: str = ""
    direction: Direction = "flat"
    quantity: float = 0
    market_value: float = 0
    cost_price: float | None = None
    leverage_factor: float = 1
    effective_long_exposure: float = 0
    effective_short_exposure: float = 0
    option_strategy_id: str | None = None
    theme_sleeve: str = ""
    portfolio_role: str = ""
    intent_guess: str = ""
    intent_confidence: IntentConfidence = "low"
    needs_human_clarification: bool = False
    warnings: list[str] = Field(default_factory=list)


class ExposureSleeveSummary(BaseModel):
    sleeve_key: str
    market_value: float = 0
    effective_long_exposure: float = 0
    effective_short_exposure: float = 0
    option_market_value: float = 0
    defined_risk_max_loss: float = 0
    short_assignment_notional: float = 0
    symbols: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExposureClarificationQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: new_id("ecq"))
    topic: str
    symbols: list[str] = Field(default_factory=list)
    question: str
    why_it_matters: str
    default_assumption: str = ""
    choices: list[str] = Field(default_factory=list)


class NormalizedExposureLedger(BaseModel):
    artifact_type: str = "normalized_exposure_ledger"
    ledger_id: str = Field(default_factory=lambda: new_id("nel"))
    generated_at: str = Field(default_factory=utc_now)
    selected_map_id: str = ""
    source: str = "futu_portfolio"
    total_assets: float = 0
    cash: float = 0
    cash_weight: float = 0
    margin_state: dict[str, Any] = Field(default_factory=dict)
    positions: list[NormalizedExposurePosition] = Field(default_factory=list)
    option_strategies: list[OptionStrategyGroup] = Field(default_factory=list)
    sleeve_exposures: list[ExposureSleeveSummary] = Field(default_factory=list)
    clarification_questions: list[ExposureClarificationQuestion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def build_exposure_ledger_from_futu_portfolio(
    futu_portfolio: dict[str, Any],
    *,
    portfolio_map: Any | None = None,
    preferences: ExposureLedgerPreferences | None = None,
    intent_policy: ExposureIntentPolicy | None = None,
    option_market_data: dict[str, Any] | None = None,
) -> NormalizedExposureLedger:
    """Normalize raw Futu portfolio JSON into an exposure ledger."""

    prefs = preferences or ExposureLedgerPreferences()
    policy = intent_policy or ExposureIntentPolicy()
    funds = futu_portfolio.get("funds") or {}
    total_assets = _safe_float(funds.get("total_assets"))
    cash = _safe_float(funds.get("cash"))
    positions_payload = futu_portfolio.get("positions") or []
    warnings: list[str] = []
    if total_assets <= 0:
        warnings.append("Futu portfolio total_assets is missing or <= 0; weights are not meaningful.")
    if cash < 0:
        warnings.append("Cash is negative; margin, option collateral, or unsettled activity must be reviewed.")

    target_sleeve_by_symbol = _target_sleeves(portfolio_map)
    stock_qty_by_symbol = _stock_quantities_by_symbol(positions_payload)
    option_market_data_by_code = _option_market_data_by_code(option_market_data)
    positions: list[NormalizedExposurePosition] = []
    option_legs: list[OptionLeg] = []
    questions: list[ExposureClarificationQuestion] = []

    for raw_position in positions_payload:
        code = str(raw_position.get("code") or "").upper()
        if not code:
            continue
        option_leg_payload = _merge_option_market_data(raw_position, option_market_data_by_code.get(code))
        option_leg = parse_option_leg(option_leg_payload)
        if option_leg:
            option_legs.append(option_leg)
            continue
        normalized = _normalize_cash_or_equity_position(
            raw_position,
            preferences=prefs,
            intent_policy=policy,
            target_sleeve_by_symbol=target_sleeve_by_symbol,
        )
        positions.append(normalized)
        if normalized.instrument_type == "leveraged_etf" and normalized.needs_human_clarification:
            questions.append(_leveraged_etf_question(normalized))

    strategies = group_option_strategies(option_legs)
    strategies = [
        _apply_option_intent_policy(strategy, policy, stock_qty_by_symbol)
        for strategy in strategies
    ]
    for strategy in strategies:
        sleeve = _sleeve_for_symbol(strategy.underlying, target_sleeve_by_symbol, prefs)
        for leg in strategy.legs:
            positions.append(
                _option_leg_position(
                    leg,
                    strategy=strategy,
                    theme_sleeve=sleeve,
                )
            )
        question = _option_strategy_question(strategy)
        if question:
            questions.append(question)

    sleeve_exposures = _summarize_sleeves(
        positions=positions,
        strategies=strategies,
        target_sleeve_by_symbol=target_sleeve_by_symbol,
        preferences=prefs,
    )
    cash_weight = cash / total_assets if total_assets > 0 else 0
    margin_state = {
        "cash_negative": cash < 0,
        "cash_weight": round(cash_weight, 6),
        "total_assets": round(total_assets, 2),
        "cash": round(cash, 2),
        "review_required": cash < 0 or any(strategy.defined_risk is False for strategy in strategies),
    }
    if cash < 0:
        if policy.negative_cash_policy == "ask":
            questions.append(
                ExposureClarificationQuestion(
                    topic="margin_policy",
                    symbols=[],
                    question="账户现金为负。你希望系统把负现金视为主动使用保证金提升资金效率，还是需要优先回补现金/降低杠杆？",
                    why_it_matters="现金为负会改变补仓能力、期权保证金风险和止盈优先级。",
                    default_assumption="Treat negative cash as a risk constraint until confirmed.",
                    choices=[
                        "主动使用保证金提升资金效率",
                        "优先回补现金/降低杠杆",
                        "暂时保持，但触发更严格风险提醒",
                    ],
                )
            )
        elif policy.negative_cash_policy == "temporary_allowed_strict_risk":
            warnings.append("Negative cash is allowed by user policy, but strict margin risk reminders are required.")
    return NormalizedExposureLedger(
        selected_map_id=selected_map_id(portfolio_map) if portfolio_map is not None else "",
        total_assets=round(total_assets, 2),
        cash=round(cash, 2),
        cash_weight=round(cash_weight, 6),
        margin_state=margin_state,
        positions=positions,
        option_strategies=strategies,
        sleeve_exposures=sleeve_exposures,
        clarification_questions=_dedupe_questions(questions),
        warnings=_dedupe(warnings),
    )


def parse_option_leg(raw_position: dict[str, Any]) -> OptionLeg | None:
    code = str(raw_position.get("code") or "").upper()
    match = OPTION_CODE_RE.match(code)
    if not match:
        return None
    market = match.group("market")
    underlying = f"{market}.{match.group('underlying')}"
    expiry = _parse_expiry(match.group("expiry"))
    option_type = "call" if match.group("option_type") == "C" else "put"
    strike = int(match.group("strike")) / 1000
    quantity = _safe_float(raw_position.get("qty"))
    side = "long" if quantity > 0 else "short"
    contract_count = abs(quantity)
    cost_price = _optional_float(raw_position.get("cost_price"))
    delta = _normalize_delta(_first_present(raw_position, "option_delta", "delta"))
    gamma = _optional_float(_first_present(raw_position, "option_gamma", "gamma"))
    vega = _optional_float(_first_present(raw_position, "option_vega", "vega"))
    theta = _optional_float(_first_present(raw_position, "option_theta", "theta"))
    rho = _optional_float(_first_present(raw_position, "option_rho", "rho"))
    implied_volatility = _optional_float(
        _first_present(raw_position, "option_implied_volatility", "implied_volatility", "iv")
    )
    option_last_price = _optional_float(_first_present(raw_position, "option_last_price", "last_price"))
    underlying_last_price = _optional_float(
        _first_present(raw_position, "underlying_last_price", "underlying_price", "stock_last_price")
    )
    opening_cashflow = None
    if cost_price is not None:
        sign = 1 if side == "short" else -1
        opening_cashflow = sign * cost_price * OPTION_MULTIPLIER * contract_count
    return OptionLeg(
        raw_code=code,
        name=str(raw_position.get("name") or ""),
        underlying=underlying,
        expiry=expiry,
        option_type=option_type,
        side=side,
        strike=strike,
        quantity=quantity,
        contract_count=contract_count,
        cost_price=cost_price,
        market_value=round(_safe_float(raw_position.get("market_val")), 2),
        option_last_price=option_last_price,
        underlying_last_price=underlying_last_price,
        implied_volatility=implied_volatility,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        opening_cashflow=round(opening_cashflow, 2) if opening_cashflow is not None else None,
        unrealized_pnl=_optional_float(raw_position.get("pl_val")),
    )


def group_option_strategies(legs: list[OptionLeg]) -> list[OptionStrategyGroup]:
    grouped: dict[tuple[str, str, str], list[OptionLeg]] = defaultdict(list)
    for leg in legs:
        grouped[(leg.underlying, leg.expiry, leg.option_type)].append(leg)

    strategies: list[OptionStrategyGroup] = []
    for (_underlying, _expiry, _option_type), group_legs in sorted(grouped.items()):
        remaining = sorted(group_legs, key=lambda item: (item.strike, item.side, item.raw_code))
        while remaining:
            if len(remaining) >= 2:
                pair = _find_vertical_pair(remaining)
                if pair:
                    first, second = pair
                    for leg in pair:
                        remaining.remove(leg)
                    strategies.append(_build_vertical_strategy([first, second]))
                    continue
            leg = remaining.pop(0)
            strategies.append(_build_single_leg_strategy(leg))
    return strategies


def _find_vertical_pair(legs: list[OptionLeg]) -> tuple[OptionLeg, OptionLeg] | None:
    for first in legs:
        for second in legs:
            if first is second:
                continue
            if first.side == second.side:
                continue
            if abs(first.contract_count - second.contract_count) > 1e-9:
                continue
            if first.strike == second.strike:
                continue
            return (first, second)
    return None


def _build_vertical_strategy(legs: list[OptionLeg]) -> OptionStrategyGroup:
    first, second = sorted(legs, key=lambda item: item.strike)
    low, high = first, second
    contract_count = min(leg.contract_count for leg in legs)
    width_cash = abs(high.strike - low.strike) * OPTION_MULTIPLIER * contract_count
    market_value = sum(leg.market_value for leg in legs)
    net_opening_credit = _sum_optional(leg.opening_cashflow for leg in legs)
    short_assignment = sum(
        leg.strike * OPTION_MULTIPLIER * leg.contract_count
        for leg in legs
        if leg.side == "short"
    )
    if first.option_type == "put":
        if high.side == "short" and low.side == "long":
            strategy_type = "bull_put_spread"
            direction = "bullish"
            intent_guess = "defined_risk_premium_income"
        elif high.side == "long" and low.side == "short":
            strategy_type = "bear_put_spread"
            direction = "bearish"
            intent_guess = "defined_risk_directional_or_hedge"
        else:
            strategy_type = "put_vertical_spread"
            direction = "unknown"
            intent_guess = "defined_risk_option_spread"
    else:
        if low.side == "long" and high.side == "short":
            strategy_type = "bull_call_spread"
            direction = "bullish"
            intent_guess = "defined_risk_directional_call_spread"
        elif low.side == "short" and high.side == "long":
            strategy_type = "bear_call_spread"
            direction = "bearish"
            intent_guess = "defined_risk_premium_income"
        else:
            strategy_type = "call_vertical_spread"
            direction = "unknown"
            intent_guess = "defined_risk_option_spread"
    max_profit = None
    max_loss = None
    if net_opening_credit is not None:
        if net_opening_credit >= 0:
            max_profit = net_opening_credit
            max_loss = max(0.0, width_cash - net_opening_credit)
        else:
            debit = abs(net_opening_credit)
            max_loss = debit
            max_profit = max(0.0, width_cash - debit)
    return OptionStrategyGroup(
        underlying=first.underlying,
        expiry=first.expiry,
        option_type=first.option_type,
        strategy_type=strategy_type,
        direction=direction,
        legs=legs,
        contract_count=contract_count,
        market_value=round(market_value, 2),
        net_opening_credit=round(net_opening_credit, 2) if net_opening_credit is not None else None,
        spread_width=round(width_cash, 2),
        defined_risk=True,
        max_profit=round(max_profit, 2) if max_profit is not None else None,
        max_loss=round(max_loss, 2) if max_loss is not None else None,
        short_assignment_notional=round(short_assignment, 2),
        intent_guess=intent_guess,
        intent_confidence="high" if strategy_type in {"bull_put_spread", "bear_call_spread"} else "medium",
        needs_human_clarification=True,
        risk_notes=[
            "Vertical spread recognized before leg-level exposure mapping.",
            "Do not treat the short leg as a naked assignment obligation.",
        ],
    )


def _build_single_leg_strategy(leg: OptionLeg) -> OptionStrategyGroup:
    market_value = leg.market_value
    short_assignment = leg.strike * OPTION_MULTIPLIER * leg.contract_count if leg.side == "short" else 0
    if leg.side == "long" and leg.option_type == "call":
        strategy_type = "long_call"
        direction = "bullish"
        intent_guess = "capital_efficient_long_exposure_or_speculation"
    elif leg.side == "long" and leg.option_type == "put":
        strategy_type = "long_put"
        direction = "bearish"
        intent_guess = "hedge_or_directional_put"
    elif leg.side == "short" and leg.option_type == "put":
        strategy_type = "short_put"
        direction = "bullish"
        intent_guess = "premium_income_or_cash_secured_put"
    else:
        strategy_type = "short_call"
        direction = "bearish"
        intent_guess = "covered_call_or_naked_call"
    delta_exposure = _delta_adjusted_exposure(leg)
    effective_long = delta_exposure if delta_exposure and delta_exposure > 0 else 0
    effective_short = abs(delta_exposure) if delta_exposure and delta_exposure < 0 else 0
    warnings = []
    if leg.side == "short":
        warnings.append("Standalone short option requires margin/assignment intent clarification.")
    if leg.side == "long" and delta_exposure is None:
        warnings.append("Long option delta exposure needs option Greeks before it can be mapped as target exposure.")
    return OptionStrategyGroup(
        underlying=leg.underlying,
        expiry=leg.expiry,
        option_type=leg.option_type,
        strategy_type=strategy_type,
        direction=direction,
        legs=[leg],
        contract_count=leg.contract_count,
        market_value=round(market_value, 2),
        net_opening_credit=leg.opening_cashflow,
        defined_risk=leg.side == "long",
        max_loss=round(abs(leg.opening_cashflow), 2)
        if leg.side == "long" and leg.opening_cashflow is not None
        else None,
        short_assignment_notional=round(short_assignment, 2),
        effective_long_exposure=round(effective_long, 2),
        effective_short_exposure=round(effective_short, 2),
        intent_guess=intent_guess,
        intent_confidence="medium",
        needs_human_clarification=True,
        warnings=warnings,
    )


def _normalize_cash_or_equity_position(
    raw_position: dict[str, Any],
    *,
    preferences: ExposureLedgerPreferences,
    intent_policy: ExposureIntentPolicy,
    target_sleeve_by_symbol: dict[str, str],
) -> NormalizedExposurePosition:
    code = str(raw_position.get("code") or "").upper()
    qty = _safe_float(raw_position.get("qty"))
    market_value = _safe_float(raw_position.get("market_val"))
    cost_price = _optional_float(raw_position.get("cost_price"))
    if _market_of(code) in intent_policy.ignored_markets:
        return NormalizedExposurePosition(
            raw_code=code,
            name=str(raw_position.get("name") or ""),
            instrument_type="etf" if code in preferences.etf_symbols else "stock",
            underlying=code,
            direction="long" if market_value > 0 else "short" if market_value < 0 else "flat",
            quantity=qty,
            market_value=round(market_value, 2),
            cost_price=cost_price,
            leverage_factor=1,
            effective_long_exposure=0,
            effective_short_exposure=0,
            theme_sleeve="ignored",
            portfolio_role="ignored_market",
            intent_guess="ignored_by_user_policy",
            intent_confidence="high",
            needs_human_clarification=False,
            warnings=["Ignored by user policy; excluded from target-map exposure matching."],
        )
    leveraged = preferences.leveraged_etf_map.get(code)
    warnings: list[str] = []
    if leveraged:
        underlying, leverage_factor = leveraged
        instrument_type = "leveraged_etf"
        portfolio_role = "leveraged_theme_expression"
        intent_guess = "leveraged_exposure_or_tactical_trade"
        intent_confidence = "medium"
        needs_clarification = True
        if intent_policy.leveraged_etf_intent == "long_term_leveraged_exposure_limited_rebalance":
            portfolio_role = "strategic_leveraged_exposure_limited_rebalance"
            intent_guess = "long_term_leveraged_exposure_limited_rebalance"
            intent_confidence = "high"
            needs_clarification = False
        elif intent_policy.leveraged_etf_intent == "tactical_trade":
            portfolio_role = "tactical_leveraged_trade"
            intent_guess = "tactical_trade"
            intent_confidence = "high"
            needs_clarification = False
        elif intent_policy.leveraged_etf_intent == "separate_overlay_budget":
            portfolio_role = "separate_overlay_budget"
            intent_guess = "separate_overlay_budget"
            intent_confidence = "high"
            needs_clarification = False
    else:
        underlying = code
        leverage_factor = 1.0
        instrument_type = "etf" if code in preferences.etf_symbols else "stock"
        portfolio_role = "target_or_extra_position"
        intent_guess = "direct_holding"
        intent_confidence = "high"
        needs_clarification = False
    direction = "long" if market_value > 0 else "short" if market_value < 0 else "flat"
    effective = market_value * leverage_factor
    effective_long = effective if effective > 0 else 0
    effective_short = abs(effective) if effective < 0 else 0
    if instrument_type == "leveraged_etf":
        warnings.append("Leveraged ETF exposure is approximated as market_value * leverage_factor.")
        if intent_policy.leveraged_etf_intent == "long_term_leveraged_exposure_limited_rebalance":
            warnings.append("User policy: count effective exposure, but treat as limited-rebalance instrument.")
    return NormalizedExposurePosition(
        raw_code=code,
        name=str(raw_position.get("name") or ""),
        instrument_type=instrument_type,
        underlying=underlying,
        direction=direction,
        quantity=qty,
        market_value=round(market_value, 2),
        cost_price=cost_price,
        leverage_factor=leverage_factor,
        effective_long_exposure=round(effective_long, 2),
        effective_short_exposure=round(effective_short, 2),
        theme_sleeve=_sleeve_for_symbol(underlying, target_sleeve_by_symbol, preferences),
        portfolio_role=portfolio_role,
        intent_guess=intent_guess,
        intent_confidence=intent_confidence,
        needs_human_clarification=needs_clarification,
        warnings=warnings,
    )


def _option_leg_position(
    leg: OptionLeg,
    *,
    strategy: OptionStrategyGroup,
    theme_sleeve: str,
) -> NormalizedExposurePosition:
    direction = "long" if leg.side == "long" else "short"
    counts_as_exposure = (
        len(strategy.legs) == 1
        and strategy.intent_guess == "long_term_capital_efficiency"
        and (strategy.effective_long_exposure or strategy.effective_short_exposure)
    )
    effective_long = strategy.effective_long_exposure if counts_as_exposure else 0
    effective_short = strategy.effective_short_exposure if counts_as_exposure else 0
    portfolio_role = "option_overlay"
    if effective_long or effective_short:
        portfolio_role = "delta_adjusted_option_exposure"
    warnings = [
        "Option leg is represented through its option_strategy_group; do not map this leg as standalone stock exposure."
    ]
    if effective_long or effective_short:
        warnings = [
            "Option leg is represented through its option_strategy_group and mapped as delta-adjusted effective exposure."
        ]
    return NormalizedExposurePosition(
        raw_code=leg.raw_code,
        name=leg.name,
        instrument_type="option_leg",
        underlying=leg.underlying,
        direction=direction,
        quantity=leg.quantity,
        market_value=round(leg.market_value, 2),
        cost_price=leg.cost_price,
        leverage_factor=1,
        effective_long_exposure=round(effective_long, 2),
        effective_short_exposure=round(effective_short, 2),
        option_strategy_id=strategy.strategy_id,
        theme_sleeve=theme_sleeve,
        portfolio_role=portfolio_role,
        intent_guess=strategy.intent_guess,
        intent_confidence=strategy.intent_confidence,
        needs_human_clarification=strategy.needs_human_clarification,
        warnings=warnings,
    )


def _apply_option_intent_policy(
    strategy: OptionStrategyGroup,
    policy: ExposureIntentPolicy,
    stock_qty_by_symbol: dict[str, float],
) -> OptionStrategyGroup:
    updated = strategy.model_copy(deep=True)
    if updated.strategy_type in {"bull_put_spread", "bear_call_spread"}:
        if policy.defined_risk_credit_spread_intent == "independent_high_iv_premium":
            updated.intent_guess = "independent_high_iv_premium_income"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
            updated.risk_notes.append(
                "User policy: defined-risk credit spread is an independent premium-income overlay, not spot target exposure."
            )
        elif policy.defined_risk_credit_spread_intent == "assignment_transition_plan":
            updated.intent_guess = "assignment_transition_plan"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
        elif policy.defined_risk_credit_spread_intent == "tactical_option_trade":
            updated.intent_guess = "tactical_option_trade"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
    if updated.strategy_type == "long_call":
        if policy.long_call_intent == "long_term_capital_efficiency":
            updated.intent_guess = "long_term_capital_efficiency"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
            if updated.effective_long_exposure or updated.effective_short_exposure:
                updated.risk_notes.append(
                    "User policy: long call is a long-term capital-efficiency tool; delta-adjusted exposure is mapped into target exposure."
                )
            else:
                updated.risk_notes.append(
                    "User policy: long call is a long-term capital-efficiency tool; delta/Greeks are required before mapping effective exposure."
                )
        elif policy.long_call_intent == "tactical_directional_trade":
            updated.intent_guess = "tactical_directional_trade"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
        elif policy.long_call_intent == "hedge_or_combo_leg":
            updated.intent_guess = "hedge_or_combo_leg"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
    if updated.strategy_type == "short_call":
        if policy.short_call_intent == "covered_call_income":
            covered_qty = stock_qty_by_symbol.get(updated.underlying, 0)
            required_qty = updated.contract_count * OPTION_MULTIPLIER
            updated.strategy_type = "covered_call"
            updated.intent_guess = "covered_call_income"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
            updated.underlying_share_quantity = covered_qty
            updated.required_underlying_share_quantity = required_qty
            updated.coverage_status = _coverage_status(covered_qty, required_qty)
            updated.defined_risk = updated.coverage_status == "covered"
            updated.risk_notes.append(
                "User policy: short call is intended as covered-call premium income/upside overwrite."
            )
            updated.warnings = _remove_option_clarification_warnings(updated.warnings)
            if updated.coverage_status != "covered":
                updated.warnings.append(
                    f"Covered-call intent is not fully covered: underlying shares {covered_qty:g} < required {required_qty:g}."
                )
        elif policy.short_call_intent == "premium_income":
            updated.intent_guess = "short_call_premium_income"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
        elif policy.short_call_intent == "tactical_bearish_trade":
            updated.intent_guess = "tactical_bearish_trade"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
    if updated.strategy_type == "short_put":
        if policy.short_put_intent == "premium_income_willing_assignment":
            updated.intent_guess = "premium_income_willing_assignment"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
            updated.risk_notes.append(
                "User policy: short put collects premium and assignment into underlying shares is acceptable."
            )
            updated.warnings = _remove_option_clarification_warnings(updated.warnings)
        elif policy.short_put_intent == "premium_income_no_assignment":
            updated.intent_guess = "premium_income_no_assignment"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
            updated.risk_notes.append(
                "User policy: short put is premium income only; assignment should be avoided or managed."
            )
            updated.warnings = _remove_option_clarification_warnings(updated.warnings)
        elif policy.short_put_intent == "tactical_bullish_trade":
            updated.intent_guess = "tactical_bullish_trade"
            updated.intent_confidence = "high"
            updated.needs_human_clarification = False
    return updated


def _summarize_sleeves(
    *,
    positions: list[NormalizedExposurePosition],
    strategies: list[OptionStrategyGroup],
    target_sleeve_by_symbol: dict[str, str],
    preferences: ExposureLedgerPreferences,
) -> list[ExposureSleeveSummary]:
    by_sleeve: dict[str, ExposureSleeveSummary] = {}
    for position in positions:
        sleeve = position.theme_sleeve or "unknown"
        summary = by_sleeve.setdefault(sleeve, ExposureSleeveSummary(sleeve_key=sleeve))
        summary.market_value += position.market_value
        summary.effective_long_exposure += position.effective_long_exposure
        summary.effective_short_exposure += position.effective_short_exposure
        if position.instrument_type == "option_leg":
            summary.option_market_value += position.market_value
        if position.underlying and position.underlying not in summary.symbols:
            summary.symbols.append(position.underlying)
        summary.warnings.extend(position.warnings)
    for strategy in strategies:
        sleeve = _sleeve_for_symbol(strategy.underlying, target_sleeve_by_symbol, preferences)
        summary = by_sleeve.setdefault(sleeve or "unknown", ExposureSleeveSummary(sleeve_key=sleeve or "unknown"))
        summary.defined_risk_max_loss += strategy.max_loss or 0
        summary.short_assignment_notional += strategy.short_assignment_notional
        if strategy.underlying not in summary.symbols:
            summary.symbols.append(strategy.underlying)
        summary.warnings.extend(strategy.warnings)
    result = []
    for summary in by_sleeve.values():
        summary.market_value = round(summary.market_value, 2)
        summary.effective_long_exposure = round(summary.effective_long_exposure, 2)
        summary.effective_short_exposure = round(summary.effective_short_exposure, 2)
        summary.option_market_value = round(summary.option_market_value, 2)
        summary.defined_risk_max_loss = round(summary.defined_risk_max_loss, 2)
        summary.short_assignment_notional = round(summary.short_assignment_notional, 2)
        summary.symbols = sorted(summary.symbols)
        summary.warnings = _dedupe(summary.warnings)
        result.append(summary)
    return sorted(result, key=lambda item: item.sleeve_key)


def _target_sleeves(portfolio_map: Any | None) -> dict[str, str]:
    if portfolio_map is None:
        return {}
    targets = extract_target_positions(portfolio_map)
    return {target.symbol: target.sleeve_key for target in targets if target.sleeve_key}


def _stock_quantities_by_symbol(positions_payload: list[dict[str, Any]]) -> dict[str, float]:
    quantities: dict[str, float] = defaultdict(float)
    for raw_position in positions_payload:
        code = str(raw_position.get("code") or "").upper()
        if not code or OPTION_CODE_RE.match(code):
            continue
        if code in DEFAULT_LEVERAGED_ETF_MAP:
            continue
        market_value = _safe_float(raw_position.get("market_val"))
        if market_value <= 0:
            continue
        quantities[code] += _safe_float(raw_position.get("qty"))
    return dict(quantities)


def _option_market_data_by_code(option_market_data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not option_market_data:
        return {}
    underlying_quotes = _underlying_quotes_by_code(option_market_data)
    rows: list[dict[str, Any]] = []
    if isinstance(option_market_data.get("data"), list):
        rows.extend(item for item in option_market_data["data"] if isinstance(item, dict))
    if isinstance(option_market_data.get("option_snapshots"), list):
        rows.extend(item for item in option_market_data["option_snapshots"] if isinstance(item, dict))
    if isinstance(option_market_data.get("option_snapshots"), dict):
        for code, item in option_market_data["option_snapshots"].items():
            if isinstance(item, dict):
                rows.append({"code": code, **item})
    if isinstance(option_market_data.get("options"), dict):
        for code, item in option_market_data["options"].items():
            if isinstance(item, dict):
                rows.append({"code": code, **item})

    by_code: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = str(row.get("code") or "").upper()
        if not code:
            continue
        parsed = parse_option_leg({"code": code, "qty": 1})
        underlying = parsed.underlying if parsed else str(row.get("underlying") or "").upper()
        enriched = dict(row)
        if underlying and _first_present(enriched, "underlying_last_price", "underlying_price", "stock_last_price") is None:
            quote = underlying_quotes.get(underlying, {})
            quote_price = _first_present(quote, "last_price", "price", "close")
            if quote_price is not None:
                enriched["underlying_last_price"] = quote_price
        by_code[code] = enriched
    return by_code


def _underlying_quotes_by_code(option_market_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_quotes = option_market_data.get("underlying_quotes") or option_market_data.get("underlyings") or {}
    if isinstance(raw_quotes, dict):
        return {
            str(code).upper(): item
            for code, item in raw_quotes.items()
            if isinstance(item, dict)
        }
    if isinstance(raw_quotes, list):
        result: dict[str, dict[str, Any]] = {}
        for item in raw_quotes:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or item.get("symbol") or "").upper()
            if code:
                result[code] = item
        return result
    return {}


def _merge_option_market_data(raw_position: dict[str, Any], market_data: dict[str, Any] | None) -> dict[str, Any]:
    if not market_data:
        return raw_position
    merged = dict(raw_position)
    for key, value in market_data.items():
        if value is None or value == "":
            continue
        merged.setdefault(key, value)
    return merged


def _sleeve_for_symbol(
    symbol: str,
    target_sleeve_by_symbol: dict[str, str],
    preferences: ExposureLedgerPreferences,
) -> str:
    symbol = symbol.upper()
    if symbol in target_sleeve_by_symbol:
        return target_sleeve_by_symbol[symbol]
    if preferences.include_default_sleeve_hints:
        return preferences.symbol_sleeve_hints.get(symbol, "")
    return ""


def _leveraged_etf_question(position: NormalizedExposurePosition) -> ExposureClarificationQuestion:
    return ExposureClarificationQuestion(
        topic="leveraged_etf_intent",
        symbols=[position.raw_code, position.underlying],
        question=f"{position.raw_code} 是长期替代 {position.underlying} 的杠杆表达，还是短线投机/交易仓？",
        why_it_matters="如果是长期表达，应按有效敞口计入版图；如果是短线交易，应放入 overlay/trading bucket。",
        default_assumption="Treat as leveraged overlay until user confirms it is strategic exposure.",
        choices=[
            "长期杠杆表达，计入目标版图有效敞口",
            "短线交易，不计入核心版图匹配",
            "保留观察，单独设置风险预算",
        ],
    )


def _option_strategy_question(strategy: OptionStrategyGroup) -> ExposureClarificationQuestion | None:
    if not strategy.needs_human_clarification:
        return None
    if strategy.strategy_type == "bull_put_spread":
        question = (
            f"{strategy.underlying} {strategy.expiry} bull put spread 是独立的高 IV 权利金策略，"
            "还是你愿意在极端情况下把它转成底层持仓的一部分？"
        )
        choices = [
            "独立 defined-risk 权利金策略，不计入现货目标仓位",
            "愿意接近 short strike 时转成底层持仓计划",
            "投机性期权仓，单独风险预算管理",
        ]
        default = "Treat as independent defined-risk premium overlay unless confirmed otherwise."
    elif strategy.strategy_type in {"long_call", "bull_call_spread"}:
        question = f"{strategy.underlying} {strategy.expiry} {strategy.strategy_type} 是资金效率工具，还是短线方向性投机？"
        choices = [
            "长期资金效率工具，按 delta-like exposure 管理",
            "短线方向性投机，单独风险预算管理",
            "对冲/组合结构的一部分",
        ]
        default = "Do not map long option as target exposure until delta/intent is confirmed."
    elif strategy.strategy_type in {"short_put", "short_call", "bear_call_spread"}:
        question = f"{strategy.underlying} {strategy.expiry} {strategy.strategy_type} 的主要目的是什么？"
        choices = [
            "收取权利金",
            "愿意接货/被行权",
            "短线方向性交易",
            "已有持仓的覆盖/对冲",
        ]
        default = "Treat as option overlay requiring explicit risk policy."
    else:
        return None
    return ExposureClarificationQuestion(
        topic="option_strategy_intent",
        symbols=[strategy.underlying, *[leg.raw_code for leg in strategy.legs]],
        question=question,
        why_it_matters="期权策略意图决定它应计入目标版图敞口、期权收益 overlay，还是投机风险预算。",
        default_assumption=default,
        choices=choices,
    )


def _remove_option_clarification_warnings(warnings: list[str]) -> list[str]:
    return [
        warning
        for warning in warnings
        if "requires margin/assignment intent clarification" not in warning
    ]


def _coverage_status(covered_qty: float, required_qty: float) -> str:
    if required_qty <= 0:
        return "not_applicable"
    if covered_qty >= required_qty:
        return "covered"
    if covered_qty > 0:
        return "partially_covered"
    return "uncovered"


def _delta_adjusted_exposure(leg: OptionLeg) -> float | None:
    if leg.delta is None or leg.underlying_last_price is None or leg.underlying_last_price <= 0:
        return None
    return leg.delta * leg.underlying_last_price * leg.multiplier * leg.quantity


def _parse_expiry(value: str) -> str:
    year = int(value[:2])
    full_year = 2000 + year
    return f"{full_year:04d}-{value[2:4]}-{value[4:6]}"


def _market_of(code: str) -> str:
    if "." not in code:
        return ""
    return code.split(".", 1)[0].upper()


def _sum_optional(values) -> float | None:
    result = 0.0
    found = False
    for value in values:
        if value is None:
            continue
        result += value
        found = True
    return result if found else None


def _safe_float(value: Any) -> float:
    parsed = _optional_float(value)
    return parsed if parsed is not None else 0.0


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _normalize_delta(value: Any) -> float | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    if abs(parsed) > 1 and abs(parsed) <= 100:
        parsed = parsed / 100
    if parsed < -1 or parsed > 1:
        return None
    return parsed


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_questions(questions: list[ExposureClarificationQuestion]) -> list[ExposureClarificationQuestion]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result: list[ExposureClarificationQuestion] = []
    for question in questions:
        key = (question.topic, tuple(sorted(question.symbols)))
        if key in seen:
            continue
        seen.add(key)
        result.append(question)
    return result


def build_exposure_ledger_from_files(
    *,
    futu_portfolio_path: str | Path,
    portfolio_map_path: str | Path | None = None,
    intent_policy_path: str | Path | None = None,
    option_market_data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> NormalizedExposureLedger:
    futu_portfolio = _read_json(Path(futu_portfolio_path))
    portfolio_map = _read_json(Path(portfolio_map_path)) if portfolio_map_path else None
    option_market_data = _read_json(Path(option_market_data_path)) if option_market_data_path else None
    intent_policy = (
        ExposureIntentPolicy.model_validate(_read_json(Path(intent_policy_path)))
        if intent_policy_path
        else None
    )
    ledger = build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        portfolio_map=portfolio_map,
        intent_policy=intent_policy,
        option_market_data=option_market_data,
    )
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "normalized_exposure_ledger.json", ledger.model_dump(mode="json"))
    return ledger


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        ledger = build_exposure_ledger_from_files(
            futu_portfolio_path=args.futu_portfolio_path,
            portfolio_map_path=args.portfolio_map_path,
            intent_policy_path=args.intent_policy_path,
            option_market_data_path=args.option_market_data_path,
            output_dir=args.output_dir,
        )
        payload = ledger.model_dump(mode="json")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"ledger_id: {ledger.ledger_id}")
            print(f"positions: {len(ledger.positions)}")
            print(f"option_strategies: {len(ledger.option_strategies)}")
            print(f"clarification_questions: {len(ledger.clarification_questions)}")
            if args.output_dir:
                print(f"output_dir: {args.output_dir}")
            if ledger.warnings:
                print("warnings:")
                for warning in ledger.warnings:
                    print(f"  - {warning}")
        return 0
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-exposure-ledger",
        description="Normalize raw Futu holdings into an exposure ledger.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build one normalized exposure ledger.")
    build.add_argument("--futu-portfolio-path", required=True, help="Raw Futu get_portfolio JSON path.")
    build.add_argument("--portfolio-map-path", help="Optional selected portfolio map JSON path.")
    build.add_argument("--intent-policy-path", help="Optional user-confirmed exposure intent policy JSON.")
    build.add_argument("--option-market-data-path", help="Optional Futu option snapshot/Greeks JSON path.")
    build.add_argument("--output-dir", help="Directory for normalized_exposure_ledger.json.")
    build.add_argument("--json", action="store_true")
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

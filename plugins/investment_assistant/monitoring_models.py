"""Typed models for portfolio monitoring and dynamic rebalance artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .schemas import CurrentHolding, SimulatedOrder
from .storage import new_id, utc_now


DriftStatus = Literal["underweight", "overweight", "within_band", "extra", "missing_price"]
TrendState = Literal[
    "uptrend",
    "extended_uptrend",
    "neutral",
    "weakening",
    "downtrend",
    "high_volatility",
    "unknown",
]
RebalanceActionType = Literal["add", "trim", "hold", "watch"]


class MonitoringPreferences(BaseModel):
    """User-tunable execution guardrails for a monitoring run."""

    core_relative_band: float = Field(default=0.20, ge=0, le=1)
    high_conviction_relative_band: float = Field(default=0.25, ge=0, le=1)
    satellite_relative_band: float = Field(default=0.35, ge=0, le=1)
    default_relative_band: float = Field(default=0.25, ge=0, le=1)
    min_absolute_band_weight: float = Field(default=0.005, ge=0, le=1)
    min_trade_value: float = Field(default=500.0, ge=0)
    max_add_cash_fraction: float = Field(default=1.0, ge=0, le=1)
    trend_overlay_enabled: bool = True
    trim_extra_positions: bool = False
    lot_size: int = Field(default=1, ge=1)
    market: str = "US"
    trd_env: str = "SIMULATE"


class TargetPosition(BaseModel):
    symbol: str
    target_weight: float = Field(ge=0, le=1)
    sleeve_key: str = ""
    role: str = ""
    rationale: str = ""
    suggested_weight_band: tuple[float, float] | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class PortfolioMonitorSnapshot(BaseModel):
    artifact_type: str = "portfolio_monitor_snapshot"
    snapshot_id: str = Field(default_factory=lambda: new_id("pms"))
    selected_map_id: str = ""
    generated_at: str = Field(default_factory=utc_now)
    data_asof: dict[str, str] = Field(default_factory=dict)
    total_assets: float = Field(gt=0)
    cash: float = Field(ge=0)
    cash_weight: float = Field(ge=0, le=1)
    positions: list[CurrentHolding] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DriftPosition(BaseModel):
    symbol: str
    target_weight: float = Field(ge=0, le=1)
    current_weight: float = Field(ge=0)
    lower_bound: float = Field(ge=0)
    upper_bound: float = Field(ge=0)
    tolerance: float = Field(ge=0)
    target_value: float = Field(ge=0)
    current_value: float = Field(ge=0)
    drift_weight: float
    drift_value: float
    quantity: int = Field(ge=0)
    can_sell_qty: int = Field(ge=0)
    status: DriftStatus
    sleeve_key: str = ""
    role: str = ""
    rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PortfolioDriftReport(BaseModel):
    artifact_type: str = "portfolio_drift_report"
    report_id: str = Field(default_factory=lambda: new_id("pdr"))
    selected_map_id: str = ""
    generated_at: str = Field(default_factory=utc_now)
    total_assets: float = Field(gt=0)
    cash: float = Field(ge=0)
    cash_weight: float = Field(ge=0)
    target_cash_weight: float = Field(ge=0, le=1)
    positions: list[DriftPosition] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PriceBar(BaseModel):
    date: str = ""
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None


class MarketPriceData(BaseModel):
    symbol: str
    last_price: float | None = None
    update_time: str = ""
    kline: list[PriceBar] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TrendSignal(BaseModel):
    symbol: str
    price: float | None = None
    trend_state: TrendState = "unknown"
    ma20: float | None = None
    ma50: float | None = None
    ma200: float | None = None
    rsi14: float | None = None
    atr14: float | None = None
    return_20d: float | None = None
    return_60d: float | None = None
    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)
    data_asof: str = ""
    warnings: list[str] = Field(default_factory=list)


class RebalanceTranche(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    weight: float = Field(ge=0, le=1)
    quantity: int = Field(ge=0)
    limit_price: float = Field(gt=0)
    trigger: str
    invalidation: str
    estimated_value: float = Field(ge=0)


class RebalanceAction(BaseModel):
    symbol: str
    action: RebalanceActionType
    reason_code: str
    target_trade_value: float = Field(ge=0)
    quantity: int = Field(ge=0)
    cash_impact: float
    current_weight: float = Field(ge=0)
    target_weight: float = Field(ge=0, le=1)
    band: tuple[float, float]
    trend_state: TrendState = "unknown"
    trigger: str = ""
    invalidation: str = ""
    tranches: list[RebalanceTranche] = Field(default_factory=list)
    simulated_order: SimulatedOrder | None = None
    requires_human_confirmation: bool = True
    evidence_refs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RebalancePlan(BaseModel):
    artifact_type: str = "rebalance_plan"
    plan_id: str = Field(default_factory=lambda: new_id("rbp"))
    selected_map_id: str = ""
    generated_at: str = Field(default_factory=utc_now)
    target_cash_weight: float = Field(ge=0, le=1)
    current_cash_weight: float = Field(ge=0)
    cash_required: float = Field(ge=0)
    cash_released: float = Field(ge=0)
    post_trade_cash: float
    actions: list[RebalanceAction] = Field(default_factory=list)
    simulated_orders: list[SimulatedOrder] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PortfolioMonitorResult(BaseModel):
    artifact_type: str = "portfolio_monitor_result"
    monitor_id: str = Field(default_factory=lambda: new_id("pmr"))
    selected_map_id: str = ""
    generated_at: str = Field(default_factory=utc_now)
    snapshot: PortfolioMonitorSnapshot
    drift_report: PortfolioDriftReport
    trend_signals: list[TrendSignal] = Field(default_factory=list)
    rebalance_plan: RebalancePlan
    warnings: list[str] = Field(default_factory=list)

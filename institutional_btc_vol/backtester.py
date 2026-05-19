from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Callable, Iterable, Sequence

EVIDENCE_STATUS = "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"


@dataclass(frozen=True)
class MarketSnapshot:
    timestamp: datetime
    data: dict[str, Any]


@dataclass(frozen=True)
class BacktestContext:
    history: tuple[MarketSnapshot, ...] = ()


@dataclass(frozen=True)
class DecisionFrame:
    timestamp: datetime
    data: MappingProxyType | dict[str, Any]
    prior_history: tuple[MarketSnapshot, ...] = ()
    source_snapshot: MarketSnapshot | None = None


@dataclass(frozen=True)
class SyntheticOrder:
    decision_time: datetime
    side: str
    quantity: float
    entry_price: float
    costs: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trade:
    entry_time: datetime
    exit_time: datetime
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    costs: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def net_pnl(self) -> float:
        if self.side in {"long", "long_spread"}:
            pnl = (self.exit_price - self.entry_price) * self.quantity
        elif self.side in {"short", "short_spread"}:
            pnl = (self.entry_price - self.exit_price) * self.quantity
        else:
            raise ValueError(f"Unsupported trade side: {self.side}")
        return pnl - self.costs


@dataclass(frozen=True)
class BacktestResult:
    trades: tuple[Trade, ...]
    equity_curve: tuple[float, ...]
    gross_pnl: float
    max_drawdown: float
    win_rate: float
    evidence_status: str = EVIDENCE_STATUS
    assumptions: MappingProxyType | dict[str, Any] = field(default_factory=dict)

    @property
    def trade_count(self) -> int:
        return len(self.trades)


def _assert_chronological(snapshots: Sequence[MarketSnapshot]) -> None:
    for previous, current in zip(snapshots, snapshots[1:]):
        if current.timestamp <= previous.timestamp:
            raise ValueError("Market snapshots must be strictly chronological; refusing to sort because that can hide data-quality defects")


def _summarize_trades(trades: list[Trade], assumptions: dict[str, Any] | None = None) -> BacktestResult:
    default_assumptions = {"execution": "screen-only synthetic fills; not executable economics"}
    merged_assumptions = {**default_assumptions, **dict(assumptions or {})}
    equity = []
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    winners = 0
    for trade in trades:
        pnl = trade.net_pnl
        if pnl > 0:
            winners += 1
        cumulative += pnl
        equity.append(cumulative)
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    win_rate = winners / len(trades) if trades else 0.0
    return BacktestResult(
        trades=tuple(trades),
        equity_curve=tuple(equity),
        gross_pnl=cumulative,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        assumptions=MappingProxyType(merged_assumptions),
    )


def run_backtest(
    snapshots: Sequence[MarketSnapshot],
    *,
    strategy: Callable[[MarketSnapshot, BacktestContext], Iterable[Trade]],
    assumptions: dict[str, Any] | None = None,
) -> BacktestResult:
    _assert_chronological(snapshots)
    trades: list[Trade] = []
    history: list[MarketSnapshot] = []
    for snapshot in snapshots:
        context = BacktestContext(history=tuple(history))
        emitted = list(strategy(snapshot, context))
        for trade in emitted:
            if trade.entry_time < snapshot.timestamp:
                raise ValueError("Strategy emitted a trade before the current decision snapshot")
            trades.append(trade)
        history.append(snapshot)
    return _summarize_trades(trades, assumptions=assumptions)


def _decision_visible_data(data: dict[str, Any]) -> MappingProxyType:
    """Return immutable decision-time data with explicit future columns removed."""
    return MappingProxyType({key: value for key, value in data.items() if not str(key).startswith("next_")})


def run_point_in_time_backtest(
    snapshots: Sequence[MarketSnapshot],
    *,
    strategy: Callable[[DecisionFrame], Iterable[SyntheticOrder]],
    execution_resolver: Callable[[SyntheticOrder, DecisionFrame, tuple[MarketSnapshot, ...]], Trade | None],
    assumptions: dict[str, Any] | None = None,
    unresolved_orders: str = "raise",
) -> BacktestResult:
    _assert_chronological(snapshots)
    trades: list[Trade] = []
    history: list[MarketSnapshot] = []
    for index, snapshot in enumerate(snapshots):
        frame = DecisionFrame(
            timestamp=snapshot.timestamp,
            data=_decision_visible_data(snapshot.data),
            prior_history=tuple(history),
            source_snapshot=snapshot,
        )
        future_snapshots = tuple(snapshots[index + 1 :])
        for order in strategy(frame):
            if order.decision_time < frame.timestamp:
                raise ValueError("Strategy emitted an order before the current decision frame")
            trade = execution_resolver(order, frame, future_snapshots)
            if trade is None:
                if unresolved_orders == "skip":
                    continue
                raise ValueError("Synthetic order could not be resolved without future market data")
            if trade.entry_time < frame.timestamp:
                raise ValueError("Execution resolver emitted a trade before the current decision frame")
            if trade.exit_time < trade.entry_time:
                raise ValueError("Execution resolver emitted a trade with exit before entry")
            trades.append(trade)
        history.append(snapshot)
    default_assumptions = {
        "execution": "screen-only synthetic fills; not executable economics",
        "anti_lookahead": "DecisionFrame excludes next_* future fields; ExecutionResolver resolves synthetic exits after decision time",
    }
    return _summarize_trades(trades, assumptions={**default_assumptions, **dict(assumptions or {})})


def hold_one_period_vol_spread_strategy(
    *,
    threshold_vol_pts: float,
    notional_vega: float,
    cost_per_trade: float = 0.0,
) -> Callable[[MarketSnapshot, BacktestContext], list[Trade]]:
    """Synthetic one-period mean-reversion model for IBIT-vs-Deribit spread evidence.

    The strategy requires the snapshot to carry an explicit `next_spread_7d_vol_pts`
    field, making the decision-time signal and the subsequent realized mark separate.
    It is useful for research/backtest hygiene only; it is not an execution model.
    """

    def strategy(snapshot: MarketSnapshot, context: BacktestContext) -> list[Trade]:
        spread = snapshot.data.get("spread_7d_vol_pts")
        next_spread = snapshot.data.get("next_spread_7d_vol_pts")
        if spread is None or next_spread is None:
            return []
        spread = float(spread)
        next_spread = float(next_spread)
        if abs(spread) < threshold_vol_pts:
            return []
        if spread > 0:
            side = "short_spread"
            signal = "IBIT rich / Deribit cheap"
        else:
            side = "long_spread"
            signal = "IBIT cheap / Deribit rich"
        return [
            Trade(
                entry_time=snapshot.timestamp,
                exit_time=snapshot.timestamp,
                side=side,
                quantity=float(notional_vega),
                entry_price=spread,
                exit_price=next_spread,
                costs=float(cost_per_trade),
                metadata={
                    "signal": signal,
                    "threshold_vol_pts": threshold_vol_pts,
                    "evidence_status": EVIDENCE_STATUS,
                    "history_rows_visible": len(context.history),
                },
            )
        ]

    return strategy


def one_period_vol_spread_signal_strategy(
    *,
    threshold_vol_pts: float,
    notional_vega: float,
    cost_per_trade: float = 0.0,
    spread_field: str = "spread_7d_vol_pts",
) -> Callable[[DecisionFrame], list[SyntheticOrder]]:
    """Decision-time signal strategy that cannot access future exit marks.

    The companion ExecutionResolver is responsible for resolving synthetic exits
    from future snapshots after this strategy has made its decision.
    """

    def strategy(frame: DecisionFrame) -> list[SyntheticOrder]:
        spread = frame.data.get(spread_field)
        if spread is None:
            return []
        spread = float(spread)
        if abs(spread) < threshold_vol_pts:
            return []
        if spread > 0:
            side = "short_spread"
            signal = "IBIT rich / Deribit cheap"
        else:
            side = "long_spread"
            signal = "IBIT cheap / Deribit rich"
        return [
            SyntheticOrder(
                decision_time=frame.timestamp,
                side=side,
                quantity=float(notional_vega),
                entry_price=spread,
                costs=float(cost_per_trade),
                metadata={
                    "signal": signal,
                    "threshold_vol_pts": threshold_vol_pts,
                    "evidence_status": EVIDENCE_STATUS,
                    "history_rows_visible": len(frame.prior_history),
                    "future_fields_visible_to_strategy": [key for key in frame.data if str(key).startswith("next_")],
                    "spread_field": spread_field,
                },
            )
        ]

    return strategy


def one_period_spread_execution_resolver(*, spread_field: str = "spread_7d_vol_pts") -> Callable[[SyntheticOrder, DecisionFrame, tuple[MarketSnapshot, ...]], Trade | None]:
    def resolver(order: SyntheticOrder, frame: DecisionFrame, future_snapshots: tuple[MarketSnapshot, ...]) -> Trade | None:
        if not future_snapshots:
            return None
        exit_snapshot = future_snapshots[0]
        if spread_field not in exit_snapshot.data:
            return None
        return Trade(
            entry_time=order.decision_time,
            exit_time=exit_snapshot.timestamp,
            side=order.side,
            quantity=order.quantity,
            entry_price=order.entry_price,
            exit_price=float(exit_snapshot.data[spread_field]),
            costs=order.costs,
            metadata={
                **order.metadata,
                "synthetic_fill": True,
                "execution_resolver": "one_period_spread_execution_resolver",
                "exit_source": "next chronological snapshot",
            },
        )

    return resolver


def run_vol_spread_backtest(
    snapshots: Sequence[MarketSnapshot],
    *,
    threshold_vol_pts: float = 5.0,
    notional_vega: float = 10_000.0,
    cost_per_trade: float = 250.0,
    spread_field: str = "spread_7d_vol_pts",
) -> BacktestResult:
    return run_point_in_time_backtest(
        snapshots,
        strategy=one_period_vol_spread_signal_strategy(
            threshold_vol_pts=threshold_vol_pts,
            notional_vega=notional_vega,
            cost_per_trade=cost_per_trade,
            spread_field=spread_field,
        ),
        execution_resolver=one_period_spread_execution_resolver(spread_field=spread_field),
        unresolved_orders="skip",
        assumptions={
            "execution": "screen-only synthetic fills; not executable economics",
            "source": "point-in-time historical spread snapshots",
            "spread_field": spread_field,
            "threshold_vol_pts": threshold_vol_pts,
            "notional_vega": notional_vega,
            "cost_per_trade": cost_per_trade,
        },
    )

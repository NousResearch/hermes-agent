from datetime import datetime, timezone

import pytest

from institutional_btc_vol.backtester import (
    BacktestContext,
    DecisionFrame,
    MarketSnapshot,
    SyntheticOrder,
    Trade,
    hold_one_period_vol_spread_strategy,
    one_period_spread_execution_resolver,
    one_period_vol_spread_signal_strategy,
    run_backtest,
    run_point_in_time_backtest,
)


def ts(hour: int) -> datetime:
    return datetime(2026, 5, 17, hour, tzinfo=timezone.utc)


def test_run_backtest_rejects_out_of_order_snapshots():
    snapshots = [
        MarketSnapshot(timestamp=ts(2), data={"btc_spot": 101}),
        MarketSnapshot(timestamp=ts(1), data={"btc_spot": 100}),
    ]

    with pytest.raises(ValueError, match="chronological"):
        run_backtest(snapshots, strategy=lambda snapshot, context: [])


def test_strategy_receives_only_prior_history_without_future_leakage():
    snapshots = [
        MarketSnapshot(timestamp=ts(1), data={"btc_spot": 100}),
        MarketSnapshot(timestamp=ts(2), data={"btc_spot": 101}),
        MarketSnapshot(timestamp=ts(3), data={"btc_spot": 102}),
    ]
    observed = []

    def recording_strategy(snapshot: MarketSnapshot, context: BacktestContext):
        observed.append((snapshot.timestamp, [row.timestamp for row in context.history]))
        return []

    run_backtest(snapshots, strategy=recording_strategy)

    assert [len(history) for _, history in observed] == [0, 1, 2]
    assert all(all(prior < current for prior in history) for current, history in observed)


def test_run_backtest_computes_deterministic_trade_metrics_and_drawdown():
    snapshots = [
        MarketSnapshot(timestamp=ts(1), data={"btc_spot": 100}),
        MarketSnapshot(timestamp=ts(2), data={"btc_spot": 110}),
        MarketSnapshot(timestamp=ts(3), data={"btc_spot": 105}),
    ]

    def fixture_strategy(snapshot: MarketSnapshot, context: BacktestContext):
        if snapshot.timestamp == ts(1):
            return [Trade(entry_time=ts(1), exit_time=ts(2), side="long", quantity=2, entry_price=100, exit_price=110, costs=1)]
        if snapshot.timestamp == ts(2):
            return [Trade(entry_time=ts(2), exit_time=ts(3), side="short", quantity=1, entry_price=110, exit_price=115, costs=0.5)]
        return []

    result = run_backtest(snapshots, strategy=fixture_strategy)

    assert result.trade_count == 2
    assert result.gross_pnl == pytest.approx(13.5)
    assert result.win_rate == pytest.approx(0.5)
    assert result.max_drawdown == pytest.approx(5.5)
    assert [trade.net_pnl for trade in result.trades] == [pytest.approx(19), pytest.approx(-5.5)]
    assert result.evidence_status == "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"


def test_hold_one_period_vol_spread_strategy_uses_prior_signal_and_next_realized_change():
    snapshots = [
        MarketSnapshot(timestamp=ts(1), data={"spread_7d_vol_pts": 6.0, "next_spread_7d_vol_pts": 4.0}),
        MarketSnapshot(timestamp=ts(2), data={"spread_7d_vol_pts": -5.5, "next_spread_7d_vol_pts": -2.5}),
        MarketSnapshot(timestamp=ts(3), data={"spread_7d_vol_pts": 1.0, "next_spread_7d_vol_pts": 0.5}),
    ]

    result = run_backtest(
        snapshots,
        strategy=hold_one_period_vol_spread_strategy(threshold_vol_pts=5.0, notional_vega=10_000, cost_per_trade=250),
    )

    assert result.trade_count == 2
    assert result.trades[0].side == "short_spread"
    assert result.trades[0].metadata["signal"] == "IBIT rich / Deribit cheap"
    assert result.trades[0].net_pnl == pytest.approx(19_750)
    assert result.trades[1].side == "long_spread"
    assert result.trades[1].metadata["signal"] == "IBIT cheap / Deribit rich"
    assert result.trades[1].net_pnl == pytest.approx(29_750)
    assert result.gross_pnl == pytest.approx(49_500)
    assert result.assumptions["execution"] == "screen-only synthetic fills; not executable economics"


def test_point_in_time_backtest_hides_future_fields_from_decision_frame():
    snapshots = [
        MarketSnapshot(timestamp=ts(1), data={"spread_7d_vol_pts": 6.0, "next_spread_7d_vol_pts": 4.0}),
        MarketSnapshot(timestamp=ts(2), data={"spread_7d_vol_pts": 4.0, "next_spread_7d_vol_pts": 2.0}),
    ]
    observed = []

    def strategy(frame: DecisionFrame):
        observed.append((frame.timestamp, dict(frame.data), [row.timestamp for row in frame.prior_history]))
        assert "next_spread_7d_vol_pts" not in frame.data
        if frame.timestamp == ts(1):
            return [
                SyntheticOrder(
                    decision_time=frame.timestamp,
                    side="short_spread",
                    quantity=10_000,
                    entry_price=frame.data["spread_7d_vol_pts"],
                    metadata={"signal": "fixture"},
                )
            ]
        return []

    def resolver(order: SyntheticOrder, frame: DecisionFrame, future_snapshots):
        return Trade(
            entry_time=order.decision_time,
            exit_time=future_snapshots[0].timestamp,
            side=order.side,
            quantity=order.quantity,
            entry_price=order.entry_price,
            exit_price=future_snapshots[0].data["spread_7d_vol_pts"],
            metadata=order.metadata,
        )

    result = run_point_in_time_backtest(snapshots, strategy=strategy, execution_resolver=resolver)

    assert result.trade_count == 1
    assert result.trades[0].exit_time == ts(2)
    assert result.trades[0].net_pnl == pytest.approx(20_000)
    assert observed[0][1] == {"spread_7d_vol_pts": 6.0}
    assert [len(history) for _, _, history in observed] == [0, 1]


def test_point_in_time_backtest_rejects_orders_before_decision_and_unresolved_future():
    snapshots = [
        MarketSnapshot(timestamp=ts(1), data={"spread_7d_vol_pts": 6.0}),
    ]

    def early_order_strategy(frame: DecisionFrame):
        return [SyntheticOrder(decision_time=ts(0), side="short_spread", quantity=1, entry_price=6.0)]

    with pytest.raises(ValueError, match="before the current decision frame"):
        run_point_in_time_backtest(snapshots, strategy=early_order_strategy, execution_resolver=lambda order, frame, future: None)

    def needs_future_strategy(frame: DecisionFrame):
        return [SyntheticOrder(decision_time=frame.timestamp, side="short_spread", quantity=1, entry_price=6.0)]

    with pytest.raises(ValueError, match="could not be resolved"):
        run_point_in_time_backtest(snapshots, strategy=needs_future_strategy, execution_resolver=lambda order, frame, future: None)


def test_resolver_based_vol_spread_strategy_uses_future_only_in_execution_resolver():
    snapshots = [
        MarketSnapshot(timestamp=ts(1), data={"spread_7d_vol_pts": 6.0, "next_spread_7d_vol_pts": 999.0}),
        MarketSnapshot(timestamp=ts(2), data={"spread_7d_vol_pts": 4.0, "next_spread_7d_vol_pts": 999.0}),
        MarketSnapshot(timestamp=ts(3), data={"spread_7d_vol_pts": -5.5, "next_spread_7d_vol_pts": 999.0}),
        MarketSnapshot(timestamp=ts(4), data={"spread_7d_vol_pts": -2.5, "next_spread_7d_vol_pts": 999.0}),
    ]

    result = run_point_in_time_backtest(
        snapshots,
        strategy=one_period_vol_spread_signal_strategy(threshold_vol_pts=5.0, notional_vega=10_000, cost_per_trade=250),
        execution_resolver=one_period_spread_execution_resolver(spread_field="spread_7d_vol_pts"),
    )

    assert result.trade_count == 2
    assert result.trades[0].side == "short_spread"
    assert result.trades[0].exit_price == 4.0
    assert result.trades[0].metadata["future_fields_visible_to_strategy"] == []
    assert result.trades[0].net_pnl == pytest.approx(19_750)
    assert result.trades[1].side == "long_spread"
    assert result.trades[1].exit_price == -2.5
    assert result.trades[1].net_pnl == pytest.approx(29_750)
    assert result.assumptions["execution"] == "screen-only synthetic fills; not executable economics"

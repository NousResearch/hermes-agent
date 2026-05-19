from __future__ import annotations

from institutional_btc_vol.historical_replay import build_venue_pair_spread_snapshots
from institutional_btc_vol.historical_schema import HistoricalOptionQuote, HistoricalSchemaError


def _quote(*, venue: str, symbol: str, dte: float, strike: float, underlying: float, mid_iv: float, ts: str, source_id: str) -> HistoricalOptionQuote:
    return HistoricalOptionQuote(
        as_of_utc=ts,
        event_ts=ts,
        available_ts=ts,
        venue=venue,
        underlying_symbol="IBIT" if venue == "OPRA" else "BTC",
        native_symbol=symbol,
        instrument_id=symbol,
        expiry="2025-01-17",
        dte=dte,
        option_type="call",
        strike_native=strike,
        strike_btc_equivalent=strike / 0.0005679 if venue == "OPRA" else strike,
        underlying_price=underlying,
        underlying_btc_equivalent=underlying / 0.0005679 if venue == "OPRA" else underlying,
        btc_per_share=0.0005679 if venue == "OPRA" else None,
        bid=1.0,
        ask=1.2,
        mid=1.1,
        bid_iv=None,
        ask_iv=None,
        mid_iv=mid_iv,
        iv_source="model_estimated_from_historical_bid_ask_mid" if venue == "OPRA" else "vendor_mark_iv",
        source_id=source_id,
        source_sha256=("a" if venue == "OPRA" else "b") * 64,
        source_confidence="licensed_vendor_api",
    )


def test_build_venue_pair_spread_snapshots_matches_nearest_tenor_and_atm():
    quotes = [
        _quote(venue="OPRA", symbol="IBIT-7D-ATM", dte=7, strike=55, underlying=55, mid_iv=0.52, ts="2025-01-02T15:30:00Z", source_id="ibit"),
        _quote(venue="OPRA", symbol="IBIT-7D-OTM", dte=7, strike=65, underlying=55, mid_iv=0.60, ts="2025-01-02T15:30:00Z", source_id="ibit"),
        _quote(venue="Deribit", symbol="BTC-7D-ATM", dte=7, strike=97000, underlying=97000, mid_iv=0.47, ts="2025-01-02T15:30:00Z", source_id="deribit"),
        _quote(venue="Deribit", symbol="BTC-12D-ATM", dte=12, strike=97000, underlying=97000, mid_iv=0.55, ts="2025-01-02T15:30:00Z", source_id="deribit"),
    ]

    snapshots = build_venue_pair_spread_snapshots(
        quotes,
        left_venue="OPRA",
        right_venue="Deribit",
        venue_pair="IBIT_DERIBIT",
        tenors=(7,),
        replay_id="hist-fixture-v1",
    )

    assert len(snapshots) == 1
    row = snapshots[0].to_dict()
    assert row["tenor"] == "7d"
    assert row["left_instrument"] == "IBIT-7D-ATM"
    assert row["right_instrument"] == "BTC-7D-ATM"
    assert row["spread_vol_pts"] == 5.0
    assert row["source_ids"] == ["ibit", "deribit"]
    assert row["evidence_status"] == "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"


def test_replay_builder_uses_available_time_as_decision_time_and_rejects_future_rows():
    quotes = [
        _quote(venue="OPRA", symbol="IBIT", dte=7, strike=55, underlying=55, mid_iv=0.52, ts="2025-01-02T15:31:00Z", source_id="ibit"),
        _quote(venue="Deribit", symbol="BTC", dte=7, strike=97000, underlying=97000, mid_iv=0.47, ts="2025-01-02T15:30:00Z", source_id="deribit"),
    ]
    snapshots = build_venue_pair_spread_snapshots(
        quotes,
        left_venue="OPRA",
        right_venue="Deribit",
        venue_pair="IBIT_DERIBIT",
        tenors=(7,),
        replay_id="hist-fixture-v1",
    )
    row = snapshots[0].to_dict()
    assert row["decision_ts"] == "2025-01-02T15:31:00Z"
    assert row["available_ts"] == "2025-01-02T15:31:00Z"

    bad = snapshots[0].with_updates(available_ts="2025-01-02T15:32:00Z")
    try:
        bad.to_dict()
    except HistoricalSchemaError as exc:
        assert "available_ts cannot be after decision_ts" in str(exc)
    else:
        raise AssertionError("future-unavailable replay row should fail")

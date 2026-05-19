from __future__ import annotations

from pathlib import Path

import pytest

from institutional_btc_vol.historical_schema import (
    HistoricalEtfHolding,
    HistoricalOptionQuote,
    HistoricalSchemaError,
    HistoricalSpreadSnapshot,
    HistoricalUnderlyingMark,
    read_jsonl_records,
    write_jsonl_records,
)


def test_historical_option_quote_validates_market_and_provenance():
    quote = HistoricalOptionQuote(
        as_of_utc="2025-01-02T15:30:00Z",
        event_ts="2025-01-02T15:30:00Z",
        available_ts="2025-01-02T15:30:01Z",
        venue="OPRA",
        underlying_symbol="IBIT",
        native_symbol="IBIT250117C00060000",
        instrument_id="IBIT-20250117-60-C",
        expiry="2025-01-17",
        dte=15.0,
        option_type="call",
        strike_native=60.0,
        strike_btc_equivalent=105650.0,
        underlying_price=55.0,
        underlying_btc_equivalent=96846.0,
        btc_per_share=0.0005679,
        bid=1.20,
        ask=1.35,
        mid=1.275,
        bid_iv=0.48,
        ask_iv=0.52,
        mid_iv=0.50,
        iv_source="model_estimated_from_historical_bid_ask_mid",
        source_id="ibit_opra_fixture",
        source_sha256="a" * 64,
        source_confidence="licensed_vendor_api",
    )

    row = quote.to_dict()
    assert row["execution_confidence"] == "screen_only_not_executable"
    assert row["quality_flags"] == ["ok"]
    assert row["mid"] == pytest.approx(1.275)

    with pytest.raises(HistoricalSchemaError, match="ask must be >= bid"):
        quote.with_updates(bid=1.40, ask=1.35).to_dict()

    with pytest.raises(HistoricalSchemaError, match="available_ts cannot precede event_ts"):
        quote.with_updates(available_ts="2025-01-02T15:29:59Z").to_dict()

    with pytest.raises(HistoricalSchemaError, match="invalid source_confidence"):
        quote.with_updates(source_confidence="magic_feed").to_dict()


def test_underlying_holding_and_spread_snapshot_roundtrip_jsonl(tmp_path: Path):
    underlying = HistoricalUnderlyingMark(
        as_of_utc="2025-01-02T15:30:00Z",
        event_ts="2025-01-02T15:30:00Z",
        available_ts="2025-01-02T15:30:01Z",
        symbol="BTC",
        price=97000.0,
        source_id="btc_reference_fixture",
        source_sha256="b" * 64,
        source_confidence="manual_fixture",
    )
    holding = HistoricalEtfHolding(
        as_of_utc="2025-01-02T21:30:00Z",
        event_ts="2025-01-02T21:30:00Z",
        available_ts="2025-01-02T22:00:00Z",
        etf_symbol="IBIT",
        btc_per_share=0.0005679,
        shares_per_btc=1760.8734,
        source_id="ibit_holdings_fixture",
        source_sha256="c" * 64,
        source_confidence="public_screen_reference",
    )
    spread = HistoricalSpreadSnapshot(
        replay_id="hist-btcvol-fixture-v1",
        decision_ts="2025-01-03T15:31:00Z",
        available_ts="2025-01-03T15:30:01Z",
        tenor="7d",
        venue_pair="IBIT_DERIBIT",
        left_mid_iv=0.52,
        right_mid_iv=0.47,
        spread_vol_pts=5.0,
        left_instrument="IBIT ATM synthetic pair",
        right_instrument="BTC-10JAN25-97000-C/P",
        source_ids=["ibit_opra_fixture", "deribit_fixture"],
        source_hashes=["a" * 64, "d" * 64],
    )

    path = tmp_path / "records.jsonl"
    write_jsonl_records([underlying, holding, spread], path)
    rows = read_jsonl_records(path)

    assert [row["record_type"] for row in rows] == ["underlying_mark", "etf_holding", "spread_snapshot"]
    assert rows[0]["execution_confidence"] == "screen_only_not_executable"
    assert rows[1]["btc_per_share"] == pytest.approx(0.0005679)
    assert rows[2]["evidence_status"] == "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"
    assert rows[2]["normalization"]["atm_rule"] == "nearest_strike"


def test_spread_snapshot_rejects_future_unavailable_decision_rows():
    spread = HistoricalSpreadSnapshot(
        replay_id="bad-future",
        decision_ts="2025-01-03T15:30:00Z",
        available_ts="2025-01-03T15:31:00Z",
        tenor="7d",
        venue_pair="IBIT_DERIBIT",
        left_mid_iv=0.52,
        right_mid_iv=0.47,
        spread_vol_pts=5.0,
        left_instrument="IBIT ATM",
        right_instrument="Deribit ATM",
        source_ids=["ibit", "deribit"],
        source_hashes=["a" * 64, "b" * 64],
    )
    # available one minute after the decision means the strategy could not know this row at decision time.
    with pytest.raises(HistoricalSchemaError, match="available_ts cannot be after decision_ts"):
        spread.to_dict()

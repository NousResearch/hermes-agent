from __future__ import annotations

import json
from pathlib import Path

import pytest

from institutional_btc_vol.historical_ingest import (
    HistoricalIngestError,
    parse_generic_deribit_options_jsonl,
    parse_generic_ibit_options_csv,
)
from institutional_btc_vol.historical_sources import source_file_sha256


def test_parse_generic_ibit_options_csv_normalizes_btc_equivalent_quote(tmp_path: Path):
    csv_path = tmp_path / "ibit_options.csv"
    csv_path.write_text(
        "ts,symbol,expiry,option_type,strike,underlying_price,btc_per_share,bid,ask,mid_iv\n"
        "2025-01-02T15:30:00Z,IBIT250117C00060000,2025-01-17,call,60,55,0.0005679,1.20,1.35,0.50\n",
        encoding="utf-8",
    )

    rows = parse_generic_ibit_options_csv(
        csv_path,
        source_id="ibit_opra_fixture",
        source_confidence="licensed_vendor_api",
        available_lag_seconds=1,
    )

    assert len(rows) == 1
    row = rows[0].to_dict()
    assert row["venue"] == "OPRA"
    assert row["underlying_symbol"] == "IBIT"
    assert row["strike_btc_equivalent"] == pytest.approx(60 / 0.0005679)
    assert row["underlying_btc_equivalent"] == pytest.approx(55 / 0.0005679)
    assert row["source_sha256"] == source_file_sha256(csv_path)
    assert row["iv_source"] == "model_estimated_from_historical_bid_ask_mid"
    assert row["execution_confidence"] == "screen_only_not_executable"


def test_parse_generic_deribit_options_jsonl_normalizes_mark_iv_quote(tmp_path: Path):
    jsonl_path = tmp_path / "deribit_options.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "timestamp": "2025-01-02T15:30:00Z",
                "instrument_name": "BTC-17JAN25-97000-C",
                "expiry": "2025-01-17",
                "option_type": "call",
                "strike": 97000,
                "underlying_price": 96850,
                "bid": 0.052,
                "ask": 0.055,
                "mark_iv": 51.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = parse_generic_deribit_options_jsonl(
        jsonl_path,
        source_id="deribit_fixture",
        source_confidence="licensed_vendor_api",
        available_lag_seconds=0,
    )

    assert len(rows) == 1
    row = rows[0].to_dict()
    assert row["venue"] == "Deribit"
    assert row["underlying_symbol"] == "BTC"
    assert row["native_symbol"] == "BTC-17JAN25-97000-C"
    assert row["mid"] == pytest.approx((0.052 + 0.055) / 2)
    assert row["mid_iv"] == pytest.approx(0.512)
    assert row["iv_source"] == "vendor_mark_iv"
    assert row["source_sha256"] == source_file_sha256(jsonl_path)


def test_ingest_skips_bad_rows_and_fails_when_no_valid_rows(tmp_path: Path):
    csv_path = tmp_path / "bad_ibit.csv"
    csv_path.write_text(
        "ts,symbol,expiry,option_type,strike,underlying_price,btc_per_share,bid,ask,mid_iv\n"
        "2025-01-02T15:30:00Z,IBIT_BAD,2025-01-17,call,60,55,0.0005679,2.00,1.00,0.50\n",
        encoding="utf-8",
    )

    with pytest.raises(HistoricalIngestError, match="no valid historical option rows"):
        parse_generic_ibit_options_csv(csv_path, source_id="bad", source_confidence="manual_fixture")

from datetime import datetime, timezone

import pytest

from institutional_btc_vol.databento_cme import (
    _parse_iso_datetime,
    _scaled_price,
    load_databento_api_key,
    parse_cme_bbo_rows,
    parse_cme_definition_rows,
)


def test_load_databento_key_from_oildesk_style_env_file_without_printing(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.delenv("DATABENTO_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('OTHER=1\nDATABENTO_API_KEY="db-test-secret"\n', encoding="utf-8")

    assert load_databento_api_key(env_file) == "db-test-secret"


def test_load_databento_key_checks_default_env_files_in_order(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.delenv("DATABENTO_KEY", raising=False)
    first = tmp_path / "missing.env"
    second = tmp_path / "oildesk.env"
    second.write_text("OTHER=1\nDATABENTO_API_KEY=db-default-secret\n", encoding="utf-8")

    assert load_databento_api_key(default_env_files=[first, second]) == "db-default-secret"


def test_scaled_price_uses_databento_display_factor_and_skips_sentinel():
    assert _scaled_price("86250000000000", "1000000000") == pytest.approx(86250.0)
    assert _scaled_price("9223372036854775807", "1000000000") is None


def test_parse_iso_datetime_accepts_databento_nanosecond_precision():
    parsed = _parse_iso_datetime("2026-05-17T09:53:02.458885000+00:00")

    assert parsed == datetime(2026, 5, 17, 9, 53, 2, 458885, tzinfo=timezone.utc)


def test_parse_cme_definitions_and_bbo_to_screen_only_rows():
    definition_csv = """ts_recv,ts_event,rtype,publisher_id,instrument_id,raw_symbol,security_update_action,instrument_class,min_price_increment,display_factor,expiration,activation,high_limit_price,low_limit_price,max_price_variation,trading_reference_price,unit_of_measure_qty,min_price_increment_amount,price_ratio,inst_attrib_value,underlying_id,raw_instrument_id,market_depth_implied,market_depth,market_segment_id,max_trade_vol,min_lot_size,min_lot_size_block,min_lot_size_round_lot,min_trade_vol,contract_multiplier,decay_quantity,original_contract_size,trading_reference_date,appl_id,maturity_year,decay_start_date,channel_id,currency,settl_currency,secsubtype,group,exchange,asset,cfi,security_type,unit_of_measure,underlying,strike_price_currency,strike_price,match_algorithm,md_security_trading_status,main_fraction,price_display_format,settl_price_type,sub_fraction,underlying_product,maturity_month,maturity_day,maturity_week,user_defined_instrument,contract_multiplier_unit,flow_schedule_type,tick_rule
1778803,1778803,29,1,42083769,BTCN6 P86250,A,P,5000000000,1000000000,1785510000000000000,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,20260515,0,2026,0,0,USD,USD,,BTC,CME,crypto,,OOF,USD,BTCN6,USD,86250000000000,,0,0,0,0,0,0,7,255,,N,0,0,
"""
    bbo_csv = """ts_recv,ts_event,rtype,publisher_id,instrument_id,side,price,size,flags,sequence,bid_px_00,ask_px_00,bid_sz_00,ask_sz_00,bid_ct_00,ask_ct_00
1778850000000000000,18446744073709551615,196,1,42083769,N,9223372036854775807,0,128,95604656,8465000000000,8725000000000,6,10,1,1
"""

    definitions = parse_cme_definition_rows(definition_csv)
    rows = parse_cme_bbo_rows(bbo_csv, definitions, as_of=datetime(2026, 5, 15, tzinfo=timezone.utc))

    assert definitions["42083769"]["raw_symbol"] == "BTCN6 P86250"
    assert definitions["42083769"]["strike_native"] == pytest.approx(86250.0)
    assert rows[0]["native_symbol"] == "BTCN6 P86250"
    assert rows[0]["option_type"] == "put"
    assert rows[0]["price_bid"] == pytest.approx(8465.0)
    assert rows[0]["price_ask"] == pytest.approx(8725.0)
    assert rows[0]["source_confidence"] == "licensed_vendor_api_databento"
    assert rows[0]["execution_confidence"] == "screen_only_not_executable"

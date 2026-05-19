import pytest

from institutional_btc_vol.run_monitor import (
    _derive_market_implied_btc_per_share,
    _find_latest_valid_ishares_cache,
    _looks_like_html,
    _select_btc_per_share_source,
)
from institutional_btc_vol.monitor import (
    compute_btc_per_share,
    estimate_black_scholes_iv,
    generate_monitor_report,
    normalize_deribit_instrument,
    normalize_ibit_option,
    quality_check_option_row,
    select_atm_by_moneyness,
)


def test_compute_btc_per_share_from_ishares_holdings_csv_with_btc_row_variants():
    csv_text = """Ticker,Name,Market Value,Weight (%),Shares,Price
BTC,Bitcoin,815000000,99.9,10000.12345678,81500
USD,Cash,100000,0.1,100000,1
Shares Outstanding,,,,17723800,
"""

    result = compute_btc_per_share(csv_text)

    assert result.btc_held == pytest.approx(10000.12345678)
    assert result.shares_outstanding == pytest.approx(17723800)
    assert result.btc_per_share == pytest.approx(10000.12345678 / 17723800)
    assert result.shares_per_btc == pytest.approx(17723800 / 10000.12345678)
    assert "BTC" in result.raw_btc_row_label


def test_compute_btc_per_share_from_real_ishares_preamble_format():
    csv_text = """iShares Bitcoin Trust ETF
Fund Holdings as of,"May 13, 2026"
Shares Outstanding,"1,448,200,000.00"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity,Market Currency,Accrual Date
"BTC","BITCOIN","-","Alternative","64,395,006,345.83","100.00","64,395,006,345.83","817,092.85700","BTC","-"
"USD","USD CASH","-","Cash","6,362.53","0.00","6,362.53","6,362.53000","USD","-"
"""

    result = compute_btc_per_share(csv_text)

    assert result.btc_held == pytest.approx(817092.857)
    assert result.shares_outstanding == pytest.approx(1448200000.0)
    assert result.btc_per_share == pytest.approx(817092.857 / 1448200000.0)


def test_normalize_deribit_instrument_accepts_single_digit_day_expiry():
    instrument = {
        "instrument_name": "BTC-5JUN26-82000-P",
        "mark_iv": 38.25,
        "bid_price": 0.012,
        "ask_price": 0.014,
        "underlying_price": 81600.0,
        "open_interest": 123.0,
        "volume": 45.0,
    }

    row = normalize_deribit_instrument(instrument, as_of="2026-05-14T19:30:00-05:00")

    assert row.expiry.isoformat() == "2026-06-05"
    assert row.option_type == "put"


def test_normalize_deribit_instrument_parses_expiry_strike_type_and_quality_flags():
    instrument = {
        "instrument_name": "BTC-26JUN26-90000-C",
        "mark_iv": 38.25,
        "bid_price": 0.012,
        "ask_price": 0.014,
        "underlying_price": 81600.0,
        "open_interest": 123.0,
        "volume": 45.0,
    }

    row = normalize_deribit_instrument(instrument, as_of="2026-05-14T19:30:00-05:00")

    assert row.native_symbol == "BTC-26JUN26-90000-C"
    assert row.expiry.isoformat() == "2026-06-26"
    assert row.option_type == "call"
    assert row.strike_native == 90000
    assert row.iv_mark == pytest.approx(0.3825)
    assert row.moneyness_spot == pytest.approx(90000 / 81600.0)
    assert row.source_confidence == "official_exchange_api"
    assert row.execution_confidence == "screen_only"
    assert "ok" in row.quality_flags


def test_normalize_ibit_option_converts_contract_to_btc_equivalent_notional():
    option = {
        "expiry": "2026-06-26",
        "strike": "46.00",
        "type": "call",
        "bid": "1.20",
        "ask": "1.35",
        "iv": "38.5%",
        "volume": "1,234",
        "open_interest": "5,678",
        "underlying_price": "46.17",
        "timestamp": "2026-05-14T19:30:00-05:00",
    }

    row = normalize_ibit_option(option, btc_per_share=0.000564212717166, as_of="2026-05-14T19:30:00-05:00")

    assert row.native_symbol == "IBIT-2026-06-26-46.0-C"
    assert row.expiry.isoformat() == "2026-06-26"
    assert row.option_type == "call"
    assert row.strike_native == pytest.approx(46.0)
    assert row.iv_mark == pytest.approx(0.385)
    assert row.contract_multiplier == 100
    assert row.btc_equivalent_per_contract == pytest.approx(100 * 0.000564212717166)
    assert row.notional_btc == pytest.approx(100 * 0.000564212717166)
    assert row.strike_btc_equivalent == pytest.approx(46.0 / 0.000564212717166)
    assert row.source_confidence == "semi_official_public_json"
    assert row.execution_confidence == "screen_only"


def test_quality_check_flags_crossed_market_and_stale_source():
    row = {
        "price_bid": 10.0,
        "price_ask": 9.5,
        "iv_mark": 0.4,
        "dte": 30,
        "timestamp_source": "2026-05-14T18:00:00-05:00",
    }

    flags = quality_check_option_row(row, as_of="2026-05-14T19:30:00-05:00", stale_minutes=30)

    assert "crossed_market" in flags
    assert "stale_source" in flags


def test_select_atm_by_moneyness_chooses_nearest_liquid_row():
    rows = [
        {"native_symbol": "low", "moneyness_spot": 0.98, "price_bid": 0, "price_ask": 0.02, "open_interest": 100},
        {"native_symbol": "atm", "moneyness_spot": 1.01, "price_bid": 0.01, "price_ask": 0.02, "open_interest": 10},
        {"native_symbol": "far", "moneyness_spot": 1.10, "price_bid": 0.01, "price_ask": 0.02, "open_interest": 100},
    ]

    selected = select_atm_by_moneyness(rows)

    assert selected["native_symbol"] == "atm"


def test_estimate_black_scholes_iv_recovers_reasonable_call_vol():
    iv = estimate_black_scholes_iv(
        option_type="call",
        spot=100.0,
        strike=100.0,
        years_to_expiry=30 / 365,
        price=4.57,
        rate=0.045,
    )

    assert iv == pytest.approx(0.38, abs=0.02)


def test_generate_monitor_report_labels_screen_only_and_cme_missing():
    report = generate_monitor_report(
        run_id="btcvol-20260514-193000",
        as_of_cst="2026-05-14 19:30:00 CDT",
        btc_spot=81600.0,
        btc_per_share=0.0005642,
        deribit_atm_rows=[{"expiry": "2026-06-26", "dte": 43, "iv_mark": 0.379, "native_symbol": "BTC-26JUN26-82000-C"}],
        ibit_atm_rows=[],
        dislocations=[{"candidate": "IBIT 30D vs Deribit 30D", "gross_iv_diff_vol_pts": 4.2, "confidence": "screen-only", "next_action": "quote"}],
        quality_warnings=["CME source missing"],
    )

    assert "btcvol-20260514-193000" in report
    assert "screen-only" in report
    assert "CME source missing" in report
    assert "quote" in report


def test_ishares_html_detection_and_cached_csv_fallback(tmp_path):
    assert _looks_like_html('<!DOCTYPE html><html><head></head><body>blocked</body></html>')
    base = tmp_path / 'data'
    bad_dir = base / 'raw' / 'newer-bad'
    good_dir = base / 'raw' / 'older-good'
    bad_dir.mkdir(parents=True)
    good_dir.mkdir(parents=True)
    (bad_dir / 'ishares_ibit_holdings.csv').write_text('<html><head></head><body>not csv</body></html>', encoding='utf-8')
    (good_dir / 'ishares_ibit_holdings.csv').write_text('''iShares Bitcoin Trust ETF
Shares Outstanding,"1,448,200,000.00"
Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity,Market Currency,Accrual Date
"BTC","BITCOIN","-","Alternative","1","100.00","1","817,092.85700","BTC","-"
''', encoding='utf-8')

    fallback = _find_latest_valid_ishares_cache(base)

    assert fallback is not None
    btc_per_share, path = fallback
    assert btc_per_share == pytest.approx(817092.857 / 1448200000.0)
    assert path.name == 'ishares_ibit_holdings.csv'


def test_market_implied_btc_per_share_uses_ibit_last_trade_over_btc_spot():
    chain = {"lastTrade": "IBIT $44.73  -0.12"}

    result = _derive_market_implied_btc_per_share(chain, btc_spot=78_760.0)

    assert result is not None
    btc_per_share, warnings = result
    assert btc_per_share == pytest.approx(44.73 / 78760.0)
    assert any("market-implied" in warning.lower() for warning in warnings)
    assert any("not official holdings" in warning.lower() for warning in warnings)


def test_btc_per_share_source_prefers_cached_official_but_cross_checks_market_implied():
    cached = (0.0005679, "cached official csv")
    market = (0.0005684, ["market-implied IBIT/BTC cross-check"])

    value, warnings = _select_btc_per_share_source(live=None, cached=cached, market_implied=market)

    assert value == pytest.approx(0.0005679)
    assert any("cached official" in warning.lower() for warning in warnings)
    assert any("independent market-implied cross-check" in warning.lower() for warning in warnings)


def test_btc_per_share_source_uses_market_implied_when_official_and_cache_missing():
    market = (0.0005684, ["market-implied IBIT/BTC fallback; not official holdings"])

    value, warnings = _select_btc_per_share_source(live=None, cached=None, market_implied=market)

    assert value == pytest.approx(0.0005684)
    assert any("market-implied" in warning.lower() for warning in warnings)
    assert any("lower-confidence" in warning.lower() for warning in warnings)

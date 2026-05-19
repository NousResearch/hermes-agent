from institutional_btc_vol.market_diagnostics import diagnose_deribit_options, diagnose_nasdaq_ibit_chain


def test_diagnose_nasdaq_ibit_chain_counts_missing_crossed_and_wide_markets():
    chain = {
        "table": {
            "rows": [
                {"expirygroup": "May 15, 2026"},
                {"strike": "50", "c_Bid": "1.00", "c_Ask": "1.20", "p_Bid": "0.90", "p_Ask": "1.10"},
                {"strike": "51", "c_Bid": "--", "c_Ask": "0.70", "p_Bid": "0.80", "p_Ask": "0.70"},
                {"strike": "52", "c_Bid": "0.10", "c_Ask": "0.50", "p_Bid": "", "p_Ask": ""},
            ]
        }
    }

    diagnostics = diagnose_nasdaq_ibit_chain(chain, wide_pct_threshold=0.50)

    assert diagnostics["source"] == "IBIT Nasdaq option-chain"
    assert diagnostics["option_markets"] == 6
    assert diagnostics["valid_bid_ask"] == 4
    assert diagnostics["missing_bid_ask"] == 2
    assert diagnostics["crossed_markets"] == 1
    assert diagnostics["wide_markets"] == 1
    assert diagnostics["grade"] == "yellow"
    assert "1 crossed market" in diagnostics["notes"]


def test_diagnose_deribit_options_counts_missing_iv_and_bid_ask_anomalies():
    rows = [
        {"instrument_name": "BTC-15MAY26-80000-C", "mark_iv": 32.0, "bid_price": 0.01, "ask_price": 0.02},
        {"instrument_name": "BTC-15MAY26-82000-C", "mark_iv": None, "bid_price": 0.03, "ask_price": 0.02},
        {"instrument_name": "BTC-15MAY26-84000-C", "mark_iv": 36.0, "bid_price": None, "ask_price": 0.04},
    ]

    diagnostics = diagnose_deribit_options(rows)

    assert diagnostics["source"] == "Deribit BTC options"
    assert diagnostics["option_markets"] == 3
    assert diagnostics["valid_bid_ask"] == 2
    assert diagnostics["missing_bid_ask"] == 1
    assert diagnostics["crossed_markets"] == 1
    assert diagnostics["missing_iv"] == 1
    assert diagnostics["grade"] == "red"
    assert "1 missing IV" in diagnostics["notes"]

from institutional_btc_vol.hedge_calculators import (
    build_miner_runway_case_study,
    build_treasury_hedge_case_study,
)


def test_treasury_hedge_case_study_models_floor_and_cap_without_executable_premium():
    result = build_treasury_hedge_case_study(
        btc_held=1000,
        spot=80000,
        hedge_ratio=0.35,
        floor_pct=0.75,
        cap_pct=1.25,
        tenor_days=90,
    )

    assert result["title"] == "Corporate BTC treasury hedge case study"
    assert result["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"
    assert result["hedged_btc"] == 350
    assert result["unhedged_btc"] == 650
    assert result["spot_value_usd"] == 80_000_000
    assert result["floor_price"] == 60_000
    assert result["cap_price"] == 100_000
    assert result["protected_value_at_floor_usd"] == 21_000_000
    assert result["scenario_rows"][0]["btc_price"] == 48_000
    assert result["scenario_rows"][0]["hedged_sleeve_value_usd"] == 21_000_000
    assert result["scenario_rows"][-1]["hedged_sleeve_value_usd"] == 35_000_000
    assert result["quote_control"] == "Premium and executable levels require two-counterparty quote verification."
    assert "premium_usd" not in result


def test_miner_runway_case_study_flags_overhedging_and_cash_runway():
    result = build_miner_runway_case_study(
        monthly_btc_production=120,
        spot=80000,
        cash_cost_per_btc=42000,
        cash_balance_usd=15_000_000,
        monthly_fixed_cost_usd=4_000_000,
        hedge_ratio=0.5,
        floor_pct=0.7,
        tenor_months=6,
    )

    assert result["title"] == "Miner runway protection case study"
    assert result["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"
    assert result["hedged_monthly_btc"] == 60
    assert result["six_month_production_btc"] == 720
    assert result["six_month_hedged_btc"] == 360
    assert result["floor_price"] == 56_000
    assert result["monthly_revenue_at_spot_usd"] == 9_600_000
    assert result["monthly_floor_revenue_on_hedged_btc_usd"] == 3_360_000
    assert result["cash_runway_months_before_hedge"] == 3.75
    assert result["warnings"] == ["Hedge only conservative production; avoid overhedging and collateral stress."]
    assert result["quote_control"] == "Indicative economics require quote verification before investor/client use."


def test_miner_runway_case_study_flags_high_hedge_ratio():
    result = build_miner_runway_case_study(
        monthly_btc_production=100,
        spot=80000,
        cash_cost_per_btc=45000,
        cash_balance_usd=10_000_000,
        monthly_fixed_cost_usd=2_000_000,
        hedge_ratio=0.8,
        floor_pct=0.7,
        tenor_months=3,
    )

    assert "Hedge ratio above 50% should be treated as aggressive until treasury policy approves it." in result["warnings"]

from __future__ import annotations

from typing import Any

EVIDENCE_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"


def _round_money(value: float) -> float:
    return round(float(value), 2)


def _round_qty(value: float) -> float:
    rounded = round(float(value), 4)
    return int(rounded) if rounded.is_integer() else rounded


def build_treasury_hedge_case_study(
    *,
    btc_held: float,
    spot: float,
    hedge_ratio: float,
    floor_pct: float,
    cap_pct: float,
    tenor_days: int,
) -> dict[str, Any]:
    hedged_btc = btc_held * hedge_ratio
    unhedged_btc = btc_held - hedged_btc
    floor_price = spot * floor_pct
    cap_price = spot * cap_pct
    scenario_prices = [spot * 0.6, floor_price, spot, cap_price, spot * 1.5]
    scenario_rows = []
    for price in scenario_prices:
        hedged_price = min(max(price, floor_price), cap_price)
        scenario_rows.append(
            {
                "btc_price": _round_money(price),
                "hedged_sleeve_value_usd": _round_money(hedged_btc * hedged_price),
                "unhedged_sleeve_value_usd": _round_money(unhedged_btc * price),
                "total_program_value_usd": _round_money((hedged_btc * hedged_price) + (unhedged_btc * price)),
            }
        )
    return {
        "title": "Corporate BTC treasury hedge case study",
        "evidence_status": EVIDENCE_STATUS,
        "structure": "put-spread collar / covered-call policy preview",
        "btc_held": _round_qty(btc_held),
        "hedged_btc": _round_qty(hedged_btc),
        "unhedged_btc": _round_qty(unhedged_btc),
        "hedge_ratio": hedge_ratio,
        "tenor_days": tenor_days,
        "spot": _round_money(spot),
        "spot_value_usd": _round_money(btc_held * spot),
        "floor_price": _round_money(floor_price),
        "cap_price": _round_money(cap_price),
        "protected_value_at_floor_usd": _round_money(hedged_btc * floor_price),
        "scenario_rows": scenario_rows,
        "quote_control": "Premium and executable levels require two-counterparty quote verification.",
        "notes": [
            "Use as board-policy illustration, not as executable pricing.",
            "Premium, strikes, and liquidity must be verified with counterparties.",
        ],
    }


def build_miner_runway_case_study(
    *,
    monthly_btc_production: float,
    spot: float,
    cash_cost_per_btc: float,
    cash_balance_usd: float,
    monthly_fixed_cost_usd: float,
    hedge_ratio: float,
    floor_pct: float,
    tenor_months: int,
) -> dict[str, Any]:
    hedged_monthly_btc = monthly_btc_production * hedge_ratio
    floor_price = spot * floor_pct
    six_month_production = monthly_btc_production * tenor_months
    six_month_hedged = hedged_monthly_btc * tenor_months
    warnings = ["Hedge only conservative production; avoid overhedging and collateral stress."]
    if hedge_ratio > 0.5:
        warnings.append("Hedge ratio above 50% should be treated as aggressive until treasury policy approves it.")
    cash_runway = cash_balance_usd / monthly_fixed_cost_usd if monthly_fixed_cost_usd else None
    return {
        "title": "Miner runway protection case study",
        "evidence_status": EVIDENCE_STATUS,
        "structure": "production floor / collar policy preview",
        "monthly_btc_production": _round_qty(monthly_btc_production),
        "hedged_monthly_btc": _round_qty(hedged_monthly_btc),
        "six_month_production_btc": _round_qty(six_month_production),
        "six_month_hedged_btc": _round_qty(six_month_hedged),
        "hedge_ratio": hedge_ratio,
        "tenor_months": tenor_months,
        "spot": _round_money(spot),
        "cash_cost_per_btc": _round_money(cash_cost_per_btc),
        "floor_price": _round_money(floor_price),
        "monthly_revenue_at_spot_usd": _round_money(monthly_btc_production * spot),
        "monthly_cash_cost_usd": _round_money(monthly_btc_production * cash_cost_per_btc),
        "monthly_floor_revenue_on_hedged_btc_usd": _round_money(hedged_monthly_btc * floor_price),
        "cash_runway_months_before_hedge": None if cash_runway is None else round(cash_runway, 2),
        "warnings": warnings,
        "quote_control": "Indicative economics require quote verification before investor/client use.",
    }

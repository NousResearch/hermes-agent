"""
utils/formatter.py
------------------
Terminal and JSON output formatting for all three NFT analytics panels.

All dictionary keys referenced here are the **spec-canonical names** produced
directly by the merged analyzer classes:

    WalletRiskAnalyzer.analyze()  →  past_tx_count, risk_score,
                                     mint_dump_pattern, wash_trading_signals,
                                     fast_flips
    SmartMoneyTracker.analyze()   →  smart_entry, avg_roi, active_months
    WalletProfiler.analyze()      →  total_nfts, avg_flip_duration,
                                     high_risk_collections, net_roi

Terminal output format (colour only in real TTY):

    ================ Wallet Risk Analyzer ================
    Wallet: D1f3...9a7
    Risk Score: HIGH
    - Past tx count: 124
    - Mint-dump pattern: YES
    - Wash trading signals: 2
    - Fast flips in same collection: YES

    ================ Smart Money Tracker =================
    Wallet: D1f3...9a7
    - Smart wallet entry: YES
    - Avg ROI: +12%
    - Active 6 months

    ================== Wallet Profiling ==================
    - Total NFTs minted: 42
    - Avg flip duration: 3 days
    - High risk collections: 3
    - Net ROI: +18%
    ======================================================
"""

from __future__ import annotations

import json
from typing import Any, Dict

from .helpers import shorten_address, yes_no, roi_str

# ── Layout ────────────────────────────────────────────────────────────────────
WIDTH     = 54
FILL      = "="
ANSI      = {
    "HIGH":   "\033[91m",   # red
    "MEDIUM": "\033[93m",   # yellow
    "LOW":    "\033[92m",   # green
    "RESET":  "\033[0m",
}


def _header(label: str) -> str:
    """Centre *label* in a '=====  label  =====' line of exactly WIDTH chars."""
    inner   = f" {label} "
    padding = WIDTH - len(inner)
    left    = padding // 2
    right   = padding - left
    return FILL * left + inner + FILL * right


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def format_report(
    risk_result:        Dict[str, Any],
    smart_money_result: Dict[str, Any],
    profiler_result:    Dict[str, Any],
    colour: bool = True,
) -> str:
    """
    Combine the three analyzer results into the canonical terminal report.

    Parameters
    ----------
    risk_result        : ``WalletRiskAnalyzer.analyze()`` output
    smart_money_result : ``SmartMoneyTracker.analyze()`` output
    profiler_result    : ``WalletProfiler.analyze()`` output
    colour             : Apply ANSI colour to the Risk Score line.
                         Pass ``False`` when stdout is not a real TTY.

    Returns
    -------
    str
        Multi-line report string ready for ``print()``.

    Key contract (all spec-canonical)
    ----------------------------------
    risk_result        : wallet, past_tx_count, risk_score,
                         mint_dump_pattern, wash_trading_signals, fast_flips
    smart_money_result : wallet, smart_entry, avg_roi, active_months
    profiler_result    : wallet, total_nfts, avg_flip_duration,
                         high_risk_collections, net_roi
    """
    wallet_short = shorten_address(risk_result["wallet"])
    risk_score   = risk_result["risk_score"]

    if colour:
        clr   = ANSI.get(risk_score, "")
        reset = ANSI["RESET"]
    else:
        clr = reset = ""

    lines = [
        "",
        _header("Wallet Risk Analyzer"),
        f"Wallet: {wallet_short}",
        f"Risk Score: {clr}{risk_score}{reset}",
        f"- Past tx count: {'1000+' if risk_result['past_tx_count'] >= 1000 else risk_result['past_tx_count']}",
        f"- Mint-dump pattern: {yes_no(risk_result['mint_dump_pattern'])}",
        f"- Wash trading signals: {risk_result['wash_trading_signals']}",
        f"- Fast flips in same collection: {yes_no(risk_result['fast_flips'])}",
        "",
        _header("Smart Money Tracker"),
        f"Wallet: {wallet_short}",
        f"- Smart wallet entry: {yes_no(smart_money_result['smart_entry'])}",
        f"- Avg ROI: {roi_str(smart_money_result['avg_roi'])}",
        f"- Active {smart_money_result['active_months']} months",
        "",
        _header("Wallet Profiling"),
        f"- Total NFTs minted: {profiler_result['total_nfts']}",
        f"- Avg flip duration: {profiler_result['avg_flip_duration']} days",
        f"- High risk collections: {profiler_result['high_risk_collections']}",
        f"- Net ROI: {roi_str(profiler_result['net_roi'])}",
        FILL * WIDTH,
        "",
    ]

    return "\n".join(lines)


def format_json(
    risk_result:        Dict[str, Any],
    smart_money_result: Dict[str, Any],
    profiler_result:    Dict[str, Any],
) -> str:
    """
    Return a pretty-printed JSON string containing all three result dicts.

    Useful for piping to other tools or persisting reports.

    The top-level keys are:
        ``wallet_risk_analyzer``, ``smart_money_tracker``, ``wallet_profiling``
    """
    payload = {
        "wallet_risk_analyzer": risk_result,
        "smart_money_tracker":  smart_money_result,
        "wallet_profiling":     profiler_result,
    }
    return json.dumps(payload, indent=2, default=str)

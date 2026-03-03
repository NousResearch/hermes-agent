"""
skills/nft-analytics/__init__.py
---------------------------------
NFT Analytics Hermes skill — package root.

Provides the full public API in one import:

    from skills.nft_analytics import (
        HeliusClient,
        WalletRiskAnalyzer, SmartMoneyTracker, WalletProfiler,
        format_report, format_json,
        validate_wallet, assert_valid_wallet,
    )

Spec-canonical output keys
--------------------------
WalletRiskAnalyzer  → wallet, past_tx_count, risk_score,
                      mint_dump_pattern, wash_trading_signals, fast_flips
SmartMoneyTracker   → wallet, smart_entry, avg_roi, active_months
WalletProfiler      → wallet, total_nfts, avg_flip_duration,
                      high_risk_collections, net_roi
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__  = "Hermes Skill Author"
__license__ = "MIT"

from .helius_client import HeliusClient          # noqa: F401

from .analyzers import (                         # noqa: F401
    WalletRiskAnalyzer,
    SmartMoneyTracker,
    WalletProfiler,
)

from .utils import (                             # noqa: F401
    format_report,
    format_json,
    validate_wallet,
    assert_valid_wallet,
)

__all__ = [
    "__version__",
    "HeliusClient",
    "WalletRiskAnalyzer",
    "SmartMoneyTracker",
    "WalletProfiler",
    "format_report",
    "format_json",
    "validate_wallet",
    "assert_valid_wallet",
]

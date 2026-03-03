"""
analyzers/__init__.py
---------------------
Public surface of the ``analyzers`` sub-package.

All three classes produce **spec-canonical key names** directly.
No wrapper classes, no parallel naming systems.

Importable as:

    from analyzers import WalletRiskAnalyzer, SmartMoneyTracker, WalletProfiler

Or from individual modules:

    from analyzers.risk        import WalletRiskAnalyzer
    from analyzers.smart_money import SmartMoneyTracker
    from analyzers.profiler    import WalletProfiler

Spec key summary
----------------
WalletRiskAnalyzer.analyze()  → wallet, past_tx_count, risk_score,
                                 mint_dump_pattern, wash_trading_signals,
                                 fast_flips

SmartMoneyTracker.analyze()   → wallet, smart_entry, avg_roi, active_months

WalletProfiler.analyze()      → wallet, total_nfts, avg_flip_duration,
                                 high_risk_collections, net_roi
"""

from __future__ import annotations

from .risk        import WalletRiskAnalyzer  # noqa: F401
from .smart_money import SmartMoneyTracker   # noqa: F401
from .profiler    import WalletProfiler      # noqa: F401

__all__ = [
    "WalletRiskAnalyzer",
    "SmartMoneyTracker",
    "WalletProfiler",
]

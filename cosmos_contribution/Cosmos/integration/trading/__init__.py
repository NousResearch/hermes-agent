"""
Cosmos Unified Trading Interface.

Provides a unified trading API that:
1. Uses Bankr as the primary trading engine for simplicity
2. Falls back to direct RPC/API when Bankr is unavailable or can't handle the request
3. Supports multiple chains: Base, Ethereum, Solana, Polygon

Priority Order:
1. Bankr Agent API (simple, handles most cases)
2. Chain-specific implementations (Jupiter, PumpPortal, etc.)
"""

from .unified_trader import UnifiedTrader, TradeRequest, TradeResponse
from .fallback_manager import FallbackManager

__all__ = [
    'UnifiedTrader',
    'TradeRequest',
    'TradeResponse',
    'FallbackManager',
    'get_trader',
]

_trader: "UnifiedTrader" = None


def get_trader() -> "UnifiedTrader":
    """Get the global unified trader instance."""
    global _trader
    if _trader is None:
        _trader = UnifiedTrader()
    return _trader

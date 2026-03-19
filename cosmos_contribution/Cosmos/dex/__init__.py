# DEXAI v2.0 - Cosmos Collective DEX Screener
# A DexScreener replacement powered by the Cosmos AI Collective

"""
DEXAI Module

AI-powered DEX screener with:
- Real-time token data from DexScreener, Birdeye, Jupiter
- AI-powered predictions from Cosmos Collective
- Token-gated boosts (pay FARNS to burn, or SOL to ecosystem)
- Collective-verified extended boosts with on-chain research
- Velocity, volume, AI prediction tabs
- Quantum simulation integration from FarSight
- Top 100+ tokens cached and auto-refreshed every 30s
- Candlestick charts via GeckoTerminal OHLCV
- WebSocket real-time updates

To run standalone:
    cd cosmos/dex
    npm install
    npm run dev

The server runs on port 3847 (configurable via DEXAI_PORT env var).

To integrate with main Cosmos server:
    from Cosmos.dex.dex_proxy import register_dex_routes
    register_dex_routes(app)

Access at /dex or /DEXAI on the main server.
"""

from .api import DexAPI, TokenPair, TrendingToken
from .dex_proxy import register_dex_routes

__all__ = ['DexAPI', 'TokenPair', 'TrendingToken', 'register_dex_routes']
__version__ = '2.0.0'

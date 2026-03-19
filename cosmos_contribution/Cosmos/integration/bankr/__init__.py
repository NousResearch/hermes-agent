"""
Bankr API Integration for Cosmos.

Provides crypto trading, DeFi operations, and Polymarket access via Bankr Agent API.

Capabilities:
- Multi-chain trading (Base, Ethereum, Solana, Polygon)
- Real-time price data and market analysis
- Polymarket prediction market access
- Portfolio tracking and management
- x402 micropayments support
"""

from .client import BankrClient, BankrError
from .trading import BankrTrading, TradeResult
from .market import BankrMarket
from .polymarket import BankrPolymarket, BetResult
from .portfolio import BankrPortfolio
from .config import BankrConfig, get_bankr_config

__all__ = [
    'BankrClient',
    'BankrError',
    'BankrTrading',
    'TradeResult',
    'BankrMarket',
    'BankrPolymarket',
    'BetResult',
    'BankrPortfolio',
    'BankrConfig',
    'get_bankr_config',
]

# Convenience function to get a configured client
_client_instance = None


def get_bankr_client() -> BankrClient:
    """Get or create the global Bankr client instance."""
    global _client_instance
    if _client_instance is None:
        config = get_bankr_config()
        _client_instance = BankrClient(api_key=config.api_key)
    return _client_instance

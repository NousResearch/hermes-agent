"""
Bankr Market Data Module.

Provides real-time price data, market analysis, and trends.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime

from .client import BankrClient
from .config import get_bankr_config

logger = logging.getLogger(__name__)


@dataclass
class TokenPrice:
    """Token price information."""
    symbol: str
    price_usd: Decimal
    change_24h: float
    volume_24h: Decimal
    market_cap: Decimal
    timestamp: datetime

    @classmethod
    def from_response(cls, data: Dict) -> "TokenPrice":
        return cls(
            symbol=data.get("symbol", ""),
            price_usd=Decimal(str(data.get("price", 0))),
            change_24h=data.get("change_24h", 0.0),
            volume_24h=Decimal(str(data.get("volume_24h", 0))),
            market_cap=Decimal(str(data.get("market_cap", 0))),
            timestamp=datetime.now(),
        )


@dataclass
class MarketTrend:
    """Market trend information."""
    direction: str  # bullish, bearish, neutral
    strength: float  # 0-1
    sentiment: str
    trending_tokens: List[str]
    top_gainers: List[Dict]
    top_losers: List[Dict]


class BankrMarket:
    """
    Market data and analysis via Bankr API.

    Provides:
    - Real-time token prices
    - Market trends and sentiment
    - Technical analysis
    - Trending tokens
    """

    def __init__(self, client: BankrClient = None):
        self.client = client
        self.config = get_bankr_config()

    async def _get_client(self) -> BankrClient:
        """Get or create client."""
        if self.client is None:
            from . import get_bankr_client
            self.client = get_bankr_client()
        return self.client

    async def get_price(self, token: str) -> TokenPrice:
        """
        Get current price for a token.

        Args:
            token: Token symbol (e.g., "ETH", "BTC")

        Returns:
            TokenPrice with current market data
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                f"Get detailed price information for {token}"
            )

            return TokenPrice.from_response({
                "symbol": token.upper(),
                "price": result.get("price", 0),
                "change_24h": result.get("change_24h", 0),
                "volume_24h": result.get("volume_24h", 0),
                "market_cap": result.get("market_cap", 0),
            })

        except Exception as e:
            logger.error(f"Price fetch failed for {token}: {e}")
            return TokenPrice(
                symbol=token.upper(),
                price_usd=Decimal(0),
                change_24h=0.0,
                volume_24h=Decimal(0),
                market_cap=Decimal(0),
                timestamp=datetime.now(),
            )

    async def get_prices(self, tokens: List[str]) -> Dict[str, TokenPrice]:
        """Get prices for multiple tokens."""
        results = {}
        for token in tokens:
            results[token] = await self.get_price(token)
        return results

    async def get_market_trends(self) -> MarketTrend:
        """
        Get current market trends and sentiment.

        Returns:
            MarketTrend with overall market analysis
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                "Analyze current crypto market trends, sentiment, and top movers"
            )

            return MarketTrend(
                direction=result.get("direction", "neutral"),
                strength=result.get("strength", 0.5),
                sentiment=result.get("sentiment", "neutral"),
                trending_tokens=result.get("trending", []),
                top_gainers=result.get("gainers", []),
                top_losers=result.get("losers", []),
            )

        except Exception as e:
            logger.error(f"Market trends fetch failed: {e}")
            return MarketTrend(
                direction="unknown",
                strength=0,
                sentiment="error",
                trending_tokens=[],
                top_gainers=[],
                top_losers=[],
            )

    async def get_technical_analysis(
        self,
        token: str,
        timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """
        Get technical analysis for a token.

        Args:
            token: Token symbol
            timeframe: Chart timeframe (1h, 4h, 1d)

        Returns:
            Technical analysis indicators
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                f"Provide technical analysis for {token} on {timeframe} timeframe"
            )

            return result

        except Exception as e:
            logger.error(f"TA fetch failed for {token}: {e}")
            return {"error": str(e)}

    async def get_trending_tokens(self, chain: str = None) -> List[Dict]:
        """
        Get trending tokens.

        Args:
            chain: Optional chain filter

        Returns:
            List of trending token information
        """
        try:
            client = await self._get_client()
            chain_filter = f" on {chain}" if chain else ""
            result = await client.execute(
                f"What are the trending tokens{chain_filter}?"
            )

            return result.get("tokens", [])

        except Exception as e:
            logger.error(f"Trending fetch failed: {e}")
            return []

    async def get_chart_data(
        self,
        token: str,
        timeframe: str = "1d",
        limit: int = 100
    ) -> List[Dict]:
        """
        Get OHLCV chart data.

        Args:
            token: Token symbol
            timeframe: Candle timeframe
            limit: Number of candles

        Returns:
            List of OHLCV candles
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                f"Get {limit} {timeframe} candles for {token}"
            )

            return result.get("candles", [])

        except Exception as e:
            logger.error(f"Chart data fetch failed for {token}: {e}")
            return []

    async def search_tokens(self, query: str) -> List[Dict]:
        """
        Search for tokens by name or symbol.

        Args:
            query: Search query

        Returns:
            List of matching tokens
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                f"Search for crypto tokens matching: {query}"
            )

            return result.get("results", [])

        except Exception as e:
            logger.error(f"Token search failed: {e}")
            return []

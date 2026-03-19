"""
cosmos Market Sentiment & Conditions.

"The vibes are currently: Bullish."
"""

import aiohttp
from loguru import logger
from typing import Dict, Any

class MarketSentimentSkill:
    def __init__(self):
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.coingecko_url = "https://api.coingecko.com/api/v3"

    async def get_fear_and_greed(self) -> Dict:
        """Fetch the current Crypto Fear & Greed Index."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.fear_greed_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", [{}])[0]
                return {}

    async def get_global_market_cap(self) -> Dict:
        """Fetch global crypto market data from CoinGecko."""
        url = f"{self.coingecko_url}/global"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", {})
                return {}

    async def get_token_price(self, token_id: str = "bitcoin") -> Dict:
        """Fetch current price and 24h change for a token."""
        url = f"{self.coingecko_url}/simple/price?ids={token_id}&vs_currencies=usd&include_24hr_change=true"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}

market_sentiment = MarketSentimentSkill()

"""
cosmos Polymarket Integration.

"The crowd is usually right, but I'm righter."
"""

import aiohttp
from loguru import logger
from typing import Dict, Any, List

class PolyMarketSkill:
    def __init__(self):
        self.gamma_url = "https://gamma-api.polymarket.com"

    async def get_active_markets(self, limit: int = 10) -> List[Dict]:
        """Fetch active prediction markets."""
        url = f"{self.base_url}/markets?active=true&limit={limit}" # Note: base_url not defined, should be gamma_url
        # Correcting the typo in the next iteration or here
        url = f"{self.gamma_url}/markets?active=true&limit={limit}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []

    async def search_markets(self, query: str) -> List[Dict]:
        """Search Polymarket events/markets."""
        url = f"{self.gamma_url}/events?search={query}&active=true"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []

    async def get_market_odds(self, market_id: str) -> Dict:
        """Get specific market details and prices/odds."""
        url = f"{self.gamma_url}/markets/{market_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}

polymarket = PolyMarketSkill()

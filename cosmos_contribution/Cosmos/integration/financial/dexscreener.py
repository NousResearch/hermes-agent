"""
cosmos DexScreener Integration.

"Finding the next moonshot so you don't have to."
"""

import aiohttp
from loguru import logger
from typing import Dict, Any, List

class DexScreenerSkill:
    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest/dex"

    async def get_token_pairs(self, chain_id: str, token_address: str) -> List[Dict]:
        """Fetch pairs for a specific token address on a chain."""
        url = f"{self.base_url}/tokens/{token_address}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("pairs", [])
                else:
                    logger.error(f"DexScreener API error: {resp.status}")
                    return []

    async def search_pairs(self, query: str) -> List[Dict]:
        """Search for pairs by token name, symbol, or address."""
        url = f"{self.base_url}/search?q={query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("pairs", [])
                else:
                    logger.error(f"DexScreener Search error: {resp.status}")
                    return []

    async def get_pair(self, chain_id: str, pair_address: str) -> Dict:
        """Get data for a specific pair."""
        url = f"{self.base_url}/pairs/{chain_id}/{pair_address}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pairs = data.get("pairs", [])
                    return pairs[0] if pairs else {}
                return {}

dex_screener = DexScreenerSkill()

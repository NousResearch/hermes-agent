"""
Cosmos DEX API - Real-time token data aggregation
Pulls from DexScreener, Birdeye, Jupiter, and more
"""

import aiohttp
import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# API Endpoints
DEXSCREENER_BASE = "https://api.dexscreener.com"
BIRDEYE_BASE = "https://public-api.birdeye.so"
JUPITER_PRICE = "https://price.jup.ag/v6"
JUPITER_TOKENS = "https://tokens.jup.ag"


@dataclass
class TokenPair:
    """Represents a trading pair"""
    pair_address: str
    base_token: str
    base_symbol: str
    base_name: str
    quote_token: str
    quote_symbol: str
    price_usd: float
    price_native: float
    volume_24h: float
    volume_6h: float
    volume_1h: float
    price_change_24h: float
    price_change_6h: float
    price_change_1h: float
    price_change_5m: float
    liquidity_usd: float
    fdv: float
    market_cap: float
    txns_24h_buys: int
    txns_24h_sells: int
    created_at: Optional[datetime] = None
    dex_id: str = "raydium"
    chain_id: str = "solana"
    url: str = ""
    image_url: str = ""
    websites: List[str] = field(default_factory=list)
    socials: Dict[str, str] = field(default_factory=dict)


@dataclass  
class TrendingToken:
    """Trending token with rank"""
    rank: int
    token: TokenPair
    trend_score: float = 0.0


class DexAPI:
    """
    Aggregated DEX data API
    Combines DexScreener, Birdeye, Jupiter for comprehensive data
    """
    
    def __init__(self, birdeye_api_key: Optional[str] = None):
        self.birdeye_api_key = birdeye_api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
        
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            
    def _is_cache_valid(self, key: str, ttl_seconds: int = 30) -> bool:
        if key not in self._cache or key not in self._cache_ttl:
            return False
        return datetime.now() - self._cache_ttl[key] < timedelta(seconds=ttl_seconds)
        
    def _set_cache(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_ttl[key] = datetime.now()
        
    async def _fetch_json(self, url: str, headers: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch JSON from URL with error handling"""
        try:
            session = await self._get_session()
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"API returned {resp.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
            
    def _parse_pair(self, data: Dict) -> TokenPair:
        """Parse DexScreener pair data into TokenPair"""
        base = data.get("baseToken", {})
        quote = data.get("quoteToken", {})
        txns = data.get("txns", {}).get("h24", {})
        price_change = data.get("priceChange", {})
        info = data.get("info", {})
        
        # Parse socials
        socials = {}
        for social in info.get("socials", []):
            socials[social.get("type", "unknown")] = social.get("url", "")
            
        created_at = None
        if data.get("pairCreatedAt"):
            try:
                created_at = datetime.fromtimestamp(data["pairCreatedAt"] / 1000)
            except:
                pass
        
        return TokenPair(
            pair_address=data.get("pairAddress", ""),
            base_token=base.get("address", ""),
            base_symbol=base.get("symbol", "???"),
            base_name=base.get("name", "Unknown"),
            quote_token=quote.get("address", ""),
            quote_symbol=quote.get("symbol", "SOL"),
            price_usd=float(data.get("priceUsd", 0) or 0),
            price_native=float(data.get("priceNative", 0) or 0),
            volume_24h=float(data.get("volume", {}).get("h24", 0) or 0),
            volume_6h=float(data.get("volume", {}).get("h6", 0) or 0),
            volume_1h=float(data.get("volume", {}).get("h1", 0) or 0),
            price_change_24h=float(price_change.get("h24", 0) or 0),
            price_change_6h=float(price_change.get("h6", 0) or 0),
            price_change_1h=float(price_change.get("h1", 0) or 0),
            price_change_5m=float(price_change.get("m5", 0) or 0),
            liquidity_usd=float(data.get("liquidity", {}).get("usd", 0) or 0),
            fdv=float(data.get("fdv", 0) or 0),
            market_cap=float(data.get("marketCap", 0) or 0),
            txns_24h_buys=int(txns.get("buys", 0) or 0),
            txns_24h_sells=int(txns.get("sells", 0) or 0),
            created_at=created_at,
            dex_id=data.get("dexId", "raydium"),
            chain_id=data.get("chainId", "solana"),
            url=data.get("url", ""),
            image_url=info.get("imageUrl", ""),
            websites=info.get("websites", []),
            socials=socials
        )
        
    # ============== Search & Lookup ==============
    
    async def search_tokens(self, query: str) -> List[TokenPair]:
        """Search for tokens by name, symbol, or address"""
        cache_key = f"search:{query}"
        if self._is_cache_valid(cache_key, 60):
            return self._cache[cache_key]
            
        url = f"{DEXSCREENER_BASE}/latest/dex/search?q={query}"
        data = await self._fetch_json(url)
        
        if not data or "pairs" not in data:
            return []
            
        pairs = [self._parse_pair(p) for p in data["pairs"][:50]]
        self._set_cache(cache_key, pairs)
        return pairs
        
    async def get_token_by_address(self, address: str) -> Optional[TokenPair]:
        """Get token info by contract address"""
        cache_key = f"token:{address}"
        if self._is_cache_valid(cache_key, 30):
            return self._cache[cache_key]
            
        url = f"{DEXSCREENER_BASE}/latest/dex/tokens/{address}"
        data = await self._fetch_json(url)
        
        if not data or "pairs" not in data or not data["pairs"]:
            return None
            
        # Get the highest liquidity pair
        pairs = sorted(data["pairs"], key=lambda x: float(x.get("liquidity", {}).get("usd", 0) or 0), reverse=True)
        pair = self._parse_pair(pairs[0])
        self._set_cache(cache_key, pair)
        return pair
        
    async def get_pair_by_address(self, pair_address: str, chain: str = "solana") -> Optional[TokenPair]:
        """Get pair info by pair address"""
        cache_key = f"pair:{chain}:{pair_address}"
        if self._is_cache_valid(cache_key, 15):
            return self._cache[cache_key]
            
        url = f"{DEXSCREENER_BASE}/latest/dex/pairs/{chain}/{pair_address}"
        data = await self._fetch_json(url)
        
        if not data or "pairs" not in data or not data["pairs"]:
            return None
            
        pair = self._parse_pair(data["pairs"][0])
        self._set_cache(cache_key, pair)
        return pair
        
    # ============== Trending & Discovery ==============
    
    async def get_trending_tokens(self, chain: str = "solana") -> List[TrendingToken]:
        """Get trending tokens (boosted profiles)"""
        cache_key = f"trending:{chain}"
        if self._is_cache_valid(cache_key, 60):
            return self._cache[cache_key]
            
        # Get boosted tokens
        url = f"{DEXSCREENER_BASE}/token-boosts/latest/v1"
        data = await self._fetch_json(url)
        
        trending = []
        if data:
            for i, item in enumerate(data[:20]):
                if item.get("chainId") != chain:
                    continue
                token_addr = item.get("tokenAddress")
                if token_addr:
                    pair = await self.get_token_by_address(token_addr)
                    if pair:
                        trending.append(TrendingToken(
                            rank=len(trending) + 1,
                            token=pair,
                            trend_score=float(item.get("amount", 0))
                        ))
                        
        self._set_cache(cache_key, trending)
        return trending
        
    async def get_new_pairs(self, chain: str = "solana", limit: int = 20) -> List[TokenPair]:
        """Get newly created pairs"""
        cache_key = f"new_pairs:{chain}"
        if self._is_cache_valid(cache_key, 30):
            return self._cache[cache_key][:limit]
            
        # DexScreener doesn't have a direct "new pairs" endpoint
        # We'll use token profiles which often includes new tokens
        url = f"{DEXSCREENER_BASE}/token-profiles/latest/v1"
        data = await self._fetch_json(url)
        
        pairs = []
        if data:
            for item in data:
                if item.get("chainId") != chain:
                    continue
                token_addr = item.get("tokenAddress")
                if token_addr and len(pairs) < limit:
                    pair = await self.get_token_by_address(token_addr)
                    if pair:
                        pairs.append(pair)
                        
        # Sort by creation time
        pairs.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        self._set_cache(cache_key, pairs)
        return pairs[:limit]
        
    async def get_top_gainers(self, chain: str = "solana", timeframe: str = "h24") -> List[TokenPair]:
        """Get top gaining tokens"""
        # This would ideally use a dedicated endpoint
        # For now we can search for popular terms and sort
        cache_key = f"gainers:{chain}:{timeframe}"
        if self._is_cache_valid(cache_key, 60):
            return self._cache[cache_key]
            
        # Get trending tokens and sort by price change
        trending = await self.get_trending_tokens(chain)
        pairs = [t.token for t in trending]
        
        # Sort by appropriate timeframe
        if timeframe == "h1":
            pairs.sort(key=lambda x: x.price_change_1h, reverse=True)
        elif timeframe == "h6":
            pairs.sort(key=lambda x: x.price_change_6h, reverse=True)
        else:
            pairs.sort(key=lambda x: x.price_change_24h, reverse=True)
            
        self._set_cache(cache_key, pairs)
        return pairs
        
    async def get_top_losers(self, chain: str = "solana", timeframe: str = "h24") -> List[TokenPair]:
        """Get top losing tokens"""
        gainers = await self.get_top_gainers(chain, timeframe)
        return list(reversed(gainers))
        
    async def get_high_volume(self, chain: str = "solana") -> List[TokenPair]:
        """Get high volume tokens"""
        cache_key = f"high_volume:{chain}"
        if self._is_cache_valid(cache_key, 60):
            return self._cache[cache_key]
            
        trending = await self.get_trending_tokens(chain)
        pairs = [t.token for t in trending]
        pairs.sort(key=lambda x: x.volume_24h, reverse=True)
        
        self._set_cache(cache_key, pairs)
        return pairs
        
    # ============== OHLCV / Chart Data ==============
    
    async def get_ohlcv(
        self, 
        pair_address: str, 
        chain: str = "solana",
        timeframe: str = "15m",
        limit: int = 100
    ) -> List[Dict]:
        """
        Get OHLCV candlestick data for charts
        Note: DexScreener doesn't provide OHLCV directly
        This uses Birdeye if API key is available
        """
        if not self.birdeye_api_key:
            # Return empty - frontend will use TradingView widget instead
            return []
            
        cache_key = f"ohlcv:{chain}:{pair_address}:{timeframe}"
        if self._is_cache_valid(cache_key, 60):
            return self._cache[cache_key]
            
        # Birdeye OHLCV endpoint
        tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
        tf = tf_map.get(timeframe, "15m")
        
        url = f"{BIRDEYE_BASE}/defi/ohlcv/pair?address={pair_address}&type={tf}&limit={limit}"
        headers = {"X-API-KEY": self.birdeye_api_key}
        data = await self._fetch_json(url, headers)
        
        if not data or "data" not in data:
            return []
            
        ohlcv = data["data"].get("items", [])
        self._set_cache(cache_key, ohlcv)
        return ohlcv
        
    # ============== Multi-token Batch ==============
    
    async def get_multiple_tokens(self, addresses: List[str]) -> List[TokenPair]:
        """Get info for multiple tokens at once"""
        # DexScreener supports comma-separated addresses
        if not addresses:
            return []
            
        cache_key = f"multi:{','.join(sorted(addresses[:30]))}"
        if self._is_cache_valid(cache_key, 30):
            return self._cache[cache_key]
            
        # Max 30 addresses per request
        addr_str = ",".join(addresses[:30])
        url = f"{DEXSCREENER_BASE}/latest/dex/tokens/{addr_str}"
        data = await self._fetch_json(url)
        
        if not data or "pairs" not in data:
            return []
            
        # Group by base token, keep highest liquidity pair each
        token_pairs: Dict[str, TokenPair] = {}
        for p in data["pairs"]:
            pair = self._parse_pair(p)
            existing = token_pairs.get(pair.base_token)
            if not existing or pair.liquidity_usd > existing.liquidity_usd:
                token_pairs[pair.base_token] = pair
                
        result = list(token_pairs.values())
        self._set_cache(cache_key, result)
        return result
        
    # ============== Jupiter Price ==============
    
    async def get_jupiter_prices(self, addresses: List[str]) -> Dict[str, float]:
        """Get prices from Jupiter (often more accurate for Solana)"""
        if not addresses:
            return {}
            
        ids = ",".join(addresses[:100])
        url = f"{JUPITER_PRICE}/price?ids={ids}"
        data = await self._fetch_json(url)
        
        if not data or "data" not in data:
            return {}
            
        return {
            addr: info.get("price", 0) 
            for addr, info in data["data"].items()
        }

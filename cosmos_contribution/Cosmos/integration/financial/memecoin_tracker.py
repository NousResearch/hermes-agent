"""
cosmos Memecoin Tracker - Pump.fun & Bags.fm.

"Tracking the degens, one bonding curve at a time."
"""

import aiohttp
from loguru import logger
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class BagsClaimMode(Enum):
    """Mode for fetching claim events."""
    OFFSET = "offset"
    TIME = "time"


@dataclass
class BagsTokenLaunch:
    """Token launch configuration for Bags.fm."""
    name: str
    symbol: str
    description: str
    image_url: Optional[str] = None
    twitter: Optional[str] = None
    telegram: Optional[str] = None
    website: Optional[str] = None


@dataclass
class BagsFeeShareConfig:
    """Fee share configuration - up to 100 fee claimers."""
    fee_claimers: List[Dict[str, Any]] = field(default_factory=list)
    use_lookup_tables: bool = False  # Required for 15+ claimers


@dataclass
class BagsSwapParams:
    """Parameters for swap transactions."""
    token_mint: str
    amount: float
    side: str  # "buy" or "sell"
    slippage_bps: int = 100  # Default 1% slippage


class MemecoinSkill:
    def __init__(self):
        self.pump_base_url = "https://frontend-api.pump.fun"  # Public frontend API
        self.bags_base_url = "https://public-api-v2.bags.fm/api/v1"  # Updated to v2
        self.bags_api_key = None
        self._rate_limit_remaining = 1000
        self._rate_limit_reset = None

    def set_bags_api_key(self, api_key: str):
        self.bags_api_key = api_key

    def _get_bags_headers(self) -> Dict[str, str]:
        """Get headers for Bags.fm API requests."""
        if not self.bags_api_key:
            return {}
        return {"x-api-key": self.bags_api_key}

    def _update_rate_limits(self, headers: Dict):
        """Update rate limit tracking from response headers."""
        if "X-RateLimit-Remaining" in headers:
            self._rate_limit_remaining = int(headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in headers:
            self._rate_limit_reset = headers["X-RateLimit-Reset"]

    async def _bags_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make authenticated request to Bags.fm API."""
        if not self.bags_api_key:
            return {"error": "Bags.fm API Key not configured."}

        url = f"{self.bags_base_url}/{endpoint.lstrip('/')}"
        headers = self._get_bags_headers()

        async with aiohttp.ClientSession() as session:
            try:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers, params=params) as resp:
                        self._update_rate_limits(resp.headers)
                        if resp.status == 200:
                            return await resp.json()
                        return {"error": f"Bags.fm error: {resp.status}", "status": resp.status}
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data) as resp:
                        self._update_rate_limits(resp.headers)
                        if resp.status in (200, 201):
                            return await resp.json()
                        return {"error": f"Bags.fm error: {resp.status}", "status": resp.status}
            except aiohttp.ClientError as e:
                logger.error(f"Bags.fm request failed: {e}")
                return {"error": str(e)}

    async def get_pump_token(self, mint_address: str) -> Dict:
        """Fetch token details and bonding curve progress from Pump.fun."""
        url = f"{self.pump_base_url}/coins/{mint_address}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Calculate bonding curve progress (simplified)
                    # Real progress uses bonding_curve account data, but API often provides it
                    return data
                else:
                    logger.error(f"Pump.fun API error: {resp.status}")
                    return {"error": f"Token {mint_address} not found on pump.fun."}

    async def get_pump_new_tokens(self, limit: int = 10) -> List[Dict]:
        """Fetch recently created tokens on Pump.fun."""
        url = f"{self.pump_base_url}/coins?offset=0&limit={limit}&sort=created_timestamp&order=DESC"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []

    # ==================== TOKEN OPERATIONS ====================

    async def get_bags_token(self, token_address: str) -> Dict:
        """Fetch token info from Bags.fm."""
        return await self._bags_request("GET", f"/tokens/{token_address}")

    async def get_bags_trending(self) -> List[Dict]:
        """Fetch trending tokens on Bags.fm."""
        result = await self._bags_request("GET", "/analytics/trending")
        return result if isinstance(result, list) else []

    async def get_token_launch_creators(self, token_mint: str) -> Dict:
        """Get creators associated with a token launch."""
        return await self._bags_request("GET", f"/tokens/{token_mint}/creators")

    async def get_token_lifetime_fees(self, token_mint: str) -> Dict:
        """Get lifetime fee information for a token."""
        return await self._bags_request("GET", f"/tokens/{token_mint}/lifetime-fees")

    async def get_token_claim_events(
        self,
        token_mint: str,
        mode: BagsClaimMode = BagsClaimMode.OFFSET,
        offset: int = 0,
        limit: int = 50,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict:
        """Get token claim events with offset or time-based pagination."""
        params = {"mode": mode.value, "limit": limit}
        if mode == BagsClaimMode.OFFSET:
            params["offset"] = offset
        else:
            if start_time:
                params["start_time"] = start_time
            if end_time:
                params["end_time"] = end_time
        return await self._bags_request("GET", f"/tokens/{token_mint}/claim-events", params=params)

    async def get_token_claim_stats(self, token_mint: str) -> Dict:
        """Get claim statistics for a token."""
        return await self._bags_request("GET", f"/tokens/{token_mint}/claim-stats")

    # ==================== TOKEN LAUNCH ====================

    async def create_token_metadata(self, launch: BagsTokenLaunch) -> Dict:
        """Create token info and metadata with optional image upload."""
        data = {
            "name": launch.name,
            "symbol": launch.symbol,
            "description": launch.description,
        }
        if launch.image_url:
            data["image_url"] = launch.image_url
        if launch.twitter:
            data["twitter"] = launch.twitter
        if launch.telegram:
            data["telegram"] = launch.telegram
        if launch.website:
            data["website"] = launch.website
        return await self._bags_request("POST", "/tokens/metadata", data=data)

    async def create_token_launch_transaction(self, token_mint: str, creator_wallet: str) -> Dict:
        """Create a token launch transaction."""
        data = {
            "token_mint": token_mint,
            "creator_wallet": creator_wallet
        }
        return await self._bags_request("POST", "/tokens/launch", data=data)

    # ==================== TRADING ====================

    async def get_trade_quote(
        self,
        token_mint: str,
        amount: float,
        side: str = "buy",
        slippage_bps: int = 100
    ) -> Dict:
        """Get a trade quote for buying or selling tokens."""
        params = {
            "token_mint": token_mint,
            "amount": amount,
            "side": side,
            "slippage_bps": slippage_bps
        }
        return await self._bags_request("GET", "/trading/quote", params=params)

    async def create_swap_transaction(self, swap: BagsSwapParams) -> Dict:
        """Create a swap transaction."""
        data = {
            "token_mint": swap.token_mint,
            "amount": swap.amount,
            "side": swap.side,
            "slippage_bps": swap.slippage_bps
        }
        return await self._bags_request("POST", "/trading/swap", data=data)

    # ==================== FEE MANAGEMENT ====================

    async def create_fee_share_config(self, config: BagsFeeShareConfig) -> Dict:
        """Create fee share configuration (up to 100 fee claimers)."""
        data = {
            "fee_claimers": config.fee_claimers,
            "use_lookup_tables": config.use_lookup_tables
        }
        return await self._bags_request("POST", "/fees/share-config", data=data)

    async def create_partner_config(self, wallet_address: str) -> Dict:
        """Create partner configuration (one per wallet)."""
        data = {"wallet_address": wallet_address}
        return await self._bags_request("POST", "/fees/partner-config", data=data)

    async def get_claimable_positions(self, wallet_address: str) -> Dict:
        """Get claimable fee positions for a wallet."""
        return await self._bags_request("GET", f"/fees/claimable/{wallet_address}")

    async def get_claim_transactions(self, wallet_address: str) -> Dict:
        """Get claim transactions for a wallet."""
        return await self._bags_request("GET", f"/fees/claims/{wallet_address}")

    async def get_partner_claim_transactions(self, wallet_address: str) -> Dict:
        """Get partner claim transactions."""
        return await self._bags_request("GET", f"/fees/partner-claims/{wallet_address}")

    async def get_partner_stats(self, wallet_address: str) -> Dict:
        """Get partner statistics."""
        return await self._bags_request("GET", f"/fees/partner-stats/{wallet_address}")

    async def get_pool_config_keys_by_vaults(self, vault_addresses: List[str]) -> Dict:
        """Get pool config keys by fee claimer vaults."""
        params = {"vaults": ",".join(vault_addresses)}
        return await self._bags_request("GET", "/fees/pool-configs", params=params)

    # ==================== WALLET / SOCIAL ====================

    async def get_fee_share_wallet(self, wallet_address: str) -> Dict:
        """Get fee share wallet info (v2)."""
        return await self._bags_request("GET", f"/wallets/{wallet_address}/fee-share")

    async def get_fee_share_wallets_bulk(self, wallet_addresses: List[str]) -> Dict:
        """Get fee share wallet info for multiple wallets (v2)."""
        data = {"wallet_addresses": wallet_addresses}
        return await self._bags_request("POST", "/wallets/fee-share/bulk", data=data)

    # ==================== UTILITY ====================

    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status."""
        return {
            "remaining": self._rate_limit_remaining,
            "reset": self._rate_limit_reset,
            "limit": 1000
        }


memecoin_tracker = MemecoinSkill()

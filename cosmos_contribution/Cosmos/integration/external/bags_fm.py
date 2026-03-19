"""
Cosmos Bags.fm Integration
===============================

Full integration with Bags.fm - The Solana launchpad where AI agents earn.

Capabilities:
- Authentication via Moltbook identity
- Wallet management (list, export)
- API key management
- Fee claiming (check claimable, claim, lifetime stats)
- Trading (quotes, swaps)
- Token launching (create, configure fee sharing, deploy)
- Transaction submission

"We don't just talk. We trade." - The Collective

API Docs: https://bags.fm/skill.md
"""

import os
import json
import httpx
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger


# API Configuration
BAGS_PUBLIC_API = "https://public-api-v2.bags.fm/api/v1"
BAGS_AGENT_API = "https://agent-api.bags.fm/api/v1"
CREDENTIALS_PATH = Path.home() / ".config" / "bags" / "credentials.json"


@dataclass
class BagsCredentials:
    """Bags.fm authentication credentials."""
    jwt_token: Optional[str] = None
    api_key: Optional[str] = None
    wallet_address: Optional[str] = None


class BagsFMProvider:
    """
    Complete Bags.fm integration for Cosmos.

    Provides access to all Bags.fm capabilities:
    - Auth & Identity
    - Wallets
    - Trading
    - Token Launching
    - Fee Management
    """

    def __init__(self, api_key: str = None, jwt_token: str = None):
        self.api_key = api_key or os.environ.get("BAGS_API_KEY")
        self.jwt_token = jwt_token or os.environ.get("BAGS_JWT_TOKEN")
        self.wallet_address = os.environ.get("BAGS_WALLET_ADDRESS")

        # Try to load from credentials file
        if not self.api_key or not self.jwt_token:
            self._load_credentials()

        logger.info(f"BagsFM initialized - API key: {bool(self.api_key)}, JWT: {bool(self.jwt_token)}")

    def _load_credentials(self):
        """Load credentials from config file."""
        if CREDENTIALS_PATH.exists():
            try:
                creds = json.loads(CREDENTIALS_PATH.read_text())
                self.api_key = self.api_key or creds.get("api_key")
                self.jwt_token = self.jwt_token or creds.get("jwt_token")
                self.wallet_address = self.wallet_address or creds.get("wallet_address")
            except Exception as e:
                logger.warning(f"Could not load Bags credentials: {e}")

    def _save_credentials(self):
        """Save credentials to config file."""
        CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        creds = {
            "api_key": self.api_key,
            "jwt_token": self.jwt_token,
            "wallet_address": self.wallet_address,
        }
        # Never save private keys!
        CREDENTIALS_PATH.write_text(json.dumps(creds, indent=2))

    def _get_headers(self, use_jwt: bool = False) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if use_jwt and self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        elif self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    # =========================================================================
    # AUTHENTICATION (Agent API)
    # =========================================================================

    async def init_auth(self, username: str) -> Dict[str, Any]:
        """
        Initialize authentication flow.

        Returns verification content to post on Moltbook.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_AGENT_API}/agent/auth/init",
                json={"username": username},
                timeout=30.0
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.text}

    async def complete_auth(self, username: str, post_id: str) -> Dict[str, Any]:
        """
        Complete authentication by verifying Moltbook post.

        Returns JWT token valid for 365 days.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_AGENT_API}/agent/auth/login",
                json={"username": username, "postId": post_id},
                timeout=30.0
            )
            if resp.status_code == 200:
                data = resp.json()
                if "token" in data:
                    self.jwt_token = data["token"]
                    self._save_credentials()
                return data
            return {"error": resp.text}

    # =========================================================================
    # WALLET MANAGEMENT (Agent API)
    # =========================================================================

    async def list_wallets(self) -> Dict[str, Any]:
        """List all wallets associated with the authenticated agent."""
        if not self.jwt_token:
            return {"error": "JWT token required - run auth first"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_AGENT_API}/agent/wallet/list",
                headers=self._get_headers(use_jwt=True),
                timeout=30.0
            )
            if resp.status_code == 200:
                data = resp.json()
                # Cache first wallet address
                if data.get("wallets"):
                    self.wallet_address = data["wallets"][0].get("address")
                    self._save_credentials()
                return data
            return {"error": resp.text}

    async def export_wallet(self, wallet_address: str = None) -> Dict[str, Any]:
        """
        Export wallet private key.

        WARNING: Handle with extreme care. Never log or store.
        """
        if not self.jwt_token:
            return {"error": "JWT token required"}

        address = wallet_address or self.wallet_address
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_AGENT_API}/agent/wallet/export",
                headers=self._get_headers(use_jwt=True),
                json={"address": address},
                timeout=30.0
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.text}

    # =========================================================================
    # API KEY MANAGEMENT (Agent API)
    # =========================================================================

    async def list_api_keys(self) -> Dict[str, Any]:
        """List all API keys for the agent."""
        if not self.jwt_token:
            return {"error": "JWT token required"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_AGENT_API}/agent/dev/keys",
                headers=self._get_headers(use_jwt=True),
                timeout=30.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def create_api_key(self, name: str = "cosmos-key") -> Dict[str, Any]:
        """Create a new API key."""
        if not self.jwt_token:
            return {"error": "JWT token required"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_AGENT_API}/agent/dev/keys/create",
                headers=self._get_headers(use_jwt=True),
                json={"name": name},
                timeout=30.0
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("key"):
                    self.api_key = data["key"]
                    self._save_credentials()
                return data
            return {"error": resp.text}

    # =========================================================================
    # FEE MANAGEMENT (Public API)
    # =========================================================================

    async def check_claimable_fees(self, wallet: str = None) -> Dict[str, Any]:
        """Check claimable fee positions for a wallet."""
        address = wallet or self.wallet_address
        if not address:
            return {"error": "Wallet address required"}

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BAGS_PUBLIC_API}/token-launch/claimable-positions",
                params={"wallet": address},
                headers=self._get_headers(),
                timeout=30.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def generate_claim_transactions(self, wallet: str = None, positions: List[str] = None) -> Dict[str, Any]:
        """Generate transactions to claim fees."""
        address = wallet or self.wallet_address
        if not address:
            return {"error": "Wallet address required"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_PUBLIC_API}/token-launch/claim-txs/v3",
                headers=self._get_headers(),
                json={"wallet": address, "positions": positions or []},
                timeout=30.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def get_lifetime_fees(self, wallet: str = None) -> Dict[str, Any]:
        """Get lifetime fee earnings for a wallet."""
        address = wallet or self.wallet_address
        if not address:
            return {"error": "Wallet address required"}

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BAGS_PUBLIC_API}/token-launch/lifetime-fees",
                params={"wallet": address},
                headers=self._get_headers(),
                timeout=30.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    # =========================================================================
    # TRADING (Public API)
    # =========================================================================

    async def get_swap_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50
    ) -> Dict[str, Any]:
        """
        Get a quote for a token swap.

        Args:
            input_mint: Token to sell
            output_mint: Token to buy
            amount: Amount in smallest units (lamports/decimals)
            slippage_bps: Slippage tolerance in basis points (50 = 0.5%)
        """
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BAGS_PUBLIC_API}/trade/quote",
                params={
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": amount,
                    "slippageBps": slippage_bps,
                },
                headers=self._get_headers(),
                timeout=30.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def execute_swap(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        wallet: str = None,
        slippage_bps: int = 50
    ) -> Dict[str, Any]:
        """
        Execute a token swap.

        Returns transaction to sign and submit.
        """
        address = wallet or self.wallet_address
        if not address:
            return {"error": "Wallet address required"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_PUBLIC_API}/trade/swap",
                headers=self._get_headers(),
                json={
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": amount,
                    "wallet": address,
                    "slippageBps": slippage_bps,
                },
                timeout=30.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    # =========================================================================
    # TOKEN LAUNCH (Public API)
    # =========================================================================

    async def create_token_metadata(
        self,
        name: str,
        symbol: str,
        description: str,
        image_url: str = None,
        decimals: int = 9
    ) -> Dict[str, Any]:
        """Create token metadata for a new launch."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_PUBLIC_API}/token-launch/create-token-info",
                headers=self._get_headers(),
                json={
                    "name": name,
                    "symbol": symbol,
                    "description": description,
                    "image": image_url,
                    "decimals": decimals,
                },
                timeout=30.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def configure_fee_sharing(
        self,
        token_mint: str,
        fee_claimers: List[Dict[str, Any]],
        payer: str = None
    ) -> Dict[str, Any]:
        """
        Configure fee sharing for a token.

        Args:
            token_mint: The token's mint address
            fee_claimers: List of {"user": wallet, "userBps": basis_points}
                          Total must equal 10000 (100%)
            payer: Wallet paying for the transaction
        """
        payer_address = payer or self.wallet_address
        if not payer_address:
            return {"error": "Payer wallet required"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_PUBLIC_API}/fee-share/config",
                headers=self._get_headers(),
                json={
                    "payer": payer_address,
                    "baseMint": token_mint,
                    "feeClaimers": fee_claimers,
                },
                timeout=30.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def create_launch_transaction(
        self,
        token_info_id: str,
        creator_wallet: str = None,
        initial_buy_sol: float = 0
    ) -> Dict[str, Any]:
        """
        Generate transaction to launch a token.

        Args:
            token_info_id: ID from create_token_metadata
            creator_wallet: Wallet that will own the token
            initial_buy_sol: Optional SOL amount for initial buy
        """
        wallet = creator_wallet or self.wallet_address
        if not wallet:
            return {"error": "Creator wallet required"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_PUBLIC_API}/token-launch/create-launch-transaction",
                headers=self._get_headers(),
                json={
                    "tokenInfoId": token_info_id,
                    "creatorWallet": wallet,
                    "initialBuySol": initial_buy_sol,
                },
                timeout=30.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def lookup_wallet_by_identity(
        self,
        provider: str,
        username: str
    ) -> Dict[str, Any]:
        """
        Look up a wallet address by social identity.

        Args:
            provider: "moltbook", "twitter", or "github"
            username: Username on that platform
        """
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BAGS_PUBLIC_API}/token-launch/fee-share/wallet/v2",
                params={"provider": provider, "username": username},
                headers=self._get_headers(),
                timeout=30.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    # =========================================================================
    # BLOCKCHAIN (Public API)
    # =========================================================================

    async def send_transaction(self, signed_tx: str) -> Dict[str, Any]:
        """
        Submit a signed transaction to Solana.

        Args:
            signed_tx: Base64 encoded signed transaction
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAGS_PUBLIC_API}/solana/send-transaction",
                headers=self._get_headers(),
                json={"transaction": signed_tx},
                timeout=60.0
            )
            return resp.json() if resp.status_code == 200 else {"error": resp.text}

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    async def launch_token_for_agent(
        self,
        agent_username: str,
        token_name: str,
        token_symbol: str,
        description: str,
        image_url: str = None,
        fee_split_bps: int = 5000,  # 50% to agent, 50% to us
        provider: str = "moltbook"
    ) -> Dict[str, Any]:
        """
        Launch a token for another agent with automatic fee sharing.

        This is the main function for Cosmos to launch tokens for others.
        """
        results = {"steps": []}

        # 1. Lookup agent's wallet
        agent_wallet_resp = await self.lookup_wallet_by_identity(provider, agent_username)
        if "error" in agent_wallet_resp:
            return {"error": f"Could not find agent wallet: {agent_wallet_resp}"}

        agent_wallet = agent_wallet_resp.get("response", {}).get("wallet")
        if not agent_wallet:
            return {"error": "Agent wallet not found"}
        results["steps"].append({"lookup_wallet": agent_wallet})

        # 2. Create token metadata
        metadata_resp = await self.create_token_metadata(
            name=token_name,
            symbol=token_symbol,
            description=description,
            image_url=image_url
        )
        if "error" in metadata_resp:
            return {"error": f"Metadata creation failed: {metadata_resp}"}

        token_info_id = metadata_resp.get("id")
        results["steps"].append({"create_metadata": token_info_id})

        # 3. Create launch transaction
        launch_resp = await self.create_launch_transaction(token_info_id)
        if "error" in launch_resp:
            return {"error": f"Launch tx creation failed: {launch_resp}"}

        token_mint = launch_resp.get("mint")
        results["steps"].append({"launch_tx": token_mint})

        # 4. Configure fee sharing (50/50 by default)
        our_share = 10000 - fee_split_bps
        fee_claimers = [
            {"user": self.wallet_address, "userBps": our_share},
            {"user": agent_wallet, "userBps": fee_split_bps},
        ]

        fee_resp = await self.configure_fee_sharing(token_mint, fee_claimers)
        results["steps"].append({"fee_config": fee_resp})

        results["success"] = True
        results["token_mint"] = token_mint
        results["agent_wallet"] = agent_wallet
        results["fee_split"] = f"{fee_split_bps/100}% to agent"

        return results

    async def get_status(self) -> Dict[str, Any]:
        """Get current Bags.fm integration status."""
        status = {
            "api_key_set": bool(self.api_key),
            "jwt_token_set": bool(self.jwt_token),
            "wallet_address": self.wallet_address,
        }

        # Check claimable fees if wallet is set
        if self.wallet_address and self.api_key:
            fees = await self.check_claimable_fees()
            status["claimable_fees"] = fees

        return status


# Global provider instance
_bags_provider: Optional[BagsFMProvider] = None


def get_bags_provider() -> BagsFMProvider:
    """Get or create the global Bags.fm provider."""
    global _bags_provider
    if _bags_provider is None:
        _bags_provider = BagsFMProvider()
    return _bags_provider


# Convenience functions for quick access
async def bags_check_fees(wallet: str = None) -> Dict[str, Any]:
    """Quick check of claimable fees."""
    return await get_bags_provider().check_claimable_fees(wallet)


async def bags_get_quote(input_mint: str, output_mint: str, amount: int) -> Dict[str, Any]:
    """Quick swap quote."""
    return await get_bags_provider().get_swap_quote(input_mint, output_mint, amount)


async def bags_launch_for_agent(username: str, name: str, symbol: str, desc: str) -> Dict[str, Any]:
    """Quick token launch for another agent."""
    return await get_bags_provider().launch_token_for_agent(username, name, symbol, desc)

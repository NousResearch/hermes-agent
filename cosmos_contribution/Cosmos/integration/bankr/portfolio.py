"""
Bankr Portfolio Module.

Provides portfolio tracking, balances, and position management.
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
class TokenBalance:
    """Token balance information."""
    symbol: str
    balance: Decimal
    value_usd: Decimal
    chain: str
    contract_address: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "TokenBalance":
        return cls(
            symbol=data.get("symbol", ""),
            balance=Decimal(str(data.get("balance", 0))),
            value_usd=Decimal(str(data.get("value_usd", 0))),
            chain=data.get("chain", ""),
            contract_address=data.get("address"),
        )


@dataclass
class NFT:
    """NFT asset information."""
    collection: str
    name: str
    token_id: str
    chain: str
    image_url: Optional[str]
    floor_price: Optional[Decimal]
    contract_address: str

    @classmethod
    def from_dict(cls, data: Dict) -> "NFT":
        return cls(
            collection=data.get("collection", ""),
            name=data.get("name", ""),
            token_id=data.get("token_id", ""),
            chain=data.get("chain", ""),
            image_url=data.get("image"),
            floor_price=Decimal(str(data.get("floor_price", 0))) if data.get("floor_price") else None,
            contract_address=data.get("address", ""),
        )


@dataclass
class Portfolio:
    """Complete portfolio summary."""
    total_value_usd: Decimal
    tokens: List[TokenBalance]
    nfts: List[NFT]
    defi_positions: List[Dict]
    chains: List[str]
    timestamp: datetime

    @classmethod
    def from_response(cls, data: Dict) -> "Portfolio":
        return cls(
            total_value_usd=Decimal(str(data.get("total_value", 0))),
            tokens=[TokenBalance.from_dict(t) for t in data.get("tokens", [])],
            nfts=[NFT.from_dict(n) for n in data.get("nfts", [])],
            defi_positions=data.get("defi_positions", []),
            chains=data.get("chains", []),
            timestamp=datetime.now(),
        )


class BankrPortfolio:
    """
    Portfolio tracking via Bankr API.

    Provides:
    - Token balance tracking across chains
    - NFT inventory
    - DeFi position tracking
    - Portfolio analytics
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

    async def get_balance(self, chain: str = None) -> List[TokenBalance]:
        """
        Get token balances.

        Args:
            chain: Optional chain filter

        Returns:
            List of token balances
        """
        try:
            client = await self._get_client()
            chain = chain or self.config.default_chain

            result = await client.execute(
                f"Show my wallet balance on {chain}"
            )

            balances = result.get("balances", [])
            return [TokenBalance.from_dict(b) for b in balances]

        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
            return []

    async def get_all_balances(self) -> Dict[str, List[TokenBalance]]:
        """
        Get balances across all chains.

        Returns:
            Dict mapping chain names to balances
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                "Show all my wallet balances across all chains"
            )

            by_chain = {}
            for balance in result.get("balances", []):
                tb = TokenBalance.from_dict(balance)
                if tb.chain not in by_chain:
                    by_chain[tb.chain] = []
                by_chain[tb.chain].append(tb)

            return by_chain

        except Exception as e:
            logger.error(f"All balances fetch failed: {e}")
            return {}

    async def get_nfts(self, chain: str = None) -> List[NFT]:
        """
        Get NFT inventory.

        Args:
            chain: Optional chain filter

        Returns:
            List of NFTs owned
        """
        try:
            client = await self._get_client()
            chain_filter = f" on {chain}" if chain else ""

            result = await client.execute(
                f"Show my NFTs{chain_filter}"
            )

            nfts = result.get("nfts", [])
            return [NFT.from_dict(n) for n in nfts]

        except Exception as e:
            logger.error(f"NFT fetch failed: {e}")
            return []

    async def get_portfolio(self) -> Portfolio:
        """
        Get complete portfolio summary.

        Returns:
            Portfolio with all assets
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                "Show my complete portfolio including tokens, NFTs, and DeFi positions"
            )

            return Portfolio.from_response(result)

        except Exception as e:
            logger.error(f"Portfolio fetch failed: {e}")
            return Portfolio(
                total_value_usd=Decimal(0),
                tokens=[],
                nfts=[],
                defi_positions=[],
                chains=[],
                timestamp=datetime.now(),
            )

    async def get_defi_positions(self) -> List[Dict]:
        """
        Get DeFi protocol positions.

        Returns:
            List of DeFi positions (staking, LP, lending)
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                "Show my DeFi positions including staking, liquidity pools, and lending"
            )

            return result.get("positions", [])

        except Exception as e:
            logger.error(f"DeFi positions fetch failed: {e}")
            return []

    async def get_transaction_history(
        self,
        chain: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get transaction history.

        Args:
            chain: Optional chain filter
            limit: Max transactions to return

        Returns:
            List of transactions
        """
        try:
            client = await self._get_client()
            chain_filter = f" on {chain}" if chain else ""

            result = await client.execute(
                f"Show my last {limit} transactions{chain_filter}"
            )

            return result.get("transactions", [])

        except Exception as e:
            logger.error(f"Transaction history fetch failed: {e}")
            return []

    async def get_total_value(self) -> Decimal:
        """Get total portfolio value in USD."""
        portfolio = await self.get_portfolio()
        return portfolio.total_value_usd

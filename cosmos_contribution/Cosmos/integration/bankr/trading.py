"""
Bankr Trading Module.

Handles buy, sell, swap operations across multiple chains.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime

from .client import BankrClient, BankrError, JobResult
from .config import get_bankr_config

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a trading operation."""
    success: bool
    trade_type: str  # buy, sell, swap, bridge
    token_in: str
    token_out: str
    amount_in: Decimal
    amount_out: Decimal
    chain: str
    tx_hash: Optional[str] = None
    gas_used: Optional[Decimal] = None
    error: Optional[str] = None
    raw_response: Dict = None
    timestamp: datetime = None

    @classmethod
    def from_job_result(cls, job: JobResult, trade_type: str) -> "TradeResult":
        """Create from a Bankr job result."""
        result = job.result
        tx = job.transactions[0] if job.transactions else {}

        return cls(
            success=job.status == "completed",
            trade_type=trade_type,
            token_in=result.get("token_in", ""),
            token_out=result.get("token_out", ""),
            amount_in=Decimal(str(result.get("amount_in", 0))),
            amount_out=Decimal(str(result.get("amount_out", 0))),
            chain=result.get("chain", ""),
            tx_hash=tx.get("hash"),
            gas_used=Decimal(str(tx.get("gas_used", 0))) if tx.get("gas_used") else None,
            raw_response=result,
            timestamp=datetime.now(),
        )

    @classmethod
    def error_result(cls, error: str, trade_type: str = "unknown") -> "TradeResult":
        """Create an error result."""
        return cls(
            success=False,
            trade_type=trade_type,
            token_in="",
            token_out="",
            amount_in=Decimal(0),
            amount_out=Decimal(0),
            chain="",
            error=error,
            timestamp=datetime.now(),
        )


class BankrTrading:
    """
    Crypto trading operations via Bankr API.

    Supports:
    - Buy tokens with USD
    - Sell tokens for USD
    - Swap between tokens
    - Bridge tokens across chains
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

    def _validate_trade_amount(self, amount_usd: float) -> bool:
        """Validate trade amount against limits."""
        if not self.config.trading_enabled:
            raise BankrError("Trading is disabled in config")

        if amount_usd > float(self.config.max_trade_usd):
            raise BankrError(
                f"Amount ${amount_usd} exceeds max trade limit ${self.config.max_trade_usd}"
            )

        return True

    async def buy_token(
        self,
        token: str,
        amount_usd: float,
        chain: str = None
    ) -> TradeResult:
        """
        Buy a token with USD amount.

        Args:
            token: Token symbol (e.g., "ETH", "BNKR")
            amount_usd: Amount in USD to spend
            chain: Blockchain (default from config)

        Returns:
            TradeResult with transaction details
        """
        try:
            self._validate_trade_amount(amount_usd)
            chain = chain or self.config.default_chain

            client = await self._get_client()
            prompt = f"Buy ${amount_usd} of {token} on {chain}"

            logger.info(f"Executing buy: {prompt}")
            job = await client.execute_trade(prompt)

            return TradeResult.from_job_result(job, "buy")

        except Exception as e:
            logger.error(f"Buy failed: {e}")
            return TradeResult.error_result(str(e), "buy")

    async def sell_token(
        self,
        token: str,
        amount: float,
        chain: str = None
    ) -> TradeResult:
        """
        Sell a token amount.

        Args:
            token: Token symbol
            amount: Amount of tokens to sell
            chain: Blockchain

        Returns:
            TradeResult with transaction details
        """
        try:
            chain = chain or self.config.default_chain

            client = await self._get_client()
            prompt = f"Sell {amount} {token} on {chain}"

            logger.info(f"Executing sell: {prompt}")
            job = await client.execute_trade(prompt)

            return TradeResult.from_job_result(job, "sell")

        except Exception as e:
            logger.error(f"Sell failed: {e}")
            return TradeResult.error_result(str(e), "sell")

    async def swap(
        self,
        from_token: str,
        to_token: str,
        amount: float,
        chain: str = None
    ) -> TradeResult:
        """
        Swap between tokens.

        Args:
            from_token: Source token symbol
            to_token: Destination token symbol
            amount: Amount of source token
            chain: Blockchain

        Returns:
            TradeResult with transaction details
        """
        try:
            chain = chain or self.config.default_chain

            client = await self._get_client()
            prompt = f"Swap {amount} {from_token} to {to_token} on {chain}"

            logger.info(f"Executing swap: {prompt}")
            job = await client.execute_trade(prompt)

            return TradeResult.from_job_result(job, "swap")

        except Exception as e:
            logger.error(f"Swap failed: {e}")
            return TradeResult.error_result(str(e), "swap")

    async def bridge(
        self,
        token: str,
        amount: float,
        from_chain: str,
        to_chain: str
    ) -> TradeResult:
        """
        Bridge tokens across chains.

        Args:
            token: Token symbol
            amount: Amount to bridge
            from_chain: Source chain
            to_chain: Destination chain

        Returns:
            TradeResult with transaction details
        """
        try:
            client = await self._get_client()
            prompt = f"Bridge {amount} {token} from {from_chain} to {to_chain}"

            logger.info(f"Executing bridge: {prompt}")
            job = await client.execute_trade(prompt)

            return TradeResult.from_job_result(job, "bridge")

        except Exception as e:
            logger.error(f"Bridge failed: {e}")
            return TradeResult.error_result(str(e), "bridge")

    async def get_price(self, token: str) -> float:
        """Get current token price in USD."""
        try:
            client = await self._get_client()
            return await client.get_price(token)
        except Exception as e:
            logger.error(f"Price fetch failed: {e}")
            return 0.0

    async def get_quote(
        self,
        from_token: str,
        to_token: str,
        amount: float,
        chain: str = None
    ) -> Dict[str, Any]:
        """
        Get a swap quote without executing.

        Args:
            from_token: Source token
            to_token: Destination token
            amount: Amount to swap
            chain: Blockchain

        Returns:
            Quote information
        """
        try:
            chain = chain or self.config.default_chain
            client = await self._get_client()

            prompt = f"Get quote for swapping {amount} {from_token} to {to_token} on {chain}"
            result = await client.execute(prompt)

            return result

        except Exception as e:
            logger.error(f"Quote failed: {e}")
            return {"error": str(e)}

    async def execute_trade(self, intent: Any) -> TradeResult:
        """
        Execute a trade from a parsed intent.

        Used by the NLP command router.
        """
        action = getattr(intent, 'action', '').lower()
        params = getattr(intent, 'parameters', {})

        if action == 'buy':
            return await self.buy_token(
                token=params.get('token', ''),
                amount_usd=params.get('amount_usd', 0),
                chain=params.get('chain'),
            )
        elif action == 'sell':
            return await self.sell_token(
                token=params.get('token', ''),
                amount=params.get('amount', 0),
                chain=params.get('chain'),
            )
        elif action == 'swap':
            return await self.swap(
                from_token=params.get('from_token', ''),
                to_token=params.get('to_token', ''),
                amount=params.get('amount', 0),
                chain=params.get('chain'),
            )
        elif action == 'bridge':
            return await self.bridge(
                token=params.get('token', ''),
                amount=params.get('amount', 0),
                from_chain=params.get('from_chain', ''),
                to_chain=params.get('to_chain', ''),
            )
        else:
            return TradeResult.error_result(f"Unknown trade action: {action}")

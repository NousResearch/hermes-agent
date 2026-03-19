"""
Bankr Polymarket Integration.

Provides prediction market access via Bankr API.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime

from .client import BankrClient, JobResult
from .config import get_bankr_config

logger = logging.getLogger(__name__)


@dataclass
class Market:
    """Prediction market information."""
    id: str
    question: str
    outcomes: List[str]
    odds: Dict[str, float]
    volume: Decimal
    end_date: Optional[datetime]
    resolved: bool
    resolution: Optional[str]

    @classmethod
    def from_response(cls, data: Dict) -> "Market":
        return cls(
            id=data.get("id", ""),
            question=data.get("question", ""),
            outcomes=data.get("outcomes", []),
            odds=data.get("odds", {}),
            volume=Decimal(str(data.get("volume", 0))),
            end_date=None,  # Parse from response
            resolved=data.get("resolved", False),
            resolution=data.get("resolution"),
        )


@dataclass
class BetResult:
    """Result of placing a bet."""
    success: bool
    market_id: str
    outcome: str
    amount_usd: Decimal
    shares: Decimal
    avg_price: Decimal
    tx_hash: Optional[str]
    error: Optional[str]
    timestamp: datetime

    @classmethod
    def from_job_result(cls, job: JobResult) -> "BetResult":
        result = job.result
        tx = job.transactions[0] if job.transactions else {}

        return cls(
            success=job.status == "completed",
            market_id=result.get("market_id", ""),
            outcome=result.get("outcome", ""),
            amount_usd=Decimal(str(result.get("amount", 0))),
            shares=Decimal(str(result.get("shares", 0))),
            avg_price=Decimal(str(result.get("avg_price", 0))),
            tx_hash=tx.get("hash"),
            error=None,
            timestamp=datetime.now(),
        )

    @classmethod
    def error_result(cls, error: str) -> "BetResult":
        return cls(
            success=False,
            market_id="",
            outcome="",
            amount_usd=Decimal(0),
            shares=Decimal(0),
            avg_price=Decimal(0),
            tx_hash=None,
            error=error,
            timestamp=datetime.now(),
        )


@dataclass
class Position:
    """User's position in a market."""
    market_id: str
    market_question: str
    outcome: str
    shares: Decimal
    avg_price: Decimal
    current_value: Decimal
    pnl: Decimal
    pnl_percent: float

    @classmethod
    def from_dict(cls, data: Dict) -> "Position":
        return cls(
            market_id=data.get("market_id", ""),
            market_question=data.get("question", ""),
            outcome=data.get("outcome", ""),
            shares=Decimal(str(data.get("shares", 0))),
            avg_price=Decimal(str(data.get("avg_price", 0))),
            current_value=Decimal(str(data.get("value", 0))),
            pnl=Decimal(str(data.get("pnl", 0))),
            pnl_percent=data.get("pnl_percent", 0.0),
        )


class BankrPolymarket:
    """
    Polymarket prediction market operations via Bankr API.

    Provides:
    - Market browsing and search
    - Odds checking
    - Bet placement
    - Position tracking
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

    def _validate_bet_amount(self, amount_usd: float) -> bool:
        """Validate bet amount against limits."""
        if not self.config.polymarket_enabled:
            raise ValueError("Polymarket is disabled in config")

        if amount_usd > float(self.config.max_bet_usd):
            raise ValueError(
                f"Amount ${amount_usd} exceeds max bet limit ${self.config.max_bet_usd}"
            )

        return True

    async def get_odds(self, market_query: str) -> Dict[str, float]:
        """
        Get odds for a prediction market.

        Args:
            market_query: Market description or ID

        Returns:
            Dict mapping outcomes to probability (0-1)
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                f"What are the Polymarket odds for {market_query}?"
            )

            return result.get("odds", {})

        except Exception as e:
            logger.error(f"Odds fetch failed: {e}")
            return {}

    async def search_markets(self, query: str) -> List[Market]:
        """
        Search for prediction markets.

        Args:
            query: Search query

        Returns:
            List of matching markets
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                f"Search Polymarket for markets about: {query}"
            )

            markets = result.get("markets", [])
            return [Market.from_response(m) for m in markets]

        except Exception as e:
            logger.error(f"Market search failed: {e}")
            return []

    async def get_market(self, market_id: str) -> Optional[Market]:
        """
        Get details for a specific market.

        Args:
            market_id: Market identifier

        Returns:
            Market details or None
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                f"Get Polymarket details for market {market_id}"
            )

            return Market.from_response(result)

        except Exception as e:
            logger.error(f"Market fetch failed: {e}")
            return None

    async def place_bet(
        self,
        market: str,
        outcome: str,
        amount_usd: float
    ) -> BetResult:
        """
        Place a bet on a prediction market.

        Args:
            market: Market description or ID
            outcome: Outcome to bet on
            amount_usd: Amount in USD

        Returns:
            BetResult with transaction details
        """
        try:
            self._validate_bet_amount(amount_usd)

            client = await self._get_client()
            prompt = f"Place ${amount_usd} bet on {outcome} for {market} on Polymarket"

            logger.info(f"Placing bet: {prompt}")
            job = await client.execute_trade(prompt)

            return BetResult.from_job_result(job)

        except Exception as e:
            logger.error(f"Bet placement failed: {e}")
            return BetResult.error_result(str(e))

    async def get_positions(self) -> List[Position]:
        """
        Get current Polymarket positions.

        Returns:
            List of current positions
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                "Show my current Polymarket positions"
            )

            positions = result.get("positions", [])
            return [Position.from_dict(p) for p in positions]

        except Exception as e:
            logger.error(f"Positions fetch failed: {e}")
            return []

    async def get_trending_markets(self) -> List[Market]:
        """
        Get trending prediction markets.

        Returns:
            List of trending markets
        """
        try:
            client = await self._get_client()
            result = await client.execute(
                "What are the trending Polymarket markets right now?"
            )

            markets = result.get("markets", [])
            return [Market.from_response(m) for m in markets]

        except Exception as e:
            logger.error(f"Trending fetch failed: {e}")
            return []

    async def execute(self, intent: Any) -> BetResult:
        """
        Execute a Polymarket operation from a parsed intent.

        Used by the NLP command router.
        """
        action = getattr(intent, 'action', '').lower()
        params = getattr(intent, 'parameters', {})

        if action in ('bet', 'place_bet', 'wager'):
            return await self.place_bet(
                market=params.get('market', ''),
                outcome=params.get('outcome', ''),
                amount_usd=params.get('amount_usd', 0),
            )
        else:
            return BetResult.error_result(f"Unknown action: {action}")

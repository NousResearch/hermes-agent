"""
x402 Payment Client.

Uses Bankr x402 SDK for automatic micropayments when calling paid APIs.
Cost: $0.01 USDC per request on Base network.
"""

import logging
from typing import Optional, Dict, Any
from decimal import Decimal
from datetime import datetime, date
from dataclasses import dataclass, field

try:
    import httpx
except ImportError:
    httpx = None

from .config import get_x402_config

logger = logging.getLogger(__name__)


@dataclass
class PaymentRecord:
    """Record of a payment made."""
    url: str
    amount_usdc: Decimal
    tx_hash: Optional[str]
    timestamp: datetime
    success: bool


@dataclass
class SpendingTracker:
    """Track daily spending to enforce limits."""
    date: date = field(default_factory=date.today)
    total_spent: Decimal = Decimal("0")
    payments: list = field(default_factory=list)

    def add_payment(self, amount: Decimal):
        """Add a payment to today's spending."""
        today = date.today()
        if self.date != today:
            # New day, reset
            self.date = today
            self.total_spent = Decimal("0")
            self.payments = []

        self.total_spent += amount
        self.payments.append({
            "amount": amount,
            "time": datetime.now().isoformat()
        })

    def can_spend(self, amount: Decimal, limit: Decimal) -> bool:
        """Check if we can spend an amount within limit."""
        today = date.today()
        if self.date != today:
            return amount <= limit
        return (self.total_spent + amount) <= limit


class X402Client:
    """
    HTTP client with automatic x402 payment handling.

    When a request returns 402 Payment Required, this client
    automatically pays via Bankr and retries the request.
    """

    def __init__(self, api_key: str = None):
        if httpx is None:
            raise ImportError("httpx is required: pip install httpx")

        self.config = get_x402_config()
        self.api_key = api_key or self.config.bankr_api_key
        self.spending = SpendingTracker()
        self._session: Optional[httpx.AsyncClient] = None

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(timeout=30.0)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.aclose()
            self._session = None

    async def request(
        self,
        method: str,
        url: str,
        auto_pay: bool = True,
        **kwargs
    ) -> httpx.Response:
        """
        Make an HTTP request with optional x402 payment handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            auto_pay: Whether to automatically pay 402 responses
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        session = await self._get_session()
        response = await session.request(method, url, **kwargs)

        # Handle 402 Payment Required
        if response.status_code == 402 and auto_pay:
            return await self._handle_payment_required(
                method, url, response, **kwargs
            )

        return response

    async def _handle_payment_required(
        self,
        method: str,
        url: str,
        response: httpx.Response,
        **kwargs
    ) -> httpx.Response:
        """Handle 402 response by paying and retrying."""
        # Parse payment requirements from headers
        payment_info = self._parse_payment_headers(response)

        if not payment_info:
            logger.warning(f"402 response but no payment info: {url}")
            return response

        amount = payment_info.get("amount", self.config.cost_per_request)

        # Check spending limit
        if not self.spending.can_spend(amount, self.config.daily_spend_limit):
            logger.error(f"Daily spending limit reached: {self.spending.total_spent}")
            return response

        # Make payment via Bankr
        payment_result = await self._make_payment(payment_info)

        if not payment_result.get("success"):
            logger.error(f"Payment failed: {payment_result.get('error')}")
            return response

        # Record spending
        self.spending.add_payment(amount)

        # Retry request with payment proof
        session = await self._get_session()
        headers = kwargs.pop("headers", {})
        headers["X-Payment"] = payment_result.get("proof", "")
        headers["X-Payment-Token"] = payment_result.get("token", "")

        return await session.request(method, url, headers=headers, **kwargs)

    def _parse_payment_headers(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse x402 payment requirements from response headers."""
        headers = response.headers

        if headers.get("X-Payment-Required") != "true":
            return {}

        return {
            "network": headers.get("X-Payment-Network", "base"),
            "amount": Decimal(headers.get("X-Payment-Amount", "0.01")),
            "currency": headers.get("X-Payment-Currency", "USDC"),
            "address": headers.get("X-Payment-Address", ""),
            "memo": headers.get("X-Payment-Memo", ""),
        }

    async def _make_payment(self, payment_info: Dict) -> Dict[str, Any]:
        """Make payment via Bankr SDK."""
        try:
            from Cosmos.integration.bankr import get_bankr_client

            client = get_bankr_client()
            amount = payment_info.get("amount", Decimal("0.01"))
            address = payment_info.get("address", "")

            # Use Bankr to make the payment
            # This is a simplified version - actual implementation would use
            # Bankr's x402 SDK for proper payment proof generation
            result = await client.execute(
                f"Pay {amount} USDC to {address} on {payment_info.get('network', 'base')}"
            )

            return {
                "success": True,
                "proof": result.get("signature", ""),
                "token": result.get("payment_token", ""),
                "tx_hash": result.get("tx_hash"),
            }

        except Exception as e:
            logger.error(f"Payment error: {e}")
            return {"success": False, "error": str(e)}

    # Convenience methods

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Make GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Make POST request."""
        return await self.request("POST", url, **kwargs)

    async def get_json(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make GET request and return JSON."""
        response = await self.get(url, **kwargs)
        response.raise_for_status()
        return response.json()

    async def post_json(self, url: str, data: Dict, **kwargs) -> Dict[str, Any]:
        """Make POST request with JSON and return JSON."""
        response = await self.post(url, json=data, **kwargs)
        response.raise_for_status()
        return response.json()

    def get_spending_summary(self) -> Dict[str, Any]:
        """Get today's spending summary."""
        return {
            "date": self.spending.date.isoformat(),
            "total_spent_usdc": str(self.spending.total_spent),
            "daily_limit_usdc": str(self.config.daily_spend_limit),
            "remaining_usdc": str(self.config.daily_spend_limit - self.spending.total_spent),
            "payment_count": len(self.spending.payments),
        }

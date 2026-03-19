"""
x402 Server Middleware.

FastAPI middleware to gate endpoints with x402 micropayments.
"""

import logging
from typing import Optional, Dict, Any, Callable, List
from decimal import Decimal
from functools import wraps
import hashlib
import time

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_x402_config
from .pricing import get_endpoint_price

logger = logging.getLogger(__name__)


class X402PaymentGate(BaseHTTPMiddleware):
    """
    FastAPI middleware that gates endpoints with x402 payments.

    Add to FastAPI app:
        app.add_middleware(X402PaymentGate)
    """

    # Endpoints that are always free
    FREE_ENDPOINTS = {
        "/",
        "/health",
        "/api/health",
        "/api/status",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    def __init__(self, app, exclude_paths: List[str] = None):
        super().__init__(app)
        self.config = get_x402_config()
        self.exclude_paths = set(exclude_paths or []) | self.FREE_ENDPOINTS

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request, checking for payment if required."""
        path = request.url.path

        # Skip excluded paths
        if path in self.exclude_paths:
            return await call_next(request)

        # Skip if x402 is disabled
        if not self.config.enabled:
            return await call_next(request)

        # Get endpoint price
        price = get_endpoint_price(path)

        # Skip free endpoints
        if price <= 0:
            return await call_next(request)

        # Check for valid payment
        payment_header = request.headers.get("X-Payment")
        payment_token = request.headers.get("X-Payment-Token")

        if not payment_header:
            return self._payment_required_response(path, price)

        # Verify payment
        if not await self._verify_payment(payment_header, payment_token, price):
            return self._payment_required_response(path, price)

        # Payment verified, proceed
        return await call_next(request)

    def _payment_required_response(self, path: str, price: Decimal) -> Response:
        """Return 402 Payment Required with payment instructions."""
        return JSONResponse(
            status_code=402,
            content={
                "error": "Payment Required",
                "message": f"This endpoint requires payment of {price} USDC",
                "endpoint": path,
                "price": str(price),
                "currency": "USDC",
                "network": self.config.network,
            },
            headers={
                "X-Payment-Required": "true",
                "X-Payment-Network": self.config.network,
                "X-Payment-Amount": str(price),
                "X-Payment-Currency": "USDC",
                "X-Payment-Address": self.config.receiver_wallet,
            }
        )

    async def _verify_payment(
        self,
        payment_proof: str,
        payment_token: str,
        expected_amount: Decimal
    ) -> bool:
        """
        Verify a payment proof.

        In production, this would:
        1. Check the signature against the payment
        2. Verify on-chain that the transaction exists
        3. Confirm the amount matches

        For now, this is a simplified verification.
        """
        if not payment_proof or not self.config.receiver_wallet:
            return False

        try:
            # Simplified verification - in production use facilitator or on-chain check
            # This would verify:
            # - Payment was made to receiver_wallet
            # - Amount >= expected_amount
            # - Payment is recent (not replayed)

            # For development, accept any non-empty proof
            if len(payment_proof) > 10:
                logger.info(f"Payment verified: {payment_proof[:20]}...")
                return True

            return False

        except Exception as e:
            logger.error(f"Payment verification error: {e}")
            return False


def x402_required(price: Decimal = None):
    """
    Decorator to require x402 payment for a specific endpoint.

    Usage:
        @app.get("/api/premium")
        @x402_required(price=Decimal("0.05"))
        async def premium_endpoint():
            return {"data": "premium content"}
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            config = get_x402_config()

            if not config.enabled:
                return await func(request, *args, **kwargs)

            endpoint_price = price or config.default_endpoint_price

            # Check payment
            payment_header = request.headers.get("X-Payment")
            if not payment_header:
                raise HTTPException(
                    status_code=402,
                    detail={
                        "error": "Payment Required",
                        "price": str(endpoint_price),
                        "currency": "USDC",
                        "network": config.network,
                    },
                    headers={
                        "X-Payment-Required": "true",
                        "X-Payment-Amount": str(endpoint_price),
                    }
                )

            return await func(request, *args, **kwargs)

        return wrapper
    return decorator


class PaymentTracker:
    """Track received payments for analytics."""

    def __init__(self):
        self.payments: List[Dict] = []
        self.total_received = Decimal("0")

    def record_payment(
        self,
        endpoint: str,
        amount: Decimal,
        payer: str = None,
        tx_hash: str = None
    ):
        """Record a received payment."""
        self.payments.append({
            "endpoint": endpoint,
            "amount": str(amount),
            "payer": payer,
            "tx_hash": tx_hash,
            "timestamp": time.time(),
        })
        self.total_received += amount

    def get_stats(self) -> Dict[str, Any]:
        """Get payment statistics."""
        by_endpoint = {}
        for p in self.payments:
            ep = p["endpoint"]
            if ep not in by_endpoint:
                by_endpoint[ep] = {"count": 0, "total": Decimal("0")}
            by_endpoint[ep]["count"] += 1
            by_endpoint[ep]["total"] += Decimal(p["amount"])

        return {
            "total_received_usdc": str(self.total_received),
            "total_payments": len(self.payments),
            "by_endpoint": {
                k: {"count": v["count"], "total": str(v["total"])}
                for k, v in by_endpoint.items()
            }
        }


# Global tracker
payment_tracker = PaymentTracker()

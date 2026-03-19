"""
x402 Endpoint Pricing Configuration.

Define prices for Cosmos API endpoints.
"""

from decimal import Decimal
from typing import Dict
import re


# Endpoint pricing in USDC
ENDPOINT_PRICING: Dict[str, Decimal] = {
    # Free tier
    "/": Decimal("0"),
    "/health": Decimal("0"),
    "/api/health": Decimal("0"),
    "/api/status": Decimal("0"),

    # Basic queries ($0.001 USDC = 0.1 cent)
    "/api/chat": Decimal("0.001"),
    "/api/memory/query": Decimal("0.001"),
    "/api/memory/search": Decimal("0.001"),

    # AI generation ($0.01 USDC = 1 cent)
    "/api/generate": Decimal("0.01"),
    "/api/analyze": Decimal("0.01"),
    "/api/summarize": Decimal("0.01"),

    # Premium features ($0.05 USDC = 5 cents)
    "/api/swarm/respond": Decimal("0.05"),
    "/api/swarm/chat": Decimal("0.05"),
    "/api/evolution/task": Decimal("0.05"),
    "/api/evolution/spawn": Decimal("0.05"),

    # High-value operations ($0.10 USDC = 10 cents)
    "/api/code/generate": Decimal("0.10"),
    "/api/code/review": Decimal("0.10"),

    # Trading operations (higher due to potential value)
    "/api/trade/execute": Decimal("0.25"),
    "/api/trade/swap": Decimal("0.25"),

    # Premium quantum trading (x402 Solana-native, handled by solana_gate)
    # Two tiers: Simulated (0.25 SOL) and Real Quantum Hardware (1 SOL)
    # Listed here for pricing catalog completeness; actual gating is in solana_gate.py
    "/api/x402/quantum/analyze": Decimal("0"),  # Gated by SOL payment (0.25 or 1 SOL), not USDC middleware
    "/api/x402/quantum/pricing": Decimal("0"),  # Free — pricing info (shows both tiers)
    "/api/x402/quantum/stats": Decimal("0"),    # Free — public stats
}

# Pattern-based pricing for dynamic endpoints
PATTERN_PRICING = [
    (re.compile(r"^/api/v\d+/.*"), Decimal("0.001")),  # Versioned API
    (re.compile(r"^/api/premium/.*"), Decimal("0.05")),  # Premium endpoints
    (re.compile(r"^/api/enterprise/.*"), Decimal("0.10")),  # Enterprise endpoints
]

# Default price for unlisted endpoints
DEFAULT_PRICE = Decimal("0.001")


def get_endpoint_price(path: str) -> Decimal:
    """
    Get the price for an endpoint.

    Args:
        path: Request path

    Returns:
        Price in USDC
    """
    # Normalize path
    path = path.rstrip("/")

    # Check exact match first
    if path in ENDPOINT_PRICING:
        return ENDPOINT_PRICING[path]

    # Check pattern matches
    for pattern, price in PATTERN_PRICING:
        if pattern.match(path):
            return price

    return DEFAULT_PRICE


def set_endpoint_price(path: str, price: Decimal):
    """
    Set or update the price for an endpoint.

    Args:
        path: Endpoint path
        price: Price in USDC
    """
    ENDPOINT_PRICING[path] = price


def get_all_pricing() -> Dict[str, str]:
    """Get all endpoint pricing as strings."""
    return {path: str(price) for path, price in ENDPOINT_PRICING.items()}


def calculate_request_cost(paths: list) -> Decimal:
    """
    Calculate total cost for multiple requests.

    Args:
        paths: List of endpoint paths

    Returns:
        Total cost in USDC
    """
    return sum(get_endpoint_price(p) for p in paths)

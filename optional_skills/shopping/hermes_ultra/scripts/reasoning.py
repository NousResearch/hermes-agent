"""Scalper Reasoning module — explains *why* a price is suspicious.

Compares MSRP (manufacturer's suggested retail price) with the current
price, analyses market context and price history, and produces a
human-readable explanation of the price deviation plus a clear
BUY / WAIT / AVOID recommendation.

Works fully rule-based (no LLM dependency) — but can be enriched
by an optional LLM for more nuanced explanations.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScalperReasoning:
    """Full reasoning output for a price analysis."""

    msrp: Optional[float] = None
    current_price: float = 0.0
    markup_pct: float = 0.0
    reasoning: str = ""
    recommendation: str = "UNKNOWN"  # BUY / WAIT / AVOID
    factors: List[str] = field(default_factory=list)
    confidence: str = "low"  # low / medium / high
    market_context: str = ""


# ---------------------------------------------------------------------------
# Factor detection helpers (private)
# ---------------------------------------------------------------------------

def _detect_supply_shortage(
    price_history: Optional[List[float]],
    market_prices: Optional[List[dict]],
) -> Optional[str]:
    """Check if prices are rising across the board (supply issue)."""
    if not market_prices:
        return None
    priced = [r["price"] for r in market_prices if r.get("price") and r["price"] > 0]
    if len(priced) < 2:
        return None

    avg = sum(priced) / len(priced)
    min_p = min(priced)
    spread_pct = ((avg - min_p) / min_p) * 100 if min_p > 0 else 0

    # If even the cheapest option is close to the average, supply is tight
    if spread_pct < 5 and avg > 0:
        return "All sellers have similar high prices — likely a supply shortage or high demand."
    return None


def _detect_new_model_released(
    price_history: Optional[List[float]],
) -> Optional[str]:
    """Check if there's a recent sharp price drop (new model incoming)."""
    if not price_history or len(price_history) < 5:
        return None
    recent = price_history[:3]  # newest first
    older = price_history[3:]
    avg_recent = sum(recent) / len(recent)
    avg_older = sum(older) / len(older)
    if avg_older > 0 and avg_recent < avg_older * 0.85:
        return "Prices dropped recently — a newer model may have been released."
    return None


def _detect_seasonal_inflation(
    markup_pct: float,
    price_history: Optional[List[float]],
) -> Optional[str]:
    """Detect if the price spike is consistent with seasonal patterns."""
    if not price_history or len(price_history) < 7:
        return None
    if 10 <= markup_pct <= 25:
        return "Moderate markup may be seasonal (holiday demand, back-to-school, etc.)."
    return None


def _detect_third_party_scalper(
    markup_pct: float,
    seller: str,
) -> Optional[str]:
    """Check if a third-party seller is inflating the price."""
    known_retailers = [
        "amazon", "best buy", "walmart", "newegg", "target",
        "ebay", "idealo", "pricespy",
    ]
    seller_lower = seller.lower() if seller else ""
    is_official = any(r in seller_lower for r in known_retailers)

    if not is_official and markup_pct > 20:
        return f"Third-party seller '{seller}' is charging {markup_pct:.0f}% above MSRP — likely scalper pricing."
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_price_reasoning(
    product_name: str,
    current_price: float,
    original_price: Optional[float] = None,
    market_prices: Optional[List[dict]] = None,
    price_history: Optional[List[float]] = None,
    seller: str = "",
) -> ScalperReasoning:
    """Analyse a product's price and explain any discrepancies.

    Args:
        product_name: Human-readable product name.
        current_price: The price being evaluated.
        original_price: MSRP or list price (if known).
        market_prices: List of ``{site, price, ...}`` from other stores.
        price_history: Price history list (newest first).
        seller: Name of the current seller.

    Returns:
        A :class:`ScalperReasoning` with structured explanation.
    """
    result = ScalperReasoning(current_price=current_price)

    if current_price <= 0:
        result.reasoning = "Price information is not available."
        result.recommendation = "UNKNOWN"
        return result

    # --- Determine MSRP ---
    msrp = original_price
    if not msrp and market_prices:
        priced = [r["price"] for r in market_prices if r.get("price") and r["price"] > 0]
        if priced:
            msrp = min(priced)
    if not msrp and price_history and len(price_history) >= 3:
        msrp = sum(price_history) / len(price_history)

    result.msrp = msrp

    if not msrp or msrp <= 0:
        result.reasoning = (
            f"{product_name} is currently priced at ${current_price:,.2f}. "
            "No MSRP or historical data available for comparison."
        )
        result.recommendation = "WAIT"
        result.confidence = "low"
        return result

    # --- Calculate markup ---
    markup_pct = ((current_price - msrp) / msrp) * 100
    result.markup_pct = round(markup_pct, 1)

    # --- Detect factors ---
    factors: List[str] = []

    # Supply shortage?
    factor = _detect_supply_shortage(price_history, market_prices)
    if factor:
        factors.append(factor)

    # New model released?
    factor = _detect_new_model_released(price_history)
    if factor:
        factors.append(factor)

    # Seasonal inflation?
    factor = _detect_seasonal_inflation(markup_pct, price_history)
    if factor:
        factors.append(factor)

    # Third-party scalper?
    factor = _detect_third_party_scalper(markup_pct, seller)
    if factor:
        factors.append(factor)

    result.factors = factors

    # --- Build reasoning text ---
    parts: List[str] = []

    if markup_pct > 0:
        parts.append(
            f"{product_name} is currently priced at ${current_price:,.2f}, "
            f"which is {markup_pct:.1f}% above the reference price of ${msrp:,.2f}."
        )
    elif markup_pct < -5:
        parts.append(
            f"{product_name} is currently priced at ${current_price:,.2f}, "
            f"which is {abs(markup_pct):.1f}% below the reference price of ${msrp:,.2f} — a discount!"
        )
    else:
        parts.append(
            f"{product_name} is priced at ${current_price:,.2f}, "
            f"close to the reference price of ${msrp:,.2f}."
        )

    if factors:
        parts.append("\nWhy this price?")
        for i, f in enumerate(factors, 1):
            parts.append(f"  {i}. {f}")

    # Market context
    if market_prices:
        priced = [r for r in market_prices if r.get("price") and r["price"] > 0]
        if priced:
            cheapest = min(priced, key=lambda r: r["price"])
            most_expensive = max(priced, key=lambda r: r["price"])
            if cheapest["price"] != most_expensive["price"]:
                result.market_context = (
                    f"Market range: ${cheapest['price']:,.2f} ({cheapest.get('site', '?')}) "
                    f"— ${most_expensive['price']:,.2f} ({most_expensive.get('site', '?')})"
                )
                parts.append(f"\n{result.market_context}")

    result.reasoning = "\n".join(parts)

    # --- Determine recommendation ---
    if markup_pct <= -10:
        result.recommendation = "BUY"
        result.confidence = "high"
    elif markup_pct <= 0:
        result.recommendation = "BUY"
        result.confidence = "medium"
    elif markup_pct <= 10:
        result.recommendation = "WAIT"
        result.confidence = "medium"
    elif markup_pct <= 30:
        result.recommendation = "WAIT"
        result.confidence = "high"
    else:
        result.recommendation = "AVOID"
        result.confidence = "high"
        if any("scalper" in f.lower() for f in factors):
            result.recommendation = "AVOID"
            result.confidence = "high"

    return result

"""Deal scoring engine — rates product deals from 0 to 100.

Factors and weights:
  - Discount rate vs target price : 40 %
  - Price history trend           : 30 %
  - Stock availability            : 15 %
  - Market comparison (orig/list) : 15 %

Score bands:
  0–30  BAD DEAL
  31–60 FAIR DEAL
  61–80 GOOD DEAL
  81–100 UNMISSABLE DEAL
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DealScore:
    """Result of a deal analysis."""
    total_score: int = 0
    discount_score: int = 0
    trend_score: int = 0
    stock_score: int = 0
    market_score: int = 0
    label: str = ""
    explanation: str = ""


_LABELS = [
    (30, "BAD DEAL", "💀"),
    (60, "FAIR DEAL", "🤔"),
    (80, "GOOD DEAL", "👍"),
    (100, "UNMISSABLE DEAL", "🔥"),
]


def _label_for_score(score: int) -> tuple:
    for threshold, label, emoji in _LABELS:
        if score <= threshold:
            return label, emoji
    return "UNMISSABLE DEAL", "🔥"


class DealScorer:
    """Computes a 0–100 deal score for a product."""

    # Weights (must sum to 1.0)
    W_DISCOUNT = 0.40
    W_TREND = 0.30
    W_STOCK = 0.15
    W_MARKET = 0.15

    def calculate(
        self,
        current_price: Optional[float],
        target_price: Optional[float] = None,
        original_price: Optional[float] = None,
        stock_status: str = "unknown",
        price_history: Optional[List[float]] = None,
    ) -> DealScore:
        """Return a DealScore for the given product data.

        Args:
            current_price: The product's current listed price.
            target_price: User's target price (what they want to pay).
            original_price: The product's original/list price before discount.
            stock_status: One of in_stock, out_of_stock, limited, unknown.
            price_history: List of recent prices (newest first).
        """
        result = DealScore()

        if current_price is None or current_price <= 0:
            result.explanation = "Price info unavailable."
            result.label = "Unknown"
            return result

        history = price_history or []

        # 1) Discount score (0–100): how close to/below target price?
        result.discount_score = self._calc_discount(current_price, target_price)

        # 2) Trend score (0–100): is price trending down?
        result.trend_score = self._calc_trend(current_price, history)

        # 3) Stock score (0–100)
        result.stock_score = self._calc_stock(stock_status)

        # 4) Market score (0–100): discount from original list price
        result.market_score = self._calc_market(current_price, original_price)

        # Weighted total
        raw = (
            result.discount_score * self.W_DISCOUNT
            + result.trend_score * self.W_TREND
            + result.stock_score * self.W_STOCK
            + result.market_score * self.W_MARKET
        )
        result.total_score = max(0, min(100, int(round(raw))))

        label, emoji = _label_for_score(result.total_score)
        result.label = f"{emoji} {label}"
        result.explanation = self._build_explanation(result, current_price, target_price)

        return result

    # ------------------------------------------------------------------
    # Sub-score calculators
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_discount(current: float, target: Optional[float]) -> int:
        """Score based on proximity to target price."""
        if target is None or target <= 0:
            return 50  # Neutral when no target set
        if current <= target:
            # At or below target — full marks, bonus for deeper savings
            savings_pct = ((target - current) / target) * 100
            return min(100, 80 + int(savings_pct))
        else:
            # Above target — score decreases with distance
            gap_pct = ((current - target) / target) * 100
            return max(0, int(80 - gap_pct * 1.5))

    @staticmethod
    def _calc_trend(current: float, history: List[float]) -> int:
        """Score based on recent price movement.
        Calibrated for USD/EUR/GBP (tighter margin since volatility is lower).
        """
        if len(history) < 2:
            return 50  # Neutral with insufficient data

        avg_price = sum(history) / len(history)
        if avg_price <= 0:
            return 50

        # Compare current to average
        change_pct = ((avg_price - current) / avg_price) * 100

        if change_pct >= 15:
            return 100  # Significant drop from average
        elif change_pct >= 8:
            return 85
        elif change_pct >= 3:
            return 70
        elif change_pct >= 0:
            return 55
        elif change_pct >= -5:
            return 35
        else:
            return 10  # Price rising significantly

    @staticmethod
    def _calc_stock(status: str) -> int:
        """Score based on stock availability."""
        scores = {
            "in_stock": 70,
            "limited": 90,  # Scarcity drives urgency
            "out_of_stock": 10,
            "unknown": 50,
        }
        return scores.get(status, 50)

    @staticmethod
    def _calc_market(current: float, original: Optional[float]) -> int:
        """Score based on discount from original/list price.
        Calibrated for USD/EUR/GBP (tighter discount expectations).
        """
        if original is None or original <= 0 or original <= current:
            return 40  # No discount visible

        discount_pct = ((original - current) / original) * 100

        if discount_pct >= 40:
            return 100
        elif discount_pct >= 20:
            return 85
        elif discount_pct >= 10:
            return 65
        elif discount_pct >= 5:
            return 50
        else:
            return 40

    @staticmethod
    def _build_explanation(score: DealScore, current: float, target: Optional[float]) -> str:
        parts = []
        if target and current <= target:
            savings = target - current
            parts.append(f"${savings:,.2f} below target price!")
        elif target:
            gap = current - target
            parts.append(f"${gap:,.2f} above target price.")

        if score.trend_score >= 70:
            parts.append("Price is on a downward trend.")
        elif score.trend_score <= 30:
            parts.append("Price is on an upward trend.")

        if score.stock_score >= 80:
            parts.append("Stock is limited — act fast!")

        return " ".join(parts) if parts else "At standard market price."

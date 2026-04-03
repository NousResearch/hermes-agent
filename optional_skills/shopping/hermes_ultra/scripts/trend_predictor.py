"""Trend prediction based on price history.

Uses simple linear regression and moving averages to predict
short-term price direction. No external ML dependencies.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrendReport:
    """Result of a trend analysis."""
    direction: str = "STABLE"  # DOWN, UP, STABLE
    direction_emoji: str = "➡️"
    predicted_change_pct: float = 0.0
    predicted_price: Optional[float] = None
    confidence: str = "low"  # low, medium, high
    analysis_text: str = ""
    moving_avg_7: Optional[float] = None
    moving_avg_30: Optional[float] = None
    data_points: int = 0
    vs_avg_pct: Optional[float] = None  # How current price compares to weekly avg
    volatility_warning: bool = False


def _linear_regression(prices: List[float]) -> tuple:
    """Simple linear regression on price data.

    Returns (slope, intercept, r_squared).
    Prices are ordered oldest-to-newest.
    """
    n = len(prices)
    if n < 2:
        return 0.0, prices[0] if prices else 0.0, 0.0

    x_vals = list(range(n))
    x_mean = sum(x_vals) / n
    y_mean = sum(prices) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, prices))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)

    if denominator == 0:
        return 0.0, y_mean, 0.0

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # R-squared
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_vals, prices))
    ss_tot = sum((y - y_mean) ** 2 for y in prices)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return slope, intercept, r_squared


def _moving_average(prices: List[float], window: int) -> Optional[float]:
    """Calculate moving average for the last *window* prices."""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


class TrendPredictor:
    """Predicts short-term price trends from history."""

    # How many data points ahead to predict (e.g., 7 = ~1 week)
    FORECAST_HORIZON = 7

    def predict(
        self,
        price_history: List[float],
        current_price: Optional[float] = None,
    ) -> TrendReport:
        """Analyze price history and predict trend direction.

        Args:
            price_history: Prices ordered newest-first (most recent → oldest).
            current_price: The latest price (optional, used for display).
        """
        report = TrendReport()
        report.data_points = len(price_history)

        if len(price_history) < 3:
            report.analysis_text = (
                f"Need at least 3 data points for trend prediction. "
                f"Currently have {len(price_history)} data point(s)."
            )
            return report

        # Reverse to oldest-first for regression
        prices = list(reversed(price_history))
        latest = current_price or prices[-1]

        # Linear regression
        slope, intercept, r_squared = _linear_regression(prices)

        # Moving averages (on oldest-first data)
        report.moving_avg_7 = _moving_average(prices, min(7, len(prices)))
        report.moving_avg_30 = _moving_average(prices, min(30, len(prices)))

        # Calculate how current price compares to weekly average
        if report.moving_avg_7 and report.moving_avg_7 > 0 and latest > 0:
            report.vs_avg_pct = round(
                ((latest - report.moving_avg_7) / report.moving_avg_7) * 100, 1
            )

        # Volatility detection (Standard Deviation / Mean > 10% indicates high volatility)
        if len(prices) >= 5:
            mean = sum(prices) / len(prices)
            if mean > 0:
                variance = sum((p - mean) ** 2 for p in prices) / len(prices)
                std_dev = math.sqrt(variance)
                if (std_dev / mean) > 0.1:  # 10% threshold
                    report.volatility_warning = True

        # Predict future price
        future_idx = len(prices) + self.FORECAST_HORIZON
        predicted = slope * future_idx + intercept
        report.predicted_price = max(0, round(predicted, 2))

        # Calculate predicted change
        if latest > 0:
            report.predicted_change_pct = round(
                ((predicted - latest) / latest) * 100, 1
            )
        else:
            report.predicted_change_pct = 0.0

        # Determine confidence
        if r_squared >= 0.7 and len(prices) >= 10:
            report.confidence = "high"
        elif r_squared >= 0.4 and len(prices) >= 5:
            report.confidence = "medium"
        else:
            report.confidence = "low"

        # Determine direction
        if report.predicted_change_pct <= -5:
            report.direction = "DOWN"
            report.direction_emoji = "📉"
        elif report.predicted_change_pct >= 5:
            report.direction = "UP"
            report.direction_emoji = "📈"
        else:
            report.direction = "STABLE"
            report.direction_emoji = "➡️"

        # Build analysis text
        report.analysis_text = self._build_text(report, latest, len(prices))
        return report

    @staticmethod
    def _build_text(report: TrendReport, current: float, n_points: int) -> str:
        parts = []

        # Weekly average comparison (the key "intelligence" line)
        if report.vs_avg_pct is not None and report.moving_avg_7:
            if report.vs_avg_pct < -1:
                parts.append(
                    f"This price is {abs(report.vs_avg_pct):.1f}% lower than "
                    f"the weekly average (${report.moving_avg_7:,.2f})."
                )
            elif report.vs_avg_pct > 1:
                parts.append(
                    f"This price is {report.vs_avg_pct:.1f}% higher than "
                    f"the weekly average (${report.moving_avg_7:,.2f})."
                )
            else:
                parts.append(
                    f"Price is in line with the weekly average "
                    f"(${report.moving_avg_7:,.2f})."
                )

        # Direction forecast
        if report.direction == "DOWN":
            parts.append(
                f"Price is trending downward — may drop ~{abs(report.predicted_change_pct):.1f}% "
                f"within the next week."
            )
            if report.predicted_price:
                parts.append(
                    f"Projected price: ${report.predicted_price:,.2f}."
                )
        elif report.direction == "UP":
            parts.append(
                f"Price is trending upward — may rise ~{report.predicted_change_pct:.1f}% "
                f"within the next week."
            )
            if report.predicted_price:
                parts.append(
                    f"Projected price: ${report.predicted_price:,.2f}."
                )
        else:
            parts.append("Price is holding steady, no significant movement expected.")

        parts.append(f"Confidence: {report.confidence} ({n_points} data points).")

        return " ".join(parts)

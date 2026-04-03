"""Resolve outstanding market predictions against actual data.

Fetches actual values via yfinance and compares against predicted values.
Called at the end of each scheduler tick to resolve any predictions whose
target resolution time has passed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Risk dashboard thresholds (from daily-market-briefing skill)
_RISK_THRESHOLDS = {
    "INR/USD": {"green_max": 83.50, "amber_max": 84.50},
    "Brent": {"green_max": 80.0, "amber_max": 90.0},
    "Crude": {"green_max": 80.0, "amber_max": 90.0},
    "US 10Y": {"green_max": 4.25, "amber_max": 4.75},
    "India VIX": {"green_max": 15.0, "amber_max": 20.0},
}

# yfinance tickers for resolution
_RESOLUTION_TICKERS = {
    "INR/USD": "INR=X",
    "Brent": "BZ=F",
    "Crude": "CL=F",
    "US 10Y": "^TNX",
    "India VIX": "^INDIAVIX",
    "NIFTY": "^NSEI",
    "SENSEX": "^BSESN",
    "Bank Nifty": "^NSEBANK",
    "market": "^NSEI",
}


def resolve_pending_predictions(db: Any) -> List[Dict[str, Any]]:
    """Resolve all predictions whose target time has passed.

    Args:
        db: SessionDB instance with prediction methods.

    Returns:
        List of resolution results for logging/reporting.
    """
    now = time.time()
    unresolved = db.get_unresolved_predictions(before_timestamp=now)

    if not unresolved:
        return []

    results = []
    for pred in unresolved:
        pred_id = pred["id"]
        pred_type = pred["prediction_type"]
        subject = pred["subject"]
        predicted_value = pred["predicted_value"]

        try:
            actual, correct = _resolve_single(pred_type, subject, predicted_value)
            if actual is not None:
                db.resolve_prediction(pred_id, actual, correct)
                results.append({
                    "id": pred_id,
                    "type": pred_type,
                    "subject": subject,
                    "predicted": predicted_value,
                    "actual": actual,
                    "correct": correct,
                })
        except Exception as e:
            logger.debug("Failed to resolve prediction %d: %s", pred_id, e)

    if results:
        correct_count = sum(1 for r in results if r["correct"])
        logger.info(
            "Resolved %d predictions: %d correct, %d incorrect",
            len(results), correct_count, len(results) - correct_count,
        )

    return results


def _resolve_single(
    pred_type: str,
    subject: str,
    predicted_value: str,
) -> Tuple[Optional[str], bool]:
    """Resolve a single prediction.

    Returns (actual_value, is_correct) or (None, False) if unable to resolve.
    """
    if pred_type == "risk_color":
        return _resolve_risk_color(subject, predicted_value)
    elif pred_type == "direction":
        return _resolve_direction(subject, predicted_value)
    elif pred_type == "fii_trend":
        return _resolve_fii_trend(subject, predicted_value)
    elif pred_type == "level":
        return _resolve_level(subject, predicted_value)
    else:
        return None, False


def _resolve_risk_color(
    subject: str,
    predicted_color: str,
) -> Tuple[Optional[str], bool]:
    """Check if a risk factor's actual value matches the predicted color."""
    ticker = _RESOLUTION_TICKERS.get(subject)
    thresholds = _RISK_THRESHOLDS.get(subject)
    if not ticker or not thresholds:
        return None, False

    actual_value = _fetch_current_price(ticker)
    if actual_value is None:
        return None, False

    green_max = thresholds["green_max"]
    amber_max = thresholds["amber_max"]

    if actual_value <= green_max:
        actual_color = "GREEN"
    elif actual_value <= amber_max:
        actual_color = "AMBER"
    else:
        actual_color = "RED"

    return actual_color, actual_color == predicted_color.upper()


def _resolve_direction(
    subject: str,
    predicted_direction: str,
) -> Tuple[Optional[str], bool]:
    """Check if market moved in the predicted direction."""
    ticker = _RESOLUTION_TICKERS.get(subject)
    if not ticker:
        # Try common mappings
        subject_lower = subject.lower()
        if "nifty" in subject_lower:
            ticker = "^NSEI"
        elif "sensex" in subject_lower:
            ticker = "^BSESN"
        else:
            ticker = "^NSEI"  # Default to Nifty

    change_pct = _fetch_day_change_pct(ticker)
    if change_pct is None:
        return None, False

    actual_direction = "bullish" if change_pct >= 0 else "bearish"
    return actual_direction, actual_direction == predicted_direction.lower()


def _resolve_fii_trend(
    subject: str,
    predicted_action: str,
) -> Tuple[Optional[str], bool]:
    """Check if FII/DII continued the predicted trend.

    This is harder to resolve automatically since FII/DII data comes from
    web sources with a 1-day lag. We use a simplified heuristic.
    """
    # FII/DII data requires web scraping which is expensive in a resolver.
    # For now, return None to skip resolution (will be resolved by next briefing).
    return None, False


def _resolve_level(
    subject: str,
    predicted_level: str,
) -> Tuple[Optional[str], bool]:
    """Check if an index reached the predicted level."""
    ticker = _RESOLUTION_TICKERS.get(subject)
    if not ticker:
        subject_lower = subject.lower()
        if "nifty" in subject_lower:
            ticker = "^NSEI"
        elif "sensex" in subject_lower:
            ticker = "^BSESN"
        else:
            return None, False

    actual_price = _fetch_current_price(ticker)
    if actual_price is None:
        return None, False

    try:
        target = float(predicted_level.replace(",", ""))
    except ValueError:
        return None, False

    actual_str = f"{actual_price:.2f}"
    # "Reached" = within 1% of the target level
    reached = abs(actual_price - target) / target < 0.01
    return actual_str, reached


# ── yfinance helpers ──────────────────────────────────────────────────

def _fetch_current_price(ticker: str) -> Optional[float]:
    """Fetch the latest price for a ticker via yfinance."""
    try:
        import yfinance as yf
        data = yf.download(ticker, period="1d", progress=False, threads=False)
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception as e:
        logger.debug("yfinance fetch failed for %s: %s", ticker, e)
        return None


def _fetch_day_change_pct(ticker: str) -> Optional[float]:
    """Fetch the day's percentage change for a ticker."""
    try:
        import yfinance as yf
        data = yf.download(ticker, period="2d", progress=False, threads=False)
        if len(data) < 2:
            return None
        prev_close = float(data["Close"].iloc[-2])
        curr_close = float(data["Close"].iloc[-1])
        if prev_close == 0:
            return None
        return ((curr_close - prev_close) / prev_close) * 100
    except Exception as e:
        logger.debug("yfinance day change failed for %s: %s", ticker, e)
        return None

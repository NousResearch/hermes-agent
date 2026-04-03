"""Extract implicit predictions from market briefing outputs.

Parses risk dashboard colors, directional language, FII trend calls,
and level forecasts from cron job outputs. These are logged as
predictions for later resolution against actual market data.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Extraction patterns ───────────────────────────────────────��─────

_RISK_COLOR_RE = re.compile(
    r"(INR/USD|Brent|Crude|US\s*10Y|India\s*VIX|FII\s*Flow|Geopolitical)"
    r"\s*[|:\-]?\s*"
    r"(🟢|🟡|🔴|GREEN|AMBER|RED)",
    re.IGNORECASE,
)

_DIRECTION_RE = re.compile(
    r"(bullish|bearish|positive|negative|upside|downside)\s+"
    r"(?:outlook|bias|pressure|sentiment|momentum|tone)"
    r"(?:\s+(?:for|on|in)\s+)?"
    r"(NIFTY|SENSEX|Bank\s*Nifty|market|\w+\s*sector)?",
    re.IGNORECASE,
)

_FII_TREND_RE = re.compile(
    r"(?:sustained|continued|persistent|consecutive)\s+"
    r"(FII|DII)\s+"
    r"(buying|selling|inflow|outflow)",
    re.IGNORECASE,
)

_LEVEL_FORECAST_RE = re.compile(
    r"(NIFTY|SENSEX|Bank\s*Nifty)\s+"
    r"(?:may|likely|expected|could|might)\s+"
    r"(?:test|reach|touch|breach|hold|cross)\s+"
    r"([\d,]+)",
    re.IGNORECASE,
)

_COLOR_MAP = {"🟢": "GREEN", "🟡": "AMBER", "🔴": "RED"}

# Market close time in IST (3:30 PM = 10:00 UTC)
_MARKET_CLOSE_OFFSET = 10 * 3600  # seconds from midnight UTC


def extract_predictions(
    output_text: str,
    job_id: str,
    job_name: str,
) -> List[Dict[str, Any]]:
    """Extract implicit predictions from a market briefing output.

    Returns a list of dicts ready for SessionDB.log_prediction().
    Each dict has: job_id, prediction_type, subject, predicted_value,
    predicted_at, resolution_target_at, confidence, source_output_path.
    """
    predictions = []
    now = time.time()
    text = output_text or ""

    # Next market close (approximate: today 3:30 PM IST if before, tomorrow if after)
    next_close = _next_market_close(now)

    # 1. Risk dashboard colors
    for match in _RISK_COLOR_RE.finditer(text):
        factor = match.group(1).strip()
        color = match.group(2).strip()
        color = _COLOR_MAP.get(color, color.upper())
        predictions.append({
            "job_id": job_id,
            "prediction_type": "risk_color",
            "subject": factor,
            "predicted_value": color,
            "predicted_at": now,
            "resolution_target_at": next_close,
            "confidence": 0.7,
        })

    # 2. Directional language
    for match in _DIRECTION_RE.finditer(text):
        direction = match.group(1).lower()
        subject = (match.group(2) or "market").strip()
        is_bullish = direction in ("bullish", "positive", "upside")
        predictions.append({
            "job_id": job_id,
            "prediction_type": "direction",
            "subject": subject,
            "predicted_value": "bullish" if is_bullish else "bearish",
            "predicted_at": now,
            "resolution_target_at": next_close,
            "confidence": 0.5,
        })

    # 3. FII/DII trend continuation
    for match in _FII_TREND_RE.finditer(text):
        entity = match.group(1).upper()
        action = match.group(2).lower()
        is_buying = action in ("buying", "inflow")
        predictions.append({
            "job_id": job_id,
            "prediction_type": "fii_trend",
            "subject": entity,
            "predicted_value": "buying" if is_buying else "selling",
            "predicted_at": now,
            "resolution_target_at": next_close + 86400,  # Check next day
            "confidence": 0.6,
        })

    # 4. Level forecasts
    for match in _LEVEL_FORECAST_RE.finditer(text):
        index_name = match.group(1).strip()
        level_str = match.group(2).replace(",", "")
        try:
            float(level_str)  # validate
        except ValueError:
            continue
        predictions.append({
            "job_id": job_id,
            "prediction_type": "level",
            "subject": index_name,
            "predicted_value": level_str,
            "predicted_at": now,
            "resolution_target_at": next_close,
            "confidence": 0.4,
        })

    if predictions:
        logger.debug(
            "Extracted %d predictions from job '%s'",
            len(predictions), job_name,
        )

    return predictions


def _next_market_close(now: float) -> float:
    """Approximate next market close timestamp (3:30 PM IST = 10:00 UTC)."""
    import datetime
    utc_now = datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc)
    today_close = utc_now.replace(hour=10, minute=0, second=0, microsecond=0)
    if utc_now.timestamp() > today_close.timestamp():
        today_close += datetime.timedelta(days=1)
    # Skip weekends
    while today_close.weekday() >= 5:
        today_close += datetime.timedelta(days=1)
    return today_close.timestamp()

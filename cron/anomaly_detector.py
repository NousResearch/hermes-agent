"""Rolling z-score anomaly detection for cron job outputs.

Compares current output statistics against a rolling baseline to detect
significant deviations. Uses pure statistics — no ML libraries required.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Regex for extracting numeric values preceded by common labels
_NUMERIC_RE = re.compile(
    r"(?:price|cost|amount|total|count|volume|change|return|yield|ratio|index|rate|"
    r"value|score|level|cap|flow|inflow|outflow|net|gross|pe|pb|eps|roe|"
    r"nifty|sensex|bse|nse|fii|dii|rbi|cpi|wpi|gdp|iip)\s*"
    r"[:\-=~]?\s*"
    r"[₹$€£]?\s*"
    r"([+-]?\d[\d,]*\.?\d*)\s*"
    r"(%|bps|cr|lakh|bn|mn|k)?",
    re.IGNORECASE,
)


@dataclass
class Anomaly:
    """A single detected anomaly in cron output."""
    field: str
    expected_mean: float
    expected_stddev: float
    actual: float
    z_score: float
    severity: str  # "warning" or "critical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "expected_mean": round(self.expected_mean, 2),
            "expected_stddev": round(self.expected_stddev, 2),
            "actual": round(self.actual, 2),
            "z_score": round(self.z_score, 2),
            "severity": self.severity,
        }


def compute_output_stats(output_text: str) -> Dict[str, Any]:
    """Compute statistical summary of a cron job output.

    Returns a dict with basic text stats and extracted numeric values.
    """
    text = output_text or ""

    char_count = len(text)
    line_count = text.count("\n") + 1 if text else 0
    word_count = len(text.split())

    # Extract numeric values
    numeric_values = {}
    for match in _NUMERIC_RE.finditer(text):
        label_end = match.start(1)
        label_start = max(0, label_end - 40)
        label_context = text[label_start:label_end].strip().lower()
        # Use last meaningful word as key
        label_words = re.findall(r"[a-z]+", label_context)
        if label_words:
            key = label_words[-1]
            value_str = match.group(1).replace(",", "")
            try:
                numeric_values[key] = float(value_str)
            except ValueError:
                pass

    entity_count = len(numeric_values)

    return {
        "char_count": char_count,
        "line_count": line_count,
        "word_count": word_count,
        "entity_count": entity_count,
        "numeric_values": numeric_values,
    }


# ── Market-specific extraction ────────────────────────────────────────

_RISK_DASHBOARD_RE = re.compile(
    r"(INR/USD|Brent|Crude|US\s*10Y|India\s*VIX|FII\s*Flow|Geopolitical)"
    r"\s*\|?\s*"
    r"(🟢|🟡|🔴|GREEN|AMBER|RED)",
    re.IGNORECASE,
)

_FII_DII_NET_RE = re.compile(
    r"(FII|DII)\s*(?:\|[^|]*){0,2}\|\s*([+-]?\s*₹?\s*[\d,]+\.?\d*)\s*(cr|crore)?",
    re.IGNORECASE,
)

_INDEX_LEVEL_RE = re.compile(
    r"(NIFTY\s*50?|SENSEX|Bank\s*Nifty|Gift\s*Nifty)\s*[|:\-]?\s*([\d,]+\.?\d*)",
    re.IGNORECASE,
)

_COLOR_MAP = {"🟢": "GREEN", "🟡": "AMBER", "🔴": "RED"}


def compute_market_output_stats(output_text: str) -> Dict[str, Any]:
    """Extended stats for market-skill cron outputs.

    Returns everything compute_output_stats() returns, plus:
    - risk_dashboard: dict mapping factor → color
    - risk_score: int (count of RED + AMBER factors, 0-6)
    - fii_net: float (FII net in crores, if found)
    - dii_net: float (DII net in crores, if found)
    - index_levels: dict mapping index name → level
    - section_count: int (## header count, completeness proxy)
    """
    base_stats = compute_output_stats(output_text)
    text = output_text or ""

    # Risk dashboard extraction
    risk_dashboard = {}
    for match in _RISK_DASHBOARD_RE.finditer(text):
        factor = match.group(1).strip()
        color = match.group(2).strip()
        color = _COLOR_MAP.get(color, color.upper())
        risk_dashboard[factor] = color

    risk_score = sum(
        1 for c in risk_dashboard.values() if c in ("RED", "AMBER")
    )

    # FII/DII net flows
    fii_net = None
    dii_net = None
    for match in _FII_DII_NET_RE.finditer(text):
        entity = match.group(1).upper()
        value_str = match.group(2).replace(",", "").replace("₹", "").strip()
        try:
            value = float(value_str)
        except ValueError:
            continue
        if entity == "FII":
            fii_net = value
        elif entity == "DII":
            dii_net = value

    # Index levels
    index_levels = {}
    for match in _INDEX_LEVEL_RE.finditer(text):
        name = match.group(1).strip().lower().replace(" ", "_")
        try:
            level = float(match.group(2).replace(",", ""))
            index_levels[name] = level
        except ValueError:
            pass

    # Section count (completeness proxy)
    section_count = text.count("\n## ") + text.count("\n### ")

    # Merge into numeric_values for anomaly detection
    numeric_values = base_stats.get("numeric_values", {})
    numeric_values["risk_score"] = risk_score
    if fii_net is not None:
        numeric_values["fii_net"] = fii_net
    if dii_net is not None:
        numeric_values["dii_net"] = dii_net
    for idx_name, level in index_levels.items():
        numeric_values[idx_name] = level
    numeric_values["section_count"] = section_count

    return {
        **base_stats,
        "numeric_values": numeric_values,
        "risk_dashboard": risk_dashboard,
        "risk_score": risk_score,
        "fii_net": fii_net,
        "dii_net": dii_net,
        "index_levels": index_levels,
        "section_count": section_count,
    }


class CronAnomalyDetector:
    """Rolling z-score anomaly detection for cron job outputs."""

    def __init__(
        self,
        db: Any,
        window_size: int = 30,
        warning_threshold: float = 2.0,
        critical_threshold: float = 3.0,
        min_baseline_size: int = 5,
    ):
        self._db = db
        self._window_size = window_size
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        self._min_baseline_size = min_baseline_size

    def check(self, job_id: str, current_stats: Dict[str, Any]) -> List[Anomaly]:
        """Compare current output stats against rolling baseline.

        Returns a list of detected anomalies (empty if output looks normal).
        """
        baseline_rows = self._db.get_cron_output_stats(
            job_id=job_id,
            limit=self._window_size,
        )

        if len(baseline_rows) < self._min_baseline_size:
            logger.debug(
                "Not enough baseline data for job %s (%d < %d)",
                job_id, len(baseline_rows), self._min_baseline_size,
            )
            return []

        anomalies = []

        # Check each numeric field
        numeric_fields = ["char_count", "line_count", "word_count", "entity_count"]
        for field in numeric_fields:
            baseline_values = [
                row[field] for row in baseline_rows
                if row.get(field) is not None
            ]
            current_value = current_stats.get(field)
            if current_value is None or len(baseline_values) < self._min_baseline_size:
                continue

            anomaly = self._check_field(field, float(current_value), baseline_values)
            if anomaly:
                anomalies.append(anomaly)

        # Check extracted numeric values if available
        current_numerics = current_stats.get("numeric_values") or {}
        if isinstance(current_numerics, str):
            try:
                current_numerics = json.loads(current_numerics)
            except (json.JSONDecodeError, TypeError):
                current_numerics = {}

        for key, current_val in current_numerics.items():
            baseline_vals = []
            for row in baseline_rows:
                row_numerics = row.get("numeric_values")
                if isinstance(row_numerics, str):
                    try:
                        row_numerics = json.loads(row_numerics)
                    except (json.JSONDecodeError, TypeError):
                        continue
                if isinstance(row_numerics, dict) and key in row_numerics:
                    baseline_vals.append(float(row_numerics[key]))

            if len(baseline_vals) >= self._min_baseline_size:
                anomaly = self._check_field(
                    f"numeric:{key}", float(current_val), baseline_vals
                )
                if anomaly:
                    anomalies.append(anomaly)

        return anomalies

    def _check_field(
        self,
        field: str,
        current: float,
        baseline: List[float],
    ) -> Optional[Anomaly]:
        """Check if a single field value is anomalous vs its baseline."""
        n = len(baseline)
        if n < 2:
            return None

        mean = sum(baseline) / n
        variance = sum((x - mean) ** 2 for x in baseline) / (n - 1)
        stddev = math.sqrt(variance) if variance > 0 else 0.0

        # If stddev is near zero, any deviation is potentially anomalous
        if stddev < 1e-10:
            if abs(current - mean) > 1e-10:
                return Anomaly(
                    field=field,
                    expected_mean=mean,
                    expected_stddev=0.0,
                    actual=current,
                    z_score=float("inf"),
                    severity="critical",
                )
            return None

        z_score = abs(current - mean) / stddev

        if z_score >= self._critical_threshold:
            severity = "critical"
        elif z_score >= self._warning_threshold:
            severity = "warning"
        else:
            return None

        return Anomaly(
            field=field,
            expected_mean=mean,
            expected_stddev=stddev,
            actual=current,
            z_score=z_score,
            severity=severity,
        )


class MarketAnomalyDetector(CronAnomalyDetector):
    """Finance-aware anomaly detection with domain-specific context.

    Uses the same z-score approach but with tighter thresholds for
    market-critical metrics where sudden changes are meaningful.
    """

    def __init__(self, db: Any, **kwargs):
        super().__init__(
            db,
            window_size=kwargs.get("window_size", 20),
            warning_threshold=kwargs.get("warning_threshold", 1.8),
            critical_threshold=kwargs.get("critical_threshold", 2.5),
            min_baseline_size=kwargs.get("min_baseline_size", 5),
        )

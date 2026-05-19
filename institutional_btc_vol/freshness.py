from __future__ import annotations

from datetime import datetime
from typing import Any

DEFAULT_MAX_AGE_MINUTES = {
    "Deribit": 15,
    "Nasdaq IBIT options": 15,
    "iShares IBIT holdings": 1440,
}


def _coerce_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def evaluate_source_freshness(
    *,
    as_of: datetime | str,
    sources: dict[str, datetime | str | None],
    max_age_minutes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate source capture freshness for the screen-only BTC vol monitor.

    This measures whether raw captures used for a run are recent enough for internal
    review. It does not convert any screen mark into an executable quote.
    """
    as_of_dt = _coerce_datetime(as_of)
    if as_of_dt is None:
        raise ValueError("as_of is required")

    thresholds = {**DEFAULT_MAX_AGE_MINUTES, **(max_age_minutes or {})}
    details: dict[str, Any] = {}
    stale_sources: list[str] = []
    missing_sources: list[str] = []

    for name, captured_at_raw in sources.items():
        max_age = int(thresholds.get(name, 15))
        captured_at = _coerce_datetime(captured_at_raw)
        if captured_at is None:
            details[name] = {
                "status": "missing",
                "captured_at": None,
                "age_minutes": None,
                "max_age_minutes": max_age,
            }
            missing_sources.append(name)
            continue

        age_minutes = max(0.0, round((as_of_dt - captured_at).total_seconds() / 60, 2))
        status = "fresh" if age_minutes <= max_age else "stale"
        details[name] = {
            "status": status,
            "captured_at": captured_at.isoformat(),
            "age_minutes": age_minutes,
            "max_age_minutes": max_age,
        }
        if status == "stale":
            stale_sources.append(name)

    penalty = len(stale_sources) + (2 * len(missing_sources))
    if penalty == 0:
        grade = "green"
    elif penalty <= 2:
        grade = "yellow"
    else:
        grade = "red"

    return {
        "grade": grade,
        "stale_sources": stale_sources,
        "missing_sources": missing_sources,
        "sources": details,
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
    }

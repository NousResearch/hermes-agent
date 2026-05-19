from __future__ import annotations

from typing import Any


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    return None if numeric is None else int(numeric)


def _latest_two_non_null(rows: list[dict[str, Any]], key: str) -> tuple[float, float] | None:
    values = [_as_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    return values[-2], values[-1]


def _nearest_iv(rows: list[dict[str, Any]], tenor: int) -> float | None:
    candidates: list[tuple[int, float]] = []
    for row in rows:
        dte = _as_int(row.get("dte"))
        iv = _as_float(row.get("iv_mark"))
        if dte is None or iv is None:
            continue
        candidates.append((abs(dte - tenor), iv))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def extract_iv_benchmarks(
    deribit_atm_rows: list[dict[str, Any]],
    ibit_atm_rows: list[dict[str, Any]],
    *,
    tenors: tuple[int, ...] = (1, 7, 30),
) -> dict[str, float | None]:
    benchmarks: dict[str, float | None] = {}
    for tenor in tenors:
        deribit_iv = _nearest_iv(deribit_atm_rows, tenor)
        ibit_iv = _nearest_iv(ibit_atm_rows, tenor)
        benchmarks[f"deribit_{tenor}d_iv"] = deribit_iv
        benchmarks[f"ibit_{tenor}d_iv"] = ibit_iv
        benchmarks[f"spread_{tenor}d_vol_pts"] = None if deribit_iv is None or ibit_iv is None else round((ibit_iv - deribit_iv) * 100, 2)
    return benchmarks


def build_trend_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(runs, key=lambda row: str(row.get("as_of_cst") or row.get("run_id") or ""))
    if not ordered:
        return {
            "run_count": 0,
            "latest_run_id": None,
            "btc_change": None,
            "quality_change": None,
            "quote_review_change": None,
            "latest_quality_grade": None,
            "btc_series": [],
            "quality_series": [],
            "spread_7d_change_vol_pts": None,
            "spread_7d_series": [],
            "latest_deribit_30d_iv": None,
            "latest_ibit_30d_iv": None,
        }

    latest = ordered[-1]
    btc_pair = _latest_two_non_null(ordered, "btc_spot")
    quality_pair = _latest_two_non_null(ordered, "quality_score")
    quote_pair = _latest_two_non_null(ordered, "quote_review_candidates")
    spread_7d_pair = _latest_two_non_null(ordered, "spread_7d_vol_pts")
    btc_series = [_as_float(row.get("btc_spot")) for row in ordered if _as_float(row.get("btc_spot")) is not None]
    quality_series = [_as_int(row.get("quality_score")) for row in ordered if _as_int(row.get("quality_score")) is not None]
    spread_7d_series = [_as_float(row.get("spread_7d_vol_pts")) for row in ordered if _as_float(row.get("spread_7d_vol_pts")) is not None]

    return {
        "run_count": len(ordered),
        "latest_run_id": latest.get("run_id"),
        "btc_change": None if btc_pair is None else btc_pair[1] - btc_pair[0],
        "quality_change": None if quality_pair is None else int(quality_pair[1] - quality_pair[0]),
        "quote_review_change": None if quote_pair is None else int(quote_pair[1] - quote_pair[0]),
        "latest_quality_grade": latest.get("quality_grade"),
        "btc_series": btc_series,
        "quality_series": quality_series,
        "spread_7d_change_vol_pts": None if spread_7d_pair is None else round(spread_7d_pair[1] - spread_7d_pair[0], 2),
        "spread_7d_series": spread_7d_series,
        "latest_deribit_30d_iv": _as_float(latest.get("deribit_30d_iv")),
        "latest_ibit_30d_iv": _as_float(latest.get("ibit_30d_iv")),
    }

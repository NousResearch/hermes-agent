from __future__ import annotations

from agent.compaction_stats import CompactionStats
from agent.conversation_compression import (
    _fmt_gross_frac,
    _format_compaction_announce,
    _format_granular_announce,
    _inturn_stats_render_eligible,
)


def _stats(value):
    if value is None or isinstance(value, CompactionStats):
        return value
    return CompactionStats(**value)


def run_case(case: dict):
    kind = case["kind"]
    if kind == "format":
        kwargs = dict(case["kwargs"])
        kwargs["stats"] = _stats(kwargs.get("stats"))
        value = _format_compaction_announce(**kwargs)
    elif kind == "granular":
        value = _format_granular_announce(
            case["head"],
            _stats(case["stats"]),
            case["model_part"],
            case["after_fallback"],
            case.get("window_from"),
            case.get("window_to"),
            basis=case.get("basis", "live"),
            wire_before=case.get("wire_before"),
            wire_after=case.get("wire_after"),
        )
    elif kind == "eligible":
        value = [
            _inturn_stats_render_eligible(status, pre, post)
            for status, pre, post in case["items"]
        ]
    elif kind == "gross":
        value = [_fmt_gross_frac(gross, pre) for gross, pre in case["items"]]
    else:
        raise AssertionError(f"unknown case kind: {kind}")
    return {
        "return": value,
        "messages": [],
        "db": [],
    }

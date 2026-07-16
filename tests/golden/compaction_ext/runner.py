from __future__ import annotations

from agent.compaction_stats import CompactionStats

# Prefer the extracted module (post-extraction tree) so the golden exercises
# agent.fork_ext.compaction_ext DIRECTLY, not the re-export chain; fall back to
# the legacy inline home for pre-extraction replay (Greptile: a re-export-only
# import would keep passing while compaction_ext drifts).
try:
    from agent.fork_ext.compaction_ext import (
        _abbrev_tokens,
        _compaction_window_label,
        _fmt_gross_frac,
        _format_compaction_announce,
        _format_granular_announce,
        _inturn_stats_render_eligible,
    )
except ModuleNotFoundError:
    from agent.conversation_compression import (
        _abbrev_tokens,
        _compaction_window_label,
        _fmt_gross_frac,
        _format_compaction_announce,
        _format_granular_announce,
        _inturn_stats_render_eligible,
    )


class _RaisingStats:
    def validate(self):
        raise RuntimeError("synthetic validate failure")


def _stats(value):
    if value is None or isinstance(value, CompactionStats):
        return value
    if isinstance(value, str) and value in {"__raise_validate", "__raise_validate__"}:
        return _RaisingStats()
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
    elif kind == "window":
        value = [_compaction_window_label(item) for item in case["items"]]
    elif kind == "abbrev":
        value = [_abbrev_tokens(item) for item in case["items"]]
    else:
        raise AssertionError(f"unknown case kind: {kind}")
    return {
        "return": value,
        "messages": [],
        "db": [],
    }

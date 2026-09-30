"""Speed budget tracker and profiler (05-final-plan.md §10)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

SPEED_BUDGET_TARGETS = {
    "first_signal_cloud_s": 2.0,
    "first_signal_local_s": 5.0,
    "spec_compute_s": 60.0,
    "approval_to_preview_s": 90.0,
    "tier1_time_to_green_s": 600.0,
    "tier2_time_to_green_cloud_s": 1500.0,
    "tier2_time_to_green_local_s": 3600.0,
    "continuation_cache_hit_pct": 80.0,
}


@dataclass
class SpeedProfileReport:
    first_signal_s: float
    spec_compute_s: float
    approval_to_preview_s: float
    time_to_green_s: float
    continuation_cache_hit_pct: float
    all_targets_met: bool
    budget_targets: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_speed_profile(
    *,
    first_signal_s: float,
    spec_compute_s: float,
    approval_to_preview_s: float,
    time_to_green_s: float,
    continuation_cache_hit_pct: float = 92.0,
    is_local: bool = False,
) -> SpeedProfileReport:
    sig_limit = SPEED_BUDGET_TARGETS["first_signal_local_s"] if is_local else SPEED_BUDGET_TARGETS["first_signal_cloud_s"]
    ttg_limit = SPEED_BUDGET_TARGETS["tier2_time_to_green_local_s"] if is_local else SPEED_BUDGET_TARGETS["tier2_time_to_green_cloud_s"]
    met = (
        first_signal_s <= sig_limit
        and spec_compute_s <= SPEED_BUDGET_TARGETS["spec_compute_s"]
        and approval_to_preview_s <= SPEED_BUDGET_TARGETS["approval_to_preview_s"]
        and time_to_green_s <= ttg_limit
        and continuation_cache_hit_pct >= SPEED_BUDGET_TARGETS["continuation_cache_hit_pct"]
    )
    return SpeedProfileReport(
        first_signal_s=round(first_signal_s, 3),
        spec_compute_s=round(spec_compute_s, 3),
        approval_to_preview_s=round(approval_to_preview_s, 3),
        time_to_green_s=round(time_to_green_s, 3),
        continuation_cache_hit_pct=round(continuation_cache_hit_pct, 1),
        all_targets_met=met,
        budget_targets=dict(SPEED_BUDGET_TARGETS),
    )

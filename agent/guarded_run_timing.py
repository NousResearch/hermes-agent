"""Per-step timing metadata for guarded desktop runs — RFC #112639.

Local timing metadata only: tool (API) durations, readiness-confirmation
durations and statuses, per-step verdicts. No screenshots, no typed text,
no secrets — records name the action, never its args.

Explicit gap: the executor path has no pre-capture to time. The SOM snapshot
capture happens before the run and its timing is not plumbed into the step
loop, so ``capture_duration_ms``/``capture_mode`` stay None unless the
caller calls :meth:`RunTimings.record_capture`. Nothing here fabricates
capture timings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Confirm statuses that actually reached the driver, plus the two ways a
# step ends without a confirmation: "no_check" (no sticky target or no
# readiness entry point) and "skipped" (step never executed).
CONFIRM_STATUSES = frozenset(
    {"satisfied", "unsatisfied", "unknown", "error", "no_check", "skipped"}
)


@dataclass
class StepTiming:
    """Timing metadata for one executed step of an admitted guarded run."""

    step_index: int
    action: str
    tool_duration_ms: Optional[float] = None
    confirm_duration_ms: Optional[float] = None
    confirm_status: str = "no_check"
    verdict: str = "continue"
    # ms from the collector's creation; non-decreasing across steps.
    started_offset_ms: float = 0.0


class RunTimings:
    """Collector for one batch's guarded-run step timings."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.steps: List[StepTiming] = []
        self.capture_duration_ms: Optional[float] = None
        self.capture_mode: Optional[str] = None

    def record_step(
        self,
        step_index: int,
        action: str,
        *,
        tool_duration_ms: Optional[float] = None,
        confirm_duration_ms: Optional[float] = None,
        confirm_status: str = "no_check",
        verdict: str = "continue",
    ) -> StepTiming:
        step = StepTiming(
            step_index=step_index,
            action=action,
            tool_duration_ms=tool_duration_ms,
            confirm_duration_ms=confirm_duration_ms,
            confirm_status=confirm_status,
            verdict=verdict,
            started_offset_ms=(time.perf_counter() - self._t0) * 1000.0,
        )
        self.steps.append(step)
        return step

    def record_capture(self, duration_ms: float, mode: str) -> None:
        # Plumb point for the SOM snapshot capture that precedes the run;
        # the executor path does not have it, so it stays None by default.
        self.capture_duration_ms = duration_ms
        self.capture_mode = mode

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def as_evidence(self) -> Dict[str, Any]:
        return {
            "step_count": len(self.steps),
            "capture_duration_ms": self.capture_duration_ms,
            "capture_mode": self.capture_mode,
            "steps": [
                {
                    "step_index": s.step_index,
                    "action": s.action,
                    "tool_duration_ms": s.tool_duration_ms,
                    "confirm_duration_ms": s.confirm_duration_ms,
                    "confirm_status": s.confirm_status,
                    "verdict": s.verdict,
                }
                for s in self.steps
            ],
        }

    def summary(self) -> str:
        # Compact evidence line: names actions only, never args or secrets.
        parts = []
        for s in self.steps:
            confirm = (
                f"{s.confirm_duration_ms:.0f}ms/{s.confirm_status}"
                if s.confirm_duration_ms is not None
                else s.confirm_status
            )
            tool = f"{s.tool_duration_ms:.0f}ms" if s.tool_duration_ms is not None else "?"
            parts.append(f"#{s.step_index} {s.action} tool={tool} confirm={confirm} {s.verdict}")
        return "run timings: " + "; ".join(parts) if parts else "run timings: none"

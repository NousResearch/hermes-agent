"""
Blast-radius control (P2-3 self-modification safety).

Gates Denji's autonomous profile edits (WS-7) with three safety layers:

1. **Per-cycle edit cap** — maximum profiles Denji can modify in one
   governance cycle.  Default 3.  Exceeded edits are deferred to the
   next cycle.

2. **Canary apply-observe-promote** — a profile edit is first applied
   to a single canary profile.  The system observes N eval runs (via
   P2-2 eval harness).  If pass-rate holds or improves, the edit is
   promoted to the remaining profiles.  If pass-rate regresses, the
   edit is reverted on the canary and discarded.

3. **Fleet-health tripwire** — aggregate success-rate and eval pass-rate
   are monitored.  If either drops below a configured threshold after
   a batch of edits, ALL further profile_editor writes are paused
   until operator review.

Usage
-----
    from hermes_cli.blast_radius import (
        EditGuard, CanaryState, FleetHealth, TripwireStatus,
    )

    guard = EditGuard(max_edits_per_cycle=3, eval_harness=harness)
    result = guard.try_edit(profile="remii-deep", patch=prompt_diff)
    if result.allowed:
        guard.apply_edit(result)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def load_eval_domains() -> dict:
    """Return the ``governance.eval_domains`` profile -> domain map.

    Single source for both the edit gate (refuse edits to unmapped
    profiles, so every autonomous edit is evaluable) and the observe
    driver (resolve the golden set). Empty on any config failure.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        gov = cfg.get("governance", {}) if isinstance(cfg, dict) else {}
        return dict(gov.get("eval_domains") or {})
    except Exception:
        return {}


def eval_domain_for(profile: str) -> "Optional[str]":
    """Domain a profile is evaluated under, or None if unmapped."""
    return load_eval_domains().get(profile)


DEFAULT_MAX_EDITS_PER_CYCLE = 3
DEFAULT_CANARY_OBSERVE_RUNS = 3
DEFAULT_TRIPWIRE_PASS_RATE = 0.60  # 60% aggregate pass rate
DEFAULT_TRIPWIRE_EVAL_PASS_RATE = 0.70  # 70% eval pass rate
TRIPWIRE_STATE_PATH = "~/.hermes/kensei/tripwire_state.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TripwireStatus(Enum):
    """Fleet-health tripwire state."""
    NORMAL = "normal"       # All clear, edits allowed
    WARNING = "warning"     # Approaching threshold
    TRIPPED = "tripped"     # Threshold breached, edits paused
    PAUSED = "paused"       # Manually paused by operator


class CanaryStage(Enum):
    """Stage of a canary edit rollout."""
    PROPOSED = "proposed"       # Edit submitted, not yet applied
    APPLIED = "applied"         # Applied to canary profile
    OBSERVING = "observing"     # Running eval runs
    PROMOTED = "promoted"       # Promoted to all profiles
    REVERTED = "reverted"       # Rolled back due to regression
    DISCARDED = "discarded"     # Operator rejected


@dataclass
class CanaryState:
    """State of a single canary edit."""
    edit_id: str
    profile: str
    stage: CanaryStage = CanaryStage.PROPOSED
    applied_at: str = ""
    observed_runs: int = 0
    required_runs: int = DEFAULT_CANARY_OBSERVE_RUNS
    baseline_pass_rate: float = 0.0
    current_pass_rate: float = 0.0
    patch_summary: str = ""
    error: Optional[str] = None
    promoted_to: list[str] = field(default_factory=list)
    # Git commit the edit landed as, so the observe loop can revert a
    # regressed canary via profile_editor.py --rollback.
    commit: str = ""


@dataclass
class FleetHealth:
    """Aggregate fleet health snapshot for tripwire monitoring."""
    timestamp: str
    total_tasks: int = 0
    success_rate: float = 1.0
    eval_pass_rate: float = 1.0
    active_profiles: int = 0
    profiles_with_regressions: list[str] = field(default_factory=list)
    tripwire: TripwireStatus = TripwireStatus.NORMAL


@dataclass
class EditResult:
    """Outcome of a try_edit() call."""
    allowed: bool
    reason: str = ""
    canary: Optional[CanaryState] = None
    defer_until: str = ""  # ISO timestamp, if deferred


# ---------------------------------------------------------------------------
# Edit Guard
# ---------------------------------------------------------------------------


@dataclass
class EditGuard:
    """Governs Denji's profile edits with caps and canary rollout.

    Usage pattern:
        guard = EditGuard(max_edits_per_cycle=3)

        for edit in denji_proposed_edits:
            result = guard.try_edit(profile=edit.profile, patch=edit.diff)
            if result.allowed:
                guard.apply_canary(result.canary)
            elif result.reason == "cycle_cap_exceeded":
                break  # defer remaining edits to next cycle
    """

    max_edits_per_cycle: int = DEFAULT_MAX_EDITS_PER_CYCLE
    canary_observe_runs: int = DEFAULT_CANARY_OBSERVE_RUNS
    tripwire_pass_rate: float = DEFAULT_TRIPWIRE_PASS_RATE
    tripwire_eval_pass_rate: float = DEFAULT_TRIPWIRE_EVAL_PASS_RATE

    # Internal state — persisted to tripwire_state.json
    _cycle_count: int = 0
    _cycle_start: str = ""
    _canaries: list[CanaryState] = field(default_factory=list)

    def __post_init__(self):
        self._load_state()

    # ------------------------------------------------------------------
    # Edit cap
    # ------------------------------------------------------------------

    def try_edit(self, profile: str, patch: str, patch_summary: str = "") -> EditResult:
        """Check whether an edit is allowed this cycle.

        Returns EditResult.allowed=False when:
        - The cycle edit cap is exhausted.
        - The fleet-health tripwire is tripped.
        """
        # Check tripwire first — if tripped, no edits at all
        health = self._read_fleet_health()
        if health.tripwire in (TripwireStatus.TRIPPED, TripwireStatus.PAUSED):
            return EditResult(
                allowed=False,
                reason=f"tripwire_{health.tripwire.value}",
            )

        # Check cycle cap
        self._maybe_rotate_cycle()
        if self._cycle_count >= self.max_edits_per_cycle:
            return EditResult(
                allowed=False,
                reason="cycle_cap_exceeded",
                defer_until=self._next_cycle_time(),
            )

        # Allowed — create canary entry
        edit_id = f"edit-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{profile}"
        canary = CanaryState(
            edit_id=edit_id,
            profile=profile,
            stage=CanaryStage.PROPOSED,
            required_runs=self.canary_observe_runs,
            patch_summary=patch_summary[:200],
        )
        return EditResult(allowed=True, canary=canary)

    # ------------------------------------------------------------------
    # Canary apply-observe-promote
    # ------------------------------------------------------------------

    def apply_canary(self, canary: CanaryState) -> None:
        """Mark a canary edit as applied to its target profile."""
        canary.stage = CanaryStage.APPLIED
        canary.applied_at = datetime.now(timezone.utc).isoformat()
        self._canaries.append(canary)
        self._cycle_count += 1
        self._save_state()

    def active_canaries(self) -> list[CanaryState]:
        """Canaries still awaiting an observe decision (applied/observing)."""
        return [
            c for c in self._canaries
            if c.stage in (CanaryStage.APPLIED, CanaryStage.OBSERVING)
        ]

    def attach_commit(self, edit_id: str, commit: str) -> bool:
        """Record the git commit an applied canary edit landed as.

        Called after profile_editor.py returns the commit, so the observe
        loop can revert this exact commit if the canary later regresses.
        """
        canary = self._find_canary(edit_id)
        if canary is None:
            return False
        canary.commit = commit
        self._save_state()
        return True

    def observe_canary(
        self,
        edit_id: str,
        eval_result,
        *,
        promote: bool = False,
    ) -> CanaryState | None:
        """Record an observation run for a canary edit.

        Args:
            edit_id: The canary edit to observe.
            eval_result: An EvalRun from the harness (P2-2).
            promote: If True, skip remaining observation runs and
                     promote immediately (operator override).

        Returns the updated CanaryState, or None if not found.
        """
        canary = self._find_canary(edit_id)
        if canary is None:
            return None

        if canary.stage == CanaryStage.APPLIED:
            canary.stage = CanaryStage.OBSERVING

        canary.observed_runs += 1
        canary.current_pass_rate = getattr(eval_result, 'pass_rate', 0.0)

        if canary.observed_runs == 1:
            canary.baseline_pass_rate = canary.current_pass_rate

        if promote or canary.observed_runs >= canary.required_runs:
            self._decide_canary(canary)

        self._save_state()
        return canary

    def _decide_canary(self, canary: CanaryState) -> None:
        """Decide whether to promote or revert a canary edit.

        Promotion criteria:
        - Current pass-rate >= baseline pass-rate (no regression)
        - Or operator forced promotion
        """
        if canary.current_pass_rate >= canary.baseline_pass_rate:
            canary.stage = CanaryStage.PROMOTED
            canary.promoted_to = [canary.profile]  # canary itself
        else:
            canary.stage = CanaryStage.REVERTED
            canary.error = (
                f"Regression detected: pass-rate dropped from "
                f"{canary.baseline_pass_rate:.1%} to {canary.current_pass_rate:.1%}"
            )

    def promote_to_profiles(self, edit_id: str, target_profiles: list[str]) -> bool:
        """Promote a canary edit to additional profiles after it passed."""
        canary = self._find_canary(edit_id)
        if canary is None or canary.stage != CanaryStage.PROMOTED:
            return False
        canary.promoted_to.extend(target_profiles)
        self._save_state()
        return True

    def revert_canary(self, edit_id: str, reason: str = "") -> CanaryState | None:
        """Force-revert a canary edit."""
        canary = self._find_canary(edit_id)
        if canary is None:
            return None
        canary.stage = CanaryStage.REVERTED
        canary.error = reason or "Operator-forced revert"
        self._save_state()
        return canary

    # ------------------------------------------------------------------
    # Fleet-health tripwire
    # ------------------------------------------------------------------

    def check_fleet_health(
        self,
        *,
        success_rate: Optional[float] = None,
        eval_pass_rate: Optional[float] = None,
        active_profiles: int = 0,
        profiles_with_regressions: Optional[list[str]] = None,
    ) -> FleetHealth:
        """Check fleet health and update tripwire status.

        Call this at the start of each governance cycle (Denji tick).
        If the tripwire trips, all profile_editor writes are paused
        until operator review.
        """
        health = FleetHealth(
            timestamp=datetime.now(timezone.utc).isoformat(),
            success_rate=success_rate or 1.0,
            eval_pass_rate=eval_pass_rate or 1.0,
            active_profiles=active_profiles,
            profiles_with_regressions=profiles_with_regressions or [],
        )

        # WARNING: approaching threshold (within 10%)
        if (
            health.success_rate < self.tripwire_pass_rate * 1.10
            or health.eval_pass_rate < self.tripwire_eval_pass_rate * 1.10
        ):
            health.tripwire = TripwireStatus.WARNING

        # TRIPPED: below threshold
        if (
            health.success_rate < self.tripwire_pass_rate
            or health.eval_pass_rate < self.tripwire_eval_pass_rate
        ):
            health.tripwire = TripwireStatus.TRIPPED

        # Persist
        self._write_fleet_health(health)
        return health

    def pause_edits(self) -> None:
        """Manually pause all profile edits (operator action)."""
        health = self._read_fleet_health()
        health.tripwire = TripwireStatus.PAUSED
        self._write_fleet_health(health)

    def resume_edits(self) -> None:
        """Manually resume profile edits (operator action)."""
        health = self._read_fleet_health()
        health.tripwire = TripwireStatus.NORMAL
        self._write_fleet_health(health)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _state_path(self) -> str:
        return os.path.expanduser(TRIPWIRE_STATE_PATH)

    def _load_state(self) -> None:
        path = self._state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._cycle_count = data.get("cycle_count", 0)
            self._cycle_start = data.get("cycle_start", "")
            self._canaries = [
                CanaryState(
                    edit_id=c["edit_id"],
                    profile=c["profile"],
                    stage=CanaryStage(c.get("stage", "proposed")),
                    applied_at=c.get("applied_at", ""),
                    observed_runs=c.get("observed_runs", 0),
                    required_runs=c.get("required_runs", self.canary_observe_runs),
                    baseline_pass_rate=c.get("baseline_pass_rate", 0.0),
                    current_pass_rate=c.get("current_pass_rate", 0.0),
                    patch_summary=c.get("patch_summary", ""),
                    error=c.get("error"),
                    promoted_to=c.get("promoted_to", []),
                    commit=c.get("commit", ""),
                )
                for c in data.get("canaries", [])
            ]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    def _save_state(self) -> None:
        path = self._state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "cycle_count": self._cycle_count,
            "cycle_start": self._cycle_start,
            "max_edits_per_cycle": self.max_edits_per_cycle,
            "canaries": [
                {
                    "edit_id": c.edit_id,
                    "profile": c.profile,
                    "stage": c.stage.value,
                    "applied_at": c.applied_at,
                    "observed_runs": c.observed_runs,
                    "required_runs": c.required_runs,
                    "baseline_pass_rate": c.baseline_pass_rate,
                    "current_pass_rate": c.current_pass_rate,
                    "patch_summary": c.patch_summary,
                    "error": c.error,
                    "promoted_to": c.promoted_to,
                    "commit": c.commit,
                }
                for c in self._canaries
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _write_fleet_health(self, health: FleetHealth) -> None:
        """Write fleet health to a separate file for the tripwire."""
        path = os.path.expanduser("~/.hermes/kensei/fleet_health.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "timestamp": health.timestamp,
                "total_tasks": health.total_tasks,
                "success_rate": health.success_rate,
                "eval_pass_rate": health.eval_pass_rate,
                "active_profiles": health.active_profiles,
                "profiles_with_regressions": health.profiles_with_regressions,
                "tripwire": health.tripwire.value,
            }, f, indent=2)

    def _read_fleet_health(self) -> FleetHealth:
        path = os.path.expanduser("~/.hermes/kensei/fleet_health.json")
        if not os.path.exists(path):
            return FleetHealth(
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        try:
            with open(path) as f:
                data = json.load(f)
            return FleetHealth(
                timestamp=data.get("timestamp", ""),
                total_tasks=data.get("total_tasks", 0),
                success_rate=data.get("success_rate", 1.0),
                eval_pass_rate=data.get("eval_pass_rate", 1.0),
                active_profiles=data.get("active_profiles", 0),
                profiles_with_regressions=data.get("profiles_with_regressions", []),
                tripwire=TripwireStatus(data.get("tripwire", "normal")),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return FleetHealth(
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    def _find_canary(self, edit_id: str) -> CanaryState | None:
        for c in self._canaries:
            if c.edit_id == edit_id:
                return c
        return None

    def _maybe_rotate_cycle(self) -> None:
        """Reset the edit counter if we're in a new calendar day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._cycle_start != today:
            self._cycle_count = 0
            self._cycle_start = today

    def _next_cycle_time(self) -> str:
        """Return the ISO timestamp when the next cycle starts."""
        # Next UTC midnight
        now = datetime.now(timezone.utc)
        next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timedelta
        next_midnight += timedelta(days=1)
        return next_midnight.isoformat()

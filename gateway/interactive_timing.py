"""Content-free timing contract for interactive gateway turns.

The record uses one process-local monotonic clock for durations and a UTC wall
clock only for the server-receipt anchor. It intentionally excludes message,
user, chat, and session identifiers so the default INFO log is safe to retain.
Channel clients rarely expose render acknowledgements, therefore visible
milestones are explicitly labelled as server-side proxies.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "hermes.interactive_turn.v1"
_RUNTIME_FIELDS = ("platform", "model", "provider", "reasoning", "harness_revision", "lane")


def _safe_runtime_value(value: Any, *, limit: int = 256) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    text = str(value).strip().replace("\n", " ").replace("\r", " ")
    return text[:limit]


def _default_harness_revision() -> str:
    configured = _safe_runtime_value(os.getenv("HERMES_HARNESS_REVISION"), limit=128)
    if configured:
        return configured
    try:
        from hermes_cli import __version__

        return _safe_runtime_value(__version__, limit=128) or "unknown"
    except Exception:
        return "unknown"


@dataclass
class InteractiveTurnTiming:
    """Idempotent process-local milestone recorder for one admitted message."""

    platform: str = "unknown"
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _wall_ns: Callable[[], int] = field(default=time.time_ns, repr=False, compare=False)
    _monotonic_ns: Callable[[], int] = field(default=time.monotonic_ns, repr=False, compare=False)
    _received_wall_ns: int = field(init=False, repr=False)
    _received_monotonic_ns: int = field(init=False, repr=False)
    _milestones_ns: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _milestone_basis: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _runtime: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _outcome: str = field(default="unknown", init=False, repr=False)
    _emitted: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._received_wall_ns = self._wall_ns()
        self._received_monotonic_ns = self._monotonic_ns()
        self.update_runtime(platform=self.platform, harness_revision=_default_harness_revision())

    def update_runtime(self, **values: Any) -> None:
        """Merge allowlisted, bounded runtime metadata; empty values do not erase known values."""
        with self._lock:
            for key in _RUNTIME_FIELDS:
                if key not in values:
                    continue
                value = _safe_runtime_value(values[key], limit=128 if key != "model" else 256)
                if value:
                    self._runtime[key] = value

    def mark(self, milestone: str, *, basis: str = "server_proxy") -> bool:
        """Record the first occurrence of an allowlisted milestone."""
        if milestone not in {"accepted", "working_state", "first_meaningful_response", "completed"}:
            raise ValueError(f"Unsupported interactive milestone: {milestone}")
        with self._lock:
            if milestone in self._milestones_ns:
                return False
            self._milestones_ns[milestone] = max(self._received_monotonic_ns, self._monotonic_ns())
            self._milestone_basis[milestone] = _safe_runtime_value(basis, limit=64) or "server_proxy"
            return True

    def fork_deferred_turn(self) -> "InteractiveTurnTiming":
        """Create an independent queued turn retaining this event's receipt anchor."""
        with self._lock:
            received_wall_ns = self._received_wall_ns
            received_monotonic_ns = self._received_monotonic_ns
            inherited_runtime = {
                key: self._runtime[key]
                for key in ("harness_revision", "lane")
                if key in self._runtime
            }
        # Seed construction with fixed anchor clocks, then restore the live monotonic clock for
        # future milestones. This avoids consuming or advancing injected clocks in tests.
        child = type(self)(
            platform=self.platform,
            _wall_ns=lambda: received_wall_ns,
            _monotonic_ns=lambda: received_monotonic_ns,
        )
        child._wall_ns = self._wall_ns
        child._monotonic_ns = self._monotonic_ns
        child.update_runtime(**inherited_runtime)
        return child

    def complete(self, outcome: str) -> None:
        with self._lock:
            self._outcome = _safe_runtime_value(outcome, limit=32) or "unknown"
        self.mark("completed", basis="server_processing_complete")

    def to_record(self) -> dict[str, Any]:
        """Return the stable JSON-compatible telemetry record."""
        with self._lock:
            milestones = dict(self._milestones_ns)
            basis = dict(self._milestone_basis)
            runtime = {key: self._runtime.get(key, "unknown") for key in _RUNTIME_FIELDS}
            outcome = self._outcome
        offsets = {
            "server_receipt": 0,
            **{
                name: max(0, (stamp - self._received_monotonic_ns) // 1_000_000)
                for name, stamp in milestones.items()
            },
        }
        return {
            "schema_version": _SCHEMA_VERSION,
            "turn_id": self.turn_id,
            "received_at": datetime.fromtimestamp(
                self._received_wall_ns / 1_000_000_000, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "milestones_ms": offsets,
            "milestone_basis": basis,
            "runtime": runtime,
            "outcome": outcome,
        }

    def emit_once(self) -> dict[str, Any] | None:
        """Log one compact machine-readable record, returning it for local consumers/tests."""
        with self._lock:
            if self._emitted:
                return None
            self._emitted = True
        record = self.to_record()
        logger.info("interactive_turn_timing %s", json.dumps(record, sort_keys=True, separators=(",", ":")))
        return record


def ensure_interactive_timing(event: Any, *, platform: Any = None) -> InteractiveTurnTiming:
    """Return the event's recorder, creating it at the generic adapter receipt boundary."""
    timing = getattr(event, "_interactive_timing", None)
    if not isinstance(timing, InteractiveTurnTiming):
        platform_name = getattr(platform, "value", platform) or "unknown"
        timing = InteractiveTurnTiming(platform=str(platform_name))
        event._interactive_timing = timing
    elif platform is not None:
        timing.update_runtime(platform=getattr(platform, "value", platform))
    return timing

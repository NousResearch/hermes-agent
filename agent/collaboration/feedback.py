"""Feedback loop — measures whether adaptations improve collaboration.

Tracks correction rate over time, detects drift, and triggers recalibration.
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent.collaboration.graph_client import GraphClient, GraphMemory, get_client

logger = logging.getLogger(__name__)


@dataclass
class SessionCorrectionRate:
    """Correction rate for a single session."""
    session_id: str
    rate: float             # corrections / total_turns
    corrections: int
    turns: int
    timestamp: str = ""


@dataclass
class DriftEvent:
    """A detected shift in user behavior."""
    type: str               # "correction_rate_increase", "new_correction_type"
    severity: str           # "low", "medium", "high"
    old_rate: float = 0.0
    new_rate: float = 0.0
    description: str = ""
    timestamp: str = ""


class CorrectionRateTracker:
    """Track correction rate over a rolling window of sessions.

    Data is stored in the fleet graph. Used to detect drift and
    validate adaptation effectiveness.
    """

    def __init__(self, window_sessions: int = 10, client: Optional[GraphClient] = None):
        self._window = window_sessions
        self._client = client or get_client()

    def load_history(self) -> List[SessionCorrectionRate]:
        """Load session correction rates from graph."""
        sessions = self._client.search_memories(
            query="category:session tag:correction_rate",
            limit=self._window * 2,
        )
        history = []
        for s in sessions:
            content = s.get("content", "")
            tags = s.get("tags", [])
            rate_tag = next((t for t in tags if t.startswith("rate:")), ":0")
            turns_tag = next((t for t in tags if t.startswith("turns:")), ":0")
            cors_tag = next((t for t in tags if t.startswith("corrections:")), ":0")
            # Parse timestamp from ts tag, fall back to graph timestamp
            ts_tag = next((t for t in tags if t.startswith("ts:")), None)
            if ts_tag:
                raw_ts = ts_tag.split(":", 1)[1] if ":" in ts_tag else ""
                try:
                    timestamp = float(raw_ts)
                except (ValueError, TypeError):
                    timestamp = 0.0
            else:
                timestamp = s.get("timestamp", 0.0)
            history.append(SessionCorrectionRate(
                session_id=s.get("sessionId", ""),
                rate=float(rate_tag.split(":")[1]),
                corrections=int(cors_tag.split(":")[1]),
                turns=int(turns_tag.split(":")[1]),
                timestamp=str(timestamp),
            ))
        # Sort by timestamp descending (newest first)
        history.sort(key=lambda x: x.timestamp, reverse=True)
        return history[:self._window]

    def record_session(self, session_id: str, corrections: int, total_turns: int) -> None:
        """Record a session's correction rate to the graph."""
        if not self._client.is_available():
            return

        rate = corrections / max(total_turns, 1)
        import time
        now = time.time()
        memory = GraphMemory(
            content=f"session:{session_id} corrections={corrections} turns={total_turns} rate={rate:.3f} ts={now}",
            category="session",
            tags=["correction_rate", f"rate:{rate:.3f}", f"corrections:{corrections}", f"turns:{total_turns}", f"ts:{now}"],
            memory_type="short_term",
            confidence=0.9,
            session_id=session_id,
            ttl=86400 * 90,  # 90 days for trend analysis
        )
        self._client.create_memory(memory)

    def trend(self) -> float:
        """Return the slope of correction rate over recent sessions.

        Negative = improving (fewer corrections).
        Positive = regressing (more corrections).

        ``load_history`` returns newest-first (sorted by timestamp desc).
        Linear regression over chronological order: reverse to oldest-first.
        """
        history = self.load_history()
        if len(history) < 3:
            return 0.0

        # ``load_history`` returns newest-first (sorted by timestamp desc).
        # Reverse to chronological order for regression: index 0 = oldest,
        # so negative slope means corrections decreasing → improving.
        rates = [h.rate for h in reversed(history)]
        n = len(rates)
        x_avg = (n - 1) / 2.0
        y_avg = sum(rates) / n

        num = sum((i - x_avg) * (r - y_avg) for i, r in enumerate(rates))
        den = sum((i - x_avg) ** 2 for i in range(n))

        if den == 0:
            return 0.0
        return num / den

    def avg_rate(self) -> float:
        """Return average correction rate across the window."""
        history = self.load_history()
        if not history:
            return 0.0
        return statistics.mean(h.rate for h in history)


class DriftDetector:
    """Detect significant changes in user behavior.

    Compares recent observations against historical patterns.
    """

    def __init__(self, tracker: CorrectionRateTracker, client: Optional[GraphClient] = None):
        self._tracker = tracker
        self._client = client or get_client()

    def check(self) -> Optional[DriftEvent]:
        """Check for drift. Returns the most severe drift event if found."""
        # 1. Correction rate increase
        trend = self._tracker.trend()
        avg_rate = self._tracker.avg_rate()

        if trend > 0.05 and avg_rate > 0.15:
            return DriftEvent(
                type="correction_rate_increase",
                severity="high",
                old_rate=avg_rate - trend * 3,
                new_rate=avg_rate,
                description=f"Correction rate up: {avg_rate:.3f} (trend: {trend:+.3f}/session)",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        if trend > 0.02:
            return DriftEvent(
                type="correction_rate_increase",
                severity="medium",
                new_rate=avg_rate,
                description=f"Mild correction rate increase: trend {trend:+.3f}/session",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        # 2. New correction types (check against known preferences)
        if self._check_new_correction_types():
            return DriftEvent(
                type="new_correction_type",
                severity="medium",
                description="New correction type detected — preferences may need update",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        return None

    def _check_new_correction_types(self) -> bool:
        """Check if new correction types have appeared recently."""
        try:
            recent = self._client.search_memories(
                "tag:correction category:observation", limit=50
            )
            if not recent:
                return False

            known_types = {"redirect", "terse_fix", "scope_shrink",
                           "format_preference", "rejection", "language_switch"}
            for c in recent:
                tags = c.get("tags", [])
                for tag in tags:
                    if tag.startswith("correction:"):
                        ctype = tag.split(":", 1)[1]
                        if ctype not in known_types:
                            return True
            return False
        except Exception:
            return False

    def should_recalibrate(self) -> bool:
        """Return True if preferences need recalibration."""
        event = self.check()
        if event is None:
            return False
        return event.severity in ("medium", "high")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_preference_impact(
    sessions_before: List[SessionCorrectionRate],
    sessions_after: List[SessionCorrectionRate],
) -> float:
    """Validate a preference's impact by comparing correction rates.

    Returns impact score: positive = improved, negative = worse, 0 = no data.
    """
    if not sessions_before or not sessions_after:
        return 0.0

    before_rate = statistics.mean(s.rate for s in sessions_before) if sessions_before else 0
    after_rate = statistics.mean(s.rate for s in sessions_after) if sessions_after else 0

    if before_rate == 0:
        return 0.0

    impact = (before_rate - after_rate) / before_rate
    return max(-1.0, min(1.0, impact))

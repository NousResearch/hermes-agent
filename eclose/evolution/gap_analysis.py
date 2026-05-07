from eclose.events.events import GapType, Severity, GapEvent, EventType


class GapAnalysisEngine:
    """Engine that identifies capability gaps from perception events."""

    def __init__(self):
        self.gaps = []

    def identify_gaps(
        self, needs: list[str], capabilities: list[str]
    ) -> list[GapEvent]:
        """Identify gaps between needs and capabilities."""
        gaps = []
        for need in needs:
            if not any(self._matches(need, cap) for cap in capabilities):
                gaps.append(
                    GapEvent(
                        type=EventType.GAP,
                        gap_type=GapType.SKILL,
                        severity=Severity.MAJOR,
                        description=f"Missing capability: {need}",
                        evidence=[{"need": need, "available": capabilities}],
                    )
                )
        return gaps

    def _matches(self, need: str, capability: str) -> bool:
        """Check if a capability matches a need."""
        need_lower = need.lower()
        cap_lower = capability.lower()
        return (
            need_lower in cap_lower
            or cap_lower in need_lower
            or any(word in cap_lower for word in need_lower.split())
        )

    def rank_gaps(self, gaps: list[GapEvent]) -> list[GapEvent]:
        """Rank gaps by priority."""
        severity_weights = {
            Severity.CRITICAL: 3,
            Severity.MAJOR: 2,
            Severity.MINOR: 1,
        }
        return sorted(
            gaps, key=lambda g: severity_weights.get(g.severity, 0), reverse=True
        )
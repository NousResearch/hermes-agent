import uuid
from dataclasses import dataclass, field
from eclose.events.events import GapEvent, EventType


@dataclass
class EvolutionProposal:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    background: dict = field(default_factory=dict)
    solution: dict = field(default_factory=dict)
    expected_impact: dict = field(default_factory=dict)
    risks: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    gap: GapEvent = None
    requires_approval: bool = True


class ProposalGenerator:
    """Generates structured evolution proposals from gaps."""

    def generate_proposal(self, gap: GapEvent) -> EvolutionProposal:
        """Generate a proposal for a given gap."""
        proposal = EvolutionProposal(
            gap=gap,
            title=self._generate_title(gap),
            background=self._generate_background(gap),
            solution=self._generate_solution(gap),
            expected_impact=self._generate_impact(gap),
            risks=self._generate_risks(gap),
        )
        # Critical gaps require approval
        proposal.requires_approval = gap.severity.value in ["critical", "major"]
        return proposal

    def _generate_title(self, gap: GapEvent) -> str:
        return f"Evolve: {gap.description}"

    def _generate_background(self, gap: GapEvent) -> dict:
        return {
            "gap": gap.description,
            "evidence": gap.evidence,
            "impact": f"Without resolution, {gap.description}",
        }

    def _generate_solution(self, gap: GapEvent) -> dict:
        return {
            "approach": "learn" if gap.gap_type.value == "skill" else "integrate",
            "steps": [
                "Research available solutions",
                "Evaluate options",
                "Implement chosen solution",
                "Test and verify",
            ],
            "timeline": "To be determined",
        }

    def _generate_impact(self, gap: GapEvent) -> dict:
        return {
            "before": "Cannot perform task requiring " + gap.description,
            "after": f"Capable of {gap.description}",
        }

    def _generate_risks(self, gap: GapEvent) -> list[dict]:
        return [
            {
                "risk": "Solution may not meet requirements",
                "likelihood": "medium",
                "mitigation": "Test thoroughly before full adoption",
            }
        ]

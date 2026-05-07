import pytest
from eclose.evolution.proposal import ProposalGenerator, EvolutionProposal
from eclose.events.events import GapType, Severity, GapEvent, EventType

def test_proposal_generator_initialization():
    gen = ProposalGenerator()
    assert gen is not None

def test_generate_proposal():
    gen = ProposalGenerator()
    gap = GapEvent(
        type=EventType.GAP,
        gap_type=GapType.SKILL,
        severity=Severity.MAJOR,
        description="Missing video processing capability",
    )
    proposal = gen.generate_proposal(gap)
    assert isinstance(proposal, EvolutionProposal)
    assert proposal.gap == gap
    assert proposal.title is not None
    assert proposal.solution is not None

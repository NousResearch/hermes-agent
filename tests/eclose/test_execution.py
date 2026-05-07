import pytest
from eclose.evolution.execution import ExecutionEngine, ExecutionResult

def test_execution_engine_initialization():
    engine = ExecutionEngine()
    assert engine is not None

def test_execute_proposal():
    from eclose.evolution.proposal import EvolutionProposal
    from eclose.events.events import EventType, GapType, Severity

    engine = ExecutionEngine()
    proposal = EvolutionProposal(
        id="test-1",
        title="Test",
        solution={
            "steps": ["echo 'test'"],
            "approach": "learn"
        }
    )
    result = engine.execute(proposal)
    assert isinstance(result, ExecutionResult)
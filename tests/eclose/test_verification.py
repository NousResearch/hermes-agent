import pytest
from eclose.evolution.verification import VerificationLayer, VerificationResult

def test_verification_layer_initialization():
    layer = VerificationLayer()
    assert layer is not None

def test_verify_execution():
    from eclose.evolution.execution import ExecutionResult

    layer = VerificationLayer()
    execution_result = ExecutionResult(
        proposal_id="test-1",
        status="success",
        results={"steps_completed": ["step1", "step2"], "steps_failed": []},
        verification={"passed": True},
    )
    result = layer.verify(execution_result)
    assert isinstance(result, VerificationResult)
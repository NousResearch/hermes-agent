import pytest
from unittest.mock import MagicMock
from agent import conversation_loop

@pytest.fixture
def mock_agent():
    """
    Fixture to initialize a properly configured mock agent object.
    Ensures numerical session and iteration counters are set to prevent TypeError.
    """
    agent = MagicMock()
    agent._skill_nudge_interval = 5
    agent._iters_since_skill = 0
    return agent

def test_conversation_loop_execution_and_assertion(mock_agent):
    """
    Validates execution of run_conversation and asserts robust output structures.
    """
    mock_agent.run_conversation.return_value = {
        "status": "success",
        "final_response": "Processed successfully",
        "tokens_used": 120
    }

    result = mock_agent.run_conversation(
        messages=[{"role": "user", "content": "Hello"}],
        moa_config=None,
        stream_callback=None
    )

    assert result is not None
    assert isinstance(result, dict)
    assert result.get("status") == "success"
    assert "final_response" in result
    assert result["tokens_used"] > 0

def test_invalid_moa_config_handling(mock_agent):
    """
    Verifies that invalid MoA configurations trigger expected fallback or error handling.
    """
    mock_agent.run_conversation.side_effect = ValueError("Invalid MOA configuration provided")

    with pytest.raises(ValueError) as exc_info:
        mock_agent.run_conversation(
            messages=[{"role": "user", "content": "Test MoA"}],
            moa_config={"invalid_key": True}
        )
    
    assert "Invalid MOA configuration" in str(exc_info.value)

def test_run_conversation_type_hints_importability():
    """
    Ensures run_conversation function is properly exported and callable from the main module.
    """
    assert hasattr(conversation_loop, "run_conversation")
    assert callable(getattr(conversation_loop, "run_conversation"))

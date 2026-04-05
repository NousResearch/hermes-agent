import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from run_agent import AIAgent

@pytest.fixture
def synthetic_agent():
    """AIAgent configured with the synthetic provider."""
    with patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI"):
        agent = AIAgent(
            provider="synthetic",
            model="hf:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            api_key="sk-synthetic-test-key-12345",
            quiet_mode=True
        )
        agent.client = MagicMock()
        return agent

def test_get_synthetic_quota_success(synthetic_agent):
    """Test successful quota retrieval from Synthetic API."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "subscription": {
            "limit": 400,
            "requests": 0,
            "renewsAt": "2026-04-06T03:58:12.790Z"
        }
    }
    
    with patch("httpx.get", return_value=mock_resp):
        quota = synthetic_agent._get_synthetic_quota()
        
    assert quota == {
        "limit": 400,
        "requests": 0,
        "renews_at": "2026-04-06T03:58:12.790Z"
    }

def test_get_synthetic_quota_failure(synthetic_agent):
    """Test handling of failed quota retrieval."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    
    with patch("httpx.get", return_value=mock_resp):
        quota = synthetic_agent._get_synthetic_quota()
        
    assert quota is None

def test_synthetic_429_backoff_calculation(synthetic_agent):
    """Test that 429 errors trigger Synthetic-specific backoff calculation."""
    # Mock a 429 error
    class MockAPIError(Exception):
        def __init__(self):
            self.status_code = 429
            self.body = {"error": {"message": "rate limit reached"}}
    
    # Mock quota renewal in 10 seconds
    renews_at = (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
    mock_quota = {
        "limit": 100,
        "requests": 100,
        "renews_at": renews_at
    }
    
    # Mock time progression to avoid infinite loops when time.sleep is mocked
    class MockTime:
        def __init__(self):
            self.curr = 1000.0
        def time(self):
            return self.curr
        def sleep(self, seconds):
            self.curr += seconds

    mt = MockTime()
    
    with patch.object(synthetic_agent, "_get_synthetic_quota", return_value=mock_quota), \
         patch("time.time", side_effect=mt.time), \
         patch("time.sleep", side_effect=mt.sleep) as mock_sleep, \
         patch("run_agent.IterationBudget.consume", return_value=True):
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success"
        mock_response.choices[0].finish_reason = "stop"
        
        # First call fails with 429, second succeeds
        synthetic_agent.client.chat.completions.create.side_effect = [MockAPIError(), mock_response]
        
        synthetic_agent.run_conversation("test")
        
        # Renewal was 10s out, so wait_time should be 12s (10 + 2 buffer)
        # Verify that we slept a total of approximately 12 seconds
        total_slept = sum(args[0] for args, kwargs in mock_sleep.call_args_list)
        assert 11 <= total_slept <= 13

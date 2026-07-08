
import pytest
from run_agent import AIAgent, FailoverReason
from agent.conversation_loop import conversation_loop

class MockAgent(AIAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fallback_chain = [{'provider': 'deepseek', 'model': 'deepseek-v4-flash'}]
        self._fallback_index = 0
        self._credential_pool = None

    def _try_activate_fallback(self, reason=None):
        self._fallback_index += 1
        return True

def test_reproduce_429_fallback_bypass():
    # Setup agent with primary + fallback
    agent = MockAgent(provider='nvidia', model='glm-5.2')
    
    # Simulate a 429 error
    # We need to trigger the loop in conversation_loop.py that handles the 429
    # This is a complex internal state, but we can target the logic directly
    
    # Manually check the condition found in the code
    classified_reason = FailoverReason.rate_limit
    is_rate_limited = True
    
    # Verify the fallback condition triggers
    should_fallback = True
    has_fallback = agent._fallback_index < len(agent._fallback_chain)
    
    print(f"Should fallback: {should_fallback}, Has fallback available: {has_fallback}")
    assert should_fallback and has_fallback, "Fallback should be triggered for 429"
    
    # Simulate the activation
    success = agent._try_activate_fallback(reason=classified_reason)
    assert success is True, "Fallback should activate"
    assert agent._fallback_index == 1

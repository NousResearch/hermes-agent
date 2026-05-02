
def old_get_usage(agent) -> dict:
    """The original logic from the repo before the fix."""
    g = lambda k, fb=None: getattr(agent, k, 0) or (getattr(agent, fb, 0) if fb else 0)
    usage = {
        "model": getattr(agent, "model", "") or "",
        "input": g("session_input_tokens", "session_prompt_tokens"),
        "output": g("session_output_tokens", "session_completion_tokens"),
        "total": g("session_total_tokens"),
        "calls": g("session_api_calls"),
    }
    return usage

def new_get_usage(agent) -> dict:
    """The new, hardened logic I implemented."""
    last_usage = getattr(getattr(agent, "last_response", None), "usage", {}) or {}
    g = lambda k, fb=None: (
        getattr(agent, k, 0) or 
        (getattr(agent, fb, 0) if fb else 0) or 
        last_usage.get(k, 0)
    )
    usage = {
        "model": getattr(agent, "model", "") or "",
        "input": g("session_input_tokens", "session_prompt_tokens") or last_usage.get("prompt_tokens", 0),
        "output": g("session_output_tokens", "session_completion_tokens") or last_usage.get("completion_tokens", 0),
        "total": g("session_total_tokens") or (
            last_usage.get("total_tokens", 0) or 
            (g("session_input_tokens", "session_prompt_tokens") + g("session_output_tokens", "session_completion_tokens"))
        ),
        "calls": g("session_api_calls") or last_usage.get("api_calls", 0),
    }
    return usage

class MockResponse:
    def __init__(self, usage):
        self.usage = usage

class MockAgent:
    def __init__(self, usage_in_last_resp=None, attrs=None):
        if usage_in_last_resp:
            self.last_response = MockResponse(usage_in_last_resp)
        else:
            self.last_response = None
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

def run_comparison():
    print("--- ⚖️ USAGE LOGIC COMPARISON ---")
    
    # Scenario: DATA VOID (Data exists in API response, but NOT in agent attributes)
    mock_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    agent = MockAgent(usage_in_last_resp=mock_usage)
    
    print("\nScenario: Data is only in 'last_response' (The Bug Scenario)")
    
    old_res = old_get_usage(agent)
    new_res = new_get_usage(agent)
    
    print(f"  Old Code Result: {old_res.get('total')} tokens")
    print(f"  New Code Result: {new_res.get('total')} tokens")
    
    if old_res.get('total') == 0 and new_res.get('total') == 150:
        print("\n✅ VERIFIED: The old code is BROKEN and the new code FIXES it.")
    else:
        print("\n❌ TEST INCONCLUSIVE: Check logic.")

if __name__ == "__main__":
    run_comparison()

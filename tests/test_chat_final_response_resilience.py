"""Regression: AIAgent.chat() must not crash when run_conversation returns an
error/failure result that omits the 'final_response' key.

Several error/failure return paths in agent.conversation_loop.run_conversation
(API error after max retries, billing/credits exhaustion, policy halt, etc.)
return a result dict with keys like {messages, completed, api_calls, error,
failed} but *no* 'final_response'. chat() previously did
``return result["final_response"]`` and raised ``KeyError: 'final_response'``
on those paths instead of surfacing the error.
"""


def _agent_with(result):
    # Bypass the heavy __init__; chat() only depends on self.run_conversation.
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.run_conversation = lambda *a, **k: result
    return agent


def test_chat_surfaces_error_when_final_response_missing():
    agent = _agent_with({
        "messages": [],
        "completed": False,
        "api_calls": 1,
        "error": "API call failed after 3 retries",
        "failed": True,
        # no "final_response" key — the regression case
    })
    assert agent.chat("hi") == "API call failed after 3 retries"


def test_chat_returns_final_response_when_present():
    agent = _agent_with({"final_response": "hello world", "error": None})
    assert agent.chat("hi") == "hello world"


def test_chat_empty_string_when_neither_final_response_nor_error():
    agent = _agent_with({"completed": True, "messages": [], "api_calls": 0})
    assert agent.chat("hi") == ""

"""Regression: AIAgent.chat() raised KeyError on run_conversation error paths.

``run_conversation`` returns a result dict WITHOUT a ``final_response`` key on
several error/exhaustion paths (invalid response after max retries,
payload-too-large after compression, compaction disabled, …). ``chat()`` read
it via ``result["final_response"]`` (subscript), so those paths crashed with
``KeyError: 'final_response'`` instead of degrading gracefully — unlike every
other call site, which uses ``.get``.
"""

from unittest.mock import MagicMock

from run_agent import AIAgent


# Mirrors the shapes returned by run_conversation's error/exhaustion paths
# (agent/conversation_loop.py) — note the absence of "final_response".
_ERROR_RESULTS = [
    {
        "messages": [], "completed": False, "api_calls": 3,
        "error": "Request payload too large: max compression attempts (3) reached.",
        "partial": True, "failed": True, "compression_exhausted": True,
    },
    {
        "messages": [], "completed": False, "api_calls": 5,
        "error": "Invalid API response after 5 retries: empty content",
        "failed": True,
    },
]


def test_chat_returns_gracefully_on_error_result():
    for err in _ERROR_RESULTS:
        agent = MagicMock()
        agent.run_conversation.return_value = err
        # Must not raise KeyError — degrade to empty string.
        out = AIAgent.chat(agent, "hello")
        assert out == ""


def test_chat_returns_final_response_on_success():
    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": "the answer",
        "messages": [], "completed": True, "api_calls": 1,
    }
    assert AIAgent.chat(agent, "hi") == "the answer"

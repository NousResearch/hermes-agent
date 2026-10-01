"""AgentRouter's intermittent thinking-replay 400 rides out on a bounded retry.

AgentRouter load-balances its OpenAI Chat Completions route across heterogeneous upstreams.
Some accept the replayed ``reasoning_content``; others translate it to Anthropic ``thinking``
blocks and reject the request with

    HTTP 400: The `content[].thinking` in the thinking mode must be passed back to the API.

even when every prior assistant tool turn already carries the reasoning the model returned. The
classifier lands that message on ``FailoverReason.format_error``, which the format-recovery ladder
did not cover, so the turn ended with the provider error — while the very same byte-identical
request succeeded minutes later (measured: 4/5 identical sends returned 200), i.e. the retry has
to be the fix, not a request rewrite (Anthropic blocks are invalid OpenAI wire format).

Contract under test:

- A burst of these 400s is retried, unchanged, a bounded number of times, pausing between
  attempts so the retry can land on another upstream.
- History is never mutated: the same complete replay is what succeeds.
- Past the budget the original error surfaces (no infinite retry).
- Another host, an incomplete replay, or a non-400 status is left alone.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent import turn_recovery
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import _recover_format_errors
from agent.turn_retry_state import TurnRetryState


class Agent:
    api_mode = "chat_completions"
    base_url = "https://agentrouter.org/v1"
    log_prefix = ""

    def _vprint(self, *args, **kwargs):
        pass


class Error(Exception):
    status_code = 400

    def __str__(self):
        return ("HTTP 400: The `content[].thinking` in the thinking mode must be passed back "
                "to the API. (request_id: 8a302c6b)")


def _complete_replay():
    """History whose assistant tool turns all carry the reasoning the model returned."""
    return [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "", "reasoning_content": "model-returned reasoning",
         "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "CONTINUE"},
    ]


def recover(agent, retry, api_messages, error=None):
    return _recover_format_errors(
        agent, error or Error(), SimpleNamespace(reason=FailoverReason.format_error),
        retry, [], api_messages,
    )


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Record the inter-attempt pauses instead of actually sleeping."""
    recorded = []
    monkeypatch.setattr(turn_recovery.time, "sleep", lambda seconds: recorded.append(seconds))
    return recorded


def test_the_message_reaches_the_recovery_ladder():
    """The seam this fix lives on: this message must stay a recoverable format_error."""
    classified = classify_api_error(Error())
    assert classified.reason == FailoverReason.format_error


def test_a_burst_of_rejections_rides_out_within_the_attempt_budget(sleeps):
    messages = _complete_replay()
    before = deepcopy(messages)
    retry = TurnRetryState()

    attempts = 0
    while recover(Agent(), retry, messages) is True:
        attempts += 1
        # The unchanged replay is what succeeds; canonical history stays byte-identical.
        assert messages == before
        assert attempts <= 10, "the retry budget must be bounded"

    assert attempts > 1  # a one-shot retry cannot ride out a burst
    assert attempts == turn_recovery._AGENTROUTER_THINKING_MAX_ATTEMPTS
    # Past the budget the original error surfaces instead of retrying forever.
    assert retry.agentrouter_thinking_retry_attempts == attempts
    assert recover(Agent(), retry, messages) is False
    assert retry.agentrouter_thinking_retry_attempts == attempts
    assert messages == before
    # Every attempt pauses first, so the retry can land on another upstream.
    assert sleeps == [turn_recovery._AGENTROUTER_THINKING_BACKOFF_SECONDS] * attempts


@pytest.mark.parametrize("base_url,reasoning,status", [
    ("https://other.example/v1", "model-returned reasoning", 400),
    ("https://evil.example/agentrouter.org/v1", "model-returned reasoning", 400),
    ("https://agentrouter.org/v1", "", 400),
    ("https://agentrouter.org/v1", "model-returned reasoning", 422),
])
def test_other_routes_or_incomplete_replays_are_not_retried(base_url, reasoning, status):
    agent = Agent()
    agent.base_url = base_url
    messages = [{"role": "assistant", "reasoning_content": reasoning, "tool_calls": [{"id": "c1"}]}]
    error = Error()
    error.status_code = status

    assert recover(agent, TurnRetryState(), messages, error) is False

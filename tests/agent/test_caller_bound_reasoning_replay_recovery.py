"""A caller-bound ``reasoning_details`` replay rejection reaches the one-shot
strip-and-retry recovery on chat completions (#109572).

The opencode relay's Console upstream seals replayed reasoning to the caller
that minted it; after its key-rotated free pool changes callers mid-session it
answers ``[invalid_request_error] reasoning `encrypted_content` was not issued
to this caller``. classify_api_error bucketed that as format_error, so the
one-shot repair never ran, every retry resent the same poisoned history, and a
long Desktop session became permanently uncontinuable. The classifier now
routes the exact envelope to invalid_encrypted_content, and this recovery
strips the sealed carrier from the persisted history and the in-flight
request."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import _recover_format_errors
from agent.turn_retry_state import TurnRetryState

_MESSAGE = (
    "Error from provider (Console): Upstream request failed: "
    "[invalid_request_error] reasoning `encrypted_content` was not issued to this caller"
)
_BODY = {"error": {"message": _MESSAGE, "type": "invalid_request_error", "param": None,
                   "code": "invalid_request_error"}}


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


def _history():
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi",
         "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque-caller-bound-blob"}]},
    ]


def _agent(api_mode="chat_completions"):
    agent = MagicMock()
    agent.api_mode = api_mode
    agent.log_prefix = "[test] "
    return agent


def test_caller_bound_replay_envelope_is_invalid_encrypted_content():
    e = MockAPIError(f"Error code: 400 - {_MESSAGE}", status_code=400, body=_BODY)
    result = classify_api_error(e, provider="opencode-free", model="muse-spark-1.3-contributor-free")
    assert result.reason == FailoverReason.invalid_encrypted_content
    # format_error's terminal hints kept — the verdict only buys the replay strip.
    assert result.retryable is False and result.should_fallback is True


def test_caller_bound_replay_envelope_stays_narrow():
    # Same wording on a 5xx is a server error, not a replay rejection.
    e = MockAPIError(f"Error code: 500 - {_MESSAGE}", status_code=500, body=_BODY)
    assert classify_api_error(e, provider="opencode-free", model="m").reason != (
        FailoverReason.invalid_encrypted_content
    )
    # A 400 naming encrypted_content without the caller wording keeps its own bucket.
    e = MockAPIError(
        "Error code: 400 - Unknown parameter: encrypted_content",
        status_code=400,
        body={"error": {"code": "unknown_parameter", "message": "Unknown parameter: encrypted_content"}},
    )
    assert classify_api_error(e, provider="opencode-free", model="m").reason != (
        FailoverReason.invalid_encrypted_content
    )


def test_chat_completions_recovery_strips_reasoning_details_and_retries():
    agent = _agent()
    messages = _history()
    api_messages = [dict(_m) for _m in messages]
    classified = SimpleNamespace(reason=FailoverReason.invalid_encrypted_content)
    _retry = TurnRetryState()

    repaired = _recover_format_errors(
        agent, MockAPIError(_MESSAGE, status_code=400, body=_BODY), classified, _retry,
        messages, api_messages,
    )

    assert repaired is True
    assert _retry.invalid_encrypted_content_retry_attempted is True
    # The poisoned carrier is gone from both the persisted history and the request.
    assert all("reasoning_details" not in _m for _m in messages)
    assert all("reasoning_details" not in _m for _m in api_messages)


def test_recovery_is_one_shot():
    agent = _agent()
    messages = _history()
    _retry = TurnRetryState()
    _retry.invalid_encrypted_content_retry_attempted = True
    classified = SimpleNamespace(reason=FailoverReason.invalid_encrypted_content)

    assert _recover_format_errors(
        agent, MockAPIError(_MESSAGE, status_code=400), classified, _retry, messages, [dict(_m) for _m in messages],
    ) is False
    assert messages[1]["reasoning_details"]  # nothing was stripped


def test_no_reasoning_details_no_new_recovery_path():
    agent = _agent()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    classified = SimpleNamespace(reason=FailoverReason.invalid_encrypted_content)
    _retry = TurnRetryState()

    assert _recover_format_errors(
        agent, MockAPIError(_MESSAGE, status_code=400), classified, _retry, messages, [dict(_m) for _m in messages],
    ) is False


def test_codex_responses_mode_is_unchanged():
    # The codex_responses branch still requires codex_reasoning_items; a history
    # carrying only reasoning_details must not enter either strip path.
    agent = _agent(api_mode="codex_responses")
    agent._codex_reasoning_replay_enabled = True
    messages = _history()
    classified = SimpleNamespace(reason=FailoverReason.invalid_encrypted_content)
    _retry = TurnRetryState()

    assert _recover_format_errors(
        agent, MockAPIError(_MESSAGE, status_code=400), classified, _retry, messages, [dict(_m) for _m in messages],
    ) is False
    assert messages[1]["reasoning_details"]

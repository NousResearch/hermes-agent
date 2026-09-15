"""An oversize rejection that arrives as HTTP 400 gets exactly one shrink attempt, then the
fallback walk — never ``max_retries`` byte-identical re-sends of the same oversized body.

``turn_recovery`` sets ``image_shrink_retry_attempted`` *before* it runs the shrink, so a set
flag at settle time means the shrink found nothing to rewrite (the excess is text, not an
inlined image). The verdict is still ``retryable``, so without the guard in
``settle_unrecovered_error`` the retry loop re-sends the identical body up to ``max_retries``
times before falling back — the behaviour these wordings had before they were routed to the
shrink path.
"""
from types import SimpleNamespace
from unittest.mock import patch

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_api_error import settle_unrecovered_error

NVIDIA_CAP = (
    "Error code: 400 - Please make sure your payload is below 26214400 bytes in size. "
    "If larger assets are required please refer to our Assets API."
)


class _Err(Exception):
    status_code = 400
    response = None

    def __init__(self, message, body=None):
        super().__init__(message)
        self.body = body or {"error": {"message": message, "type": "invalid_request_error",
                                       "code": "invalid_image_format"}}


class _Agent:
    log_prefix = ""
    verbose = False
    provider = "nvidia"
    _fallback_chain = [object()]
    _fallback_index = 0
    _credential_pool = None
    # ``__getattr__`` answers every unknown attribute with a truthy lambda, which the backoff
    # helper would read as an interrupt request; pin the flags the retry path consults.
    _interrupt_requested = False
    _shutdown_requested = False

    def __init__(self, has_fallback=True):
        self.activated = []
        self._has_fallback = has_fallback

    def _has_pending_fallback(self):
        return self._has_fallback

    def _try_activate_fallback(self, **kwargs):
        if not self._has_fallback:
            return False
        self.activated.append(True)
        return True

    def _summarize_api_error(self, error):
        return str(error)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _settle(agent, err, *, shrink_attempted, provider="nvidia"):
    retry = SimpleNamespace(
        copilot_stale_cred_retry_attempted=False, primary_recovery_attempted=False,
        image_shrink_retry_attempted=shrink_attempted,
        restart_with_redirected_messages=False, has_retried_429=False,
    )
    classified = classify_api_error(err, provider=provider)
    with patch("agent.conversation_loop._is_copilot_provider", lambda a: False):
        return settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=400, error_msg=str(err),
            is_context_length_error=False, is_rate_limited=False, _is_zai_coding_overload=False,
            _provider=provider, _base="https://integrate.api.nvidia.com/v1", _model="meta/llama-3.2-11b-vision-instruct",
            messages=[], api_messages=[], api_kwargs={}, active_system_prompt="", conversation_history=None,
            approx_tokens=10, retry_count=0, max_retries=3, compression_attempts=0, api_call_count=1,
        )


def test_payload_cap_400_is_an_image_verdict():
    """The routing this file's guard depends on."""
    assert classify_api_error(_Err(NVIDIA_CAP), provider="nvidia").reason is FailoverReason.image_too_large


def test_spent_shrink_falls_back_instead_of_resending_the_same_body():
    agent = _Agent()
    verdict = _settle(agent, _Err(NVIDIA_CAP), shrink_attempted=True)
    assert verdict.action == "break"
    assert agent.activated == [True]


def test_first_image_rejection_still_retries_after_the_shrink():
    agent = _Agent()
    verdict = _settle(agent, _Err(NVIDIA_CAP), shrink_attempted=False)
    assert verdict.action == "fallthrough"
    assert agent.activated == []


def test_spent_shrink_without_a_fallback_is_terminal_not_a_retry():
    verdict = _settle(_Agent(has_fallback=False), _Err(NVIDIA_CAP), shrink_attempted=True)
    assert verdict.action == "return"
    assert verdict.result["failed"] is True


def test_other_reasons_are_untouched_by_the_guard():
    """A rate limit with the image flag set must still take the ordinary retry path."""
    err = _Err("Error code: 429 - rate limit exceeded")
    err.status_code = 429
    agent = _Agent()
    assert _settle(agent, err, shrink_attempted=True).action == "fallthrough"

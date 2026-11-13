"""A connect-flavored 401 from a self-hosted proxy must take the retry ladder; a real
credential 401 must still fail fast.

Cron fires (board-harness-watch, disposition-closer, plan-adherence, daily-briefing) died on
``HTTP 401: Authentication Error, All connection attempts failed`` — the LiteLLM proxy's relay
of an unreachable vLLM deployment. Hermes read the 401 status as an auth refusal
(``_V_AUTH_ROTATE``, ``retryable=False``), so one blip burned the whole fire and bumped
``failure_streak``. The proxy's genuine bad-credential refusal is worded differently
(``Invalid proxy server token passed`` / ``type: token_not_found_in_db``), so the two bodies
must land on opposite verdicts on the same route.

Module-level ``import run_agent`` pulls ``agent.conversation_loop`` (and its ``hermes_bootstrap``
install-state read) in at COLLECTION time, before the ``home_io_guard`` autouse fixture arms.
Lazy-importing it from inside a test instead reads the real ``~/.hermes/manifest.json`` under the
guard; this is the idiom the other main-loop tests (test_auth_provider_failover,
test_api_max_retries_config, ...) use for exactly this reason.
"""
from unittest.mock import patch

import run_agent  # loads agent.conversation_loop pre-guard — see module docstring
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from agent.turn_recovery import route_classified_error
from agent.turn_retry_state import TurnRetryState


class _ProxyError(Exception):
    """Simulates ``openai.AuthenticationError`` from the LiteLLM proxy."""
    status_code = 401
    response = None

    def __init__(self, body):
        super().__init__(f"Error code: 401 - {body}")
        self.body = body


# Verbatim relay of an unreachable upstream deployment (observed in the gateway's errors.log).
CONNECT_401 = {"error": {
    "message": "Authentication Error, All connection attempts failed",
    "type": "auth_error", "param": "None", "code": "401",
}}
# Verbatim refusal of an unknown/vanished virtual key from the same proxy.
BAD_KEY_401 = {"error": {
    "message": "Invalid proxy server token passed",
    "type": "token_not_found_in_db", "param": "None", "code": "401",
}}
LITELLM = "http://proxy.internal:4000/v1"
PROVIDER = "litellm_proxy"


class _Agent:
    """Minimal seam: fallback activation and the credential-pool presence are real, the rest no-op."""
    log_prefix = ""
    verbose = False
    provider = PROVIDER
    base_url = LITELLM
    model = "main_model"
    _fallback_chain = [object()]
    _fallback_index = 0
    _credential_pool = None

    def __init__(self):
        self.activated = []
        self.buffers = []

    def _has_pending_fallback(self):
        return True

    def _try_activate_fallback(self, **kwargs):
        self.activated.append(kwargs.get("reason"))
        return True

    def _summarize_api_error(self, error):
        return str(error)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _settle(agent, err, *, retry_count=0):
    retry = TurnRetryState()
    classified = classify_api_error(err, provider=PROVIDER, model="main_model", base_url=LITELLM)
    with patch("agent.conversation_loop._credential_refresh_profile", lambda a: None):
        verdict = settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=401,
            error_msg=str(err), is_context_length_error=False, is_rate_limited=False,
            _is_zai_coding_overload=False, _provider=PROVIDER, _base=LITELLM, _model="main_model",
            messages=[], api_messages=[], api_kwargs={}, active_system_prompt="",
            conversation_history=None, approx_tokens=10, retry_count=retry_count, max_retries=3,
            compression_attempts=0, api_call_count=1,
        )
    return verdict, classified


def _route(agent, err, *, retry_count):
    retry = TurnRetryState()
    classified = classify_api_error(err, provider=PROVIDER, model="main_model", base_url=LITELLM)
    verdict = route_classified_error(
        agent, err, classified, retry, error_msg=str(err), error_context={}, recovered_with_pool=False,
        base_url=LITELLM, model="main_model", messages=[], api_messages=[], system_message=None,
        active_system_prompt="", conversation_history=None, retry_count=retry_count, max_retries=3,
        compression_attempts=0, max_compression_attempts=3, api_call_count=1, effective_task_id=None,
    )
    return verdict, classified


def test_connect_flavored_401_retries_instead_of_settling_as_a_client_error():
    """The classification-time bug: ``not retryable and not should_compress and reason not in
    RETRYABLE_CLIENT_REASONS`` is what routed the fire straight to the terminal result. With the
    connect wording classified as transport, the error must fall through to the backoff path."""
    agent = _Agent()
    verdict, classified = _settle(agent, _ProxyError(CONNECT_401))
    assert classified.reason is FailoverReason.timeout
    assert classified.retryable is True
    assert verdict.action == "fallthrough"          # -> compute_error_backoff + next attempt
    assert agent.activated == []                    # no eager fallback switch on attempt 1


def test_true_credential_401_still_settles_as_a_client_error_on_attempt_1():
    """No regression: a real bad-credential 401 fails fast (one attempt, then fallback), it does
    not loop the retry ladder."""
    agent = _Agent()
    verdict, classified = _settle(agent, _ProxyError(BAD_KEY_401))
    assert classified.reason is FailoverReason.auth
    assert classified.retryable is False
    assert verdict.action == "break"                # fallback activated, turn restarts there
    assert len(agent.activated) == 1


def test_connect_flavored_401_reaches_backoff_on_every_attempt_below_the_ceiling():
    """The retry ladder must actually run: attempts 1..max_retries-1 all reach the backoff path
    (the old behaviour ended the fire on attempt 1)."""
    for retry_count in (0, 1, 2):
        agent = _Agent()
        verdict, _ = _settle(agent, _ProxyError(CONNECT_401), retry_count=retry_count)
        assert verdict.action == "fallthrough", retry_count


def test_persistent_connect_flavored_401_falls_back_like_a_transport_failure():
    """After the retry budget, the cascade runs with the transport reason rather than a
    credential-benching auth reason."""
    agent = _Agent()
    verdict, classified = _settle(agent, _ProxyError(CONNECT_401), retry_count=3)
    assert verdict.action == "break"
    assert agent.activated == [classified.reason] == [FailoverReason.timeout]


def test_connect_flavored_401_does_not_take_the_eager_auth_fallback_branch():
    """``route_classified_error`` switches provider on a persistent auth verdict; a proxy that
    simply could not reach its upstream must keep trying it, not bench the route on attempt 1."""
    agent = _Agent()
    verdict, _ = _route(agent, _ProxyError(CONNECT_401), retry_count=0)
    assert verdict.action == "fallthrough"
    assert agent.activated == []


def test_true_credential_401_still_takes_the_auth_fallback_branch():
    """The auth branch itself is untouched: a persistent auth verdict still escalates to fallback."""
    agent = _Agent()
    verdict, _ = _route(agent, _ProxyError(BAD_KEY_401), retry_count=0)
    assert verdict.action == "break"
    assert agent.activated == [FailoverReason.auth]


def test_connect_flavored_401_is_not_routed_to_credential_guidance():
    """A connect-flavored 401 is not ``is_auth``, so the terminal path's auth/billing guard
    (agent/turn_recovery.py) will not send an operator debugging a dead GPU toward their key."""
    classified = classify_api_error(_ProxyError(CONNECT_401), provider=PROVIDER, base_url=LITELLM)
    assert classified.is_auth is False
    assert not (classified.is_auth or classified.reason == FailoverReason.billing)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

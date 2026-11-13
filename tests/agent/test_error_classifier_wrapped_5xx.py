"""A 401 whose body reports an auth gateway's own 5xx is an outage, not a rejected credential.

OpenAI's auth gateway answered ``401 authentication failed (status=503): upstream connect error``
during an outage. Read as ``auth`` it printed "sign in again", redeemed the single-use refresh
token, benched the pool entry and parked the Kanban task with exit 78, on a healthy login.
"""
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import TurnRetryState, recover_after_classification
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
from hermes_cli.turn_exit import turn_exit_code


class _APIError(Exception):
    def __init__(self, message, status_code=401, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


# The observed text, verbatim up to where the worker log truncates it.
_OBSERVED = (
    "authentication failed (status=503): upstream connect error or disconnect/reset before headers. "
    "retried and the latest reset reason: remote connection failure, transport failure reason: "
    "TLS_error:|268435581:SSL routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED:verify cert failed: "
    "SAN matcher:TLS_er"
)


@pytest.mark.parametrize("provider", ["openai-codex", "openrouter", ""])
@pytest.mark.parametrize("message", [
    _OBSERVED,
    f"HTTP 401: {_OBSERVED}",
    "authentication failed (status=502)",
    "Authentication Failed (STATUS=500): bad gateway",
    "upstream connect error or disconnect/reset before headers. reset reason: connection failure",
])
def test_401_wrapping_a_gateway_5xx_is_server_error(message, provider):
    result = classify_api_error(_APIError(message), provider=provider)
    assert result.reason == FailoverReason.server_error
    assert result.retryable is True
    assert result.should_rotate_credential is False
    assert result.is_auth is False


def test_401_wrapping_a_gateway_5xx_in_the_body_is_server_error():
    error = _APIError("Unauthorized", body={"error": {"message": _OBSERVED}})
    assert classify_api_error(error, provider="openai-codex").reason == FailoverReason.server_error


@pytest.mark.parametrize("message, body", [
    ("Unauthorized", None),
    ("Incorrect API key provided", None),
    ("authentication failed (status=401)", None),
    ("authentication failed (status=403)", None),
    ("authentication failed (status=5000)", None),
    ("substatus=503: invalid token", None),
    ("Your authentication token has been invalidated.", {"error": {"code": "token_invalidated"}}),
])
def test_a_real_401_stays_auth(message, body):
    result = classify_api_error(_APIError(message, body=body), provider="openai-codex")
    assert result.reason == FailoverReason.auth
    assert result.retryable is False
    assert result.should_rotate_credential is True


def test_kanban_worker_requeues_instead_of_parking():
    reason = classify_api_error(_APIError(_OBSERVED), provider="openai-codex").reason.value
    code = turn_exit_code({"failed": True, "failure_reason": reason}, kanban_worker=True)
    assert code == KANBAN_RATE_LIMIT_EXIT_CODE


def test_credential_pool_is_left_alone():
    classified = classify_api_error(_APIError(_OBSERVED), provider="openai-codex")
    pool = MagicMock(provider="openai-codex")
    agent = MagicMock(
        _credential_pool=pool, provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex",
        api_key="k",
    )
    agent._is_entitlement_failure.return_value = False
    agent._auth_pool_refresh_counts = None
    recovered, _ = recover_with_credential_pool(
        agent, status_code=401, has_retried_429=False, classified_reason=classified.reason,
        error_context={"message": _OBSERVED},
    )
    assert recovered is False
    pool.try_refresh_matching.assert_not_called()
    pool.mark_exhausted_and_rotate.assert_not_called()


@pytest.fixture
def codex_agent(monkeypatch):
    """A Codex agent past the recovery rungs that run ahead of the 401 refresh."""
    monkeypatch.setattr("agent.turn_recovery._recover_welcome_tier", lambda *a, **k: False)
    monkeypatch.setattr("agent.turn_recovery._clamp_to_affordable_budget", lambda *a, **k: False)
    monkeypatch.setattr("agent.turn_recovery._reject_all_turns", lambda *a, **k: False)
    agent = MagicMock(api_mode="codex_responses", provider="openai-codex", _credential_pool=None)
    agent._recover_with_credential_pool.return_value = (False, False)
    agent._try_refresh_codex_client_credentials.return_value = True
    return agent


def _recover(agent, error):
    classified = classify_api_error(error, provider="openai-codex")
    return recover_after_classification(
        agent, error, classified, TurnRetryState(), status_code=401, error_context={}, messages=[],
        api_messages=[],
    )


def test_outage_does_not_redeem_the_single_use_refresh_token(codex_agent):
    assert _recover(codex_agent, _APIError(_OBSERVED))[0] is False
    codex_agent._try_refresh_codex_client_credentials.assert_not_called()


def test_real_401_still_refreshes(codex_agent):
    assert _recover(codex_agent, _APIError("Unauthorized"))[0] is True
    codex_agent._try_refresh_codex_client_credentials.assert_called_once_with(force=True)

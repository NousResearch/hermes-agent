"""Rejected xAI OAuth bearers reach bounded recovery before generic auth failure (#82052, #108122).

Drive the real classifier, error-context extraction, pool dispatch and recovery phase. The
credential exchange is the test boundary; client adoption is covered by its lifecycle tests.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from agent.agent_runtime_helpers import _is_entitlement_403, extract_api_error_context, recover_with_credential_pool
from agent.api_error_summary import ApiErrorSummaryMixin
from agent.error_classifier import classify_api_error
from agent.turn_recovery import recover_after_classification
from agent.turn_retry_state import TurnRetryState


class _WireError(Exception):
    def __init__(self, status, message="Forbidden", body=None):
        super().__init__(message)
        self.status_code = status
        self.body = body


_STALE_BODY = {
    "code": "unauthenticated:bad-credentials",
    "error": "The OAuth2 access token could not be validated.",
}
_STALE_ERRORS = [
    pytest.param(403, "Forbidden", _STALE_BODY, id="structured-xai-body"),
    pytest.param(403, "Forbidden", {"code": "unauthenticated:bad-credentials"}, id="structured-code-only"),
    pytest.param(403, "Forbidden", {"code": "The caller does not have permission to execute the specified operation",
                                      "error": "unauthenticated:bad-credentials"}, id="stale-marker-over-permission-text"),
    pytest.param(403, "Forbidden", {"error": {"message": _STALE_BODY["error"]}}, id="nested-sdk-body"),
    pytest.param(403, _STALE_BODY["error"], None, id="message-only"),
    pytest.param(None, "Stream refused [WKE=unauthenticated:bad-credentials]", None, id="sse-marker"),
    pytest.param(None, "Stream refused", {"error": {"code": "unauthenticated:bad-credentials"}}, id="sse-code"),
]


class _Agent(ApiErrorSummaryMixin):
    api_mode = "codex_responses"
    base_url = "https://api.x.ai/v1"
    model = "grok-test"
    _credential_pool = None
    _recover_with_credential_pool = recover_with_credential_pool
    _extract_api_error_context = staticmethod(extract_api_error_context)

    def __init__(self, provider="xai-oauth", *, refresh_ok=True):
        self.provider = provider
        self.refresh_ok = refresh_ok
        self.refreshes = []
        self.notices = []

    def _get_transport(self):
        return None

    def _try_refresh_codex_client_credentials(self, *, force):
        self.refreshes.append(force)
        return self.refresh_ok

    def _buffer_vprint(self, message):
        self.notices.append(message)


def _recover(agent, error, retry, messages):
    return recover_after_classification(
        agent, error, classify_api_error(error, provider=agent.provider, model=agent.model), retry,
        status_code=error.status_code, error_context=extract_api_error_context(error),
        messages=messages, api_messages=deepcopy(messages),
    )


@pytest.mark.parametrize("status,message,body", _STALE_ERRORS)
@pytest.mark.parametrize("refresh_ok", [True, False], ids=["recovered", "refresh-failed"])
def test_stale_oauth_recovery_is_once_and_preserves_the_request(status, message, body, refresh_ok):
    agent = _Agent(refresh_ok=refresh_ok)
    retry = TurnRetryState()
    error = _WireError(status, message, body)
    messages = [{"role": "system", "content": "Keep this prefix."}, {"role": "user", "content": "Continue."}]
    before = deepcopy(messages)

    assert _recover(agent, error, retry, messages) == (refresh_ok, False)
    assert agent.refreshes == [True]
    assert retry.codex_auth_retry_attempted
    assert _recover(agent, error, retry, messages) == (False, False)
    # A later 401 shares the same budget; a rejected refresh cannot spin either.
    assert _recover(agent, _WireError(401, "Unauthorized"), retry, messages) == (False, False)
    assert agent.refreshes == [True]
    assert messages == before
    assert bool(agent.notices) is refresh_ok
    # The pool must agree, including when the only signal is a structured code.
    assert not _is_entitlement_403(agent, status, extract_api_error_context(error))


@pytest.mark.parametrize("provider,status,message,body", [
    pytest.param("openai-codex", 403, "Forbidden", _STALE_BODY, id="other-oauth-provider"),
    pytest.param("xai", 403, "Forbidden", _STALE_BODY, id="api-key-provider"),
    pytest.param("xai-oauth", 403, "Forbidden", None, id="generic-forbidden"),
    pytest.param("xai-oauth", 403, "This model requires a SuperGrok subscription", None, id="subscription"),
    pytest.param("xai-oauth", 403, "Your team has reached its spending limit", {
        "code": "personal-team-blocked:spending-limit",
    }, id="billing"),
    pytest.param("xai-oauth", 403, "Forbidden", {
        "code": "personal-team-blocked:spending-limit", "error": _STALE_BODY["error"],
    }, id="billing-code-wins-over-text"),
    pytest.param("xai-oauth", 403, "OAuth authentication is currently not allowed for this organization", None,
                 id="organization-policy"),
    pytest.param("xai-oauth", 429, "Rate limited", _STALE_BODY, id="throttled-not-stale"),
    pytest.param("xai-oauth", 500, "Error code: 403 - unauthenticated:bad-credentials", None,
                 id="real-status-wins-over-text"),
    pytest.param("xai-oauth", None, "unauthenticated request", None, id="unqualified-auth-text"),
])
def test_unrelated_errors_never_refresh_xai_credentials(provider, status, message, body):
    agent = _Agent(provider)
    retry = TurnRetryState()
    assert _recover(agent, _WireError(status, message, body), retry, []) == (False, False)
    assert agent.refreshes == []
    assert not retry.codex_auth_retry_attempted


@pytest.mark.parametrize("body,expected_status", [
    pytest.param({"code": "personal-team-blocked:spending-limit", "error": _STALE_BODY["error"]},
                 "exhausted", id="explicit-billing"),
    pytest.param({"code": "permission_denied", "error":
                  "OAuth authentication is currently not allowed for this organization. " + _STALE_BODY["error"]},
                 None, id="explicit-organization-policy"),
])
def test_explicit_refusals_do_not_refresh_a_real_oauth_pool(tmp_path, monkeypatch, body, expected_status):
    from agent.credential_pool import load_pool

    auth_home = tmp_path / "profile"
    auth_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(auth_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    (auth_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "credential_pool": {"xai-oauth": [{
            "id": "account-a", "label": "test account", "source": "manual:device_code",
            "auth_type": "oauth", "access_token": "test-bearer", "refresh_token": "test-refresh",
        }]},
    }), encoding="utf-8")
    pool = load_pool("xai-oauth")
    agent = _Agent()
    agent.api_key = "test-bearer"
    agent._credential_pool = pool
    agent._credential_pool_entry_id = "account-a"

    def unexpected_refresh(**_):
        pytest.fail("a billing/policy refusal must not redeem an OAuth refresh token")

    monkeypatch.setattr(pool, "try_refresh_matching", unexpected_refresh)
    monkeypatch.setattr(agent, "_try_refresh_codex_client_credentials", unexpected_refresh)

    assert _recover(agent, _WireError(403, "Forbidden", body), TurnRetryState(), []) == (False, False)
    (entry,) = pool.entries()
    assert entry.last_status == expected_status
    if expected_status == "exhausted":
        assert entry.failure_reason == "billing"
        assert entry.last_error_reason == body["code"]

"""Regression tests: xAI OAuth stale-token 403 must trigger the one-shot refresh+retry.

Bug (observed on a long-lived gateway, 2026-09-11): xAI signals a stale/rejected OAuth
bearer as HTTP 403 ``unauthenticated:bad-credentials`` / "The OAuth2 access token could
not be validated." — not as a 401. ``_refresh_credentials_after_401`` gated every
per-provider refresh on ``status_code == 401``, so the recovery chain never fired for
these bodies: the pool refresh matched no entry (a profile borrowing the root's grant has
no local pool rows) and the turn aborted as non-retryable while the auth store held a
working grant — replaying the dumped request with the stored token returned 200 seconds
after the 403.

The classifier already keeps exactly these bodies refreshable (``_is_entitlement_403``,
#29344); these tests pin the loop-side contract that the one-shot credential refresh
fires for them too, and ONLY for them (entitlement/subscription 403s must not trigger
a refresh that cannot fix them).
"""

import types

import agent.turn_recovery as turn_recovery
from agent.turn_retry_state import TurnRetryState


def _xai_agent(**overrides):
    agent = types.SimpleNamespace(
        provider="xai-oauth",
        api_mode="codex_responses",
        refreshed=[],
    )
    agent._try_refresh_codex_client_credentials = lambda *, force=True: (
        agent.refreshed.append(force) or True
    )
    agent._buffer_vprint = lambda *_a, **_k: None
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


_STALE_BODY = (
    "Error code: 403 - {'code': 'unauthenticated:bad-credentials', "
    "'error': 'The OAuth2 access token could not be validated.'}"
)

_SSE_VARIANT = "stream error [WKE=unauthenticated:expired_token] mid-response"


class TestStaleXaiOauth403Refresh:
    def test_stale_token_403_fires_one_shot_refresh(self):
        """403 + stale-token marker → the xai-oauth branch of the 401 refresh runs."""
        agent = _xai_agent()
        api_error = PermissionError(_STALE_BODY)  # message carries the body
        retry = TurnRetryState()
        ok = turn_recovery._refresh_credentials_after_401(agent, api_error, retry, 403)
        assert ok is True
        assert agent.refreshed == [True]  # force=True

    def test_sse_unauthenticated_marker_also_fires(self):
        """The status-less SSE variant ``[WKE=unauthenticated:...]`` is the same signal."""
        agent = _xai_agent()
        retry = TurnRetryState()
        ok = turn_recovery._refresh_credentials_after_401(
            agent, PermissionError(_SSE_VARIANT), retry, None
        )
        assert ok is True

    def test_entitlement_403_does_not_fire(self):
        """A plain 403 without a stale-token marker (entitlement/subscription shape)
        must NOT burn the one-shot refresh — a refresh cannot fix entitlement."""
        agent = _xai_agent()
        retry = TurnRetryState()
        ok = turn_recovery._refresh_credentials_after_401(
            agent, PermissionError("Error code: 403 - forbidden"), retry, 403
        )
        assert ok is False
        assert agent.refreshed == []

    def test_other_provider_403_does_not_fire(self):
        """Provider scoping: only xai-oauth gets the 403 treatment."""
        agent = _xai_agent(provider="openai-codex")
        retry = TurnRetryState()
        ok = turn_recovery._refresh_credentials_after_401(
            agent, PermissionError(_STALE_BODY), retry, 403
        )
        assert ok is False
        assert agent.refreshed == []

    def test_one_shot_guard_shared_with_401(self):
        """The 403 path shares ``codex_auth_retry_attempted`` with the 401 path —
        a 401-then-403 (or vice versa) within one attempt refreshes exactly once."""
        agent = _xai_agent()
        retry = TurnRetryState()
        assert turn_recovery._refresh_credentials_after_401(agent, PermissionError("401"), retry, 401) is True
        assert turn_recovery._refresh_credentials_after_401(agent, PermissionError(_STALE_BODY), retry, 403) is False
        assert agent.refreshed == [True]


class TestIsStaleXaiOauthError:
    def test_json_body_shape(self):
        assert turn_recovery._is_stale_xai_oauth_error(403, _STALE_BODY) is True

    def test_sse_shape_without_status(self):
        assert turn_recovery._is_stale_xai_oauth_error(None, _SSE_VARIANT) is True

    def test_plain_403(self):
        assert turn_recovery._is_stale_xai_oauth_error(403, "forbidden") is False

    def test_401_with_marker_string(self):
        """A non-403 STATUS (e.g. 401) with a marker-bearing message that does not claim
        403 anywhere must not enter the 403 path — the regular 401 branch owns it."""
        assert turn_recovery._is_stale_xai_oauth_error(401, _SSE_VARIANT) is False
        assert turn_recovery._is_stale_xai_oauth_error(500, _SSE_VARIANT) is False

    def test_empty_message(self):
        assert turn_recovery._is_stale_xai_oauth_error(403, "") is False

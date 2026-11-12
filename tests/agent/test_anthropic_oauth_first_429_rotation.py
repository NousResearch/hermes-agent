"""Anthropic OAuth pools can rotate on the first 429 without a 600s retry.

All tests use synthetic credentials and do not access Anthropic or the auth store.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.agent_runtime_helpers import _recover_rate_limit


def _entry(credential_id, key, *, kind="oauth"):
    return SimpleNamespace(
        id=credential_id,
        auth_type=kind,
        runtime_api_key=key,
        last_status=None,
    )


def _first_429(provider, entries, *, context=None, second_is_available=True):
    pool = SimpleNamespace(
        provider=provider,
        entries=lambda: entries,
        current=lambda: entries[0],
    )
    rotate = Mock(return_value=second_is_available)
    recovered, retried = _recover_rate_limit(
        pool,
        has_retried_429=False,
        error_context=context or {"message": "This request would exceed your account's rate limit."},
        api_key_hint=entries[0].runtime_api_key,
        credential_id=entries[0].id,
        rotate_and_swap=rotate,
    )
    return (recovered, retried), rotate


def test_multi_account_anthropic_oauth_rotates_on_first_429():
    result, rotate = _first_429("anthropic", [
        _entry("first", "fake-oauth-one"),
        _entry("second", "fake-oauth-two"),
    ])
    assert result == (True, False)
    rotate.assert_called_once_with(429, "anthropic OAuth rate limit")


@pytest.mark.parametrize("provider, entries", [
    ("anthropic", [_entry("only", "fake-oauth-one")]),
    ("anthropic", [_entry("first", "same-key"), _entry("alias", "same-key")]),
    ("anthropic", [_entry("first", "fake-api-one", kind="api_key"), _entry("second", "fake-api-two", kind="api_key")]),
    ("openai-codex", [_entry("first", "fake-one"), _entry("second", "fake-two")]),
])
def test_single_account_alias_api_key_or_other_provider_retains_retry_once(provider, entries):
    result, rotate = _first_429(provider, entries, context={"message": "temporary throttle"})
    assert result == (False, True)
    rotate.assert_not_called()


def test_pool_exhaustion_does_not_claim_success():
    result, rotate = _first_429("anthropic", [
        _entry("first", "fake-oauth-one"),
        _entry("second", "fake-oauth-two"),
    ], second_is_available=False)
    assert result == (False, True)
    rotate.assert_called_once_with(429, "anthropic OAuth rate limit")

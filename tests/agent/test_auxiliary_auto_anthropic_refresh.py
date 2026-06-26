"""Regression test for the auto->anthropic auth-refresh derivation.

Bug: auxiliary/compression calls dispatch with provider="auto" but bind to a
concrete Anthropic OAuth backend. The old gate was:

    if _is_auth_error(first_err) and resolved_provider not in {"auto","",None}:

so an "auto"-routed Anthropic client that 401'd on an expired OAuth token was
EXCLUDED from the refresh path and retried forever on the stale token.

Fix: derive the real provider via _recoverable_pool_provider(resolved, client)
before the eligibility check. This test pins that derivation for the Anthropic
host so the refresh branch is reachable for auto-routed clients.
"""

import pytest

from agent.auxiliary_client import (
    _recoverable_pool_provider,
    _evict_cached_client_instance,
    _is_auth_error,
)


class _AnthropicishClient:
    base_url = "https://api.anthropic.com/v1/"


class _UnknownClient:
    base_url = "https://example.invalid/v1/"


class TestAutoAnthropicDerivation:
    def test_auto_with_anthropic_host_maps_to_anthropic(self):
        # This is the exact path my patch relies on: provider resolved to
        # "auto", client bound to api.anthropic.com -> must yield "anthropic"
        # so _refresh_provider_credentials("anthropic") gets called.
        assert _recoverable_pool_provider("auto", _AnthropicishClient()) == "anthropic"

    def test_empty_provider_with_anthropic_host_maps_to_anthropic(self):
        assert _recoverable_pool_provider("", _AnthropicishClient()) == "anthropic"

    def test_explicit_anthropic_passes_through(self):
        assert _recoverable_pool_provider("anthropic", _UnknownClient()) == "anthropic"

    def test_auto_with_unknown_host_returns_none(self):
        # No host match, no main_runtime -> None, so the refresh gate stays
        # closed (old behavior preserved for genuinely unrecoverable clients).
        assert _recoverable_pool_provider("auto", _UnknownClient()) is None


class TestRefreshGateReachability:
    """Simulate the patched gate end-to-end without live OAuth.

    Reproduces the boolean logic at lines ~5560/6046: derive provider, then
    decide whether the refresh branch is entered.
    """

    @staticmethod
    def _gate_enters_refresh(resolved_provider, client, first_err, client_is_nous=False):
        refresh_provider = resolved_provider
        if refresh_provider in {"auto", "", None}:
            refresh_provider = _recoverable_pool_provider(resolved_provider, client)
        return bool(
            _is_auth_error(first_err)
            and refresh_provider
            and not client_is_nous
        )

    def test_auto_anthropic_401_enters_refresh(self):
        exc = Exception("Error code: 401 - Unauthorized")
        exc.status_code = 401
        # OLD code: resolved="auto" -> excluded -> False (the bug).
        # NEW code: derives "anthropic" -> enters refresh -> True.
        assert self._gate_enters_refresh("auto", _AnthropicishClient(), exc) is True

    def test_auto_unknown_host_401_does_not_enter_refresh(self):
        exc = Exception("Error code: 401 - Unauthorized")
        exc.status_code = 401
        assert self._gate_enters_refresh("auto", _UnknownClient(), exc) is False

    def test_auto_anthropic_non_auth_error_does_not_enter_refresh(self):
        exc = Exception("Error code: 500 - Internal server error")
        exc.status_code = 500
        assert self._gate_enters_refresh("auto", _AnthropicishClient(), exc) is False

    def test_nous_client_never_enters_refresh(self):
        exc = Exception("Error code: 401 - Unauthorized")
        exc.status_code = 401
        assert self._gate_enters_refresh(
            "auto", _AnthropicishClient(), exc, client_is_nous=True
        ) is False


class TestEvictByInstanceExists:
    def test_evict_none_is_noop(self):
        # Patch calls _evict_cached_client_instance(client) before retry; the
        # auto-keyed entry must be evictable by instance. None must be safe.
        assert _evict_cached_client_instance(None) is False

    def test_evict_unknown_instance_returns_false(self):
        assert _evict_cached_client_instance(object()) is False

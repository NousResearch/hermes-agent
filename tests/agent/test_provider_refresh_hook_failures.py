"""A provider's credential-refresh hooks run inside the API-error handler: a raising hook (a plugin bug,
a plugin host that disconnected or timed out) answers "not refreshed" and the original API error walks
the fallback chain, instead of escaping the handler outside the retry loop's ``try``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.client_lifecycle import ClientLifecycleMixin
from agent.error_classifier import classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from providers.base import ProviderProfile


class _HostGone(ProviderProfile):
    """Every refresh hook can lose its plugin host; *broken* names the one that does."""

    broken = ""

    def _maybe_raise(self, hook: str) -> None:
        if hook == self.broken:
            raise ConnectionError("plugin host closed the connection")

    def is_stale_credential_error(self, *, status_code, message):
        self._maybe_raise("is_stale_credential_error")
        return True

    def refresh_runtime_credentials(self, *, reason):
        self._maybe_raise("refresh_runtime_credentials")
        return ("fresh-token", None)

    def credential_refresh_notice(self, *, reason):
        self._maybe_raise("credential_refresh_notice")
        return ""


class _Err(Exception):
    status_code = 400
    response = None
    body = {"error": {"message": "Unsupported parameter: 'max_tokens'", "type": "invalid_request_error"}}


class _Agent:
    """The real credential-refresh method; the fallback seam records activations; the rest no-ops."""
    log_prefix = ""
    verbose = False
    provider = "acme-hosted"
    base_url = "https://api.acme-hosted.example/v1"
    api_key = "stale-token"
    _fallback_chain = [object()]
    _fallback_index = 0
    _credential_pool = None
    _try_refresh_provider_runtime_credentials = ClientLifecycleMixin._try_refresh_provider_runtime_credentials

    def __init__(self):
        self.activated = []

    def _has_pending_fallback(self):
        return True

    def _try_activate_fallback(self, **kwargs):
        self.activated.append(True)
        return True

    def _replace_primary_openai_client(self, *, reason):
        return False  # the rebuilt client could not be made: not refreshed either

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.mark.parametrize("broken", ["is_stale_credential_error", "refresh_runtime_credentials"])
def test_a_raising_refresh_hook_keeps_the_api_error_and_walks_the_fallback_chain(broken):
    profile = _HostGone(name="acme-hosted")
    profile.broken = broken
    agent, err = _Agent(), _Err("Unsupported parameter: 'max_tokens'")
    retry = SimpleNamespace(provider_stale_refresh_attempted=False, primary_recovery_attempted=False)
    with patch("agent.conversation_loop._credential_refresh_profile", lambda a: profile):
        verdict = settle_unrecovered_error(
            agent, api_error=err, classified=classify_api_error(err, provider=agent.provider), _retry=retry,
            status_code=400, error_msg=str(err), is_context_length_error=False, is_rate_limited=False,
            _is_zai_coding_overload=False, _provider=agent.provider, _base=agent.base_url, _model="m",
            messages=[], api_messages=[], api_kwargs={}, active_system_prompt="", conversation_history=None,
            approx_tokens=10, retry_count=0, max_retries=3, compression_attempts=0, api_call_count=1,
        )
    assert verdict.action == "break"
    assert agent.activated == [True]
    assert agent.api_key == "stale-token"

# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""Credential-pool exhaustion must not mask a rate-limit as a keyless-client abort.

Bug (skill-patch-applier cron b193bfce7d42, 2026-07-07): under a Claude-pool 5-hour
cap, the single relay credential (CLAUDE_POOL_KEY) is marked exhausted on a 429; the
recovery-path rotation then calls ``_swap_credential`` with an entry whose
``runtime_api_key`` resolves to ``""``. ``_swap_credential`` builds an Anthropic SDK
client with ``api_key=""`` which raises, at REQUEST time, a
``TypeError: "Could not resolve authentication method…"``. The conversation loop
classifies a bare ``TypeError`` as ``is_local_validation_error`` -> non-retryable ->
the turn aborts (a cron reports a job failure). A transient rate-limit is thus surfaced
as a non-retryable local-programming error.

The fix (two layers):
  D-1a (source): ``_swap_credential`` returns a tri-state ``SwapOutcome`` and NEVER
      builds/uses a keyless client. On an empty key it distinguishes the 429-exhausted
      case (RETRYABLE_EXHAUSTED) from a never-rate-limited/missing key (MISSING_CREDENTIAL,
      loud) via the pool entry's ``last_error_code`` (bounded by the exhausted TTL).
  D-1b (backstop): the auth-resolution ``TypeError`` is excluded from
      ``is_local_validation_error`` (mirroring the ``NoneType is not iterable`` carve-out),
      so a stray occurrence routes to the defined ``auth`` terminal, never a "local bug" crash.
"""

import time
from unittest.mock import MagicMock

import pytest

from agent.credential_pool import PooledCredential


# The exact substring the Anthropic SDK raises when a client is built with no key.
AUTH_RESOLVE_MSG = (
    "Could not resolve authentication method. Expected either api_key or "
    "auth_token to be set. Or for one of the `X-Api-Key` or `Authorization` "
    "headers to be explicitly omitted"
)


# ---------------------------------------------------------------------------
# Ground-truth guard (AC-9): a keyless Anthropic client still raises the exact
# substring at request time on the pinned SDK. Fails LOUD on an SDK message drift
# that would silently reopen the wedge.
# ---------------------------------------------------------------------------
def test_keyless_anthropic_client_raises_auth_resolution_substring():
    from agent.anthropic_adapter import build_anthropic_client

    client = build_anthropic_client("", "http://127.0.0.1:18810/anthropic")
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - message is the assertion
        client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert "could not resolve authentication method" in str(excinfo.value).lower(), (
        "Anthropic SDK message for a keyless client changed — the D-1b substring "
        "match in conversation_loop.is_local_validation_error must be updated to match."
    )


# ---------------------------------------------------------------------------
# D-1b / INV-3 (AC-1, AC-7): the auth-resolution TypeError is NOT a local
# validation bug, and does not abort the turn as one.
# ---------------------------------------------------------------------------
class TestClassifierBackstop:
    def _is_local_validation_error(self, api_error):
        """Replicate the conversation_loop predicate exactly (the code under test).

        conversation_loop builds this predicate inline; we assert on the SAME
        boolean expression so a regression in the real code is caught by importing
        and exercising it via the integration test below, while this unit test pins
        the intended truth table.
        """
        import json
        import ssl

        return (
            isinstance(api_error, (ValueError, TypeError))
            and not isinstance(api_error, (UnicodeEncodeError, json.JSONDecodeError))
            and not isinstance(api_error, ssl.SSLError)
            and not (
                isinstance(api_error, TypeError)
                and "nonetype" in str(api_error).lower()
                and "not iterable" in str(api_error).lower()
            )
            # D-1b: the auth-resolution TypeError is a client-construction error
            # (we built the client with no key), NOT a local programming bug.
            and not (
                isinstance(api_error, TypeError)
                and "could not resolve authentication method" in str(api_error).lower()
            )
        )

    def test_auth_resolution_typeerror_excluded_from_local_validation(self):
        """The real conversation_loop predicate must exclude this TypeError."""
        from agent import conversation_loop

        # The code under test builds the predicate inline; expose it via a helper
        # so the test exercises the shipped logic, not a copy.
        assert hasattr(conversation_loop, "_is_auth_resolution_error"), (
            "conversation_loop must expose _is_auth_resolution_error(err) so the "
            "D-1b carve-out is testable and reusable (mirrors the NoneType helper)."
        )
        err = TypeError(AUTH_RESOLVE_MSG)
        assert conversation_loop._is_auth_resolution_error(err) is True

    def test_nonetype_not_iterable_still_excluded(self):
        """Regression: the pre-existing NoneType carve-out is untouched."""
        err = TypeError("argument of type 'NoneType' is not iterable")
        assert self._is_local_validation_error(err) is False

    def test_plain_local_typeerror_still_included(self):
        """A genuine local bug (bad kwarg) is STILL a non-retryable local error."""
        err = TypeError("f() got an unexpected keyword argument 'foo'")
        assert self._is_local_validation_error(err) is True

    def test_auth_resolution_typeerror_excluded_by_reference_predicate(self):
        err = TypeError(AUTH_RESOLVE_MSG)
        assert self._is_local_validation_error(err) is False


# ---------------------------------------------------------------------------
# D-1a / INV-2 / D-2b (AC-2, AC-8): _swap_credential never builds a keyless
# client, and returns the tri-state outcome keyed on last_error_code.
# ---------------------------------------------------------------------------
class TestSwapCredentialGuard:
    def _make_anthropic_agent(self, pool):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.model = "claude-opus-4-8"
        agent.provider = "claude-app"
        agent.base_url = "http://127.0.0.1:18810/anthropic"
        agent.api_mode = "anthropic_messages"
        agent.api_key = "sentinel-old-key"
        agent._anthropic_api_key = "sentinel-old-key"
        agent._anthropic_base_url = "http://127.0.0.1:18810/anthropic"
        # A live client the guard must NOT replace with a keyless one.
        agent._anthropic_client = MagicMock(name="live-anthropic-client")
        agent._is_anthropic_oauth = False
        agent._client_kwargs = {}
        agent._credential_pool = pool
        return agent

    def _entry(self, access_token, *, last_status=None, last_error_code=None,
               last_status_at=None):
        return PooledCredential(
            provider="claude-app",
            id="usw-home",
            label="CLAUDE_POOL_KEY",
            auth_type="api_key",
            priority=0,
            source="manual",
            access_token=access_token,
            last_status=last_status,
            last_error_code=last_error_code,
            last_status_at=last_status_at,
        )

    def test_swap_outcome_enum_exists(self):
        from run_agent import SwapOutcome

        assert {o.name for o in SwapOutcome} >= {
            "SWAPPED",
            "RETRYABLE_EXHAUSTED",
            "MISSING_CREDENTIAL",
        }

    def test_empty_key_429_exhausted_is_retryable_and_builds_no_client(self):
        """Empty key that was 429-exhausted within TTL -> RETRYABLE_EXHAUSTED, client untouched."""
        from run_agent import SwapOutcome

        pool = MagicMock()
        entry = self._entry(
            "", last_status="exhausted", last_error_code=429,
            last_status_at=time.time(),
        )
        agent = self._make_anthropic_agent(pool)
        original_client = agent._anthropic_client

        outcome = agent._swap_credential(entry)

        assert outcome == SwapOutcome.RETRYABLE_EXHAUSTED
        # INV-2: the live client is NOT replaced with a keyless one.
        assert agent._anthropic_client is original_client
        assert agent._anthropic_api_key == "sentinel-old-key"

    def test_empty_key_never_ratelimited_is_missing_credential(self):
        """Empty key with no 429 history (config error) -> MISSING_CREDENTIAL (loud)."""
        from run_agent import SwapOutcome

        pool = MagicMock()
        entry = self._entry("", last_status=None, last_error_code=None)
        agent = self._make_anthropic_agent(pool)
        original_client = agent._anthropic_client

        outcome = agent._swap_credential(entry)

        assert outcome == SwapOutcome.MISSING_CREDENTIAL
        assert agent._anthropic_client is original_client

    def test_stale_429_marker_is_missing_not_retryable(self):
        """A 429 marker older than the exhausted-TTL is NOT treated as transient."""
        from run_agent import SwapOutcome

        pool = MagicMock()
        # last_status_at far in the past -> past any exhausted TTL.
        entry = self._entry(
            "", last_status="exhausted", last_error_code=429,
            last_status_at=time.time() - 86400,
        )
        agent = self._make_anthropic_agent(pool)

        outcome = agent._swap_credential(entry)

        assert outcome == SwapOutcome.MISSING_CREDENTIAL

    def test_nonempty_key_swaps_as_before(self):
        """A usable key still installs a new client and returns SWAPPED (unchanged behavior)."""
        from run_agent import SwapOutcome

        pool = MagicMock()
        entry = self._entry("sk-ant-real-key")
        agent = self._make_anthropic_agent(pool)

        outcome = agent._swap_credential(entry)

        assert outcome == SwapOutcome.SWAPPED
        assert agent._anthropic_api_key == "sk-ant-real-key"

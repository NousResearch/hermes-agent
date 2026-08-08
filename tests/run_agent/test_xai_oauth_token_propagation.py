"""E2E regression tests for xAI OAuth token propagation through _swap_credential.

SmelterLabs suspected a second-layer bug (PR #29344): ``_swap_credential`` might
update the credential pool entry but NOT propagate the new token into the live
OpenAI SDK client / HTTP transport instance.  If true, refresh-then-retry would
still use the stale token and the user would remain stuck.

Investigation showed the propagation IS correct.  These tests pin the invariant
so a future refactor can't silently break it.

Propagation chain (for non-Anthropic providers like xai-oauth):

    _swap_credential(entry)
      → self.api_key = runtime_key
      → self._client_kwargs["api_key"] = runtime_key      (line ~2910)
      → self._client_kwargs["base_url"] = runtime_base    (line ~2911)
      → self._replace_primary_openai_client(reason=…)     (line ~2913)
          → self.client = OpenAI(**self._client_kwargs)   (new client with new key)

Then on the next API call:

    _create_request_openai_client(reason=…)
      → request_kwargs = dict(self._client_kwargs)        (line ~2574)
      → OpenAI(**request_kwargs)                          (fresh per-request client)

Both the shared ``self.client`` AND the per-request client receive the refreshed
token.  The model-bounce workaround (``/model deepseek → /model grok-4.3``) was
never required for token propagation — it was required because the *classifier*
blocked the refresh path before cc93053b.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xai_agent(*, api_key: str = "stale-token-abc", base_url: str = "https://api.x.ai/v1"):
    """Build a minimal AIAgent wired for xAI OAuth codex_responses."""
    from run_agent import AIAgent

    agent = AIAgent(
        api_key=api_key,
        base_url=base_url,
        model="grok-4.3",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    agent.provider = "xai-oauth"
    agent._interrupt_requested = False
    return agent


def _make_refresh_pool(*, refreshed_key: str, refreshed_base: str):
    """Build a fake credential pool that returns a refreshed entry."""

    class _FakeEntry:
        access_token = refreshed_key
        base_url = refreshed_base
        id = "entry_refreshed"

        @property
        def runtime_api_key(self):
            return self.access_token

        @property
        def runtime_base_url(self):
            return self.base_url

    class _FakePool:
        def try_refresh_current(self):
            return _FakeEntry()

        def mark_exhausted_and_rotate(self, **kwargs):
            return None

        def has_available(self):
            return False

    return _FakePool()


def _bad_credentials_error_context():
    """The normalised error body for a stale xAI OAuth token."""
    return {
        "reason": (
            "The caller does not have permission to execute the specified operation"
        ),
        "message": (
            "The OAuth2 access token could not be validated. "
            "[WKE=unauthenticated:bad-credentials]"
        ),
    }


# ---------------------------------------------------------------------------
# Propagation invariant tests
# ---------------------------------------------------------------------------


class TestSwapCredentialPropagatesToken:
    """_swap_credential must propagate the new token into self._client_kwargs
    and self.api_key so subsequent API calls pick up the refreshed credential."""

    def test_swap_credential_updates_api_key_on_agent(self):
        """After _swap_credential, agent.api_key holds the new token."""
        agent = _make_xai_agent()
        entry = MagicMock()
        entry.runtime_api_key = None
        entry.access_token = "fresh-token-xyz"
        entry.runtime_base_url = None
        entry.base_url = "https://api.x.ai/v1"

        # Patch the client rebuild to avoid needing a real OpenAI constructor
        with patch.object(agent, "_replace_primary_openai_client", return_value=True):
            agent._swap_credential(entry)

        assert agent.api_key == "fresh-token-xyz"

    def test_swap_credential_updates_client_kwargs_api_key(self):
        """After _swap_credential, self._client_kwargs['api_key'] has the new token."""
        agent = _make_xai_agent()
        assert agent._client_kwargs.get("api_key") == "stale-token-abc"

        entry = MagicMock()
        entry.runtime_api_key = None
        entry.access_token = "fresh-token-xyz"
        entry.runtime_base_url = None
        entry.base_url = "https://api.x.ai/v1"

        with patch.object(agent, "_replace_primary_openai_client", return_value=True):
            agent._swap_credential(entry)

        assert agent._client_kwargs["api_key"] == "fresh-token-xyz"

    def test_swap_credential_updates_client_kwargs_base_url(self):
        """After _swap_credential, self._client_kwargs['base_url'] reflects any
        URL rotation (even though xAI typically keeps the same URL)."""
        agent = _make_xai_agent()

        entry = MagicMock()
        entry.runtime_api_key = None
        entry.access_token = "fresh-token-xyz"
        entry.runtime_base_url = None
        entry.base_url = "https://api.x.ai/v2"

        with patch.object(agent, "_replace_primary_openai_client", return_value=True):
            agent._swap_credential(entry)

        assert agent._client_kwargs["base_url"] == "https://api.x.ai/v2"

    def test_swap_credential_calls_replace_primary_openai_client(self):
        """_swap_credential must rebuild the shared client — the core of the
        propagation chain."""
        agent = _make_xai_agent()

        entry = MagicMock()
        entry.runtime_api_key = None
        entry.access_token = "fresh-token-xyz"
        entry.runtime_base_url = None
        entry.base_url = "https://api.x.ai/v1"

        with patch.object(
            agent, "_replace_primary_openai_client", return_value=True
        ) as mock_rebuild:
            agent._swap_credential(entry)

        mock_rebuild.assert_called_once_with(reason="credential_rotation")


class TestRecoveryPathPropagatesToken:
    """End-to-end: _recover_with_credential_pool → _swap_credential → propagation.

    These tests exercise the actual _swap_credential implementation (not a stub)
    and verify the refreshed token lands in the right fields for the next API call.
    """

    def test_recover_with_credential_pool_propagates_refreshed_token(self):
        """Full recovery path: stale 403 → refresh → _swap_credential → new token
        in self._client_kwargs and self.api_key.

        This is the E2E regression test for the suspected (but disproven)
        propagation gap.  If _swap_credential is ever refactored to skip
        _replace_primary_openai_client or stop updating _client_kwargs, this
        test will catch it.
        """
        from agent.error_classifier import FailoverReason

        agent = _make_xai_agent()
        assert agent.api_key == "stale-token-abc"

        pool = _make_refresh_pool(
            refreshed_key="refreshed-token-456",
            refreshed_base="https://api.x.ai/v1",
        )
        agent._credential_pool = pool

        # Patch the client rebuild to avoid needing a real OpenAI constructor.
        # This is the ONLY mock — _swap_credential itself runs for real.
        with patch.object(agent, "_replace_primary_openai_client", return_value=True):
            recovered, _retried_429 = agent._recover_with_credential_pool(
                status_code=403,
                has_retried_429=False,
                classified_reason=FailoverReason.auth,
                error_context=_bad_credentials_error_context(),
            )

        assert recovered is True, "Bad-credentials 403 must trigger recovery"

        # The key assertion: the new token propagated through the full chain.
        assert agent.api_key == "refreshed-token-456", (
            "agent.api_key must reflect the refreshed token after recovery"
        )
        assert agent._client_kwargs["api_key"] == "refreshed-token-456", (
            "self._client_kwargs['api_key'] must reflect the refreshed token "
            "so _create_request_openai_client picks it up on the next call"
        )

    def test_recover_with_credential_pool_propagates_rotated_base_url(self):
        """If refresh returns a different base_url (e.g. failover endpoint),
        it must propagate too."""
        from agent.error_classifier import FailoverReason

        agent = _make_xai_agent()

        pool = _make_refresh_pool(
            refreshed_key="refreshed-token-789",
            refreshed_base="https://api.x.ai/failover/v1",
        )
        agent._credential_pool = pool

        with patch.object(agent, "_replace_primary_openai_client", return_value=True):
            recovered, _ = agent._recover_with_credential_pool(
                status_code=403,
                has_retried_429=False,
                classified_reason=FailoverReason.auth,
                error_context=_bad_credentials_error_context(),
            )

        assert recovered is True
        assert agent._client_kwargs["base_url"] == "https://api.x.ai/failover/v1"

    def test_recover_with_credential_pool_rotation_fallback_propagates(self):
        """When try_refresh_current returns None (refresh failed) and rotation
        picks a different pool entry, the rotated entry's token must still
        propagate through _swap_credential."""
        from agent.error_classifier import FailoverReason

        agent = _make_xai_agent()

        class _RotatedEntry:
            access_token = "rotated-token-from-next-entry"
            base_url = "https://api.x.ai/v1"
            id = "entry_rotated"

            @property
            def runtime_api_key(self):
                return self.access_token

            @property
            def runtime_base_url(self):
                return self.base_url

        class _PoolRefreshFailed:
            def try_refresh_current(self):
                return None  # refresh failed

            def mark_exhausted_and_rotate(self, **kwargs):
                return _RotatedEntry()

            def has_available(self):
                return True

        agent._credential_pool = _PoolRefreshFailed()

        with patch.object(agent, "_replace_primary_openai_client", return_value=True):
            recovered, _ = agent._recover_with_credential_pool(
                status_code=403,
                has_retried_429=False,
                classified_reason=FailoverReason.auth,
                error_context=_bad_credentials_error_context(),
            )

        assert recovered is True
        assert agent.api_key == "rotated-token-from-next-entry"
        assert agent._client_kwargs["api_key"] == "rotated-token-from-next-entry"


class TestRequestClientPicksUpRefreshedToken:
    """Verify that _create_request_openai_client reads from _client_kwargs
    (the propagation sink) so the per-request client gets the refreshed token."""

    def test_create_request_client_uses_updated_client_kwargs(self):
        """After _swap_credential, _create_request_openai_client must create
        a client with the new api_key, not the stale one."""
        agent = _make_xai_agent()
        assert agent._client_kwargs["api_key"] == "stale-token-abc"

        # Simulate what _swap_credential does to _client_kwargs
        agent._client_kwargs["api_key"] = "refreshed-token-def"
        agent._client_kwargs["base_url"] = "https://api.x.ai/v1"

        # _create_request_openai_client copies _client_kwargs and builds
        # a new OpenAI client.  We mock OpenAI to capture what it receives.
        captured_kwargs = {}

        def _capture_openai(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with (
            patch("agent.agent_runtime_helpers._ra") as mock_ra,
        ):
            mock_ra.return_value.OpenAI = _capture_openai
            mock_ra.return_value.logger = MagicMock()

            # This is what the conversation loop calls on the next iteration
            client = agent._create_request_openai_client(
                reason="test_propagation",
                api_kwargs={},
            )

        # The per-request client was created with the refreshed token
        assert captured_kwargs.get("api_key") == "refreshed-token-def", (
            "_create_request_openai_client must forward the refreshed "
            "api_key from self._client_kwargs to the new OpenAI client"
        )
        assert captured_kwargs.get("max_retries") == 0, (
            "per-request clients must set max_retries=0 (agent loop owns retries)"
        )


class TestReplacePrimaryClientUsesUpdatedKwargs:
    """Verify _replace_primary_openai_client builds from updated _client_kwargs."""

    def test_replace_primary_client_gets_refreshed_key(self):
        """_replace_primary_openai_client creates a new self.client from
        self._client_kwargs — after _swap_credential updates it, the new
        client must carry the refreshed key."""
        agent = _make_xai_agent()

        # Simulate _swap_credential's updates
        agent._client_kwargs["api_key"] = "refreshed-via-swap"
        agent._client_kwargs["base_url"] = "https://api.x.ai/v1"

        captured_kwargs = {}

        def _capture_openai(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch("agent.agent_runtime_helpers._ra") as mock_ra:
            mock_ra.return_value.OpenAI = _capture_openai
            mock_ra.return_value.logger = MagicMock()

            result = agent._replace_primary_openai_client(reason="credential_rotation")

        assert result is True
        assert captured_kwargs.get("api_key") == "refreshed-via-swap", (
            "_replace_primary_openai_client must create the new client with "
            "the refreshed api_key from self._client_kwargs"
        )


class TestStaleTokenToSuccessE2E:
    """Full E2E scenario: stale token → 403 → refresh → retry → verify new token
    is in place for the next API call.  No stub on _swap_credential."""

    def test_full_stale_to_refreshed_flow(self):
        """Simulate the exact user-facing scenario:

        1. Agent starts with stale token
        2. API returns 403 with [WKE=unauthenticated:bad-credentials]
        3. _recover_with_credential_pool classifies as auth (not entitlement)
        4. Pool refreshes the token
        5. _swap_credential propagates into _client_kwargs
        6. The next _create_request_openai_client picks up the new token
        """
        from agent.error_classifier import FailoverReason

        agent = _make_xai_agent(api_key="stale-stale-stale")
        assert agent.api_key == "stale-stale-stale"

        pool = _make_refresh_pool(
            refreshed_key="brand-new-shiny-token",
            refreshed_base="https://api.x.ai/v1",
        )
        agent._credential_pool = pool

        # Step 2-4: Recovery
        with patch.object(agent, "_replace_primary_openai_client", return_value=True):
            recovered, _ = agent._recover_with_credential_pool(
                status_code=403,
                has_retried_429=False,
                classified_reason=FailoverReason.auth,
                error_context=_bad_credentials_error_context(),
            )

        assert recovered is True

        # Step 5: Propagation verified
        assert agent.api_key == "brand-new-shiny-token"
        assert agent._client_kwargs["api_key"] == "brand-new-shiny-token"

        # Step 6: Next request client picks up new token
        captured_kwargs = {}

        def _capture_openai(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch("agent.agent_runtime_helpers._ra") as mock_ra:
            mock_ra.return_value.OpenAI = _capture_openai
            mock_ra.return_value.logger = MagicMock()

            agent._create_request_openai_client(
                reason="retry_after_refresh",
                api_kwargs={},
            )

        assert captured_kwargs.get("api_key") == "brand-new-shiny-token", (
            "The per-request client after recovery must use the refreshed token"
        )

"""Regression tests for #52727: switch_model() must reload the credential pool
when the provider changes.

When the desktop model picker swaps providers mid-session, ``switch_model``
mutates ``agent.model``/``agent.provider``/``agent.base_url``/``agent.api_key``
but never refreshes ``agent._credential_pool``. The pool stays bound to the
ORIGINAL provider. ``recover_with_credential_pool`` then sees a
``pool.provider != agent.provider`` mismatch and skips rotation entirely —
the 401 burns the whole retry cycle with no recovery, and the original
provider's pool entry gets marked ``STATUS_EXHAUSTED`` and persisted in
``auth.json`` (issue #52727).

The fix reloads the pool via ``load_pool(new_provider)`` inside ``switch_model``
whenever the provider changes (or the pool is missing). This keeps the
defensive mismatch guard in ``recover_with_credential_pool`` intact while
making it impossible for a legitimate same-call switch to trip the guard.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import switch_model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_agent(current_provider, current_model, current_pool):
    """Bare agent object with the minimum attributes switch_model touches.

    Uses ``MagicMock`` so ``_anthropic_prompt_cache_policy(...)`` and
    ``_ensure_lmstudio_runtime_loaded()`` work without real implementations.
    """
    agent = MagicMock(name=f"Agent[{current_provider}]")
    agent.provider = current_provider
    agent.model = current_model
    agent.base_url = f"https://{current_provider}.example/v1"
    agent.api_key = f"{current_provider}-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="Client")
    agent._client_kwargs = {
        "api_key": "***",
        "base_url": f"https://{current_provider}.example/v1",
    }
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._is_anthropic_oauth = False
    agent._config_context_length = None
    agent._transport_cache = {}
    agent._cached_system_prompt = "cached-system-prompt"
    agent.context_compressor = None
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._credential_pool = current_pool
    # Real-ish instance methods that switch_model calls
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    agent._ensure_lmstudio_runtime_loaded = MagicMock()
    return agent


def _make_pool(provider):
    pool = MagicMock(name=f"Pool[{provider}]")
    pool.provider = provider
    return pool


# ---------------------------------------------------------------------------
# the fix
# ---------------------------------------------------------------------------


class TestSwitchModelReloadsCredentialPool:
    """Issue #52727: switch_model must refresh _credential_pool on provider change."""

    def test_switch_to_different_provider_reloads_pool(self):
        """opencode-go -> groq must replace the agent's pool with a groq pool."""
        old_pool = _make_pool("opencode-go")
        new_pool = _make_pool("groq")
        agent = _make_agent("opencode-go", "qwen-coder", old_pool)

        with patch(
            "agent.credential_pool.load_pool",
            return_value=new_pool,
        ) as load_pool_mock:
            switch_model(
                agent,
                new_model="llama-3.3-70b",
                new_provider="groq",
                api_key="groq-key-new",
                base_url="https://api.groq.com/openai/v1",
                api_mode="chat_completions",
            )

        # The agent's pool must now point at the NEW provider's pool.
        assert agent._credential_pool is new_pool, (
            f"agent._credential_pool was not reloaded on provider switch "
            f"(still references {old_pool.provider})"
        )
        assert agent._credential_pool.provider == "groq"
        assert agent._credential_pool is not old_pool
        # load_pool MUST have been called with the new provider.
        load_pool_mock.assert_called_once_with("groq")

    def test_switch_to_same_provider_does_not_reload_pool(self):
        """Re-selecting the current provider must NOT churn the pool reference."""
        existing_pool = _make_pool("opencode-go")
        agent = _make_agent("opencode-go", "qwen-coder", existing_pool)

        load_pool_mock = MagicMock(name="load_pool")

        with patch("agent.credential_pool.load_pool", load_pool_mock):
            switch_model(
                agent,
                new_model="qwen-coder",
                new_provider="opencode-go",  # SAME provider
                api_key="opencode-go-key-new",
                base_url="https://opencode.example/v1",
                api_mode="chat_completions",
            )

        # Pool must remain the same object — no churn for same-provider switch.
        assert agent._credential_pool is existing_pool
        load_pool_mock.assert_not_called()

    def test_switch_creates_pool_when_agent_had_none(self):
        """An agent without a pool that switches providers must acquire one."""
        new_pool = _make_pool("groq")
        agent = _make_agent("opencode-go", "qwen-coder", None)

        with patch("agent.credential_pool.load_pool", return_value=new_pool):
            switch_model(
                agent,
                new_model="llama-3.3-70b",
                new_provider="groq",
                api_key="groq-key-new",
                base_url="https://api.groq.com/openai/v1",
                api_mode="chat_completions",
            )

        assert agent._credential_pool is new_pool
        assert agent._credential_pool.provider == "groq"

    def test_recover_pool_mismatch_guard_no_longer_trips_after_switch(self):
        """End-to-end: after a provider switch, recover_with_credential_pool
        must not skip rotation due to a provider mismatch.

        Before the fix: pool.provider=='opencode-go', agent.provider=='groq'
        → mismatch guard fires → recovery skipped → 401 burns the cycle.
        After the fix: switch_model reloaded the pool to groq, so the guard
        is a no-op and recovery proceeds.
        """
        from agent.agent_runtime_helpers import recover_with_credential_pool
        from agent.error_classifier import FailoverReason

        old_pool = _make_pool("opencode-go")
        new_pool = _make_pool("groq")
        new_pool.mark_exhausted_and_rotate.return_value = None
        agent = _make_agent("opencode-go", "qwen-coder", old_pool)

        with patch("agent.credential_pool.load_pool", return_value=new_pool):
            switch_model(
                agent,
                new_model="llama-3.3-70b",
                new_provider="groq",
                api_key="groq-key-new",
                base_url="https://api.groq.com/openai/v1",
                api_mode="chat_completions",
            )

        # After the switch, the pool's provider matches the agent's provider.
        # A 429 on groq should now reach pool.mark_exhausted_and_rotate.
        recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.rate_limit,
        )

        # The guard would have returned (False, has_retried_429) early
        # without touching the pool. After the fix, the pool is consulted.
        assert new_pool.current.called, (
            "pool.current() was never called — mismatch guard short-circuited"
        )

    def test_pool_reload_failure_does_not_block_switch(self):
        """If load_pool raises (e.g. corrupt auth.json), switch_model must
        still complete — the pool will simply be missing for this turn, and
        the next turn can re-attempt. Crashing the whole switch is worse
        than a transient pool gap."""
        agent = _make_agent("opencode-go", "qwen-coder", _make_pool("opencode-go"))

        with patch(
            "agent.credential_pool.load_pool",
            side_effect=RuntimeError("simulated corrupt auth.json"),
        ):
            # Should NOT raise — pool reload failure is logged+swallowed.
            switch_model(
                agent,
                new_model="llama-3.3-70b",
                new_provider="groq",
                api_key="groq-key-new",
                base_url="https://api.groq.com/openai/v1",
                api_mode="chat_completions",
            )

        # The switch itself completed (provider/model updated) even though
        # the pool reload failed, without retaining the old provider's pool.
        assert agent.provider == "groq"
        assert agent.model == "llama-3.3-70b"
        assert agent._credential_pool is None

"""A temporarily-benched credential must not permanently disqualify a fallback.

When a provider's only credential is cooling down (e.g. DeepSeek 402 Insufficient
Balance, benched for an hour), ``resolve_provider_client`` returns ``None``. At the
call site that is indistinguishable from a provider the user never set up, so the
chain both logged it as "provider not configured" and added the entry to the
session-scoped ``_unavailable_fallback_keys`` memo — which is only cleared by a
config edit. A one-hour billing bench therefore removed the entry for the whole
lifetime of the cached agent.

``credential_pool`` already separates the two cases: ``has_credentials()`` means
the user configured something, ``has_available()`` means one is usable right now.
"""

from __future__ import annotations

from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = None
        return agent


def _mock_client(base_url="https://api.deepseek.com/v1", api_key="fb-key"):
    mock = type("Client", (), {})()
    mock.base_url = base_url
    mock.api_key = api_key
    mock.chat = type("Chat", (), {})()
    mock.chat.completions = type("Completions", (), {})()
    mock.chat.completions.create = lambda *args, **kwargs: None
    return mock


class _Pool:
    """Stand-in for credential_pool.CredentialPool."""

    def __init__(self, *, configured: bool, available: bool):
        self._configured = configured
        self._available = available

    def has_credentials(self) -> bool:
        return self._configured

    def has_available(self) -> bool:
        return self._available

    def next_available_at(self):
        return None


CHAIN = [
    {"provider": "deepseek", "model": "deepseek-v4-pro"},
    {"provider": "openai-codex", "model": "gpt-5.5"},
]
KEY = ("deepseek", "deepseek-v4-pro", "")


def _activate(agent, *, configured: bool, available: bool):
    def resolve(provider, **kwargs):
        if provider == "deepseek":
            return (None, None)  # benched or unconfigured — both look like this
        return (_mock_client(), "gpt-5.5")

    with (
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve),
        patch(
            "agent.credential_pool.load_pool",
            return_value=_Pool(configured=configured, available=available),
        ),
    ):
        return agent._try_activate_fallback(None)


class TestBenchedCredentialIsNotPermanentlyUnavailable:
    def test_benched_provider_is_not_memoized_as_unavailable(self):
        """Configured but cooling down: skip this turn, stay retryable later."""
        agent = _make_agent(fallback_model=list(CHAIN))

        assert _activate(agent, configured=True, available=False) is True
        assert agent.model == "gpt-5.5"
        assert KEY not in getattr(agent, "_unavailable_fallback_keys", set()), (
            "a temporarily-benched credential must stay retryable for later turns"
        )

    def test_genuinely_unconfigured_provider_is_still_memoized(self):
        """No credentials at all: keep the existing per-session suppression."""
        agent = _make_agent(fallback_model=list(CHAIN))

        assert _activate(agent, configured=False, available=False) is True
        assert KEY in getattr(agent, "_unavailable_fallback_keys", set()), (
            "an unconfigured provider should not be retried every turn"
        )


class TestUnavailableMemoExpires:
    """The memo must not outlive the condition that created it.

    Suppressing an unconfigured provider avoids retrying it every turn, but the
    memo was only cleared by a ``fallback_providers`` content edit. Credentials
    are commonly added via ``hermes auth`` (which writes auth.json, not
    config.yaml), so a provider configured mid-uptime stayed suppressed for the
    whole life of the cached agent.
    """

    def test_memo_entry_expires_so_newly_configured_provider_is_retried(self):
        agent = _make_agent(fallback_model=list(CHAIN))

        # First pass: genuinely unconfigured, so it gets memoized.
        assert _activate(agent, configured=False, available=False) is True
        memo = getattr(agent, "_unavailable_fallback_keys")
        assert KEY in memo

        # Simulate the memo aging past its retry window.
        import agent.chat_completion_helpers as ch

        ch._expire_unavailable_entry_for_test(memo, KEY)
        assert KEY not in memo, "an aged-out memo entry must not keep suppressing"

        # Credentials have since been added; the entry is reconsidered.
        agent2 = _make_agent(fallback_model=list(CHAIN))
        agent2._unavailable_fallback_keys = memo

        def resolve(provider, **kwargs):
            return (_mock_client(), "deepseek-v4-pro")

        with (
            patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve),
            patch(
                "agent.credential_pool.load_pool",
                return_value=_Pool(configured=True, available=True),
            ),
        ):
            assert agent2._try_activate_fallback(None) is True
        assert agent2.model == "deepseek-v4-pro", (
            "a provider configured after being memoized must become usable again"
        )

    def test_memo_still_suppresses_within_the_retry_window(self):
        """Back-to-back activations must not re-probe an unconfigured provider."""
        agent = _make_agent(fallback_model=list(CHAIN))
        assert _activate(agent, configured=False, available=False) is True

        calls = []

        def resolve(provider, **kwargs):
            calls.append(provider)
            return (_mock_client(), "gpt-5.5")

        agent._fallback_index = 0
        agent._fallback_activated = False
        with (
            patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve),
            patch(
                "agent.credential_pool.load_pool",
                return_value=_Pool(configured=False, available=False),
            ),
        ):
            agent._try_activate_fallback(None)
        assert "deepseek" not in calls, (
            "within the retry window the memo should short-circuit before resolving"
        )

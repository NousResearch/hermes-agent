"""Auth-failure provider failover (conversation loop).

A 401/403 that survives the per-provider credential-refresh attempt
(revoked OAuth, blocked/expired key, an account pinned to a dead/staging
endpoint) must escalate to the configured fallback chain instead of
thrashing on the same dead credential every turn.

Before the fix, the conversation loop's generic failover dispatch only
fired for ``{rate_limit, billing}`` reasons; ``auth`` / ``auth_permanent``
fell through to "switch providers manually" advice and never called
``_try_activate_fallback()``. These tests pin:

  1. 401/403 classify as auth (``classified.is_auth`` True).
  2. ``_try_activate_fallback`` advances the chain on an auth reason.
  3. The one-shot guard flag exists on TurnRetryState.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.error_classifier import classify_api_error, FailoverReason
from agent.turn_retry_state import TurnRetryState


def _make_agent(fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


def _auth_error(status=401, msg="Your API key is invalid, blocked or out of funds."):
    err = Exception(f"Error code: {status} - {msg}")
    err.status_code = status
    return err


class TestAuthErrorClassification:
    def test_401_is_auth(self):
        c = classify_api_error(_auth_error(401))
        assert c.reason in {FailoverReason.auth, FailoverReason.auth_permanent}
        assert c.is_auth is True

    def test_403_is_auth(self):
        c = classify_api_error(_auth_error(403, "forbidden"))
        assert c.is_auth is True

    def test_500_is_not_auth(self):
        err = Exception("Error code: 500 - internal server error")
        err.status_code = 500
        c = classify_api_error(err)
        assert c.is_auth is False


class TestAuthFailoverGuardFlag:
    def test_flag_defaults_false(self):
        assert TurnRetryState().auth_failover_attempted is False


class TestAuthFailoverActivation:
    """The decision the loop makes on a persistent auth failure: when a
    fallback chain exists and the guard hasn't fired, escalate to it."""

    def _should_failover(self, agent, classified, retry):
        # Mirror the exact gating condition added to conversation_loop.py.
        return (
            classified.is_auth
            and not retry.auth_failover_attempted
            and agent._fallback_index < len(agent._fallback_chain)
        )

    def test_auth_failover_fires_when_chain_present(self):
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        retry = TurnRetryState()
        classified = classify_api_error(_auth_error(401))
        assert self._should_failover(agent, classified, retry) is True
        # And the activation primitive actually advances on an auth reason.
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "gpt-4o"),
        ):
            advanced = agent._try_activate_fallback(reason=classified.reason)
        assert advanced is True
        assert agent._fallback_index == 1

    def test_no_failover_without_chain(self):
        """A user with no fallback configured (the common case for the
        original incident) does NOT failover — falls through to the
        existing terminal handling + troubleshooting advice."""
        agent = _make_agent(fallback_model=None)
        retry = TurnRetryState()
        classified = classify_api_error(_auth_error(401))
        assert self._should_failover(agent, classified, retry) is False

    def test_guard_blocks_repeat_failover(self):
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        retry = TurnRetryState()
        retry.auth_failover_attempted = True  # already escalated this attempt
        classified = classify_api_error(_auth_error(401))
        assert self._should_failover(agent, classified, retry) is False

    def test_non_auth_error_does_not_trigger_auth_failover(self):
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        retry = TurnRetryState()
        err = Exception("Error code: 500 - internal server error")
        err.status_code = 500
        classified = classify_api_error(err)
        assert self._should_failover(agent, classified, retry) is False


class TestFallbackBaseUrlAwareSkip:
    """A fallback entry that shares (provider, model) with the current backend
    but points at a DIFFERENT ``base_url`` is a distinct endpoint and must be
    activated, not skipped as a self-fallback (#54251). An entry with the SAME
    base_url is still skipped (preserves the self-loop guard)."""

    def test_failover_activates_for_same_provider_model_different_base_url(self):
        agent = _make_agent(
            fallback_model=[{
                "provider": "custom",
                "model": "my-model",
                "base_url": "https://b.example/v1",
                "api_key": "k",
            }]
        )
        # Primary backend: same provider+model, but a DIFFERENT endpoint.
        agent.provider = "custom"
        agent.model = "my-model"
        agent.base_url = "https://a.example/v1"
        agent._fallback_index = 0

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(base_url="https://b.example/v1"), "my-model"),
        ):
            advanced = agent._try_activate_fallback(reason=FailoverReason.auth)
        # Before the fix the entry is wrongly skipped as a self-fallback and,
        # with no further entries, activation returns False without advancing.
        assert advanced is True
        assert agent._fallback_index == 1

    def test_failover_skips_same_provider_model_same_base_url(self):
        agent = _make_agent(
            fallback_model=[{
                "provider": "custom",
                "model": "my-model",
                "base_url": "https://a.example/v1",
                "api_key": "k",
            }]
        )
        # Primary backend: identical provider+model+base_url — a true self-loop.
        agent.provider = "custom"
        agent.model = "my-model"
        agent.base_url = "https://a.example/v1"
        agent._fallback_index = 0

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(base_url="https://a.example/v1"), "my-model"),
        ):
            advanced = agent._try_activate_fallback(reason=FailoverReason.auth)
        # The lone entry is the same endpoint → skipped → chain exhausted.
        assert advanced is False

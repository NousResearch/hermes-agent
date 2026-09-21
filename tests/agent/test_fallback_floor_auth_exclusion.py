"""The implicit fallback floor is a rate-limit/overload/transport recovery
path, never an auth-failure escape.

``apply_fallback_floor`` (agent/agent_init.py) appends a last-resort entry to
every agent's ``_fallback_chain`` at init. Activating that implicit floor on an
auth/credential failure would silently switch the user's intended provider and
mask a broken key/OAuth instead of surfacing it. So:

  * an auth reason must NOT activate a floor entry (falls through to terminal
    auth handling), while
  * a rate-limit/overload/transport reason still uses the floor, and
  * a user-configured (non-floor) fallback still failovers on auth — that is an
    explicit opt-in, not a silent switch.

These tests deliberately do NOT neutralise the floor (unlike the pure
chain-semantics tests in test_provider_fallback.py); they exercise the real
floor-present behaviour.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.agent_init import apply_fallback_floor
from agent.error_classifier import FailoverReason

FLOOR = {"provider": "opencode-go", "model": "minimax-m3"}


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


class TestFloorTagging:
    def test_apply_fallback_floor_tags_entry(self):
        chain = apply_fallback_floor([], FLOOR)
        assert chain == [{"provider": "opencode-go", "model": "minimax-m3", "is_floor": True}]

    def test_configured_duplicate_is_not_tagged_as_floor(self):
        # A user who explicitly lists the floor backend keeps their own
        # (un-tagged) entry — an explicit opt-in, eligible for auth failover.
        configured = [{"provider": "opencode-go", "model": "minimax-m3"}]
        chain = apply_fallback_floor(configured, FLOOR)
        assert chain == configured
        assert not chain[0].get("is_floor")

    def test_floor_present_on_agent_without_config(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == [
            {"provider": "opencode-go", "model": "minimax-m3", "is_floor": True}
        ]


class TestFloorNotEligibleForAuth:
    def test_auth_does_not_activate_floor(self):
        agent = _make_agent(fallback_model=None)
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "minimax-m3"),
        ):
            assert agent._try_activate_fallback(reason=FailoverReason.auth) is False
        assert agent._fallback_activated is False

    def test_auth_permanent_does_not_activate_floor(self):
        agent = _make_agent(fallback_model=None)
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "minimax-m3"),
        ):
            assert agent._try_activate_fallback(reason=FailoverReason.auth_permanent) is False

    def test_pending_non_floor_fallback_false_when_floor_only(self):
        agent = _make_agent(fallback_model=None)
        assert agent._has_pending_non_floor_fallback() is False
        # The plain pending check still sees the floor (rate-limit may use it).
        assert agent._has_pending_fallback() is True


class TestFloorEligibleForNonAuth:
    def test_rate_limit_activates_floor(self):
        agent = _make_agent(fallback_model=None)
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "minimax-m3"),
        ):
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
        assert agent.model == "minimax-m3"
        assert agent._fallback_activated is True

    def test_overloaded_activates_floor(self):
        agent = _make_agent(fallback_model=None)
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "minimax-m3"),
        ):
            assert agent._try_activate_fallback(reason=FailoverReason.overloaded) is True


class TestConfiguredFallbackStillFailsOverOnAuth:
    def test_auth_uses_configured_entry_then_stops_at_floor(self):
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        # chain = [configured openai/gpt-4o, floor opencode-go/minimax-m3]
        assert agent._has_pending_non_floor_fallback() is True
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "gpt-4o"),
        ):
            # First auth failover: the explicit, user-configured entry.
            assert agent._try_activate_fallback(reason=FailoverReason.auth) is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 1
            # Second auth failover: only the floor remains -> refused.
            assert agent._try_activate_fallback(reason=FailoverReason.auth) is False

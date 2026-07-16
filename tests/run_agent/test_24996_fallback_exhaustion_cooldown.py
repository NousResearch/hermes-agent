"""Regression tests for fallback exhaustion cooldowns."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.error_classifier import FailoverReason
from agent.chat_completion_helpers import _FALLBACK_EXHAUSTED_COOLDOWN_S
from agent.cooldown_manager import (
    CooldownManager,
    build_cooldown_key,
    set_cooldown_manager,
)


def _make_agent(fallback_model=None, provider="openrouter"):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider=provider,
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


def _mock_response(content="ok", finish_reason="stop"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


class _RateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate limit exceeded")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "rate limit exceeded"}}


def _fresh_mgr():
    mgr = CooldownManager(storage_path=False)
    set_cooldown_manager(mgr)
    return mgr


class TestExhaustionArmsCooldown:
    def test_non_retryable_exhaustion_arms_cooldown(self):
        """Non-rate-limit exhaustion should arm a cooldown."""
        mgr = _fresh_mgr()
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(_mock_client(), "resolved")),
        ):
            assert agent._try_activate_fallback() is True   # -> entry 0
            assert agent._try_activate_fallback() is True   # -> entry 1
            assert agent._try_activate_fallback() is False

        status = mgr.get_cooldown_status()
        assert len(status["cooling"]) >= 1

    def test_exhaustion_uses_primary_snapshot_key_not_active_fallback_key(self):
        """A distinct fallback credential must not make next-turn primary
        restoration miss the exhaustion cooldown.
        """
        from agent.cooldown_manager import build_cooldown_key

        mgr = _fresh_mgr()
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        primary = agent._primary_runtime["api_key"]
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client(api_key="fallback-key"), "resolved")):
            assert agent._try_activate_fallback() is True
            assert agent.api_key == "fallback-key"
            assert agent._try_activate_fallback() is False

        primary_key = build_cooldown_key(agent._primary_runtime["provider"], primary, "rate_limit")
        fallback_key = build_cooldown_key(agent._primary_runtime["provider"], "fallback-key", "rate_limit")
        assert mgr.is_cooling(primary_key)
        assert not mgr.is_cooling(fallback_key)

    def test_no_chain_does_not_arm_cooldown(self):
        """An empty chain (no fallback configured) must not arm a cooldown."""
        mgr = _fresh_mgr()
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False
        status = mgr.get_cooldown_status()
        assert status["cooling"] == []

    def test_rate_limit_exhaustion_arms_cooldown(self):
        """A rate-limit failure arms an exponential cooldown via CooldownManager."""
        mgr = _fresh_mgr()
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(_mock_client(), "resolved")),
        ):
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is False

        status = mgr.get_cooldown_status()
        assert len(status["cooling"]) >= 1

    def test_cooldown_armed_only_once_for_same_provider(self):
        """Chain-switching while already on fallback must not re-arm cooldown
        for the primary provider."""
        mgr = _fresh_mgr()
        fbs = [
            {"provider": "openrouter", "model": "model-a"},
            {"provider": "anthropic", "model": "model-b"},
        ]
        agent = _make_agent(fallback_model=fbs, provider="custom")
        with (
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(_mock_client(), "resolved")),
        ):
            agent._try_activate_fallback(reason=FailoverReason.rate_limit)
            first_cooling = set(mgr.get_cooldown_status()["cooling"])

            agent._try_activate_fallback(reason=FailoverReason.rate_limit)
            second_cooling = set(mgr.get_cooldown_status()["cooling"])

        assert first_cooling == second_cooling


class TestPrimaryCooldownClearOnValidatedSuccess:
    def test_primary_success_path_clears_escalated_cooldown_and_resets_delay(self):
        mgr = _fresh_mgr()
        agent = _make_agent()
        primary_key = build_cooldown_key(
            agent._primary_runtime["provider"],
            agent._primary_runtime["api_key"],
            "rate_limit",
        )

        first_delay = mgr.mark_failure(primary_key, "rate_limit")
        second_delay = mgr.mark_failure(primary_key, "rate_limit")
        assert second_delay > first_delay
        assert mgr.get_all_states()[primary_key]["count"] == 2

        agent._interruptible_api_call = lambda api_kwargs: _mock_response("primary recovered")

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "primary recovered"
        assert primary_key not in mgr.get_all_states()

        reset_delay = mgr.mark_failure(primary_key, "rate_limit")
        assert reset_delay == first_delay
        assert mgr.get_all_states()[primary_key]["count"] == 1

    def test_fallback_success_does_not_clear_primary_cooldown_history(self):
        mgr = _fresh_mgr()
        agent = _make_agent(
            fallback_model=[{"provider": "openrouter", "model": "gpt-4o"}],
        )
        primary_key = build_cooldown_key(
            agent._primary_runtime["provider"],
            agent._primary_runtime["api_key"],
            "rate_limit",
        )

        first_delay = mgr.mark_failure(primary_key, "rate_limit")
        second_delay = mgr.mark_failure(primary_key, "rate_limit")
        mgr.mark_failure(agent._primary_runtime["provider"], "billing")
        assert second_delay > first_delay
        assert mgr.get_all_states()[primary_key]["count"] == 2
        assert mgr.get_all_states()[agent._primary_runtime["provider"]]["count"] == 1

        def _fake_api_call(api_kwargs):
            if not agent._fallback_activated:
                raise _RateLimitError()
            return _mock_response("fallback recovered")

        agent._interruptible_api_call = _fake_api_call

        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(api_key="fallback-key"), "gpt-4o"),
            ),
            patch("run_agent.time.sleep", return_value=None),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "fallback recovered"
        assert mgr.get_all_states()[primary_key]["count"] == 3
        assert mgr.get_all_states()[agent._primary_runtime["provider"]]["count"] == 1

        next_delay = mgr.mark_failure(primary_key, "rate_limit")
        assert next_delay > second_delay
        assert mgr.get_all_states()[primary_key]["count"] == 4

"""Follow-up for the cross-turn stream-stale circuit breaker (#58962).

The breaker latches: once ``_consecutive_stale_streams`` reaches the give-up
threshold, ``interruptible_streaming_api_call`` raises BEFORE any stream is
attempted — so the "reset on successful stream" path can never run again on
its own. The breaker's error message tells the user to "switch models …
then retry", and the provider-fallback chain swaps providers on the same
agent object, so BOTH swap paths must clear the streak or a healthy new
provider would keep short-circuiting forever:

- ``switch_model()``   (user-initiated /model swap)
- ``try_activate_fallback()``  (automatic provider fallback)
- ``restore_primary_runtime()``  (turn-start restore back to the primary)

The non-streaming sibling ``interruptible_api_call`` shares the same
breaker (guard at entry, bump on stale_call_kill, reset on success) —
quiet-mode / subagent sessions take that path and had the identical
infinite stale-retry class.
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_agent_openrouter():
    """Minimal openrouter agent (skips __init__), mirroring
    tests/run_agent/test_switch_model_rollback.py."""
    agent = AIAgent.__new__(AIAgent)

    agent.provider = "openrouter"
    agent.model = "x-ai/grok-4"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "or-key-original"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="OriginalClient")
    agent._client_kwargs = {
        "api_key": "or-key-original",
        "base_url": "https://openrouter.ai/api/v1",
    }
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None

    return agent


def _make_fallback_agent(fallback_model):
    """Full-constructor agent for the fallback path, mirroring
    tests/run_agent/test_24996_fallback_exhaustion_cooldown.py."""
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


def test_switch_model_resets_stale_streak():
    """A user-initiated /model swap must clear the latched streak so the new
    provider gets a real stream attempt instead of an instant short-circuit."""
    agent = _make_agent_openrouter()
    agent._consecutive_stale_streams = 7  # past any reasonable threshold

    agent._create_openai_client = MagicMock(return_value=MagicMock(name="NewClient"))

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        agent.switch_model(
            new_model="openai/gpt-5",
            new_provider="openrouter",
            api_key="or-key-new",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent._consecutive_stale_streams == 0


def test_switch_model_failure_does_not_reset_streak():
    """A failed swap rolls back — the agent is still on the wedged provider,
    so the breaker must stay latched (reset happens after the rebuild)."""
    agent = _make_agent_openrouter()
    agent._consecutive_stale_streams = 7

    def boom(*_a, **_kw):
        raise RuntimeError("simulated client build failure")

    agent._create_openai_client = boom

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        try:
            agent.switch_model(
                new_model="openai/gpt-5",
                new_provider="openrouter",
                api_key="or-key-new",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )
        except RuntimeError:
            pass

    assert agent._consecutive_stale_streams == 7


def test_switch_model_late_finalization_failure_keeps_stale_streak():
    """The breaker resets only after every fallible switch step commits."""
    agent = _make_agent_openrouter()
    agent._consecutive_stale_streams = 7
    agent._create_openai_client = MagicMock(return_value=MagicMock(name="NewClient"))
    agent._close_openai_client = MagicMock()

    class ExplodingFallback:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("late fallback finalization failure")

    setattr(agent, "_fallback_chain", [ExplodingFallback()])

    with (
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        pytest.raises(RuntimeError, match="late fallback finalization failure"),
    ):
        agent.switch_model(
            new_model="Hermes-4-70B",
            new_provider="nous",
            api_key="nous-key-new",
            base_url="https://inference-api.nousresearch.com/v1",
            api_mode="chat_completions",
        )

    assert agent._consecutive_stale_streams == 7


def test_fallback_activation_resets_stale_streak():
    """Automatic provider fallback swaps to a different backend; the streak
    measured the OLD provider and must not wedge the new one."""
    fbs = [{"provider": "openai", "model": "gpt-4o"}]
    agent = _make_fallback_agent(fallback_model=fbs)
    agent._consecutive_stale_streams = 7

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(_mock_client(), "resolved"),
    ):
        assert agent._try_activate_fallback() is True

    assert agent._consecutive_stale_streams == 0


def test_fallback_exhaustion_keeps_stale_streak():
    """When the chain is exhausted (no swap happened), the streak stays
    latched — the session is still wedged on the same provider."""
    agent = _make_fallback_agent(fallback_model=[])
    agent._consecutive_stale_streams = 7

    assert agent._try_activate_fallback() is False
    assert agent._consecutive_stale_streams == 7


def test_restore_primary_runtime_resets_stale_streak():
    """Turn-start restore back to the primary is the third provider-swap
    path: the streak measured the fallback we're leaving, so the restored
    primary must get a fresh attempt instead of an instant short-circuit."""
    fbs = [{"provider": "openai", "model": "gpt-4o"}]
    agent = _make_fallback_agent(fallback_model=fbs)

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(_mock_client(), "resolved"),
    ):
        assert agent._try_activate_fallback() is True

    # Streak accumulated while wedged on the FALLBACK provider.
    agent._consecutive_stale_streams = 7

    with patch("run_agent.OpenAI", return_value=MagicMock()):
        assert agent._restore_primary_runtime() is True

    assert agent._consecutive_stale_streams == 0


def test_no_fallback_restore_noop_keeps_stale_streak():
    """When no fallback was activated, restore is a no-op (returns False)
    and must NOT clear the streak — the session never left the wedged
    primary, so the breaker's cross-turn latch has to survive turn starts."""
    agent = _make_fallback_agent(fallback_model=[])
    agent._consecutive_stale_streams = 7

    assert agent._restore_primary_runtime() is False
    assert agent._consecutive_stale_streams == 7


class TestNonStreamingSibling:
    """interruptible_api_call carries the same breaker (#58962)."""

    def test_non_streaming_short_circuits_at_threshold(self, monkeypatch):
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")
        agent = _make_fallback_agent(fallback_model=[])
        agent._consecutive_stale_streams = 3

        with pytest.raises(RuntimeError, match="unresponsive"):
            agent._interruptible_api_call({})

        # The client is never touched on the short-circuit path.
        agent.client.chat.completions.create.assert_not_called()
        assert agent._consecutive_stale_streams == 3

    def test_non_streaming_success_resets_streak(self, monkeypatch):
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")
        agent = _make_fallback_agent(fallback_model=[])
        agent._consecutive_stale_streams = 2  # below threshold
        agent.client.chat.completions.create.return_value = MagicMock(
            name="resp", choices=[MagicMock()]
        )

        resp = agent._interruptible_api_call({"model": "m", "messages": []})
        assert resp is not None
        assert agent._consecutive_stale_streams == 0

"""Per-entry ``disable_streaming`` on fallback chain entries (#21522).

Some fallback backends advertise an OpenAI-compatible ``/chat/completions``
route but do not actually serve SSE; they answer ``stream=True`` with a
200 that carries no usable deltas.  ``try_activate_fallback()`` swaps model,
provider, base_url, api_mode, credential pool and timeouts, but never touched
the streaming decision, so the conversation loop kept dispatching streaming
requests to a backend that cannot serve them.

Scope is deliberately the fallback chain only: entries already flow through
``hermes_cli/fallback_config.py`` as verbatim dict copies, so the flag needs
no new configuration plumbing.  Named custom providers are a separate surface
and are not covered here.

These tests drive the real dispatch decision in
``agent/conversation_loop.py`` rather than inspecting ``_disable_streaming``,
and assert on the kwargs that reach ``chat.completions.create``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(fallback_model):
    """Agent with a live stream consumer, so streaming is the default path."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="primary/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
            stream_delta_callback=lambda text: None,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_completion(content="Final answer"):
    """A plain (non-streaming) ChatCompletion-shaped response."""
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="fallback/model", usage=None)


def _mock_fb_client():
    client = MagicMock()
    client.base_url = "https://fallback.example/v1"
    client.api_key = "fb-key"
    return client


def _activate_fallback(agent):
    """Run the real fallback swap with provider resolution stubbed out.

    Uses the rate-limit reason so the primary stays in cooldown and
    ``restore_primary_runtime()`` leaves the fallback active for the turn,
    the same state a session is in after the primary 429s.
    """
    from agent.error_classifier import FailoverReason

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(_mock_fb_client(), "fallback/model"),
    ):
        assert agent._try_activate_fallback(FailoverReason.rate_limit) is True
    assert agent.provider == "customshim"


def _run_turn(agent):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("hello")


def test_fallback_entry_disable_streaming_dispatches_non_streaming():
    """A fallback entry with ``disable_streaming: true`` must reach the
    provider as a plain completion request, and the turn must still return a
    normal non-streaming completion."""
    agent = _make_agent([
        {
            "provider": "customshim",
            "model": "fallback/model",
            "disable_streaming": True,
        }
    ])
    _activate_fallback(agent)

    fb_client = agent.client
    fb_client.chat.completions.create.return_value = _mock_completion()
    result = _run_turn(agent)

    assert fb_client.chat.completions.create.called
    kwargs = fb_client.chat.completions.create.call_args.kwargs
    assert "stream" not in kwargs or kwargs["stream"] is False
    assert result["final_response"] == "Final answer"
    assert result["completed"] is True


def test_fallback_entry_without_flag_still_streams():
    """Fallback entries that say nothing about streaming keep the existing
    streaming behavior. The flag is opt-in."""
    agent = _make_agent([
        {"provider": "customshim", "model": "fallback/model"}
    ])
    _activate_fallback(agent)

    fb_client = agent.client
    fb_client.chat.completions.create.return_value = iter([])
    _run_turn(agent)

    assert fb_client.chat.completions.create.call_args.kwargs["stream"] is True


def test_restoring_primary_re_enables_streaming():
    """The opt-out belongs to the fallback entry, not the session; the
    restored primary must stream again."""
    agent = _make_agent([
        {
            "provider": "customshim",
            "model": "fallback/model",
            "disable_streaming": True,
        }
    ])
    _activate_fallback(agent)
    assert agent._disable_streaming is True

    agent._rate_limited_until = 0
    assert agent._restore_primary_runtime() is True

    assert agent._disable_streaming is False
    assert agent.provider == "openrouter"


def test_reactive_disable_survives_primary_restore():
    """A provider that signalled "stream not supported" mid-session stays
    disabled across the restore; only config-driven opt-outs are undone."""
    agent = _make_agent([
        {"provider": "customshim", "model": "fallback/model"}
    ])
    _activate_fallback(agent)
    agent._disable_streaming = True  # as the streaming path sets it

    agent._rate_limited_until = 0
    assert agent._restore_primary_runtime() is True

    assert agent._disable_streaming is True

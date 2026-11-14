"""grok-4.7 has its own xAI entries: 500K context, reasoning effort up to ``xhigh``, the reasoning
stale-timeout floor — and no Priority Processing tier (that stays with the Grok 4.6 family)."""
import pytest

from agent.transports import get_transport


@pytest.mark.parametrize("prefix", ["", "xai/", "x-ai/"])
def test_grok_47_context_effort_and_timeout(prefix, tmp_path, monkeypatch):
    """A stale 256K cache entry from the ``grok-4`` catch-all is replaced, reasoning effort is
    sent, and the reasoning stale-timeout floor applies."""
    import agent.model_metadata as mm
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    model = f"{prefix}grok-4.7"
    base = "https://api.x.ai/v1"
    mm.save_context_length(model, base, 256_000)
    assert mm.get_model_context_length(model, base_url=base, provider="xai") == 500_000
    assert mm.grok_supports_reasoning_effort(model)
    assert get_reasoning_stale_timeout_floor(model) == 300.0


@pytest.mark.parametrize("model", ["grok-4.7", "xai/grok-4.7", "x-ai/grok-4.7"])
@pytest.mark.parametrize("effort,expected", [("low", "low"), ("high", "high"), ("xhigh", "xhigh"), ("ultra", "xhigh")])
def test_grok_47_keeps_xhigh_but_not_the_priority_tier(model, effort, expected):
    import agent.transports.codex  # registers the transport

    kw = get_transport("codex_responses").build_kwargs(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        is_xai_responses=True,
        reasoning_config={"effort": effort},
        request_overrides={"service_tier": "priority"},
    )

    assert kw["reasoning"] == {"effort": expected}
    assert "service_tier" not in kw

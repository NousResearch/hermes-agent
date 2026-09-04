"""Focused real-path coverage for the GPT-6 Astra baseline contract."""

from types import SimpleNamespace


def test_explicit_astra_resolves_and_uses_official_responses(monkeypatch, tmp_path):
    """A fresh profile resolves metadata and routes the official endpoint without live I/O."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *args, **kwargs: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *args, **kwargs: [])

    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-6-astra",
        provider="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        platform="cli",
        max_iterations=2,
        quiet_mode=True,
        skip_memory=True,
    )

    assert agent.api_mode == "codex_responses"
    assert agent.context_compressor.context_length == 1_050_000
    kwargs = agent._get_transport().build_kwargs(
        model=agent.model,
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        provider=agent.provider,
        base_url=agent.base_url,
        reasoning_config={"enabled": False, "effort": "none"},
    )
    assert kwargs["reasoning"]["effort"] == "low"
    assert kwargs["prompt_cache_options"] == {"ttl": "30m"}

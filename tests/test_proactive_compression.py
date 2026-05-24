from types import SimpleNamespace

from agent.model_metadata import estimate_request_tokens_rough


def _agent(*, context_length=100_000, threshold_tokens=75_000, enabled=True, ratio=0.20):
    compressor = SimpleNamespace(
        context_length=context_length,
        threshold_tokens=threshold_tokens,
    )
    return SimpleNamespace(
        compression_enabled=enabled,
        context_compressor=compressor,
        tools=[{"type": "function", "function": {"name": "noop", "description": "x" * 40}}],
        _proactive_compression_enabled=True,
        _proactive_compression_next_turn_reserve_ratio=ratio,
    )


def test_proactive_compression_pressure_uses_context_window_ratio():
    from agent.conversation_compression import estimate_proactive_compression_pressure

    messages = [{"role": "user", "content": "x" * 55_000 * 4}]
    system_prompt = "system"
    agent = _agent(context_length=100_000, threshold_tokens=75_000, ratio=0.20)

    pressure = estimate_proactive_compression_pressure(
        agent, messages, system_prompt=system_prompt
    )

    current = estimate_request_tokens_rough(
        messages, system_prompt=system_prompt, tools=agent.tools
    )
    assert pressure["current_tokens"] == current
    assert pressure["reserve_tokens"] == 20_000
    assert pressure["projected_tokens"] == current + 20_000
    assert pressure["threshold_tokens"] == 75_000
    assert pressure["should_compress"] is True


def test_proactive_compression_pressure_respects_disabled_setting():
    from agent.conversation_compression import estimate_proactive_compression_pressure

    messages = [{"role": "user", "content": "x" * 70_000 * 4}]
    agent = _agent(enabled=False, ratio=0.20)

    pressure = estimate_proactive_compression_pressure(agent, messages)

    assert pressure["enabled"] is False
    assert pressure["should_compress"] is False


def test_proactive_reserve_ratio_is_clamped():
    from agent.conversation_compression import estimate_proactive_compression_pressure

    messages = [{"role": "user", "content": "x"}]
    agent = _agent(context_length=100_000, threshold_tokens=75_000, ratio=2.0)

    pressure = estimate_proactive_compression_pressure(agent, messages)

    assert pressure["reserve_ratio"] == 0.5
    assert pressure["reserve_tokens"] == 50_000

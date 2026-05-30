from types import SimpleNamespace

from agent.anthropic_adapter import convert_messages_to_anthropic
from agent.orchestration_mode import (
    MODE_ENTER,
    MODE_EXIT,
    is_enabled,
    reminder_for_next_turn,
    set_enabled,
    supports_mid_conversation_system_messages,
)


def test_orchestration_enable_sets_xhigh_and_entry_notice():
    agent = SimpleNamespace(reasoning_config={"enabled": True, "effort": "low"})

    set_enabled(agent, True)

    assert is_enabled(agent)
    assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}
    assert reminder_for_next_turn(agent) == MODE_ENTER
    assert reminder_for_next_turn(agent) == ""


def test_orchestration_disable_restores_reasoning_and_exit_notice():
    previous = {"enabled": True, "effort": "low"}
    agent = SimpleNamespace(reasoning_config=previous)

    set_enabled(agent, True)
    reminder_for_next_turn(agent)
    set_enabled(agent, False)

    assert not is_enabled(agent)
    assert agent.reasoning_config == previous
    assert reminder_for_next_turn(agent) == MODE_EXIT


def test_mid_system_support_is_narrow_to_native_opus_48():
    good = SimpleNamespace(
        api_mode="anthropic_messages",
        provider="anthropic",
        model="claude-opus-4-8",
        base_url="https://api.anthropic.com",
    )
    bad = SimpleNamespace(
        api_mode="anthropic_messages",
        provider="openrouter",
        model="claude-opus-4-8",
        base_url="https://openrouter.ai/api/v1",
    )

    assert supports_mid_conversation_system_messages(good)
    assert not supports_mid_conversation_system_messages(bad)


def test_anthropic_converter_preserves_mid_system_for_opus_48_native():
    system, messages = convert_messages_to_anthropic(
        [
            {"role": "system", "content": "root"},
            {"role": "user", "content": "do it"},
            {"role": "system", "content": "mid reminder"},
        ],
        base_url="https://api.anthropic.com",
        model="claude-opus-4-8",
    )

    assert system == "root"
    assert messages[-1] == {"role": "system", "content": "mid reminder"}


def test_anthropic_converter_keeps_legacy_single_system_elsewhere():
    system, messages = convert_messages_to_anthropic(
        [
            {"role": "system", "content": "root"},
            {"role": "user", "content": "do it"},
            {"role": "system", "content": "later"},
        ],
        base_url="https://openrouter.ai/api/v1",
        model="claude-opus-4-8",
    )

    assert system == "later"
    assert all(m.get("role") != "system" for m in messages)

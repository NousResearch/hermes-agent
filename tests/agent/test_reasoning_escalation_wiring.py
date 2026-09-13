def test_wire_reasoning_config_honours_pinned_turn_override():
    from types import SimpleNamespace

    from agent.chat_completion_helpers import _reasoning_config_for_wire

    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-6-astra",
        reasoning_config={"enabled": True, "effort": "medium"},
        _turn_reasoning_config_override={"enabled": True, "effort": "high"},
        _turn_reasoning_escalation_target=("openai-codex", "gpt-6-astra"),
    )
    assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "high"}

    # A fallback route (different provider/model tuple) must not inherit the escalation.
    agent.model = "gpt-5.6-luna"
    assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "medium"}

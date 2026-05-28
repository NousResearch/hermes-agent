from agent.reasoning_efforts import resolve_reasoning_effort


def test_resolve_reasoning_effort_passes_allowed_value():
    assert resolve_reasoning_effort(
        "HIGH",
        allowed={"low", "medium", "high"},
    ) == "high"


def test_resolve_reasoning_effort_applies_alias_before_allowed_check():
    assert resolve_reasoning_effort(
        "max",
        allowed={"low", "medium", "high", "xhigh"},
        aliases={"max": "xhigh"},
    ) == "xhigh"


def test_resolve_reasoning_effort_returns_none_for_unsupported_value():
    assert resolve_reasoning_effort(
        "max",
        allowed={"low", "medium", "high"},
    ) is None


def test_resolve_reasoning_effort_ignores_blank_values():
    assert resolve_reasoning_effort("", allowed={"low"}) is None
    assert resolve_reasoning_effort(None, allowed={"low"}) is None

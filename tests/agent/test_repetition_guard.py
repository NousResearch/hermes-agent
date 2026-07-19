"""Assistant-turn repetition guard primitive tests."""

from agent.repetition_guard import (
    AssistantRepetitionGuard,
    RepetitionGuardConfig,
    assistant_turn_signature,
)


def _turn(content="Verifying the scene:", args='{"action": "screenshot"}'):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "browser", "arguments": args},
            }
        ],
    }


def test_signature_ignores_tool_call_ids_and_reasoning():
    a = _turn()
    a["reasoning"] = "first attempt"
    b = _turn()
    b["reasoning"] = "second attempt, different thoughts"
    b["tool_calls"][0]["id"] = "call_zzz999"

    assert assistant_turn_signature(a) == assistant_turn_signature(b)


def test_signature_distinguishes_different_arguments():
    a = _turn(args='{"action": "screenshot"}')
    b = _turn(args='{"action": "click", "x": 10}')

    assert assistant_turn_signature(a) != assistant_turn_signature(b)


def test_signature_flattens_list_content():
    a = _turn(content=[{"type": "text", "text": "hello"}])
    b = _turn(content="hello")

    assert assistant_turn_signature(a) == assistant_turn_signature(b)


def test_default_config_nudges_but_never_aborts():
    guard = AssistantRepetitionGuard()

    assert guard.observe(_turn()) == "ok"
    assert guard.observe(_turn()) == "ok"
    assert guard.observe(_turn()) == "nudge"
    # Post-nudge repeats keep nudging; abort stays off by default.
    for _ in range(10):
        assert guard.observe(_turn()) == "nudge"


def test_varied_turns_never_trigger():
    guard = AssistantRepetitionGuard()

    for i in range(20):
        assert guard.observe(_turn(args=f'{{"step": {i}}}')) == "ok"


def test_abort_fires_after_post_nudge_repeats_when_enabled():
    guard = AssistantRepetitionGuard(
        RepetitionGuardConfig(abort_enabled=True)
    )

    assert guard.observe(_turn()) == "ok"
    assert guard.observe(_turn()) == "ok"
    assert guard.observe(_turn()) == "nudge"
    assert guard.observe(_turn()) == "nudge"
    assert guard.observe(_turn()) == "abort"


def test_progress_between_repeats_defers_the_nudge():
    guard = AssistantRepetitionGuard()

    assert guard.observe(_turn()) == "ok"
    assert guard.observe(_turn(args='{"action": "click"}')) == "ok"
    assert guard.observe(_turn()) == "ok"
    assert guard.observe(_turn(args='{"action": "type"}')) == "ok"
    # Third identical within the 5-turn window → nudge.
    assert guard.observe(_turn()) == "nudge"


def test_old_repeats_age_out_of_the_window():
    guard = AssistantRepetitionGuard()

    assert guard.observe(_turn()) == "ok"
    assert guard.observe(_turn()) == "ok"
    for i in range(5):
        assert guard.observe(_turn(args=f'{{"step": {i}}}')) == "ok"
    # The two early repeats have aged out — this is 1 of 5, not 3 of 5.
    assert guard.observe(_turn()) == "ok"


def test_reset_for_turn_clears_state():
    guard = AssistantRepetitionGuard()

    guard.observe(_turn())
    guard.observe(_turn())
    guard.observe(_turn())
    guard.reset_for_turn()
    assert guard.observe(_turn()) == "ok"
    assert guard.observe(_turn()) == "ok"


def test_disabled_guard_returns_ok():
    guard = AssistantRepetitionGuard(RepetitionGuardConfig(enabled=False))

    for _ in range(10):
        assert guard.observe(_turn()) == "ok"


def test_config_from_mapping_parses_and_falls_back():
    cfg = RepetitionGuardConfig.from_mapping(
        {
            "enabled": "yes",
            "window": 7,
            "nudge_after": 2,
            "abort_enabled": True,
            "abort_after": "3",
        }
    )
    assert cfg.enabled is True
    assert cfg.window == 7
    assert cfg.nudge_after == 2
    assert cfg.abort_enabled is True
    assert cfg.abort_after == 3

    defaults = RepetitionGuardConfig.from_mapping(None)
    assert defaults.enabled is True
    assert defaults.abort_enabled is False
    assert defaults.window == 5
    assert defaults.nudge_after == 3
    assert defaults.abort_after == 2

    garbage = RepetitionGuardConfig.from_mapping(
        {"window": -3, "nudge_after": "lots", "abort_after": 0}
    )
    assert garbage.window == 5
    assert garbage.nudge_after == 3
    assert garbage.abort_after == 2

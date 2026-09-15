"""Unit tests for the plugin ``halt_turn`` directive parser (#107665)."""

from hermes_cli.plugins import (
    get_plugin_halt_turn_response,
    get_pre_tool_call_directive,
)


def test_valid_halt_turn_from_response():
    results = [{"action": "halt_turn", "response": "stopped by plugin"}]
    assert get_plugin_halt_turn_response(results) == "stopped by plugin"


def test_valid_halt_turn_falls_back_to_message():
    results = [{"action": "halt_turn", "message": "halt via message"}]
    assert get_plugin_halt_turn_response(results) == "halt via message"


def test_response_wins_over_message():
    results = [
        {"action": "halt_turn", "response": "from response", "message": "from message"},
    ]
    assert get_plugin_halt_turn_response(results) == "from response"


def test_first_valid_halt_wins():
    results = [
        {"action": "halt_turn", "response": ""},
        {"action": "halt_turn", "response": "first valid"},
        {"action": "halt_turn", "response": "second valid"},
    ]
    assert get_plugin_halt_turn_response(results) == "first valid"


def test_ignore_whitespace_only_response():
    assert get_plugin_halt_turn_response([{"action": "halt_turn", "response": "   "}]) is None
    assert get_plugin_halt_turn_response([{"action": "halt_turn", "message": "\n"}]) is None


def test_ignore_unknown_action():
    assert get_plugin_halt_turn_response([{"action": "block", "message": "nope"}]) is None
    assert get_plugin_halt_turn_response([{"action": "stop", "response": "nope"}]) is None
    assert get_plugin_halt_turn_response([{"response": "nope"}]) is None


def test_action_is_stripped():
    results = [{"action": "  halt_turn  ", "response": "padded action"}]
    assert get_plugin_halt_turn_response(results) == "padded action"


def test_ignore_non_dict_and_non_str():
    assert get_plugin_halt_turn_response([None, "halt_turn", {"action": "halt_turn", "response": 1}]) is None


def test_pre_tool_call_directive_accepts_halt_turn(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kwargs: [{"action": "halt_turn", "response": "halt now"}],
    )
    assert get_pre_tool_call_directive("web_search", {}) == ("halt_turn", "halt now")


def test_pre_tool_call_invalid_halt_is_not_block(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kwargs: [{"action": "halt_turn", "response": ""}],
    )
    assert get_pre_tool_call_directive("web_search", {}) == (None, None)

import copy

import model_tools
from tools.registry import registry
from tools.send_message_tool import SEND_MESSAGE_SCHEMA


def _send_message_schema(tool_defs):
    for tool_def in tool_defs:
        function = tool_def.get("function", {})
        if function.get("name") == "send_message":
            return function
    raise AssertionError("send_message schema not found")


def _force_send_message_available(monkeypatch):
    entry = registry.get_entry("send_message")
    assert entry is not None
    monkeypatch.setattr(entry, "check_fn", lambda: True)


def _clear_tool_caches():
    model_tools._clear_tool_defs_cache()
    try:
        from tools.registry import invalidate_check_fn_cache

        invalidate_check_fn_cache()
    except Exception:
        pass


def test_send_message_components_schema_hidden_without_discord(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(model_tools, "_discord_components_schema_enabled", lambda: False)
    _force_send_message_available(monkeypatch)
    _clear_tool_caches()

    tool_defs = model_tools.get_tool_definitions(enabled_toolsets=["messaging"], quiet_mode=True)

    properties = _send_message_schema(tool_defs)["parameters"]["properties"]
    assert "components" not in properties


def test_send_message_components_schema_restored_when_discord_enabled(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    _force_send_message_available(monkeypatch)
    _clear_tool_caches()

    monkeypatch.setattr(model_tools, "_discord_components_schema_enabled", lambda: False)
    without_discord = model_tools.get_tool_definitions(enabled_toolsets=["messaging"], quiet_mode=True)
    assert "components" not in _send_message_schema(without_discord)["parameters"]["properties"]

    model_tools._clear_tool_defs_cache()
    monkeypatch.setattr(model_tools, "_discord_components_schema_enabled", lambda: True)
    with_discord = model_tools.get_tool_definitions(enabled_toolsets=["messaging"], quiet_mode=True)

    assert "components" in _send_message_schema(with_discord)["parameters"]["properties"]
    assert "components" in SEND_MESSAGE_SCHEMA["parameters"]["properties"]


def test_strip_send_message_components_schema_does_not_mutate_input():
    original = {
        "type": "function",
        "function": copy.deepcopy(SEND_MESSAGE_SCHEMA),
    }

    stripped = model_tools._without_send_message_components_schema(original)

    assert "components" not in stripped["function"]["parameters"]["properties"]
    assert "components" in original["function"]["parameters"]["properties"]

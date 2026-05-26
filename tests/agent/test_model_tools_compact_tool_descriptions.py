from copy import deepcopy

import model_tools


def _demo_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "demo_tool",
                "description": "Please use this tool to carefully inspect the current page and helpfully return the result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Exact path"},
                    },
                },
            },
        }
    ]


def test_get_tool_definitions_compaction_disabled(monkeypatch):
    base = _demo_tools()
    monkeypatch.setattr(model_tools, "_compact_tool_descriptions_enabled", lambda: False)
    monkeypatch.setattr(model_tools.registry, "get_definitions", lambda tools_to_include, quiet=False: deepcopy(base))
    monkeypatch.setattr(model_tools, "resolve_toolset", lambda name: ["demo_tool"])
    monkeypatch.setattr(model_tools, "validate_toolset", lambda name: True)

    out = model_tools._compute_tool_definitions(enabled_toolsets=["demo"], disabled_toolsets=[], quiet_mode=True)
    assert out[0]["function"]["description"] == base[0]["function"]["description"]
    assert out[0]["function"]["name"] == "demo_tool"


def test_get_tool_definitions_compaction_enabled(monkeypatch):
    base = _demo_tools()
    monkeypatch.setattr(model_tools, "_compact_tool_descriptions_enabled", lambda: True)
    monkeypatch.setattr(model_tools.registry, "get_definitions", lambda tools_to_include, quiet=False: deepcopy(base))
    monkeypatch.setattr(model_tools, "resolve_toolset", lambda name: ["demo_tool"])
    monkeypatch.setattr(model_tools, "validate_toolset", lambda name: True)

    out = model_tools._compute_tool_definitions(enabled_toolsets=["demo"], disabled_toolsets=[], quiet_mode=True)
    assert out[0]["function"]["name"] == "demo_tool"
    assert out[0]["function"]["parameters"] == base[0]["function"]["parameters"]
    assert len(out[0]["function"]["description"]) < len(base[0]["function"]["description"])


def test_get_tool_definitions_cache_key_reflects_toggle(monkeypatch):
    calls = {"n": 0}
    base = _demo_tools()

    monkeypatch.setattr(model_tools.registry, "_generation", 999, raising=False)
    monkeypatch.setattr(model_tools.registry, "get_definitions", lambda tools_to_include, quiet=False: calls.__setitem__("n", calls["n"] + 1) or deepcopy(base))
    monkeypatch.setattr(model_tools, "resolve_toolset", lambda name: ["demo_tool"])
    monkeypatch.setattr(model_tools, "validate_toolset", lambda name: True)
    monkeypatch.setattr(model_tools, "_tool_defs_cache", {})
    monkeypatch.setattr(model_tools, "_compact_tool_descriptions_enabled", lambda: False)

    first = model_tools.get_tool_definitions(enabled_toolsets=["demo"], quiet_mode=True)
    second = model_tools.get_tool_definitions(enabled_toolsets=["demo"], quiet_mode=True)
    assert calls["n"] == 1
    assert first[0]["function"]["description"] == second[0]["function"]["description"]

    monkeypatch.setattr(model_tools, "_compact_tool_descriptions_enabled", lambda: True)
    third = model_tools.get_tool_definitions(enabled_toolsets=["demo"], quiet_mode=True)
    assert calls["n"] == 2
    assert len(third[0]["function"]["description"]) < len(base[0]["function"]["description"])

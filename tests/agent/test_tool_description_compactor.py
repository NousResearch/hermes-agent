from agent.tool_description_compactor import compact_description, compact_tool_definitions


def test_compact_description_removes_filler_but_keeps_core_instruction():
    text = "Please use this tool to carefully inspect the current page and helpfully return the result."
    out = compact_description(text)
    assert len(out) < len(text)
    assert "Please" not in out
    assert "carefully" not in out
    assert "helpfully" not in out
    assert "tool" in out
    assert "page" in out


def test_compact_description_preserves_url_path_inline_code_and_flag():
    text = (
        "Use this tool to inspect `/tmp/demo.py` and open https://example.com/docs "
        "with `python run.py` using --verbose for additional detail."
    )
    out = compact_description(text)
    assert "`/tmp/demo.py`" in out
    assert "https://example.com/docs" in out
    assert "`python run.py`" in out
    assert "--verbose" in out


def test_compact_description_returns_original_when_too_short():
    text = "Read file path exactly."
    assert compact_description(text) == text


def test_compact_tool_definitions_only_changes_description_field():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "demo_tool",
                "description": "Please use this tool to carefully inspect the current page and helpfully return the result.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Exact path"}},
                },
            },
        }
    ]
    out = compact_tool_definitions(tools)
    assert out[0]["function"]["name"] == "demo_tool"
    assert out[0]["function"]["parameters"] == tools[0]["function"]["parameters"]
    assert len(out[0]["function"]["description"]) < len(tools[0]["function"]["description"])
    assert tools[0]["function"]["description"].startswith("Please use this tool")

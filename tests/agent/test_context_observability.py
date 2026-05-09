"""Tests for agent/context_observability.py."""

from agent.context_observability import build_context_breakdown, format_context_breakdown


def test_breakdown_separates_system_messages_and_tool_schemas():
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "tool", "tool_name": "read_file", "content": "large tool result"},
    ]
    tools = [
        {"type": "function", "function": {"name": "read_file", "description": "Read files"}},
        {"type": "function", "function": {"name": "web_search", "description": "Search web"}},
    ]

    report = build_context_breakdown(
        messages,
        tools=tools,
        system_prompt="Injected system prefix",
        model="openai/gpt-5.5",
        provider="openrouter",
        context_window=100_000,
        enabled_toolsets=["file", "web"],
    )

    section_names = [section.name for section in report.sections]
    assert section_names == [
        "system_prompt",
        "system_messages",
        "conversation_messages",
        "tool_results",
        "tool_schemas",
    ]
    assert report.total_estimated_tokens == sum(section.estimated_tokens for section in report.sections)
    assert report.tool_count == 2
    assert report.enabled_toolsets == ["file", "web"]
    assert report.cache.supported is True
    assert report.cache.configured is False
    assert report.sections[-1].cacheability == "stable"
    assert report.sections[-2].cacheability == "volatile"


def test_breakdown_warns_when_tool_schemas_dominate():
    messages = [{"role": "user", "content": "tiny"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{idx}",
                "description": "x" * 400,
                "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
            },
        }
        for idx in range(5)
    ]

    report = build_context_breakdown(messages, tools=tools)

    assert any("Tool schemas dominate" in warning for warning in report.warnings)
    assert any("narrower toolsets" in suggestion for suggestion in report.suggestions)


def test_format_context_breakdown_outputs_safe_summary_without_raw_content():
    messages = [
        {"role": "system", "content": "secret system prompt with sk-test-should-not-print"},
        {"role": "user", "content": "my password is hunter2"},
    ]
    report = build_context_breakdown(messages, tools=[], context_window=1000)

    output = format_context_breakdown(report)

    assert "Context Composition" in output
    assert "system_messages" in output
    assert "conversation_messages" in output
    assert "sk-test-should-not-print" not in output
    assert "hunter2" not in output
    assert "Estimated request" in output

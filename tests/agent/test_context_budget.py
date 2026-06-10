"""Tests for context budget diagnostics."""

from agent.context_budget import build_context_budget_report, format_context_budget_report


def test_context_budget_splits_memory_and_tool_schemas():
    messages = [
        {"role": "system", "content": "system rules" * 100},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer" * 200},
        {
            "role": "user",
            "content": (
                "current question\n\n"
                "<memory-context>\nremembered fact " + ("x" * 800) + "\n</memory-context>"
            ),
        },
    ]
    tools = [{"type": "function", "function": {"name": "big_tool", "description": "d" * 1000}}]

    report = build_context_budget_report(
        api_messages=messages,
        tools=tools,
        model="test-model",
        provider="test-provider",
        context_length=100_000,
        session_id="sess",
        api_call_count=2,
    )

    assert report["total_tokens"] > 0
    assert report["tool_count"] == 1
    assert report["buckets"]["system_prompt"] > 0
    assert report["buckets"]["memory_context"] > 0
    assert report["buckets"]["tool_schemas"] > 0
    assert report["memory_context_chars"] > 0

    lines = format_context_budget_report(report)
    rendered = "\n".join(lines)
    assert "Context budget:" in rendered
    assert "memory_context" in rendered
    assert "tool_schemas" in rendered

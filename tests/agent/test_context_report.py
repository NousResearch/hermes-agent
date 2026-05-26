from types import SimpleNamespace

from agent.context_report import build_context_report, format_context_report


def test_context_report_breaks_down_prompt_tools_messages_and_skills():
    messages = [
        {"role": "user", "content": "please inspect the repo"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_skill",
                    "function": {
                        "name": "skill_view",
                        "arguments": '{"name": "hermes-agent"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_skill",
            "content": '{"success": true, "name": "hermes-agent", "content": "'
            + ("skill guidance " * 200)
            + '"}',
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_terminal",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "rg context"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_terminal",
            "content": "terminal output\n" + ("line\n" * 500),
        },
    ]
    system_parts = {
        "stable": (
            "base guidance\n"
            "<available_skills>\n"
            "  dev:\n"
            "    - hermes-agent: Hermes Agent operations and config\n"
            "    - tiny: Small helper\n"
            "</available_skills>"
        ),
        "context": "AGENTS.md instructions",
        "volatile": "Conversation started: Tuesday, May 26, 2026",
    }
    agent = SimpleNamespace(
        model="gpt-5.5",
        provider="openai-codex",
        tools=[
            {"type": "function", "function": {"name": "terminal", "description": "x" * 800}},
            {"type": "function", "function": {"name": "skill_view", "description": "x" * 200}},
        ],
        context_compressor=SimpleNamespace(
            context_length=272_000,
            threshold_tokens=217_600,
            compression_count=1,
            last_prompt_tokens=68_529,
        ),
        _cached_system_prompt="\n\n".join(system_parts.values()),
        _session_messages=messages,
        _build_system_prompt_parts=lambda _system_message=None: system_parts,
    )

    report = build_context_report(agent, messages)
    rendered = format_context_report(report)

    assert report["components"]["system_prompt"] > 0
    assert report["components"]["tools_schema"] > 0
    assert report["components"]["messages"] > 0
    assert report["message_info"]["loaded_skills"][0]["name"] == "hermes-agent"
    top_tools = {item["tool"] for item in report["message_info"]["top_tool_results"]}
    assert {"skill_view", "terminal"} <= top_tools
    assert report["skills_index"]["tokens"] > 0

    assert "Context composition" in rendered
    assert "Top-level buckets" in rendered
    assert "Largest tool schemas" in rendered
    assert "Largest tool results in messages" in rendered
    assert "Skill-related context" in rendered
    assert "hermes-agent" in rendered

    compact = format_context_report(report, compact=True)
    assert "🧠 Context Window" in compact
    assert "📦 Buckets" in compact
    assert "System prompt:" in compact
    assert "tok (" in compact
    assert "[#" not in compact
    assert "🎯 Skills" in compact
    assert "hermes-agent" in compact

"""Tests for live session context breakdown."""

from unittest.mock import MagicMock, patch

from agent.context_breakdown import compute_session_context_breakdown


def _make_agent(
    *,
    stable: str = "identity and guidance",
    context: str = "",
    volatile: str = "timestamp line",
    tools: list | None = None,
    context_length: int = 200_000,
    last_prompt_tokens: int = 0,
):
    agent = MagicMock()
    agent.model = "openai/gpt-5.4"
    agent.tools = tools or [
        {"type": "function", "function": {"name": "terminal", "description": "run"}},
        {"type": "function", "function": {"name": "mcp_demo_tool", "description": "mcp"}},
        {"type": "function", "function": {"name": "delegate_task", "description": "spawn"}},
    ]
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = True
    agent.context_compressor = MagicMock(
        context_length=context_length,
        last_prompt_tokens=last_prompt_tokens,
    )
    return agent, {"stable": stable, "context": context, "volatile": volatile}


def test_breakdown_includes_major_categories():
    stable = (
        "base guidance\n"
        "<available_skills>\n  demo:\n    - hello: hi\n</available_skills>"
    )
    context = "# Project Context\nFollow AGENTS.md"
    volatile = "Current time: now"
    history = [{"role": "user", "content": "hello there"}]
    agent, parts = _make_agent(stable=stable, context=context, volatile=volatile)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, history)

    ids = {item["id"] for item in data["categories"]}
    assert {"system_prompt", "tool_definitions", "rules", "skills", "mcp", "subagent_definitions", "conversation"} <= ids
    assert data["context_max"] == 200_000
    assert data["estimated_total"] > 0


def test_conversation_is_reported_at_zero_on_an_empty_transcript():
    """An empty conversation must read as zero, not as a missing category.

    Dropping the row makes "nothing said yet" indistinguishable from "the
    breakdown never measured the conversation", which is what #87903 reports
    on a fresh session.
    """
    agent, parts = _make_agent()

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    conversation = [item for item in data["categories"] if item["id"] == "conversation"]
    assert len(conversation) == 1
    assert conversation[0]["tokens"] == 0
    assert conversation[0]["label"] == "Conversation"


def test_conversation_is_reported_at_zero_when_history_is_none():
    """``messages=None`` is the same empty conversation, not a missing one."""
    agent, parts = _make_agent()

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, None)

    assert [item["id"] for item in data["categories"]].count("conversation") == 1


def test_structurally_absent_categories_are_still_dropped():
    """Only the conversation is exempt: an unconfigured category stays hidden.

    Otherwise the panel grows permanent zero rows for every feature the
    operator has not turned on.
    """
    agent, parts = _make_agent(
        stable="base guidance",  # no <available_skills> block
        context="",  # no rules
        tools=[{"type": "function", "function": {"name": "terminal"}}],
    )

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    ids = {item["id"] for item in data["categories"]}
    assert "conversation" in ids
    assert {"skills", "mcp", "subagent_definitions", "rules", "memory"} & ids == set()


def test_zero_conversation_does_not_inflate_the_estimated_total():
    agent, parts = _make_agent()
    history = [{"role": "user", "content": "hello there"}]

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        empty = compute_session_context_breakdown(agent, [])
        spoken = compute_session_context_breakdown(agent, history)

    assert empty["estimated_total"] < spoken["estimated_total"]


# ── /context renderers (pure functions over the payload) ────────────────────

from agent.context_breakdown import (  # noqa: E402
    compute_context_details,
    render_context_breakdown_lines,
    render_context_category_lines,
    render_context_details_lines,
    render_context_grid,
)


def _payload(**overrides):
    base = {
        "categories": [
            {"id": "system_prompt", "label": "System prompt", "tokens": 10_000},
            {"id": "tool_definitions", "label": "Tool definitions", "tokens": 20_000},
            {"id": "skills", "label": "Skills", "tokens": 5_000},
            {"id": "conversation", "label": "Conversation", "tokens": 15_000},
        ],
        "context_max": 200_000,
        "context_percent": 25,
        "context_used": 50_000,
        "estimated_total": 50_000,
        "model": "openai/gpt-test",
    }
    base.update(overrides)
    return base


def test_grid_is_5x20_and_mostly_free():
    rows = render_context_grid(_payload())
    assert len(rows) == 5
    cells = " ".join(rows).split(" ")
    assert len(cells) == 100
    # 50k / 200k → 25 used cells, 75 free
    assert cells.count("·") == 75
    # Category glyphs proportional: 10k→5, 20k→10, 5k→2-3, 15k→7-8 cells
    assert cells.count("■") == 5
    assert cells.count("▣") == 10










def test_breakdown_lines_grid_toggle():
    with_grid = render_context_breakdown_lines(_payload(), grid=True)
    without = render_context_breakdown_lines(_payload(), grid=False)
    assert any("·" in line for line in with_grid[:5])
    assert not any("·" in line for line in without[:2])
    # Both include the window summary and the expand hint
    for lines in (with_grid, without):
        text = "\n".join(lines)
        assert "Context window: 50,000 / 200,000 tokens (25%)" in text
        assert "/context all" in text




def test_details_lines_caps_listing():
    details = {
        "skills": [
            {"name": f"skill-{i}", "index_tokens": 10, "skill_md_tokens": 100}
            for i in range(20)
        ],
        "toolsets": [],
    }
    lines = render_context_details_lines(details)
    assert any("… and 5 more" in line for line in lines)



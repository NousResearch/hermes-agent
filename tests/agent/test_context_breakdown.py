"""Tests for live session context breakdown."""

from unittest.mock import MagicMock, patch

from agent.context_breakdown import _chars_to_tokens, compute_session_context_breakdown


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
    # Skills index lives in the volatile tier (#37117), not stable.
    stable = "base guidance"
    context = "# Project Context\nFollow AGENTS.md"
    volatile = (
        "<available_skills>\n  demo:\n    - hello: hi\n</available_skills>\n\n"
        "Current time: now"
    )
    history = [{"role": "user", "content": "hello there"}]
    agent, parts = _make_agent(stable=stable, context=context, volatile=volatile)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, history)

    ids = {item["id"] for item in data["categories"]}
    assert {"system_prompt", "tool_definitions", "rules", "skills", "mcp", "subagent_definitions", "conversation"} <= ids
    assert data["context_max"] == 200_000
    assert data["estimated_total"] > 0


def test_skills_index_falls_back_to_stable_for_legacy_sessions():
    """A session whose cached system prompt predates #37117 (skills still in
    the stable tier) must still attribute a "skills" category."""
    stable = (
        "base guidance\n"
        "<available_skills>\n  demo:\n    - hello: hi\n</available_skills>"
    )
    volatile = "Current time: now"
    agent, parts = _make_agent(stable=stable, volatile=volatile)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    ids = {item["id"] for item in data["categories"]}
    assert "skills" in ids


def test_skills_index_not_double_counted_in_system_prompt():
    """The skills block must be attributed only to the "skills" category,
    not also folded into "system_prompt" via the volatile tail."""
    skills_block = "<available_skills>\n  demo:\n    - hello: hi\n</available_skills>"
    stable = "base guidance"
    remainder = "Current time: now"
    volatile = f"{skills_block}\n\n{remainder}"
    agent, parts = _make_agent(stable=stable, volatile=volatile)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    by_id = {item["id"]: item["tokens"] for item in data["categories"]}
    expected_system_prompt = _chars_to_tokens(
        "\n\n".join((stable, remainder))
    )
    assert by_id["system_prompt"] == expected_system_prompt
    assert by_id["skills"] == _chars_to_tokens(skills_block)


def test_context_details_finds_skills_in_volatile_tier():
    """compute_context_details() (the /context all per-skill listing) must
    also look in the volatile tier, not just stable."""
    from agent.context_breakdown import compute_context_details

    stable = "base guidance"
    volatile = "<available_skills>\n  demo:\n    - hello: hi\n</available_skills>"
    agent, parts = _make_agent(stable=stable, volatile=volatile)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        details = compute_context_details(agent)

    names = {entry["name"] for entry in details["skills"]}
    assert names == {"hello"}



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



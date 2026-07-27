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


def test_breakdown_uses_measured_context_when_available():
    agent, parts = _make_agent(last_prompt_tokens=42_000)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    assert data["context_used"] == 42_000
    assert data["context_percent"] == 21

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


def test_grid_nonzero_category_never_invisible():
    payload = _payload(
        categories=[{"id": "memory", "label": "Memory", "tokens": 10}],
        estimated_total=10,
        context_used=10,
    )
    rows = render_context_grid(payload)
    assert "▧" in " ".join(rows)


def test_grid_without_context_max_is_all_free():
    rows = render_context_grid(_payload(context_max=0))
    cells = " ".join(rows).split(" ")
    assert set(cells) == {"·"}


def test_category_lines_include_tokens_percent_and_free_space():
    lines = render_context_category_lines(_payload())
    text = "\n".join(lines)
    assert "Estimated usage by category" in text
    assert "System prompt" in text and "10,000 tokens" in text
    assert "5.0%" in text  # 10k / 200k
    assert "Free space" in text and "150,000 tokens" in text


def test_category_lines_no_categories():
    lines = render_context_category_lines(_payload(categories=[]))
    assert any("no data yet" in line for line in lines)


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


def test_breakdown_lines_with_details_omits_hint():
    details = {
        "skills": [
            {"name": "alpha", "index_tokens": 25, "skill_md_tokens": 800},
            {"name": "beta", "index_tokens": 30, "skill_md_tokens": None},
        ],
        "toolsets": [
            {"toolset": "terminal", "tool_count": 3, "schema_tokens": 4_000},
        ],
    }
    lines = render_context_breakdown_lines(_payload(), details=details, grid=False)
    text = "\n".join(lines)
    assert "Toolsets by schema cost" in text
    assert "terminal" in text and "4,000 tokens" in text
    assert "Skills by cost" in text
    assert "alpha" in text and "beta" in text
    assert "n/a" in text  # unmapped SKILL.md renders n/a, not a crash
    assert "Use /context all" not in text


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


def test_compute_context_details_maps_bytes_to_tokens():
    agent, parts = _make_agent(
        stable=(
            "base\n<available_skills>\n  demo:\n"
            "    - hello: a demo skill\n</available_skills>"
        ),
    )
    fake_skills = [{
        "name": "hello",
        "index_line_bytes": 40,
        "index_line_total_bytes": 40,
        "index_line_shared_bytes": 0,
        "index_line_skill_count": 1,
        "skill_md_bytes": 401,
        "path": "/tmp/hello/SKILL.md",
    }]
    fake_toolsets = [{"toolset": "terminal", "tool_count": 2, "json_bytes": 399}]
    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts), \
         patch("hermes_cli.prompt_size._compute_skills_breakdown", return_value=fake_skills), \
         patch("hermes_cli.prompt_size._compute_toolsets_breakdown", return_value=fake_toolsets):
        details = compute_context_details(agent)

    assert details["skills"] == [
        {"name": "hello", "index_tokens": 10, "skill_md_tokens": 101},
    ]
    assert details["toolsets"] == [
        {"toolset": "terminal", "tool_count": 2, "schema_tokens": 100},
    ]

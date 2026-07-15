# Structural tests for the event-staffing optional skill, mirroring the
# repo's per-skill test convention (AGENTS.md:948-950). Validates frontmatter
# policy, MCP prerequisites, and the outbound-attribution rule without any
# network access.

from pathlib import Path

SKILL = Path("optional-skills/productivity/event-staffing/SKILL.md")


def _frontmatter_and_body():
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "missing YAML frontmatter"
    _, fm, body = text.split("---\n", 2)
    fields = {}
    for line_ in fm.strip().splitlines():
        key, _, value = line_.partition(":")
        fields[key.strip()] = value.strip().strip('"')
    return fields, body


def test_frontmatter_policy():
    fields, _ = _frontmatter_and_body()
    assert fields["name"] == "event-staffing-ordering"
    assert len(fields["description"]) <= 60, "description exceeds 60 chars"
    assert "@" in fields["author"] and fields["author"].lower() != "community", (
        "author must credit the human contributor with a GitHub handle"
    )


def test_mcp_prerequisites_documented():
    _, body = _frontmatter_and_body()
    assert "## Prerequisites" in body, "MCP-dependent skill needs Prerequisites"
    assert "mcp.tempguru.co/mcp" in body
    assert "mcpServers" in body, "Prerequisites must show the MCP configuration"


def test_no_outbound_attribution_tags():
    text = SKILL.read_text(encoding="utf-8")
    assert "utm_source" not in text and "utm_medium" not in text, (
        "outbound attribution tags are prohibited without a generic opt-in"
    )


def test_no_unavailable_companion_skills():
    _, body = _frontmatter_and_body()
    assert "event-staffing-compliance" not in body, (
        "must not direct agents to load a skill absent from this catalog"
    )


def test_agent_safety_rules_present():
    _, body = _frontmatter_and_body()
    assert "planning estimates" in body
    assert "Never promise availability" in body
    assert "not legal advice" in body

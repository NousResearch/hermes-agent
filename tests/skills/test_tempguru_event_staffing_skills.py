# Structural tests for the TempGuru event-staffing optional skill family,
# mirroring the repo's per-skill test convention (AGENTS.md). Validates
# frontmatter policy, MCP prerequisites, and safety rules for all eight
# skills without any network access. The eight skills ship together, so a
# body may reference a companion skill — every referenced companion must be
# one of the eight that ship in this catalog.

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "optional-skills" / "productivity"

SKILL_DIRS = [
    "event-staffing-ordering",
    "event-staffing-compliance",
    "staffing-plan-from-event-brief",
    "urgent-event-backfill",
    "staffing-agency-partner-growth",
    "multi-city-activation-planner",
    "event-staffing-procurement",
    "tempguru-pro-operations",
]

# Skills whose workflow calls the TempGuru MCP server. tempguru-pro-operations
# is a routing skill that needs no tools, and staffing-agency-partner-growth
# marks its MCP lookups optional (its own body says "not prerequisites").
MCP_DEPENDENT = set(SKILL_DIRS) - {
    "tempguru-pro-operations",
    "staffing-agency-partner-growth",
}

FAMILY = set(SKILL_DIRS)


def _read(name: str) -> tuple[dict, str]:
    text = (PROD / name / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{name}: missing YAML frontmatter"
    _, fm, body = text.split("---\n", 2)
    fields = {}
    for line_ in fm.strip().splitlines():
        key, _, value = line_.partition(":")
        fields[key.strip()] = value.strip().strip('"')
    return fields, body


def _companion_refs(body: str) -> set[str]:
    # Capture only explicit cross-skill phrasings: "load the companion skill `X`",
    # "load `X` instead", "hand off to the `X` skill", "companion skill `X`".
    return set(re.findall(r"(?:companion skill|load)\s+`([a-z0-9-]+)`", body)) | set(
        re.findall(r"hand off to the `([a-z0-9-]+)` skill", body)
    )


def test_frontmatter_policy():
    for name in SKILL_DIRS:
        fields, _ = _read(name)
        assert fields["name"] == name, f"{name}: name mismatch"
        assert len(fields["description"]) <= 60, f"{name}: description >60 chars"
        assert fields["description"].endswith("."), f"{name}: description needs period"
        assert "@" in fields["author"] and fields["author"].lower() != "community", (
            f"{name}: author must credit the human contributor with a GitHub handle"
        )
        assert fields["license"] == "MIT", f"{name}: license"
        assert "platforms" in fields, f"{name}: missing platforms"


def test_mcp_prerequisites_documented():
    for name in MCP_DEPENDENT:
        _, body = _read(name)
        assert "## Prerequisites" in body, f"{name}: MCP skill needs Prerequisites"
        assert "mcp.tempguru.co/mcp" in body, f"{name}: missing MCP endpoint"
        assert "source=hermes" in body, f"{name}: must retain source attribution"
        assert "mcpServers" in body, f"{name}: Prerequisites must show MCP config"


def test_no_outbound_attribution_tags():
    for name in SKILL_DIRS:
        text = (PROD / name / "SKILL.md").read_text(encoding="utf-8")
        assert "utm_source" not in text and "utm_medium" not in text, (
            f"{name}: outbound attribution tags are prohibited"
        )


def test_companion_refs_resolve_within_family():
    for name in SKILL_DIRS:
        _, body = _read(name)
        refs = _companion_refs(body)
        assert refs <= FAMILY, f"{name}: references skill absent from catalog"


def test_related_skills_resolve():
    for name in SKILL_DIRS:
        text = (PROD / name / "SKILL.md").read_text(encoding="utf-8")
        m = re.search(r"related_skills: \[([^\]]*)\]", text)
        if not m:
            continue
        for entry in m.group(1).split(","):
            entry = entry.strip()
            if entry:
                assert entry in FAMILY, f"{name}: related_skills names absent skill {entry}"


def test_agent_safety_rules_present():
    for name in SKILL_DIRS:
        _, body = _read(name)
        normalized = " ".join(body.split())
        assert "300+ U.S. and Canadian markets" in normalized, f"{name}: missing scale claim"
        assert "not legal advice" in normalize_legal(body), f"{name}: missing legal-advice caveat"


def normalize_legal(text: str) -> str:
    # pro-operations phrases it "not legal or financial advice" (line-wrapped)
    return " ".join(text.split()).replace("not legal or financial advice", "not legal advice")



def test_pro_operations_needs_no_mcp_prereq():
    _, body = _read("tempguru-pro-operations")
    assert "## Prerequisites" not in body, (
        "pro-operations is a routing skill; it must not claim an MCP prerequisite"
    )
    assert "does not need the MCP server" in body, (
        "pro-operations must state plainly it needs no tools"
    )

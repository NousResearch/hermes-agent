"""The Hermes skill is a compact router, not a full manual."""

from pathlib import Path

from agent.markdown_sections import markdown_headings


SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "hermes-agent"
)


def test_hermes_agent_router_is_compact_and_routes_to_focused_references():
    router = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    expected_references = {
        "references/cli-reference.md",
        "references/configuration-and-context.md",
        "references/security-and-privacy.md",
        "references/automation-and-surfaces.md",
        "references/troubleshooting.md",
        "references/contributing.md",
        "references/windows.md",
        "references/native-mcp.md",
        "references/webhooks.md",
    }

    assert len(router) <= 9_000
    for reference in expected_references:
        assert reference in router
        assert (SKILL_DIR / reference).is_file()


def test_split_preserves_each_original_operational_section_once():
    migrated_references = [
        "references/cli-reference.md",
        "references/configuration-and-context.md",
        "references/security-and-privacy.md",
        "references/automation-and-surfaces.md",
        "references/windows.md",
        "references/troubleshooting.md",
        "references/contributing.md",
    ]
    expected_sections = {
        "Scope & Verification",
        "Quick Start",
        "CLI Reference",
        "Slash Commands (In-Session)",
        "Key Paths & Config",
        "Project Context Files",
        "Security & Privacy Toggles",
        "Voice & Transcription",
        "Spawning Additional Hermes Instances",
        "Durable & Background Systems",
        "Surfaces & Other Capabilities",
        "Windows-Specific Quirks",
        "Troubleshooting",
        "Where to Find Things",
        "Contributor Quick Reference",
    }
    paths = [SKILL_DIR / "SKILL.md", *(SKILL_DIR / path for path in migrated_references)]
    titles = [
        heading.title
        for path in paths
        for heading in markdown_headings(path.read_text(encoding="utf-8"))
    ]

    for section in expected_sections:
        assert titles.count(section) == 1, section

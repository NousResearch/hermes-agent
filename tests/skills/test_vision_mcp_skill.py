from __future__ import annotations

import re
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "mcp" / "vision-mcp"
SKILL_PATH = SKILL_DIR / "SKILL.md"


def _frontmatter_text() -> str:
    src = SKILL_PATH.read_text(encoding="utf-8")
    match = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return match.group(1)


def _frontmatter_value(key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", _frontmatter_text(), re.MULTILINE)
    assert match, f"missing frontmatter key: {key}"
    return match.group(1).strip().strip('"')


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert SKILL_PATH.is_file()


def test_frontmatter_matches_optional_skill_standards() -> None:
    assert _frontmatter_value("name") == "vision-mcp"
    description = _frontmatter_value("description")
    assert description.endswith(".")
    assert len(description) <= 60
    assert _frontmatter_value("license") == "MIT"


def test_author_credits_human_contributor_first() -> None:
    # AGENTS.md contributed-skill standard: the external contributor's real
    # name + GitHub handle come first; "Hermes Agent" is the secondary
    # collaborator, never the primary author.
    author = _frontmatter_value("author")
    assert author == "Haruyuki (Haruhiyuki), Hermes Agent"


def test_platforms_gate_real_supported_desktops() -> None:
    platforms = _frontmatter_value("platforms")
    assert "macos" in platforms
    assert "windows" in platforms
    assert "linux" not in platforms


def test_modern_sections_are_present_in_order() -> None:
    src = SKILL_PATH.read_text(encoding="utf-8")
    # AGENTS.md skill standard: modern `# <Skill> Skill` title form.
    assert re.search(r"^# Vision-MCP Skill$", src, re.MULTILINE)
    sections = [
        "# Vision-MCP Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [src.index(section) for section in sections]
    assert positions == sorted(positions)


def test_skill_names_hermes_mcp_tools_and_setup_commands() -> None:
    src = SKILL_PATH.read_text(encoding="utf-8")
    required = [
        "hermes mcp install vision-mcp",
        "hermes mcp configure vision-mcp",
        "mcp_vision_mcp_vision_map_list_apps",
        "mcp_vision_mcp_vision_map_run_workflow",
        "mcp_vision_mcp_vision_map_repair_minimal",
        "mcp_vision_mcp_capsule_attach_window",
    ]
    for text in required:
        assert text in src


def test_skill_calls_out_low_level_and_mutating_tool_consent() -> None:
    src = SKILL_PATH.read_text(encoding="utf-8")
    for tool in [
        "vision_map.click_at",
        "vision_map.type_text",
        "vision_map.apply_patch",
        "vision_map.commit_state",
        "vision_map.commit_workflow",
    ]:
        assert tool in src
    assert "explicit user consent" in src


def test_setup_commands_match_the_catalog_manifest_pin() -> None:
    # The skill duplicates the manifest's npx setup commands. Catalog policy
    # pins the CLI release in the manifest; the skill must reference the same
    # exact version so the two files cannot drift (and never float `@latest`).
    manifest = (
        SKILL_DIR.parents[2] / "optional-mcps" / "vision-mcp" / "manifest.yaml"
    ).read_text(encoding="utf-8")
    manifest_refs = set(re.findall(r"@vision-mcp/cli@(\S+?)[\"\s]", manifest))
    assert len(manifest_refs) == 1, f"manifest pins one CLI version: {manifest_refs}"
    version = manifest_refs.pop()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)

    src = SKILL_PATH.read_text(encoding="utf-8")
    skill_refs = set(re.findall(r"@vision-mcp/cli@([\w.<>-]+)", src))
    # `@<version>` placeholders in upgrade guidance are allowed; every literal
    # version must equal the manifest pin.
    literal_refs = {r for r in skill_refs if r != "<version>"}
    assert literal_refs == {version}, (
        f"SKILL.md references {literal_refs}, manifest pins {version}"
    )

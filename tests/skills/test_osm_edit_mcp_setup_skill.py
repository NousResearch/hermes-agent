"""Contract tests for the OSM Edit MCP setup optional skill."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = ROOT / "optional-skills" / "mcp" / "osm-edit-mcp-setup" / "SKILL.md"


def _content() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter(content: str) -> dict[str, object]:
    yaml = pytest.importorskip("yaml")
    match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
    assert match is not None
    parsed = yaml.safe_load(match.group(1))
    assert isinstance(parsed, dict)
    return parsed


def test_osm_edit_mcp_setup_frontmatter() -> None:
    content = _content()
    metadata = _frontmatter(content)

    assert metadata["name"] == "osm-edit-mcp-setup"
    assert len(str(metadata["description"])) <= 60
    assert str(metadata["description"]).endswith(".")
    assert str(metadata["author"]).startswith("Petr Korolev,")
    assert metadata["platforms"] == ["linux", "macos"]
    # category is derived from the directory path (optional-skills/<category>/<name>)
    hermes_meta = metadata["metadata"]["hermes"]
    assert "osm" in hermes_meta["tags"] or "osm" in str(hermes_meta).lower()


def test_osm_edit_mcp_setup_has_modern_section_order() -> None:
    sections = re.findall(r"^## (.+)$", _content(), re.MULTILINE)
    required = [
        "Why This Skill Exists",
        "When to Use",
        "Prerequisites",
        "Security Invariants",
        "Configuration Model",
        "Procedure",
        "Pitfalls",
        "Verification",
    ]

    assert [sections.index(name) for name in required] == sorted(
        sections.index(name) for name in required
    )


def test_osm_edit_mcp_setup_preserves_safety_contract() -> None:
    content = _content()

    for required in (
        "OSM_USE_DEV_API",
        "OSM_WRITE_PROFILE=safe",
        "OSM_REQUIRE_HOST_CONFIRMATION=true",
        "raw_write_tools_registered",
        "get_edit_capabilities",
        "check_authentication",
        "background=true, pty=true",
        'process(action="submit")',
        "/reload-mcp",
    ):
        assert required in content

    # the safe profile must never be silently downgraded to expert
    assert re.search(r"OSM_WRITE_PROFILE\s*=\s*expert", content) is None
    assert "OSM_WRITE_PROFILE=expert" not in content
    assert "/opt/data" not in content
    assert "/home/" not in content


def test_osm_edit_mcp_setup_contains_no_example_secrets() -> None:
    content = _content()
    secret_assignment = re.compile(
        r"(?i)(client_secret|access_token|authorization_code|password)"
        r"\s*[:=]\s*(?!<)[A-Za-z0-9_.~-]{16,}"
    )
    github_token = re.compile(r"(?:gh[opusr]_|github_pat_)[A-Za-z0-9_]{20,}")

    assert secret_assignment.search(content) is None
    assert github_token.search(content) is None

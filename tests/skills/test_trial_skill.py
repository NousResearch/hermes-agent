"""Policy and integration tests for the optional Trial skill."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "optional-skills" / "software-development" / "trial"
OLD_SKILL_DIR = ROOT / "optional-skills" / "autonomous-ai-agents" / "trial"
OLD_SKILL_PATH = OLD_SKILL_DIR / "SKILL.md"
SKILL_PATH = SKILL_DIR / "SKILL.md"
DOC_PATH = (
    ROOT
    / "website"
    / "docs"
    / "user-guide"
    / "skills"
    / "optional"
    / "software-development"
    / "software-development-trial.md"
)
OLD_DOC_PATH = (
    ROOT
    / "website"
    / "docs"
    / "user-guide"
    / "skills"
    / "optional"
    / "autonomous-ai-agents"
    / "autonomous-ai-agents-trial.md"
)


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _frontmatter_field(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}:\s*(.+)$",
        _skill_text(),
        re.MULTILINE,
    )
    assert match, f"missing frontmatter field: {name}"
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def test_skill_is_in_software_development_only() -> None:
    assert SKILL_PATH.is_file()
    assert not OLD_SKILL_PATH.exists()
    assert "category: autonomous-ai-agents" not in _skill_text()


def test_frontmatter_meets_hardline_policy() -> None:
    assert _frontmatter_field("name") == "trial"
    description = _frontmatter_field("description")
    assert description == "Use before final responses to block unproven completion."
    assert len(description) <= 60
    assert description.endswith(".")
    assert _frontmatter_field("version") == "0.5.1"
    assert (
        _frontmatter_field("author")
        == "Abdulrahman Qasem (Da7-Tech), Hermes Agent"
    )
    assert _frontmatter_field("license") == "MIT"
    assert _frontmatter_field("platforms") == "[linux, macos, windows]"


def test_body_uses_native_hermes_tools() -> None:
    body = _skill_text().split("---", 2)[-1]
    assert "`search_files`" in body
    assert "`delegate_task`" in body
    assert "`terminal`" in body
    assert re.search(r"(?<![\w-])grep(?![\w-])", body, re.IGNORECASE) is None


def test_modern_sections_are_complete_and_ordered() -> None:
    text = _skill_text()
    headings = [
        "# Trial Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_verification_claim_links_to_the_measured_source() -> None:
    text = _skill_text()
    normalized = re.sub(r"\s+", " ", text)
    assert "https://github.com/Da7-Tech/trial" in normalized
    assert "version 0.4 receipt-and-coverage" in normalized
    assert "6/6 with Trial versus 4/6 without it" in normalized
    assert "auditable evidence, not cryptographic proof" in normalized
    assert "historical" in normalized
    assert "cannot be independently" in normalized
    assert "Version 0.5" in normalized
    assert "has not yet been" in normalized


def test_pre_delivery_gate_is_fail_closed() -> None:
    text = _skill_text()
    normalized = re.sub(r"\s+", " ", text)

    draft = normalized.index("Draft privately")
    judge = normalized.index("Judge", draft)
    release = normalized.index("Release", judge)
    assert draft < judge < release
    assert "final response remains a private draft" in normalized
    assert (
        "Release the response only after every visible claim is accepted"
        in normalized
    )
    assert "keep the draft internal" in normalized
    assert (
        "Negative verdicts remain internal and return the agent to work"
        in normalized
    )
    assert "Do not expose internal courtroom labels" in normalized
    assert "telling the user that the agent almost lied" in normalized


def test_generated_docs_use_the_new_install_identifier() -> None:
    assert DOC_PATH.is_file()
    assert not OLD_DOC_PATH.exists()
    doc = DOC_PATH.read_text(encoding="utf-8")
    assert "official/software-development/trial" in doc
    assert "`optional-skills/software-development/trial`" in doc
    assert "`0.5.1`" in doc
    assert "Abdulrahman Qasem (Da7-Tech), Hermes Agent" in doc
    assert "final response remains a private draft" in doc
    assert "Negative verdicts remain internal" in doc


def test_catalog_and_sidebar_use_software_development_route() -> None:
    catalog = (
        ROOT / "website" / "docs" / "reference" / "optional-skills-catalog.md"
    ).read_text(encoding="utf-8")
    sidebar = (ROOT / "website" / "sidebars.ts").read_text(encoding="utf-8")
    new_route = "software-development/software-development-trial"
    old_route = "autonomous-ai-agents/autonomous-ai-agents-trial"
    assert new_route in catalog
    assert new_route in sidebar
    assert old_route not in catalog
    assert old_route not in sidebar

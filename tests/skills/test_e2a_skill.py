"""Static compliance tests for the optional e2a skill.

These tests validate the contribution-guide hardlines and operational safety
without making live network calls.
"""

import re
from pathlib import Path

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "email"
    / "e2a"
    / "SKILL.md"
)

REQUIRED_SECTIONS = (
    "# e2a Skill",
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
)


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    match = re.search(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    assert match, "frontmatter must close with ---"
    return match.group(1), content[match.end() :]


def _field(frontmatter: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*[\"']?(.*?)[\"']?\s*$", frontmatter, re.MULTILINE)
    assert match, f"missing frontmatter field: {name}"
    return match.group(1).rstrip("\"'")


def test_skill_file_exists():
    assert SKILL_PATH.is_file()


def test_frontmatter_required_fields():
    frontmatter, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert re.search(rf"^{field}:", frontmatter, re.MULTILINE), (
            f"missing frontmatter field: {field}"
        )
    assert _field(frontmatter, "name") == "e2a"
    assert re.search(r"^metadata:\n\s+hermes:", frontmatter, re.MULTILINE)
    assert re.search(r"^\s+tags:\s*\[[^]]+\]", frontmatter, re.MULTILINE)
    assert re.search(r"^\s+related_skills:", frontmatter, re.MULTILINE)


def test_description_hardline():
    frontmatter, _ = _frontmatter_and_body()
    description = _field(frontmatter, "description")
    assert len(description) <= 60
    assert description.endswith(".")
    assert _field(frontmatter, "name").lower() not in description.lower()


def test_author_credits_human_first():
    frontmatter, _ = _frontmatter_and_body()
    author = _field(frontmatter, "author")
    assert not author.startswith("Hermes Agent")
    assert "jiashuoz" in author


def test_required_section_order():
    _, body = _frontmatter_and_body()
    positions = [body.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions)


def test_mcp_prerequisites_are_explicit_and_secret_safe():
    _, body = _frontmatter_and_body()
    assert "https://api.e2a.dev/mcp" in body
    assert "auth: oauth" in body
    assert 'Authorization: "Bearer ${E2A_API_KEY}"' in body
    assert "hermes mcp login e2a" in body


def test_email_operations_preserve_threading_and_durable_acceptance():
    _, body = _frontmatter_and_body()
    assert "reply_to_message" in body
    for status in ("accepted", "scheduled", "pending_review"):
        assert f"`{status}`" in body
    assert "Do not resend" in body
    assert "send_at" in body and "scheduled_at" in body


def test_verification_uses_skill_and_dynamic_mcp_toolsets_read_only():
    _, body = _frontmatter_and_body()
    assert "--toolsets skills,mcp-e2a" in body
    assert "whoami" in body
    assert "without changing anything" in body


def test_no_machine_local_paths_or_hosted_customer_addresses():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "/home/" not in content
    assert not re.search(r"[A-Z]:\\\\Users", content)
    assert not re.search(r"[A-Z0-9._%+-]+@agents\.e2a\.dev", content, re.IGNORECASE)

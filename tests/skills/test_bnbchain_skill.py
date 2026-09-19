from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "blockchain" / "bnbchain"
SKILL_PATH = SKILL_DIR / "SKILL.md"
OLD_SKILL_PATH = REPO_ROOT / "skills" / "bnbchain" / "SKILL.md"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> str:
    match = re.match(r"^---\n(.*?)\n---\n", skill_text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    return match.group(1)


def frontmatter_value(frontmatter: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", frontmatter, re.MULTILINE)
    assert match, f"frontmatter is missing {key!r}"
    return match.group(1).strip().strip("\"'")


def test_skill_uses_optional_blockchain_layout() -> None:
    assert SKILL_PATH.is_file()
    assert not OLD_SKILL_PATH.exists()


@pytest.mark.parametrize(
    "reference",
    [
        "erc8004-tools-reference.md",
        "evm-tools-reference.md",
        "greenfield-tools-reference.md",
        "prompts-reference.md",
    ],
)
def test_reference_files_are_shipped(reference: str) -> None:
    assert (SKILL_DIR / "references" / reference).is_file()


def test_frontmatter_contract(frontmatter: str) -> None:
    description = frontmatter_value(frontmatter, "description")

    assert len(description) <= 60
    assert description.endswith(".")
    assert frontmatter_value(frontmatter, "author") == (
        "Korede Odubanjo (@koredeBNB), Hermes Agent"
    )
    assert frontmatter_value(frontmatter, "license") == "MIT"
    assert frontmatter_value(frontmatter, "platforms") == "[linux, macos, windows]"
    assert re.search(r"^\s{4}category:\s*blockchain$", frontmatter, re.MULTILINE)
    assert re.search(r"^\s{4}related_skills:\s*\[evm\]$", frontmatter, re.MULTILINE)
    assert re.search(r"^\s{4}tags:\s*\[.+\]$", frontmatter, re.MULTILINE)


def test_modern_sections_are_present_in_order(skill_text: str) -> None:
    sections = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]

    positions = [skill_text.index(section) for section in sections]
    assert positions == sorted(positions)


def test_uses_hermes_native_mcp_setup(skill_text: str) -> None:
    assert (
        "hermes mcp add bnbchain-mcp --command npx --args -y @bnb-chain/mcp@latest"
        in skill_text
    )
    assert "hermes mcp test bnbchain-mcp" in skill_text
    assert "--env PRIVATE_KEY=" in skill_text
    assert skill_text.index("--env PRIVATE_KEY=") < skill_text.index(
        "--args -y @bnb-chain/mcp@latest", skill_text.index("--env PRIVATE_KEY=")
    )
    assert "Claude Desktop" not in skill_text
    assert "Cursor MCP" not in skill_text
    assert '"mcpServers"' not in skill_text


def test_write_safety_is_explicit(skill_text: str) -> None:
    safety_section = skill_text[skill_text.index("## Pitfalls") :]

    for required_term in ("network", "recipient", "amount", "confirm"):
        assert required_term in safety_section.lower()
    assert "do not default to mainnet" in safety_section.lower()
    assert "never" in safety_section.lower() and "private key" in safety_section.lower()

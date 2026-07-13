"""Hermetic contract checks for the read-only Web3 permission skill."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "security"
    / "web3-permission-audit"
    / "SKILL.md"
)
DOC_PATH = (
    REPO_ROOT
    / "website"
    / "docs"
    / "user-guide"
    / "skills"
    / "optional"
    / "security"
    / "security-web3-permission-audit.md"
)
CATALOG_PATH = REPO_ROOT / "website" / "docs" / "reference" / "optional-skills-catalog.md"

SECTION_ORDER = (
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
)


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _frontmatter_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    assert match, f"missing frontmatter field: {key}"
    return match.group(1).strip().strip("'\"")


def test_skill_uses_the_modern_instruction_contract():
    text = _skill_text()
    description = _frontmatter_value(text, "description")

    assert len(description) <= 60
    assert description.endswith(".")
    assert _frontmatter_value(text, "author").startswith("Ahmet Osrak")
    assert "Use the Hermes `terminal` tool" in text
    assert "related_skills: [blockchain/solana, blockchain/evm]" in text
    assert "--chain base" in text
    assert "blockchain/base" not in text

    positions = [text.index(section) for section in SECTION_ORDER]
    assert positions == sorted(positions)

    # Shell utilities may appear inside terminal examples, but not as the
    # skill's headline interaction surface.
    assert not re.search(r"(?m)^- `(?:curl|python3)`", text)


def test_skill_keeps_writes_and_credentials_out_of_scope():
    text = _skill_text()

    for forbidden in (
        "private key",
        "seed phrase",
        "wallet file",
        "signing authority",
        "transaction submission",
    ):
        assert forbidden in text
    assert "read-only" in text
    assert "user approval" in text


def test_generated_page_and_catalog_point_to_the_skill():
    assert DOC_PATH.is_file(), "generated Web3 permission docs page is missing"
    assert CATALOG_PATH.is_file(), "optional skills catalog is missing"

    docs = DOC_PATH.read_text(encoding="utf-8")
    catalog = CATALOG_PATH.read_text(encoding="utf-8")
    assert "## How to Run" in docs
    assert "## Pitfalls" in docs
    assert "security-web3-permission-audit" in catalog

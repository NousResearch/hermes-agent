from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = ROOT / "optional-skills" / "blockchain" / "zcash" / "SKILL.md"
HELPER_PATH = (
    ROOT
    / "optional-skills"
    / "blockchain"
    / "zcash"
    / "scripts"
    / "zcash_client.py"
)


def read_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_frontmatter_description_matches_hardline_contract():
    text = read_skill()
    match = re.search(r"^description: (.+)$", text, re.MULTILINE)

    assert match is not None
    description = match.group(1)
    assert len(description) <= 60
    assert description.endswith(".")
    assert description.count(".") == 1


def test_skill_uses_required_modern_section_order():
    text = read_skill()
    sections = [
        "When to Use",
        "Prerequisites",
        "How to Run",
        "Quick Reference",
        "Procedure",
        "Pitfalls",
        "Verification",
    ]

    positions = [text.index(f"\n## {section}\n") for section in sections]
    assert positions == sorted(positions)


def test_skill_uses_hermes_native_mcp_configuration():
    text = read_skill()

    assert "\nmcp_servers:\n  zcash:\n" in text
    assert 'command: "npx"' in text
    assert 'args: ["-y", "@frontiercompute/zcash-mcp@1.4.0"]' in text
    assert "mcpServers" not in text


def test_skill_keeps_wallet_and_secret_boundary_explicit():
    text = read_skill().lower()

    for phrase in (
        "not a wallet",
        "private keys",
        "raw prompts",
        "balance scanning",
        "shielded-spend construction",
        "broadcasting",
        "does not prove",
    ):
        assert phrase in text


def test_unreferenced_duplicate_helper_is_removed():
    assert not HELPER_PATH.exists()

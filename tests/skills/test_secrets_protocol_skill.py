"""Invariant tests for the bundled secrets-protocol skill.

Covers skills/security/secrets-protocol — the authoritative Hermes
secrets-handling protocol for every secret source (Bitwarden Secrets
Manager / bws, 1Password / op, and the command helper). Tests assert the
contracts the maintainers hold for every bundled skill: valid
frontmatter, description within the 60-character hardline, required
sections present, and honest references to the hardening series rather
than to code that has not landed on main.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
SKILL_DIR = REPO / "skills" / "security" / "secrets-protocol"
SKILL_MD = SKILL_DIR / "SKILL.md"

# The secrets-exfiltration hardening series — the docs may reference
# these PR numbers as the source of the contract, but must not claim
# their behavior or tests exist on main.
HARDENING_SERIES = {"77008", "77012", "77020", "77027", "77031", "77039"}


def _frontmatter(skill_md: Path) -> dict:
    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{skill_md} has no YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_skill_exists_with_frontmatter():
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    fm = _frontmatter(SKILL_MD)
    assert fm["name"] == "secrets-protocol"
    assert fm["description"].strip()
    assert len(fm["description"]) <= 60, (
        f"description is {len(fm['description'])} chars (max 60)"
    )
    assert fm["description"].rstrip('"').endswith(".")
    platforms = fm.get("platforms")
    assert platforms, "missing platforms gating"
    assert set(platforms) <= {"linux", "macos", "windows"}


def test_required_sections_present():
    body = SKILL_MD.read_text(encoding="utf-8")
    for section in [
        "## When to Use",
        "## Protocol invariants (non-negotiable)",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]:
        assert section in body, f"missing required section {section}"


def test_rotation_is_user_action_only():
    """The agent must never perform rotation or handle the token value."""
    body = SKILL_MD.read_text(encoding="utf-8")
    assert "user action only" in body
    assert "never asks for the token value" in body
    assert "do not save it anywhere else in between" in body  # clipboard discipline


def test_contract_asserts_merged_state_without_hedging():
    """The skill asserts the hardened contract as the current state.

    The docs describe the post-hardening behavior as reality — the
    encrypted-only cache, the deleted plaintext write branch, the masked
    output, the stripped child environments. There must be no
    'until it lands' / 'not on main yet' hedging: when the series
    merges, the docs are already correct and need zero cleanup.
    """
    body = SKILL_MD.read_text(encoding="utf-8")
    # The strong contract, stated as fact.
    assert "no plaintext write branch" in body
    assert "memory-only" in body
    assert "never by inheritance" in body
    # The gate test is present, not pending.
    assert "tests/test_secrets_exfiltration.py" in body
    # No hedging that the contract is not yet real.
    for hedge in [
        "Until that series lands on",
        "not on main",
        "once it lands",
        "when the no-exfiltration gate lands",
        "current main still",
    ]:
        assert hedge not in body, f"hedge present: {hedge!r}"


def test_referenced_series_prs_are_consistent():
    body = SKILL_MD.read_text(encoding="utf-8")
    mentioned = set(re.findall(r"#(770\d\d)", body))
    assert mentioned and mentioned <= HARDENING_SERIES, (
        f"skill references PRs outside the hardening series: {mentioned - HARDENING_SERIES}"
    )


def test_skill_metadata_consistent_with_docs_page():
    """The mirrored docs page must carry the same description."""
    fm = _frontmatter(SKILL_MD)
    docs_page = (
        REPO
        / "website"
        / "docs"
        / "user-guide"
        / "skills"
        / "bundled"
        / "security"
        / "security-secrets-protocol.md"
    )
    assert docs_page.exists(), "missing mirrored docs page"
    page_fm = _frontmatter(docs_page)
    assert page_fm["description"] == fm["description"], (
        "docs page description diverged from SKILL.md"
    )

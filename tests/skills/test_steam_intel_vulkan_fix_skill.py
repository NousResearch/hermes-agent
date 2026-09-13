"""Tests for the `steam-intel-vulkan-fix` skill.

The skill's whole job is routing a very specific crash signature (Access
Violation in igvk64.dll) to a specific, safe remediation (force D3D11 via
Steam launch options). These tests keep that routing honest: the fault
signature and the fix must both be present and unambiguous, and the skill
must never bake in machine-local paths that break for other users.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "gaming" / "steam-intel-vulkan-fix"
SKILL_MD = SKILL_DIR / "SKILL.md"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_frontmatter_has_required_fields(skill_text):
    """A skill that fails to load is invisible to every agent."""
    assert skill_text.startswith("---")
    assert re.search(r"\n---\s*\n", skill_text[3:]), "frontmatter not closed"
    for field in ("name:", "description:", "version:", "author:", "license:", "platforms:"):
        assert field in skill_text.split("---", 2)[1], f"missing frontmatter field: {field}"


def test_description_within_hardline_budget(skill_text):
    """Repo review rejects descriptions over 60 chars (index truncates at 57)."""
    fm = skill_text.split("---", 2)[1]
    m = re.search(r'description:\s*"?([^"\n]+)"?', fm)
    assert m, "no description found"
    desc = m.group(1).strip()
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline: 60)"
    assert desc.endswith("."), "description must end with a period"


def test_fault_signature_is_routed(skill_text):
    """The skill must name the exact crash signature it exists to fix."""
    assert "igvk64.dll" in skill_text
    assert "0xc0000005" in skill_text
    assert "+r_rhirenderfamily d3d11" in skill_text


def test_launch_options_persistence_is_documented(skill_text):
    """The fix must include the persistent (localconfig.vdf) route, not only the UI route."""
    assert "localconfig.vdf" in skill_text
    assert "LaunchOptions" in skill_text
    assert "backup" in skill_text.lower()


def test_no_machine_local_paths(skill_text):
    """Committed skills must not bake in the author's machine paths."""
    banned = re.compile(r"/home/|/Users/(?!.*Steam)|C:\\Users\\[^S]|AppData/Local/hermes")
    for line in skill_text.splitlines():
        assert not banned.search(line), f"machine-local path in line: {line}"


def test_body_has_verification_section(skill_text):
    """Every repo skill must say how to prove it worked."""
    assert re.search(r"## Verification", skill_text)

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "social-media" / "dataverse" / "SKILL.md"

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def _text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter() -> str:
    text = _text()
    assert text.startswith("---"), "frontmatter must start at byte 0"
    end = text.index("\n---", 3)
    return text[3:end]


def test_description_is_one_short_sentence():
    m = re.search(r"^description: (.+)$", _frontmatter(), re.MULTILINE)
    assert m, "frontmatter description missing"
    desc = m.group(1).strip().strip('"')
    assert len(desc) <= 60, f"description is {len(desc)} chars, max 60"
    assert desc.endswith(".")


def test_frontmatter_declares_prerequisites_and_platforms():
    fm = _frontmatter()
    assert re.search(r"^name: dataverse$", fm, re.MULTILINE)
    assert "platforms: [linux, macos]" in fm
    assert "commands: [dv]" in fm
    assert "MC_API" in fm


def test_required_sections_present_in_order():
    text = _text()
    assert "# Dataverse Skill" in text
    positions = [text.find(h) for h in REQUIRED_SECTIONS]
    missing = [h for h, p in zip(REQUIRED_SECTIONS, positions) if p == -1]
    assert not missing, f"missing sections: {missing}"
    assert positions == sorted(positions), "sections out of required order"


def test_documents_real_cli_surface():
    text = _text()
    for snippet in (
        "dv status",
        "dv auth",
        "dv search x",
        "dv search reddit",
        "dv gravity create",
        "dv gravity build",
        "dv gravity dataset",
        "-o json",
        "--from",
        "--mode",
        "`terminal`",
    ):
        assert snippet in text, f"expected {snippet!r} in SKILL.md"


def test_does_not_document_nonexistent_flags():
    text = _text()
    # dv uses --from/--to; these flags don't exist on dv search
    assert "--start" not in text
    assert "--end " not in text
    # subreddits are targeted via an r/<name> keyword, not a flag
    assert "r/MachineLearning" in text


def test_compares_against_xurl_not_retired_xitter():
    text = _text()
    assert "xitter" not in text, "xitter was replaced by xurl on main"
    assert "## Comparison with xurl" in text
    assert "OAuth 2.0" in text

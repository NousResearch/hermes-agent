"""Tests for the scrollcraft optional skill (ported from nateherkai/scroll-craft)."""

import re
from pathlib import Path

import yaml

_STAGING = Path(__file__).resolve().parents[2] / "optional-skills/web-development/scrollcraft"
_FALLBACK = (
    Path.home() / ".hermes/hermes-agent/optional-skills/web-development/scrollcraft"
)
SKILL_DIR = _STAGING if _STAGING.is_dir() else _FALLBACK
SKILL_MD = SKILL_DIR / "SKILL.md"


def _frontmatter() -> dict:
    text = SKILL_MD.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    data = yaml.safe_load(match.group(1))
    assert isinstance(data, dict)
    return data


def test_frontmatter_parses():
    fm = _frontmatter()
    assert fm["name"] == "scrollcraft"


def test_description_length_and_period():
    desc = _frontmatter()["description"]
    assert isinstance(desc, str)
    assert len(desc) <= 60, f"description is {len(desc)} chars"
    assert desc.endswith(".")


def test_platforms_present():
    fm = _frontmatter()
    assert "platforms" in fm
    assert isinstance(fm["platforms"], list) and fm["platforms"]


def test_license_is_mit():
    assert _frontmatter()["license"] == "MIT"


def test_license_file_exists():
    lic = SKILL_DIR / "LICENSE.txt"
    assert lic.is_file()
    assert "MIT License" in lic.read_text(encoding="utf-8")


def test_mentioned_paths_exist_or_marked_upstream():
    """Every vendored-tree relative path mentioned in SKILL.md exists on disk,
    or its line is marked as upstream / not vendored."""
    path_re = re.compile(
        r"(?:references|engine|scripts|templates)/[\w.-]+\.(?:md|html|mjs|css|js)"
    )
    for line in SKILL_MD.read_text(encoding="utf-8").splitlines():
        for rel in path_re.findall(line):
            if (SKILL_DIR / rel).is_file():
                continue
            low = line.lower()
            assert "upstream" in low or "not vendored" in low, (
                f"SKILL.md mentions missing path {rel!r} without an "
                f"upstream/not-vendored marker on the same line: {line!r}"
            )


def test_no_foreign_agent_residue():
    """Prose files carry no upstream-agent-specific residue."""
    residue = re.compile(r"claude|allowed-tools|AskUserQuestion", re.IGNORECASE)
    prose = [SKILL_MD] + sorted((SKILL_DIR / "references").glob("*.md"))
    for path in prose:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert not residue.search(line), f"{path.name}:{lineno}: {line!r}"


def test_scripts_present_and_nonempty():
    scripts_dir = SKILL_DIR / "scripts"
    mjs = sorted(scripts_dir.glob("*.mjs"))
    expected = {
        "doctor.mjs",
        "kie.mjs",
        "serve.mjs",
        "shoot.mjs",
        "workspace.mjs",
        "worldflight-assert.mjs",
    }
    assert {p.name for p in mjs} == expected
    for p in mjs:
        assert p.stat().st_size > 0, f"{p.name} is empty"

"""Bundled SKILL.md must not hardcode ~/.hermes for executable script paths (#93152)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / "skills"

# Copied-and-run assignment: S=~/.hermes/skills/...
_ASSIGN = re.compile(r"(?m)^[A-Za-z_][A-Za-z0-9_]*=(?:['\"])?~/\.hermes/skills/")
# Copied-and-run invocation: python ~/.hermes/skills/...
_INVOKE = re.compile(
    r"(?m)(?:^|[\s`])(?:python3?|bash|uv)\s+(?:['\"])?~/\.hermes/skills/"
)


def test_bundled_skill_md_does_not_hardcode_default_home_for_scripts():
    offenders: list[str] = []
    for skill_md in SKILLS.rglob("SKILL.md"):
        text = skill_md.read_text(encoding="utf-8")
        if _ASSIGN.search(text) or _INVOKE.search(text):
            offenders.append(str(skill_md.relative_to(REPO).as_posix()))
    assert offenders == [], (
        "SKILL.md documents a ~/.hermes executable path; use "
        "${HERMES_HOME:-$HOME/.hermes} instead: " + ", ".join(offenders)
    )

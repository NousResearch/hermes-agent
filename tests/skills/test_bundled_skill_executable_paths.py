"""Bundled SKILL.md must not hardcode ~/.hermes for executable script paths (#93152)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / "skills"

# Copied-and-run assignment: S=~/.hermes/skills/...
_ASSIGN = re.compile(r"(?m)^[A-Za-z_][A-Za-z0-9_]*=(?:['\"])?~/\.hermes/skills/")
# Copied-and-run invocation: python/bash/sh/uv ~/.hermes/skills/...
# ``sh`` is included so a shell-script skill does not slip the net (#93172 review).
# Direct execution (``~/.hermes/skills/foo.sh`` with no interpreter) and paths
# embedded mid-sentence stay out of this best-effort net.
# optional-skills/ is not scanned: those docs describe the installed location,
# which is not the repo layout, so the same glob would false-fail.
_INVOKE = re.compile(
    r"(?m)(?:^|[\s`])(?:python3?|bash|sh|uv)\s+(?:['\"])?~/\.hermes/skills/"
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


def test_invoke_pattern_catches_sh_as_well_as_python_bash_uv():
    samples = (
        "python ~/.hermes/skills/foo.py",
        "python3 ~/.hermes/skills/foo.py",
        "bash ~/.hermes/skills/foo.sh",
        "sh ~/.hermes/skills/foo.sh",
        "`uv ~/.hermes/skills/foo.py`",
        "sh '~/.hermes/skills/foo.sh'",
    )
    for sample in samples:
        assert _INVOKE.search(sample), sample
    assert _INVOKE.search("bash ~/.hermes/skills/foo.sh")
    # A longer interpreter must not be swallowed by the ``sh`` alternative.
    assert _INVOKE.search("bash ~/.hermes/skills/foo.sh").group(0).lstrip("` ").startswith("bash")

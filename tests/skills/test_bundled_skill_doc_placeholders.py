"""Bundled skills must not carry angle-bracket tool placeholders.

skill_view returns a skill's full text, so any bundled SKILL.md containing
the literal placeholder token reported in #132504 (an angle-bracket tool
name, the exact shape provider gateways classify as role-tag injection)
makes every later request that carries that skill content unrecoverable on
affected providers. Documented placeholders use brace form instead,
matching the precedent in
skills/autonomous-ai-agents/hermes-agent/references/native-mcp.md.

The scanned token is assembled from fragments so this source file never
contains the literal itself.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ROLE_TAG_PLACEHOLDER = re.compile("<" + "tool" + ">")


def bundled_skill_files() -> list[Path]:
    paths = sorted(
        p
        for p in (REPO_ROOT / "skills").rglob("*.md")
        if ".hub" not in p.parts
    )
    optional = REPO_ROOT / "optional-skills"
    if optional.is_dir():
        paths += sorted(
            p
            for p in optional.rglob("*.md")
            if ".hub" not in p.parts
        )
    assert paths, "bundled skills tree not found under skills/ or optional-skills/"
    return paths


def test_bundled_skills_have_no_role_tag_tool_placeholder():
    offenders = []
    for path in bundled_skill_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if ROLE_TAG_PLACEHOLDER.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
    assert offenders == [], (
        "bundled skill docs contain the angle-bracket tool placeholder"
        " reported in #132504; use the brace form ({tool}) instead:"
        f" {offenders}"
    )

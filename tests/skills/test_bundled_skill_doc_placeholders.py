"""Bundled skill files must not carry bare chat-role tags.

skill_view returns a skill's text verbatim, so a bundled doc containing an
angle-bracketed chat role name (the placeholder shape reported in #132504)
lands in every later request of the session; OpenRouter's gateway classifies
it as role-tag injection and 403s the whole conversation. Document such
placeholders in brace form instead, matching the precedent in
skills/autonomous-ai-agents/hermes-agent/references/native-mcp.md.

The pattern is assembled from fragments so this file never contains the literal.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ROLE_TAG = re.compile("<" + "/?(?:tool|system|assistant)" + ">", re.IGNORECASE)
# Every extension skill_view will serve (tools.skills_tool_plugin._SKILL_FILE_EXTS) plus asset scripts.
SERVED_EXTS = {".md", ".py", ".yaml", ".yml", ".json", ".tex", ".sh", ".mjs", ".js", ".txt"}


def bundled_skill_files() -> list[Path]:
    paths = [
        p
        for root in ("skills", "optional-skills")
        if (REPO_ROOT / root).is_dir()
        for p in sorted((REPO_ROOT / root).rglob("*"))
        if p.suffix in SERVED_EXTS and p.is_file() and ".hub" not in p.parts
    ]
    assert paths, "bundled skills tree not found under skills/ or optional-skills/"
    return paths


def test_bundled_skills_have_no_role_tag_placeholder():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path in bundled_skill_files()
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if ROLE_TAG.search(line)
    ]
    assert offenders == [], (
        "bundled skill files contain an angle-bracketed chat role name (#132504);"
        f" use the brace form (e.g. {{tool}}) instead: {offenders}"
    )

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


def bundled_skill_texts() -> list[tuple[Path, str]]:
    # skill_view serves ANY file under a skill root that decodes as UTF-8 text, whatever its
    # suffix, so scan everything and skip only what fails that same strict decode.
    texts = []
    for root in ("skills", "optional-skills", "plugins"):
        if not (REPO_ROOT / root).is_dir():
            continue
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if not path.is_file() or ".hub" in path.parts:
                continue
            try:
                texts.append((path, path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                continue
    assert texts, "bundled skills tree not found under skills/, optional-skills/ or plugins/"
    return texts


def test_bundled_skills_have_no_role_tag_placeholder():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path, text in bundled_skill_texts()
        for lineno, line in enumerate(text.splitlines(), start=1)
        if ROLE_TAG.search(line)
    ]
    assert offenders == [], (
        "bundled skill files contain an angle-bracketed chat role name (#132504);"
        f" use the brace form (e.g. {{tool}}) instead: {offenders}"
    )


def test_tool_error_strings_have_no_role_tag():
    # Tool results enter history and are re-sent every turn, like a skill_view load.
    from tools.connectors.gateway.merge import partition_calls
    from tools.tool_search_validation import not_deferrable_error

    messages = [
        not_deferrable_error("no_such_tool_xyz"),
        partition_calls([{"name": "connectors__bad", "arguments": {}}]).errors[0]["error"]["message"],
    ]
    assert [m for m in messages if ROLE_TAG.search(m)] == []

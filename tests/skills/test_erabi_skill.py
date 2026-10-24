"""Hermetic tests for the erabi skill.

Stdlib + pytest only; NO live network. Validates that the SKILL.md matches
Hermes conventions (config key, discovered tool-name namespace, authoring
standards). Run with:

    scripts/run_tests.sh tests/skills/test_erabi_skill.py -q
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "autonomous-ai-agents" / "erabi" / "SKILL.md"

TOOLS = [
    "register",
    "discover",
    "intent",
    "report_outcome",
    "pending_outcomes",
    "confirm_outcome",
    "my_reputation",
    "my_earnings",
]


def _text():
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "SKILL.md must start with a YAML frontmatter block"
    return m.group(1)


def test_description_is_short_one_sentence():
    m = re.search(r"^description: (.*)$", _text(), re.MULTILINE)
    assert m, "missing description"
    desc = m.group(1).strip()
    assert len(desc) <= 60, f"description is {len(desc)} chars (max 60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"
    # No marketing words, and don't repeat the skill name.
    lowered = desc.lower()
    for banned in ("powerful", "comprehensive", "seamless", "advanced", "erabi"):
        assert banned not in lowered, f"description should not contain {banned!r}"


def test_author_credits_human_first():
    fm = _frontmatter(_text())
    m = re.search(r"^author: (.*)$", fm, re.MULTILINE)
    assert m, "missing author"
    author = m.group(1)
    assert "@HMAKT99" in author, "human contributor + GitHub handle must be credited"
    assert author.index("HMAKT99") < author.find("Hermes Agent") or "Hermes Agent" not in author, (
        "the human contributor must be listed before 'Hermes Agent'"
    )


def test_uses_hermes_config_key_not_generic_json():
    text = _text()
    assert "mcp_servers:" in text, "must document the Hermes `mcp_servers` config key"
    # The generic Claude/Cursor-style `mcpServers` JSON must not be the headline setup.
    assert "mcpServers" not in text, "must not use the generic `mcpServers` JSON block"


def test_tools_use_hermes_mcp_namespace():
    text = _text()
    for tool in TOOLS:
        assert f"mcp__erabi__{tool}" in text, f"tool {tool} must appear as mcp__erabi__{tool}"
    # Bare tool names must not headline the prose (e.g. a lone `register` backtick).
    for tool in TOOLS:
        assert f"`{tool}`" not in text, f"bare `{tool}` should be namespaced as mcp__erabi__{tool}"


def test_has_required_sections():
    text = _text()
    for section in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert section in text, f"missing required section: {section}"


def test_platforms_declared():
    fm = _frontmatter(_text())
    assert re.search(r"^platforms: \[.*\]$", fm, re.MULTILINE), "must declare platforms"


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)

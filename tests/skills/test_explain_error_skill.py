"""Static checks for optional-skills/software-development/explain-error/SKILL.md.

Verifies the skill meets the Hermes authoring standards (AGENTS.md) and that
its Python traceback / exception-chaining guidance is technically correct.
Pure stdlib + pytest, no network, no imports from the skill.
"""

import re
from pathlib import Path

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "software-development"
    / "explain-error"
    / "SKILL.md"
)


def _read() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _norm(text: str) -> str:
    """Collapse whitespace so line-wrapped quoted phrases match as substrings."""
    return re.sub(r"\s+", " ", text)


def _frontmatter(text: str) -> dict:
    """Parse the leading YAML frontmatter into a flat dict of top-level keys."""
    assert text.startswith("---"), "SKILL.md must start with a frontmatter block"
    end = text.index("\n---", 3)
    block = text[3:end]
    fields = {}
    for line in block.splitlines():
        # Only top-level (non-indented) `key: value` lines with an inline value.
        m = re.match(r"^([A-Za-z_][\w.]*): (.+)$", line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def test_skill_md_exists():
    assert SKILL_MD.is_file(), f"missing skill file: {SKILL_MD}"


def test_description_is_short_single_sentence():
    desc = _frontmatter(_read()).get("description", "")
    assert desc, "description field is required"
    # Reject a YAML block scalar; the description must be one inline sentence.
    assert desc not in ("|", ">", "|-", ">-"), "description must not be a block scalar"
    assert len(desc) <= 60, f"description is {len(desc)} chars (max 60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"
    assert desc.count(".") == 1, "description must be a single sentence"


def test_description_has_no_marketing_words():
    desc = _frontmatter(_read()).get("description", "").lower()
    for word in ("powerful", "comprehensive", "seamless", "advanced", "ultimate", "blazing"):
        assert word not in desc, f"description uses marketing word: {word!r}"
    # Must not simply repeat the skill name.
    assert "explain error" not in desc


def test_author_credits_human_first():
    author = _frontmatter(_read()).get("author", "")
    assert "@HeLLGURD" in author, "author must credit the human contributor's handle"
    if "Hermes" in author:
        assert author.index("HeLLGURD") < author.index("Hermes"), "human must be credited first"


def test_required_sections_present_and_ordered():
    text = _read()
    required = [
        "# Explain Error Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    last = -1
    for heading in required:
        idx = text.find(heading)
        assert idx != -1, f"missing required section: {heading!r}"
        assert idx > last, f"section out of order: {heading!r}"
        last = idx


def test_prose_points_at_native_tools_not_raw_shell():
    """Reading/searching must reference native tools, not raw shell utilities."""
    text = _read()
    assert "`read_file`" in text, "reading source should point at read_file"
    assert "`search_files`" in text, "searching should point at search_files"
    for bad in ("sed -n", "ls -la", "cat ", "grep ", "head -", "tail -"):
        assert bad not in text, f"raw shell util should be a native tool: {bad!r}"


def test_python_traceback_ordering_is_correct():
    """The exception type/message sits at the BOTTOM of a Python traceback."""
    norm = _norm(_read())
    # Corrected statement: for Python the message is LAST / at the very bottom.
    assert "message LAST, at the very bottom" in norm
    assert "Read it bottom-up" in norm
    # Guard against a regression to the reversed (buggy) wording, which put the
    # Python exception type/message "at the top".
    low = norm.lower()
    assert "at the top (python)" not in low
    assert "message at the top" not in low


def test_chained_exception_ordering_is_correct():
    """Original/cause is the FIRST/top block; final raised is the LAST/bottom."""
    norm = _norm(_read())
    assert "During handling of the above exception, another exception occurred" in norm
    assert "The above exception was the direct cause of the following exception" in norm
    # Corrected wording: the original cause is the first/top block.
    assert "original cause is the FIRST/top block" in norm
    assert "LAST/bottom block" in norm
    # The buggy version labelled the original exception "the bottom one".
    assert "the bottom one" not in norm
    assert "original exception first (the bottom one)" not in norm

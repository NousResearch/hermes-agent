"""Contract tests: requesting-code-review must stay read-only unless authorized.

Review / verify / inspect / audit / report requests must not default to
auto-fix or `git add -A` / `git commit`. Commit remains available behind an
explicit authorization gate, and then only intended paths may be staged.
"""

from __future__ import annotations

import re
from pathlib import Path

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "requesting-code-review"
    / "SKILL.md"
)
SKILL = SKILL_PATH.read_text(encoding="utf-8")


def _section(text: str, heading_needle: str) -> str:
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("## ") and heading_needle in line),
        None,
    )
    assert start is not None, f"missing section containing {heading_needle!r}"
    end = next(
        (j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_skill_file_exists() -> None:
    assert SKILL_PATH.is_file()


def test_review_only_default_path_does_not_commit() -> None:
    """SKILL must gate commit behind explicit authorization."""
    # origin/main recommends this unguarded mutation as Step 8:
    #   git add -A && git commit -m "[verified] <description>"
    assert "git add -A && git commit" not in SKILL
    # Actionable command form (start of a line), not a later prohibition.
    assert not re.search(r"^git add -A\b", SKILL, re.MULTILINE)

    assert re.search(
        r"(?i)review-only|read-only|explicit.*commit|authoriz",
        SKILL,
    )


def test_all_passed_does_not_unconditionally_commit() -> None:
    """After 'All passed', review-only must report and stop — not jump to commit."""
    step6 = _section(SKILL, "Step 6")
    # origin/main unguarded success path:
    assert "Proceed to Step 8 (commit)." not in step6
    assert re.search(r"(?i)authoriz|review-only|read-only", step6)


def test_failures_do_not_unconditionally_auto_fix() -> None:
    """Auto-fix mutates the tree; review-only must report and stop."""
    step6 = _section(SKILL, "Step 6")
    # origin/main unguarded failure path:
    assert "then proceed to Step 7 (auto-fix)." not in step6
    step7 = _section(SKILL, "Step 7")
    assert re.search(r"(?i)authoriz|explicit|review-only", step7)


def test_authorized_commit_stages_intended_paths_only() -> None:
    """Fail-open: gated commit stays; never recommend staging the whole tree."""
    step8 = _section(SKILL, "Step 8")
    assert "git commit" in step8
    assert re.search(r"(?i)authoriz|explicit", step8)
    assert re.search(r"git add -- ", step8) or re.search(r"(?i)intended paths", step8)
    assert "git add -A &&" not in step8


def test_when_to_use_separates_review_only_from_precommit() -> None:
    when = _section(SKILL, "When to Use")
    # origin/main lumps verify with commit/push/ship:
    #   When user says "commit", "push", "ship", "done", "verify", or "review before merge"
    assert '"done", "verify"' not in when
    assert re.search(r"(?i)review-only", when)
    assert re.search(r"(?i)read-only|do not .*commit|not .*commit", when)

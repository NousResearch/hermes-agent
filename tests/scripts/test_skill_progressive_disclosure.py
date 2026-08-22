"""Progressive disclosure enforcement for in-repo skills.

A skill's SKILL.md must be lean: the always-loaded surface carries the
trigger, the steps, and the completion criteria. Bulk reference material
(maps, recipes, worked receipts, deep API tables) lives in
references/ and is loaded only when needed.

This test enforces the shape the hermes-agent-skill-authoring skill
documents: SKILL.md under the lean threshold, and any skill whose
SKILL.md exceeds it must push bulk into references/ instead.
"""

from __future__ import annotations

import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"

# The lean SKILL.md ceiling. Peer skills sit at 8-15k chars (~150-400
# lines). The authoring skill's hard cap is 100k chars; the lean bar is
# much lower so the always-loaded surface stays cheap.
SKILL_MD_MAX_LINES = 500
SKILL_MD_MAX_CHARS = 60_000

# Skills that legitimately exceed the lean bar AND carry references/
# (progressive disclosure satisfied: bulk lives in references/, SKILL.md
# is the always-loaded surface). Enumerated 2026-08-05 from the repo tree.
KNOWN_LARGE_SKILLS = {
    "research/research-paper-writing": 2377,
    "creative/comfyui": 612,
    "mlops/evaluation/weights-and-biases": 598,
    "creative/p5js": 556,
    "github/github-repo-management": 516,
}

# Skills that CURRENTLY violate the lean bar AND lack references/ —
# they fail until they are restructured (progressive disclosure is
# mandatory, not aspirational). This list shrinks as they are fixed;
# it must never grow.
DISCLOSURE_VIOLATIONS = {
    "autonomous-ai-agents/claude-code": 745,
    "creative/claude-design": 650,
    "creative/humanizer": 647,
    "research/llm-wiki": 507,
}


def _skills() -> list[tuple[str, pathlib.Path]]:
    out = []
    for root, dirs, files in os.walk(SKILLS_DIR):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        root_p = pathlib.Path(root)
        for f in files:
            if f == "SKILL.md":
                rel = root_p.relative_to(SKILLS_DIR).as_posix().replace("\\", "/")
                out.append((rel, root_p / f))
    return sorted(out)


def _line_count(p: pathlib.Path) -> int:
    with open(p, "rb") as fh:
        return sum(1 for _ in fh)


def _char_count(p: pathlib.Path) -> int:
    return p.stat().st_size


def test_every_skill_has_lean_skilli_md():
    """SKILL.md stays under the lean bar unless explicitly known-large or
    already tracked as a disclosure violation.

    KNOWN_LARGE = over bar but structured (references/). DISCLOSURE_VIOLATIONS
    = over bar without references/, tracked by their own monotonic-shrink test.
    A skill NOT in either list crossing the bar is a NEW violation — the hard
    failure this test exists for.
    """
    violations = []
    for rel, p in _skills():
        if rel in KNOWN_LARGE_SKILLS or rel in DISCLOSURE_VIOLATIONS:
            continue
        lines = _line_count(p)
        chars = _char_count(p)
        if lines > SKILL_MD_MAX_LINES or chars > SKILL_MD_MAX_CHARS:
            violations.append((rel, lines, chars))
    assert not violations, (
        "SKILL.md over the progressive-disclosure bar (NEW, untracked):\n"
        + "\n".join(f"  {rel}: {lines} lines / {chars} chars" for rel, lines, chars in violations)
        + "\nMove bulk into references/ — SKILL.md is the always-loaded surface. "
        "Or if this is a legitimately large skill, track it in KNOWN_LARGE_SKILLS."
    )


def test_known_large_skills_do_not_grow():
    """The known-large list is frozen — those skills must not grow, and no
    new skill joins without a kill-track justification (edit this test)."""
    for rel, recorded in KNOWN_LARGE_SKILLS.items():
        p = SKILLS_DIR / pathlib.Path(*rel.split("/")) / "SKILL.md"
        if p.exists():
            n = _line_count(p)
            assert n <= recorded + 5 or recorded == 0, (
                f"{rel} grew past its frozen size (recorded {recorded}, now {n})"
            )


def test_disclosure_violations_shrink_monotonically():
    """The violation list is the restructure track — it must shrink, never grow.

    claude-design, humanizer, and claude-code exceed the lean bar without
    references/ today. When each is restructured (bulk moved to
    references/), its entry is REMOVED from this list — the shrinking is
    the completion record, the same shape as the 2K-law manifest.
    """
    for rel, recorded in DISCLOSURE_VIOLATIONS.items():
        p = SKILLS_DIR / pathlib.Path(*rel.split("/")) / "SKILL.md"
        if not p.exists():
            continue  # already removed from the repo — resolved
        n = _line_count(p)
        refs = (SKILLS_DIR / pathlib.Path(*rel.split("/")) / "references").is_dir()
        if refs and n <= SKILL_MD_MAX_LINES:
            raise AssertionError(
                f"{rel} is now lean ({n}L) with references/ — REMOVE it from "
                "DISCLOSURE_VIOLATIONS (the list must shrink, not sit stale)."
            )
        assert n <= recorded + 5, (
            f"{rel} GREW past its recorded violation size ({recorded} -> {n}) — "
            "restructure it, don't let it grow."
        )


def test_large_skills_have_references_dir():
    """Known-large skills must keep the bulk in references/, not SKILL.md."""
    for rel in KNOWN_LARGE_SKILLS:
        skill_dir = SKILLS_DIR / pathlib.Path(*rel.split("/"))
        refs = skill_dir / "references"
        assert refs.is_dir(), f"{rel} is known-large but has no references/ dir"
        ref_files = [f for f in refs.iterdir() if f.is_file()]
        assert ref_files, f"{rel} has an empty references/ dir"


def test_reference_files_honor_the_2k_law():
    """references/ files are also code/docs — the 2K law applies to them."""
    violations = []
    for root, dirs, files in os.walk(SKILLS_DIR):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        root_p = pathlib.Path(root)
        for f in files:
            p = root_p / f
            if p.suffix.lower() in {".py", ".sh", ".js", ".ts"}:
                n = _line_count(p)
                if n > 2000:
                    rel = p.relative_to(REPO_ROOT).as_posix().replace("\\", "/")
                    violations.append((n, rel))
    assert not violations, (
        "Skill code files over 2,000 lines:\n"
        + "\n".join(f"  {n:>6}  {rel}" for n, rel in sorted(violations, reverse=True))
    )


def test_kill_locks_skill_is_lean():
    """The campaign-operations-kill-locks skill must stay under the bar."""
    p = SKILLS_DIR / "github" / "campaign-operations-kill-locks" / "SKILL.md"
    assert p.exists(), "campaign-operations-kill-locks SKILL.md missing"
    assert _line_count(p) <= SKILL_MD_MAX_LINES, "kill-locks SKILL.md grew past the lean bar"

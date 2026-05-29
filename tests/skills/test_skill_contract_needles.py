"""Skill contract needle / pitfall linter.

For a small allowlist of Gond-authoritative SKILL.md files, assert that
required top-section needles exist (e.g., memory-hygiene rules, runtime
LAW lines from the top-loaded Contract section) and that forbidden /
legacy patterns do not reappear in skill examples.

Provenance: grafted from ``mvanhorn/last30days-skill`` commit
``1e03af19e0ad435ee6d227a3593b0c6e5d2ecbe8`` ``tests/test_plugin_contract.py``
plus the ``docs/solutions/RELEASE_CONSISTENCY_TEST_CASCADE_*`` lesson
(MIT, copyright 2026 Matt Van Horn). Adapted subset: lightweight skill
contract-surface checks. Rejected subset: monolithic SKILL.md style,
multi-harness installer expansion, two-file consistency cascade.

Anti-cascade discipline (from the source's own postmortem): this file is
the single source of truth. The allowlist + needle/forbidden registries
live here so unrelated branches that touch other SKILL.md files cannot
red CI. We deliberately do NOT compare against the regenerated copies
under ``website/docs/``; ``tests/website/test_generate_skill_docs.py``
already owns the generator-freshness invariant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Pattern

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SkillContract:
    """A Gond-authoritative skill and the contract surface it ships."""

    path: Path
    required_top_needles: tuple[tuple[str, Pattern[str]], ...]
    forbidden_patterns: tuple[tuple[str, Pattern[str]], ...] = ()
    top_section_lines: int = 120


# Patterns for install/scaffold paths that have been removed. Keep this
# list empty until a path is *actually* deprecated — speculative entries
# break unrelated branches without preventing any real failure.
_LEGACY_PATH_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = ()


GOND_AUTHORITATIVE_SKILLS: tuple[SkillContract, ...] = (
    SkillContract(
        path=REPO_ROOT
        / "skills"
        / "software-development"
        / "hermes-agent-skill-authoring"
        / "SKILL.md",
        required_top_needles=(
            (
                "top-loaded Contract heading (`## Contract` / `## Non-Negotiable Rules`)",
                re.compile(
                    r"^##\s+(?:Contract|Non-Negotiable Rules|Runtime Contract)\b",
                    re.MULTILINE,
                ),
            ),
            (
                "memory-hygiene rule (procedural lessons → solution notes, not memory)",
                re.compile(
                    r"procedural lessons?[^\n]*(?:solution notes?|skills?)[^\n]*"
                    r"(?:not|never)[^\n]*memory",
                    re.IGNORECASE,
                ),
            ),
            (
                "provenance LAW for external pattern grafts",
                re.compile(
                    r"provenance[^\n]*law|external pattern grafts?[^\n]*"
                    r"(?:source|license|provenance)",
                    re.IGNORECASE,
                ),
            ),
        ),
        forbidden_patterns=_LEGACY_PATH_PATTERNS,
        top_section_lines=120,
    ),
)


def _top_section(text: str, lines: int) -> str:
    return "\n".join(text.splitlines()[:lines])


def _find_violations(contract: SkillContract) -> tuple[list[str], list[str]]:
    """Return (missing_required, forbidden_hits) for one contract."""
    text = contract.path.read_text(encoding="utf-8")
    top = _top_section(text, contract.top_section_lines)

    missing = [
        label
        for label, pat in contract.required_top_needles
        if not pat.search(top)
    ]
    forbidden_hits: list[str] = []
    for label, pat in contract.forbidden_patterns:
        m = pat.search(text)
        if m:
            forbidden_hits.append(f"{label}: matched {m.group(0)!r}")
    return missing, forbidden_hits


# ── Live-tree assertions ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "contract",
    GOND_AUTHORITATIVE_SKILLS,
    ids=lambda c: str(c.path.relative_to(REPO_ROOT)),
)
def test_gond_authoritative_skill_has_required_needles(
    contract: SkillContract,
) -> None:
    if not contract.path.exists():
        pytest.skip(
            f"{contract.path.relative_to(REPO_ROOT)} not present on this branch"
        )
    missing, _ = _find_violations(contract)
    assert not missing, (
        f"{contract.path.relative_to(REPO_ROOT)} is missing required "
        f"top-section needle(s): {missing}. These rules must survive "
        f"truncation; if you reworded one, update the regex in this test "
        f"rather than dropping it."
    )


@pytest.mark.parametrize(
    "contract",
    GOND_AUTHORITATIVE_SKILLS,
    ids=lambda c: str(c.path.relative_to(REPO_ROOT)),
)
def test_gond_authoritative_skill_has_no_forbidden_patterns(
    contract: SkillContract,
) -> None:
    if not contract.path.exists():
        pytest.skip(
            f"{contract.path.relative_to(REPO_ROOT)} not present on this branch"
        )
    _, forbidden = _find_violations(contract)
    assert not forbidden, (
        f"{contract.path.relative_to(REPO_ROOT)} reintroduced legacy / "
        f"removed pattern(s): {forbidden}. Remove the example, or — if "
        f"the path was un-deprecated — update _LEGACY_PATH_PATTERNS in "
        f"this test file."
    )


# ── Failure-mode fixtures ──────────────────────────────────────────────────
#
# Synthetic SKILL.md content exercises the linter primitives to prove each
# documented failure mode is caught. These fixtures never read from the real
# tree so they cannot cascade-red unrelated branches.


_FAKE_PASSING_SKILL = (
    "---\n"
    "name: example\n"
    "description: Use when ...\n"
    "---\n\n"
    "# Example\n\n"
    "## Overview\n\nIntro.\n\n"
    "## When to Use\n\n- trigger\n\n"
    "## Contract\n\n"
    "1. **Provenance law:** every external pattern graft must record "
    "source and license.\n"
    "2. **Solution-note law:** reusable procedural lessons belong in "
    "solution notes, not in durable personal memory.\n"
)


def _ad_hoc_contract(
    tmp_path: Path,
    body: str,
    *,
    forbidden: tuple[tuple[str, Pattern[str]], ...] = (),
) -> SkillContract:
    path = tmp_path / "SKILL.md"
    path.write_text(body, encoding="utf-8")
    return SkillContract(
        path=path,
        required_top_needles=GOND_AUTHORITATIVE_SKILLS[0].required_top_needles,
        forbidden_patterns=forbidden,
        top_section_lines=120,
    )


def test_linter_accepts_skill_with_all_needles(tmp_path: Path) -> None:
    contract = _ad_hoc_contract(tmp_path, _FAKE_PASSING_SKILL)
    missing, forbidden = _find_violations(contract)
    assert missing == []
    assert forbidden == []


def test_linter_flags_missing_memory_hygiene_needle(tmp_path: Path) -> None:
    body = _FAKE_PASSING_SKILL.replace(
        "**Solution-note law:** reusable procedural lessons belong in "
        "solution notes, not in durable personal memory.",
        "**Helpful tip:** write things down somewhere.",
    )
    contract = _ad_hoc_contract(tmp_path, body)
    missing, _ = _find_violations(contract)
    assert any("memory-hygiene" in label for label in missing), missing


def test_linter_flags_missing_contract_heading(tmp_path: Path) -> None:
    body = _FAKE_PASSING_SKILL.replace("## Contract", "## Notes")
    contract = _ad_hoc_contract(tmp_path, body)
    missing, _ = _find_violations(contract)
    assert any("Contract heading" in label for label in missing), missing


def test_linter_flags_reintroduced_legacy_install_path(
    tmp_path: Path,
) -> None:
    forbidden = (
        (
            "legacy `hermes-cli skills add` install command (fixture only)",
            re.compile(r"hermes-cli\s+skills\s+add", re.IGNORECASE),
        ),
    )
    body = (
        _FAKE_PASSING_SKILL
        + "\n```\nhermes-cli skills add my-skill\n```\n"
    )
    contract = _ad_hoc_contract(tmp_path, body, forbidden=forbidden)
    _, forbidden_hits = _find_violations(contract)
    assert forbidden_hits, (
        "linter should flag the reintroduced legacy install path"
    )


def test_linter_ignores_text_below_top_section_window(
    tmp_path: Path,
) -> None:
    """Needles must appear in the top window.

    Failure mode from the source spike §2: late critical rules are
    fragile, models stop reading before reaching them.
    """
    body = (
        "---\nname: x\ndescription: y\n---\n\n"
        "# Title\n\n## Overview\n\nLong intro.\n\n"
        + "intro line\n" * 200
        + "## Contract\n\n"
        + "**Solution-note law:** procedural lessons belong in solution "
        + "notes, not in durable personal memory. Provenance law applies.\n"
    )
    contract = _ad_hoc_contract(tmp_path, body)
    missing, _ = _find_violations(contract)
    assert missing, (
        "linter should flag laws that drift below the top-section window"
    )

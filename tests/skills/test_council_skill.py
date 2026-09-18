"""Structural invariants for the council skill.

The council protocol is prose, but it is backed by structured data: 18 persona
files whose frontmatter drives panel selection, tie-breaking, and duo pairing.
The failure mode is drift between that data and the roster prose that the
coordinator reads -- a member documented in a profile but not tagged for it, or
a polarity pair named in one direction only. Both happened upstream. These
tests pin the relationships, not the wording.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "council"
)
PERSONA_DIR = SKILL_DIR / "references" / "personas"
ROSTER = SKILL_DIR / "references" / "roster.md"

REQUIRED_PERSONA_KEYS = {
    "figure",
    "domain",
    "polarity",
    "reasoning_method",
    "polarity_pairs",
    "triads",
    "duo_keywords",
    "profiles",
}


def parse_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---"), f"{path.name}: no frontmatter"
    _, raw, _ = text.split("---", 2)
    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict), f"{path.name}: frontmatter is not a mapping"
    return parsed


@pytest.fixture(scope="module")
def personas() -> dict[str, dict]:
    files = sorted(PERSONA_DIR.glob("council-*.md"))
    return {f.stem.removeprefix("council-"): parse_frontmatter(f) for f in files}


@pytest.fixture(scope="module")
def roster_text() -> str:
    return ROSTER.read_text(encoding="utf-8")


def test_skill_frontmatter_meets_bundled_skill_bar():
    fm = parse_frontmatter(SKILL_DIR / "SKILL.md")
    assert fm["name"] == "council"

    description = fm["description"]
    # Bundled skills are listed in the system prompt; long descriptions dilute
    # attention across the whole catalog.
    assert len(description) <= 60, f"description is {len(description)} chars"
    assert description.endswith("."), "description must be one sentence"

    for key in ("version", "author", "license", "platforms"):
        assert fm.get(key), f"missing {key}"


def test_related_skills_resolve_in_repo():
    # The catalog resolves related_skills by skill id. An entry that names no
    # shipped skill is a dead link nobody notices until a user follows it.
    fm = parse_frontmatter(SKILL_DIR / "SKILL.md")
    repo_root = SKILL_DIR.parents[2]
    related = fm["metadata"]["hermes"]["related_skills"]
    assert related, "related_skills is empty"

    for name in related:
        hits = (
            list(repo_root.glob(f"skills/*/{name}/SKILL.md"))
            + list(repo_root.glob(f"optional-skills/*/{name}/SKILL.md"))
            + list(repo_root.glob(f"skills/*/*/{name}/SKILL.md"))
        )
        assert hits, f"related_skills entry does not resolve in-repo: {name}"


def test_every_persona_carries_the_fields_selection_depends_on(personas):
    # Roster size is asserted where it is *derived*: the `classic` profile
    # heading in roster.md, checked by test_profile_sizes_match_the_members_
    # tagged_for_them. Pinning the literal here would only duplicate it.
    assert personas, "no persona files found"

    # The required set is a contract -- each of these keys drives a step of
    # panel selection. Extra keys are allowed, but only if every member carries
    # them, so the schema stays uniform across the roster without freezing a
    # literal that a new field would have to be threaded through.
    reference = sorted(next(iter(personas.values())))
    for name, fm in personas.items():
        missing = REQUIRED_PERSONA_KEYS - set(fm)
        assert not missing, f"{name}: missing selection keys {sorted(missing)}"
        assert sorted(fm) == reference, (
            f"{name}: key set differs from the rest of the roster "
            f"({sorted(set(fm) ^ set(reference))})"
        )
        for key, value in fm.items():
            assert value, f"{name}: {key} is empty"


def test_reasoning_methods_are_distinct(personas):
    # Method diversity is the mechanism the council buys. Two seats sharing a
    # reasoning method is a silently degraded panel, not a cosmetic issue.
    methods = [fm["reasoning_method"] for fm in personas.values()]
    assert len(set(methods)) == len(methods), "duplicate reasoning_method"


def test_polarity_pairs_resolve_and_are_mutual(personas):
    for name, fm in personas.items():
        for peer in fm["polarity_pairs"]:
            assert peer in personas, f"{name} pairs with unknown member {peer}"
            assert name in personas[peer]["polarity_pairs"], (
                f"{name} -> {peer} is not reciprocal"
            )


def test_polarity_pairs_match_the_roster_prose(personas, roster_text):
    # The coordinator picks duo pairings from the prose list; panel selection
    # reads the frontmatter. A pair present in only one of them is invisible
    # from whichever side you happen to be reading.
    def slug_of(prose_name):
        folded = re.sub(r"[^a-z]", "", prose_name.lower())
        return next(
            (n for n in personas if re.sub(r"[^a-z]", "", n) == folded), None
        )

    section = roster_text.split("## Polarity pairs", 1)[1].split("\n## ", 1)[0]
    prose = set()
    for left, right in re.findall(r"^- \*\*(.+?) vs (.+?)\*\*", section, re.MULTILINE):
        a, b = slug_of(left), slug_of(right)
        assert a and b, f"unknown member in polarity pair '{left} vs {right}'"
        prose.add(frozenset((a, b)))

    declared = {
        frozenset((name, peer))
        for name, fm in personas.items()
        for peer in fm["polarity_pairs"]
    }
    assert prose == declared, (
        f"only in frontmatter {sorted(map(sorted, declared - prose))}, "
        f"only in roster prose {sorted(map(sorted, prose - declared))}"
    )


def test_roster_documents_exactly_the_members_that_exist(personas, roster_text):
    documented = set(re.findall(r"`council-([a-z-]+)`", roster_text))
    assert documented == set(personas), (
        f"roster/persona mismatch: "
        f"only in roster {sorted(documented - set(personas))}, "
        f"only on disk {sorted(set(personas) - documented)}"
    )


def test_profile_sizes_match_the_members_tagged_for_them(personas, roster_text):
    # Each profile heading states its own size ("### `execution-lean` — 5
    # members"). That number and the frontmatter tags are two encodings of one
    # fact; drift between them silently changes who gets seated.
    declared = {
        name: int(size)
        for name, size in re.findall(
            r"^### `([a-z-]+)` (?:—|--) (?:all )?(\d+) members", roster_text, re.M
        )
    }
    assert declared, "no profile headings found in roster"

    for profile, expected in declared.items():
        tagged = [n for n, fm in personas.items() if profile in fm["profiles"]]
        assert len(tagged) == expected, (
            f"profile {profile}: roster says {expected}, "
            f"{len(tagged)} members tagged ({sorted(tagged)})"
        )


def test_persona_profiles_and_triads_exist_in_the_roster(personas, roster_text):
    known_profiles = set(re.findall(r"^### `([a-z-]+)` (?:—|--)", roster_text, re.M))
    known_triads = set(re.findall(r"^\| `([a-z-]+)` \|", roster_text, re.M))

    for name, fm in personas.items():
        for profile in fm["profiles"]:
            assert profile in known_profiles, f"{name}: unknown profile {profile}"
        for triad in fm["triads"]:
            assert triad in known_triads, f"{name}: unknown triad {triad}"


def test_skill_references_resolve():
    skill_md = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"`(references/[a-z0-9_/-]+\.md)`", skill_md))
    assert referenced, "SKILL.md references no support files"

    for rel in referenced:
        if "<name>" in rel:
            continue
        assert (SKILL_DIR / rel).is_file(), f"SKILL.md points at missing {rel}"


def test_no_foreign_harness_coupling_survived_the_port():
    # Ported from a Claude Code plugin. These tokens are the tell that a path or
    # a spawn primitive from that host leaked through.
    forbidden = ("subagent_type", "CLAUDE_PLUGIN_ROOT", "~/.claude", "cursor-agent")

    for path in SKILL_DIR.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} still references {token}"

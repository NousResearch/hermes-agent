"""Document contracts for the rnd-execution optional research skill.

The skill is method-only prose, so its shippable surface is an interface: the
literals its artifacts are stamped with (mutation-slip dispositions, the
``superseded: <slip>`` stamp, NOT-LANDED, the source-tag enum) and the
frontmatter the loader and hub rely on. Renaming a disposition or dropping a
stamp format silently invalidates every receipt written under the old name,
so these checks pin that vocabulary. Obedience to the discipline itself is
review/exercise territory, not a unit-test surface.
"""
from pathlib import Path

import pytest

from agent.skill_utils import parse_frontmatter

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "optional-skills" / "research" / "rnd-execution" / "SKILL.md"

SOURCE_TAGS = "measured / read-from-code / recalled"
SLIP_DISPOSITIONS = ("BREAKS", "RE-TEST", "SUPERSEDE", "UNTOUCHED")


@pytest.fixture(scope="module")
def document():
    return parse_frontmatter(SKILL.read_text(encoding="utf-8"))


def test_frontmatter_loader_contract(document):
    """Loader + hub essentials: name matches the directory, description stays
    inside the 60-char hardline, and the human contributor is credited first."""
    metadata, _ = document
    assert metadata["name"] == SKILL.parent.name
    assert len(metadata["description"]) <= 60
    assert metadata["description"].rstrip().endswith(".")
    assert metadata["author"].startswith("Abdulbaki (TopcuAbdulbaki)")
    assert not metadata["author"].startswith("Hermes Agent")
    assert metadata["license"] == "MIT"
    assert metadata["platforms"]


def test_mutation_slip_dispositions_are_pinned(document):
    """Every dependent line of a slip is classified with one of exactly four
    dispositions, and superseded results carry their stamp in place."""
    _, body = document
    for disposition in SLIP_DISPOSITIONS:
        assert disposition in body, disposition
    assert "superseded: <slip>" in body


def test_record_stamps_survive_edits(document):
    """The NOT-LANDED stamp marks records whose producing script has no repo
    home; losing the literal strands those records."""
    _, body = document
    assert "NOT-LANDED" in body


def test_source_tag_enum_is_complete(document):
    """Rule 4's tag vocabulary is a closed enum agents emit into records; the
    exact triple must stay stable and every claim type covered by it."""
    _, body = document
    assert SOURCE_TAGS in body
    assert "recalled" in body  # the 'unverified' counterpart for external claims


def test_external_claims_route_to_grounded_citations(document):
    """Web/literature claims must keep their routing to the bundled
    grounded-citations skill, with the manual fallback spelled out."""
    _, body = document
    assert "grounded-citations" in body
    assert "Sources" in body


def test_noise_floor_and_cost_gate_promises(document):
    """The two hard gates: headline claims need ≥3 seeds, and long runs must
    name the decision they will change before they start."""
    _, body = document
    assert "≥3 seeds" in body
    assert "which decision will this change" in body


def test_cover_test_acceptance_ratio(document):
    """Routing is accepted only when two independent fresh agents both find
    the single right answer — the stated 2/2 criterion."""
    _, body = document
    assert "2/2" in body


def test_related_skills_resolve_to_shipped_skills(document):
    """related_skills entries must name skills that ship in this tree state —
    a dangling entry points loaders at nothing."""
    metadata, _ = document
    shipped = {
        p.parent.name
        for root in (REPO / "skills", REPO / "optional-skills")
        for p in root.rglob("SKILL.md")
    }
    for related in metadata.get("related_skills") or []:
        assert related in shipped, related
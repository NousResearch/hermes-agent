"""Integration tests for the AutoResearch optional skill."""

from tools.skills_hub import OptionalSkillSource


def test_autoresearch_skill_is_searchable_from_official_source():
    source = OptionalSkillSource()

    results = source.search("autoresearch", limit=20)
    identifiers = {item.identifier for item in results}

    assert "official/research/autoresearch" in identifiers


def test_autoresearch_skill_bundle_contains_helper_script():
    source = OptionalSkillSource()

    bundle = source.fetch("official/research/autoresearch")

    assert bundle is not None
    assert bundle.name == "autoresearch"
    assert "SKILL.md" in bundle.files
    assert "scripts/autoresearch.py" in bundle.files

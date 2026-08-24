"""Tests for blog.schema_contract — enforce the Astro schema at write time."""
import pytest

import blog.schema_contract as sc


REAL_REPO = None  # use the real content.config.ts via SAHILBLOG_REPO


def test_loads_contract_from_real_schema():
    c = sc.load_contract(REAL_REPO)
    # Sanity: the real schema defines these.
    assert "manual" in c["sources"]
    assert "research-paper" in c["sources"]
    assert {"ai", "pm", "builder"} <= c["tiers"]
    assert "essay" in c["formats"]
    assert "evals" in c["tags"]
    assert "ai" in c["tags"]


def test_source_alias_manual_queue():
    fm = sc.normalise_frontmatter({"source": "manual_queue", "tags": []})
    assert fm["source"] == "manual"


def test_source_unknown_clamps_to_manual():
    fm = sc.normalise_frontmatter({"source": "wat", "tags": []})
    assert fm["source"] == "manual"


def test_source_valid_passthrough():
    fm = sc.normalise_frontmatter({"source": "research-paper", "tags": []})
    assert fm["source"] == "research-paper"


def test_tag_alias_evaluation_to_evals():
    fm = sc.normalise_frontmatter({"tags": ["ai", "evaluation"]})
    assert "evals" in fm["tags"]
    assert "evaluation" not in fm["tags"]


def test_invalid_tag_dropped():
    fm = sc.normalise_frontmatter({"tags": ["ai", "totally-made-up-tag-xyz"]})
    assert "ai" in fm["tags"]
    assert "totally-made-up-tag-xyz" not in fm["tags"]


def test_tags_deduped():
    fm = sc.normalise_frontmatter({"tags": ["ai", "ai", "AI"]})
    assert fm["tags"].count("ai") == 1


def test_tier_and_format_clamp():
    fm = sc.normalise_frontmatter(
        {"tier": "nonsense", "format": "nonsense", "tags": []})
    assert fm["tier"] == "pm"
    assert fm["format"] == "essay"


def test_valid_tier_format_passthrough():
    fm = sc.normalise_frontmatter(
        {"tier": "builder", "format": "blueprint", "tags": []})
    assert fm["tier"] == "builder"
    assert fm["format"] == "blueprint"


def test_normalise_preserves_other_fields():
    fm = sc.normalise_frontmatter(
        {"title": "X", "description": "Y", "approved": False,
         "source": "manual_queue", "tags": ["ai"]})
    assert fm["title"] == "X"
    assert fm["description"] == "Y"
    assert fm["approved"] is False


def test_tags_passthrough_when_schema_unreadable(tmp_path):
    # Point at a repo with no content.config.ts → allowed set empty → passthrough.
    sc.load_contract.cache_clear()
    kept, dropped = sc.normalise_tags(["anything", "goes"], frozenset())
    assert kept == ["anything", "goes"]
    assert dropped == []


# -- Approach A research-roundup frontmatter clamps ---------------------------

def test_research_tier_clamped_to_pm():
    """tier='research' (new enum) is clamped to 'pm' until the Astro schema is updated."""
    fm = sc.normalise_frontmatter({"tier": "research", "tags": []})
    assert fm["tier"] == "pm"


def test_research_source_clamped_to_manual():
    """source='curated-roundup' (new enum) is clamped to 'manual'."""
    fm = sc.normalise_frontmatter({"source": "curated-roundup", "tags": []})
    assert fm["source"] == "manual"


def test_research_source_alias_variants_clamped():
    """All curated-roundup alias spellings clamp to 'manual'."""
    for src in ("curated-roundup", "curated_roundup", "curatedroundup", "roundup"):
        fm = sc.normalise_frontmatter({"source": src, "tags": []})
        assert fm["source"] == "manual", f"{src!r} should clamp to manual"


def test_research_format_clamped_to_essay():
    """format='roundup' (new enum) is clamped to 'essay'."""
    fm = sc.normalise_frontmatter({"format": "roundup", "tags": []})
    assert fm["format"] == "essay"


def test_research_frontmatter_full_clamp():
    """A full research-roundup frontmatter dict clamps every new value."""
    fm = sc.normalise_frontmatter({
        "tier": "research",
        "source": "curated-roundup",
        "format": "roundup",
        "tags": ["research", "roundup"],
    })
    assert fm["tier"] == "pm"
    assert fm["source"] == "manual"
    assert fm["format"] == "essay"


def test_valid_tier_still_passthrough_after_alias_added():
    """Adding the research alias must not disturb existing valid tiers."""
    fm = sc.normalise_frontmatter({"tier": "builder", "tags": []})
    assert fm["tier"] == "builder"

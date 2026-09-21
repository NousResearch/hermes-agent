"""Tests for the provider-free P11 visual-plan contract."""

import json
from pathlib import Path

import pytest

from blog import reference_catalog as rc
from blog import visual_plan as vp


def _record(reference_id, roles, *, provenance="sahil_curated"):
    return rc.ReferenceRecord(
        reference_id=reference_id,
        path=f"refs/{reference_id}.png",
        sha256=(reference_id * 64)[:64],
        record_schema_version="2",
        provenance_class=provenance,
        ownership_or_usage_basis="fixture-only review required",
        usage_classification="review-required",
        allowed_roles=tuple(roles),
        parent_reference_id=None,
        collection="fixture",
        source="fixture",
        provenance_state="fixture",
        is_core=True,
    )


@pytest.fixture
def catalog():
    records = {
        "layout-hero": _record("layout-hero", ("layout",)),
        "layout-section": _record("layout-section", ("layout",)),
        "style-editorial": _record("style-editorial", ("style", "palette")),
        "composition-systems": _record("composition-systems", ("composition", "subject")),
    }
    return rc.ReferenceCatalog(root=Path("/fixtures"), baseline=records, core=records)


_HERO = {
    "role": "hero",
    "key": "hero",
    "reference_assignments": {
        "layout": ["layout-hero"],
        "style": ["style-editorial"],
        "composition": ["composition-systems"],
    },
    "layout": "architectural cross-section",
    "style": "ninth-observatory",
    "palette": "stone-and-brass",
    "motif": "control hall",
}

_SEC1 = {
    "role": "section",
    "key": "sec-mechanism",
    "reference_assignments": {
        "layout": ["layout-section"],
        "style": ["style-editorial"],
    },
    "layout": "bento-grid",
    "style": "ninth-observatory",
    "palette": "stone-and-brass",
    "motif": "control hall",
    "section_heading": "The mechanism",
}


def _good_plan_dict():
    return {
        "article_id": "art-001",
        "art_brief": "crossover point between local and API inference",
        "assets": [_HERO, _SEC1],
    }


def _build(catalog, **overrides):
    payload = _good_plan_dict()
    payload.update(overrides)
    return vp.build_visual_plan(catalog=catalog, **payload)


def test_build_valid_plan_preserves_roles_and_provenance(catalog):
    plan = _build(catalog)
    assert plan.version == vp.VISUAL_PLAN_VERSION
    assert plan.article_id == "art-001"
    assert [asset.layout for asset in plan.assets] == [
        "architectural cross-section", "bento-grid"
    ]
    assignment = next(
        item for item in plan.assets[0].reference_assignments
        if item.visual_role == "layout"
    )
    assert assignment.reference_id == "layout-hero"
    assert assignment.provenance_class == "sahil_curated"


def test_plan_requires_exactly_one_hero(catalog):
    with pytest.raises(vp.VisualPlanError, match="exactly one hero"):
        _build(catalog, assets=[_SEC1])


def test_multi_asset_plan_requires_distinct_layouts(catalog):
    duplicate = dict(_SEC1, layout=_HERO["layout"])
    with pytest.raises(vp.VisualPlanError, match="distinct layouts"):
        _build(catalog, assets=[_HERO, duplicate])


def test_plan_rejects_unknown_reference_id(catalog):
    bad = dict(_HERO, reference_assignments={"layout": ["missing"]})
    with pytest.raises(vp.VisualPlanError, match="unknown reference"):
        _build(catalog, assets=[bad])


def test_plan_rejects_role_not_allowed_by_catalogue(catalog):
    bad = dict(_HERO, reference_assignments={"layout": ["style-editorial"]})
    with pytest.raises(vp.VisualPlanError, match="not allowed"):
        _build(catalog, assets=[bad])


def test_plan_rejects_missing_reference_assignments(catalog):
    bad = dict(_HERO)
    del bad["reference_assignments"]
    with pytest.raises(vp.VisualPlanError, match="reference_assignments"):
        _build(catalog, assets=[bad])


def test_plan_locks_style_palette_and_motif(catalog):
    out_of_family = dict(_SEC1, palette="neon-hud")
    with pytest.raises(vp.VisualPlanError, match="does not match family palette"):
        _build(catalog, assets=[_HERO, out_of_family])


def test_plan_json_is_deterministic_and_round_trips(catalog):
    first = _build(catalog)
    second = _build(catalog)
    assert first.to_json() == second.to_json()
    restored = vp.VisualPlan.from_dict(json.loads(first.to_json()))
    assert restored.to_json() == first.to_json()
    assert first.digest() == second.digest()


def test_save_visual_plan_is_deterministic_and_explicit(catalog, tmp_path):
    plan = _build(catalog)
    output = tmp_path / "article.visual-plan.json"
    vp.save_visual_plan(plan, output)
    assert json.loads(output.read_text()) == plan.to_dict()
    assert output.read_text() == plan.to_json() + "\n"


def test_save_visual_plan_refuses_a_directory(catalog, tmp_path):
    with pytest.raises(vp.VisualPlanError, match="file path"):
        vp.save_visual_plan(_build(catalog), tmp_path)


def test_hero_only_plan_is_valid(catalog):
    plan = _build(catalog, assets=[_HERO])
    assert len(plan.assets) == 1
    assert plan.assets[0].role == "hero"


def test_two_heroes_are_rejected(catalog):
    duplicate = dict(_HERO, key="hero-2", layout="bento-grid")
    with pytest.raises(vp.VisualPlanError, match="exactly one hero"):
        _build(catalog, assets=[_HERO, duplicate])


def test_duplicate_asset_keys_are_rejected(catalog):
    duplicate = dict(_SEC1, key="hero")
    with pytest.raises(vp.VisualPlanError, match="duplicate asset keys"):
        _build(catalog, assets=[_HERO, duplicate])


def test_hero_cannot_have_a_section_heading(catalog):
    bad = dict(_HERO, section_heading="Introduction")
    with pytest.raises(vp.VisualPlanError, match="must not have section_heading"):
        _build(catalog, assets=[bad])


def test_section_requires_a_section_heading(catalog):
    bad = dict(_SEC1)
    del bad["section_heading"]
    with pytest.raises(vp.VisualPlanError, match="must have section_heading"):
        _build(catalog, assets=[_HERO, bad])


def test_empty_layout_is_rejected(catalog):
    bad = dict(_HERO, layout="")
    with pytest.raises(vp.VisualPlanError, match="layout is required"):
        _build(catalog, assets=[bad])


def test_empty_article_id_or_brief_is_rejected(catalog):
    with pytest.raises(vp.VisualPlanError, match="article_id is required"):
        _build(catalog, article_id="")
    with pytest.raises(vp.VisualPlanError, match="art_brief is required"):
        _build(catalog, art_brief=" ")


def test_unknown_plan_version_is_rejected(catalog):
    data = _build(catalog).to_dict()
    data["version"] = "999"
    with pytest.raises(vp.VisualPlanError, match="unsupported visual plan version"):
        vp.VisualPlan.from_dict(data)

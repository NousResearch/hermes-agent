"""Unit tests for hermes_cli.skill_catalog (no live Hub)."""

from tools.skills_hub import SkillMeta

from hermes_cli.skill_catalog.normalize import dedupe_metas, derive_categories, meta_to_item
from hermes_cli.skill_catalog.pagination import filter_by_category, paginate, sort_items


def test_derive_categories_matches_git_tag():
    meta = SkillMeta(
        name="Git Helper",
        description="Automate pull requests",
        source="github",
        identifier="foo/git-helper",
        trust_level="community",
        tags=["git"],
    )
    cats = derive_categories(meta)
    codes = [c["code"] for c in cats]
    assert "git_github" in codes


def test_derive_categories_allows_multiple_matches():
    meta = SkillMeta(
        name="React Pull Request Helper",
        description="Frontend automation for GitHub reviews",
        source="github",
        identifier="foo/react-pr-helper",
        trust_level="community",
        tags=["react", "git"],
    )
    codes = [c["code"] for c in derive_categories(meta)]
    assert "web_frontend" in codes
    assert "git_github" in codes


def test_dedupe_prefers_higher_trust():
    low = SkillMeta(
        name="Same",
        description="",
        source="github",
        identifier="same/skill",
        trust_level="community",
    )
    high = SkillMeta(
        name="Same",
        description="",
        source="official",
        identifier="same/skill",
        trust_level="builtin",
    )
    out = dedupe_metas([low, high])
    assert len(out) == 1
    assert out[0].trust_level == "builtin"


def test_filter_by_category():
    items = [
        {"name": "a", "categories": [{"code": "web_frontend", "name": "Web"}]},
        {"name": "b", "categories": [{"code": "git_github", "name": "Git"}]},
    ]
    git_only = filter_by_category(items, "git_github")
    assert len(git_only) == 1
    assert git_only[0]["name"] == "b"


def test_paginate_second_page():
    rows = [{"n": i} for i in range(25)]
    chunk, meta = paginate(rows, page=2, page_size=10)
    assert len(chunk) == 10
    assert meta["total"] == 25
    assert meta["hasNext"] is True
    assert meta["hasPrev"] is True


def test_sort_items_by_name():
    rows = [
        {"name": "zebra", "categories": [], "_order": 0},
        {"name": "alpha", "categories": [], "_order": 1},
    ]
    out = sort_items(rows, "name")
    assert out[0]["name"] == "alpha"


def test_meta_to_item_strips_order():
    meta = SkillMeta(
        name="T",
        description="D",
        source="official",
        identifier="official/x",
        trust_level="builtin",
        tags=[],
    )
    row = meta_to_item(meta, order=5)
    assert row["_order"] == 5
    sorted_rows = sort_items([row], "relevance")
    assert "_order" not in sorted_rows[0]

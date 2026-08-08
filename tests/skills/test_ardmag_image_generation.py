"""Tests for optional-skills/creative/ardmag-image-generation/scripts/ardmag_brief.py.

Cover the behaviors codified after the 2026-05-26 Delta Research hero
incident: specific products preferred over generic categories, cache-busting
filename when the current hero is generic, and a refusal-style brief when the
detector finds nothing usable.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "ardmag-image-generation"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import ardmag_brief  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────


DELTA_ARTICLE = """---
title: "Tratamente pentru piatră naturală: cum alegi soluția potrivită"
description: "Ghid practic Delta Research..."
kicker: "Ghid tehnic · Delta Research"
publishedAt: "2026-05-26"
author: "Echipa ardmag"
tags: ["delta research", "tratamente", "impregnare", "piatra naturala", "ghid"]
heroImage: "/blog/delta-tratamente-piatra-naturala/hero.webp"
---

Articolul grupează criteriile de alegere și produsele **Delta Research** relevante.

Pentru suprafețele orizontale: **Seal**, **Quasar**, **Wet Seal**, **Nano Wet**,
**Eco Stone Pro** și **Eco Toner**.

Pentru verticale: **Idrorep** și **Total Wet**.

Notă: efectul vizual contează — alegi **natural** sau efect umed.
"""


GENERIC_ONLY_ARTICLE = """---
title: "Generic article"
tags: ["tratamente"]
heroImage: "/blog/foo/hero.webp"
---

Pentru articolul ăsta se foloseste **Tratamente Specifice** și nimic concret.
"""


@pytest.fixture
def delta_article(tmp_path: Path) -> Path:
    p = tmp_path / "delta-tratamente-piatra-naturala.md"
    p.write_text(DELTA_ARTICLE, encoding="utf-8")
    return p


@pytest.fixture
def generic_article(tmp_path: Path) -> Path:
    p = tmp_path / "generic-foo.md"
    p.write_text(GENERIC_ONLY_ARTICLE, encoding="utf-8")
    return p


@pytest.fixture
def assets_root(tmp_path: Path) -> Path:
    """Build a fake backend/static/images/ tree with the slugs the Delta
    article will resolve against, plus the generic category we want to avoid."""
    root = tmp_path / "images"
    root.mkdir()
    for slug in [
        "seal",
        "quasar",
        "wet-seal",
        "nano-wet",
        "eco-stone-pro",
        "eco-toner",
        "idrorep",
        "total-wet",
        "solutii-delta",  # generic category — must be flagged
        "ager",  # unrelated, must not be matched
    ]:
        sub = root / slug
        sub.mkdir()
        (sub / f"{slug}-001.jpg").write_bytes(b"\x00")
    return root


@pytest.fixture
def catalog_csv(tmp_path: Path) -> Path:
    p = tmp_path / "catalog.csv"
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["handleId", "name", "brand"])
        writer.writerow(["product_seal", "SEAL", "Delta Research"])
        writer.writerow(["product_quasar", "QUASAR", "Delta Research"])
        writer.writerow(["product_wet_seal", "WET SEAL", "Delta Research"])
        writer.writerow(["product_idrorep", "IDROREP", "Delta Research"])
        writer.writerow(["product_eco_stone_pro", "ECO STONE PRO", "Delta Research"])
        writer.writerow(["product_eco_toner", "ECO TONER", "Delta Research"])
        writer.writerow(["product_total_wet", "TOTAL WET", "Delta Research"])
        writer.writerow(["product_nano_wet", "NANO WET", "Delta Research"])
        writer.writerow(["product_solutii_delta", "SOLUTII DELTA", "Delta Research"])
    return p


# ── helpers ──────────────────────────────────────────────────────────────────


def _run(capsys, argv: list[str]) -> dict:
    with mock.patch("sys.argv", ["ardmag_brief"] + argv):
        ardmag_brief.main()
    return json.loads(capsys.readouterr().out)


# ── slugify / frontmatter primitives ────────────────────────────────────────


class TestSlugify:
    def test_basic(self):
        assert ardmag_brief.slugify("Eco Stone Pro") == "eco-stone-pro"

    def test_diacritics_stripped(self):
        assert ardmag_brief.slugify("Tratămente pentru piatră") == "tratamente-pentru-piatra"

    def test_collapses_punctuation(self):
        assert ardmag_brief.slugify("WET-SEAL!!!") == "wet-seal"

    def test_empty(self):
        assert ardmag_brief.slugify("---") == ""


class TestFrontmatter:
    def test_parses_tags_list(self):
        fm, body = ardmag_brief.parse_frontmatter(DELTA_ARTICLE)
        assert fm["tags"] == [
            "delta research",
            "tratamente",
            "impregnare",
            "piatra naturala",
            "ghid",
        ]
        assert fm["heroImage"] == "/blog/delta-tratamente-piatra-naturala/hero.webp"
        assert "Articolul grupează" in body

    def test_no_frontmatter(self):
        fm, body = ardmag_brief.parse_frontmatter("just body\n")
        assert fm == {}
        assert body == "just body\n"


# ── detect-products ─────────────────────────────────────────────────────────


class TestDetectProducts:
    def test_extracts_delta_products_from_body(self, capsys, delta_article):
        result = _run(capsys, ["detect-products", "--article", str(delta_article)])
        names = [p["name"] for p in result["products"]]
        for required in ["Seal", "Quasar", "Wet Seal", "Eco Stone Pro", "Idrorep", "Total Wet"]:
            assert required in names, f"missing {required}: got {names}"

    def test_infers_delta_brand_from_tags(self, capsys, delta_article):
        result = _run(capsys, ["detect-products", "--article", str(delta_article)])
        assert result["brand"] == "delta-research"

    def test_skips_non_product_bold_words(self, capsys, delta_article):
        result = _run(capsys, ["detect-products", "--article", str(delta_article)])
        slugs = [p["slug"] for p in result["products"]]
        assert "natural" not in slugs
        assert "interior" not in slugs

    def test_catalog_match_adds_handle(self, capsys, delta_article, catalog_csv):
        result = _run(
            capsys,
            ["detect-products", "--article", str(delta_article), "--catalog", str(catalog_csv)],
        )
        by_slug = {p["slug"]: p for p in result["products"]}
        assert by_slug["seal"]["in_catalog"] is True
        assert by_slug["seal"]["catalog_handle"] == "product_seal"
        assert by_slug["eco-stone-pro"]["catalog_brand"] == "Delta Research"

    def test_generic_only_article_brand_none(self, capsys, generic_article):
        result = _run(capsys, ["detect-products", "--article", str(generic_article)])
        assert result["brand"] == "none"


# ── resolve-assets ──────────────────────────────────────────────────────────


class TestResolveAssets:
    def test_maps_each_product_to_its_asset_dir(self, capsys, delta_article, assets_root):
        result = _run(
            capsys,
            [
                "resolve-assets",
                "--article",
                str(delta_article),
                "--assets-root",
                str(assets_root),
            ],
        )
        by_slug = {p["slug"]: p for p in result["products"]}
        assert by_slug["seal"]["asset_dir"].endswith("/seal")
        assert by_slug["seal"]["sample_image"].endswith(".jpg")
        assert by_slug["eco-stone-pro"]["asset_dir"].endswith("/eco-stone-pro")
        assert by_slug["idrorep"]["asset_dir"].endswith("/idrorep")

    def test_unknown_product_has_no_asset(self, capsys, tmp_path, assets_root):
        article = tmp_path / "mystery.md"
        article.write_text(
            "---\ntitle: x\ntags: [tenax]\nheroImage: /blog/x/hero.webp\n---\n"
            "Folosim **Whoknows**.\n",
            encoding="utf-8",
        )
        result = _run(
            capsys,
            ["resolve-assets", "--article", str(article), "--assets-root", str(assets_root)],
        )
        assert result["products"][0]["asset_dir"] is None
        assert result["products"][0]["sample_image"] is None

    def test_generic_category_flagged(self, capsys, generic_article, assets_root):
        # Generic article mentions **Tratamente Specifice** which slugifies
        # to a generic-category slug; this should still resolve to no specific
        # asset_dir (it is not in our fake tree) and is_generic must be False
        # because the slug is "tratamente-specifice" — present in
        # GENERIC_CATEGORY_SLUGS — so is_generic must be True.
        result = _run(
            capsys,
            ["resolve-assets", "--article", str(generic_article), "--assets-root", str(assets_root)],
        )
        by_slug = {p["slug"]: p for p in result["products"]}
        assert by_slug["tratamente-specifice"]["is_generic"] is True


# ── suggest-filename ────────────────────────────────────────────────────────


class TestSuggestFilename:
    def test_replaces_generic_hero_name(self, capsys, delta_article):
        result = _run(capsys, ["suggest-filename", "--article", str(delta_article)])
        assert result["suggested_filename"].endswith(".webp")
        assert result["needs_cache_busting"] is True
        assert result["suggested_filename"] != "hero.webp"
        # Brand and topic must be present in the new name.
        assert "delta-research" in result["suggested_filename"]

    def test_already_semantic_filename_not_busted(self, capsys, tmp_path):
        article = tmp_path / "article.md"
        article.write_text(
            '---\ntitle: T\ntags: [delta research]\n'
            'heroImage: "/blog/article/hero-delta-research-tratamente.webp"\n---\n'
            'Folosim **Seal**.\n',
            encoding="utf-8",
        )
        result = _run(capsys, ["suggest-filename", "--article", str(article)])
        assert result["needs_cache_busting"] is False


# ── build-brief ─────────────────────────────────────────────────────────────


class TestBuildBrief:
    def test_full_brief_for_delta_article(
        self, capsys, delta_article, assets_root, catalog_csv
    ):
        result = _run(
            capsys,
            [
                "build-brief",
                "--article",
                str(delta_article),
                "--assets-root",
                str(assets_root),
                "--catalog",
                str(catalog_csv),
            ],
        )
        # The brief must surface the real products, the new filename and a
        # validation checklist that includes the production HTML check.
        assert result["brand"] == "delta-research"
        product_slugs = [p["slug"] for p in result["products"]]
        for required in ["seal", "quasar", "idrorep", "eco-stone-pro"]:
            assert required in product_slugs

        # Filename is semantic and points away from the generic hero.webp.
        assert result["suggested_filename"] != "hero.webp"
        assert result["frontmatter_patch"].startswith("/blog/delta-tratamente-piatra-naturala/")
        assert result["frontmatter_patch"].endswith(result["suggested_filename"])
        assert result["needs_cache_busting"] is True

        # Prompt encodes the anti-collage rules and references the real products.
        prompt_lower = result["prompt"].lower()
        assert "hard prohibitions" in prompt_lower
        assert "collage" in prompt_lower
        assert "cutout" in prompt_lower
        assert "Seal" in result["prompt"]
        assert "Quasar" in result["prompt"]
        # Delta Research brand label is present.
        assert "Delta Research" in result["prompt"]

        # Validation checklist is non-empty and mentions production HTML.
        assert any("Production article HTML" in step for step in result["validation_checklist"])
        assert any(result["suggested_filename"] in step for step in result["validation_checklist"])

        # No blockers for a well-formed article.
        assert result["blockers"] == []

    def test_no_products_yields_blocker(self, capsys, tmp_path, assets_root):
        article = tmp_path / "empty.md"
        article.write_text(
            "---\ntitle: nothing\ntags: []\nheroImage: /blog/nothing/hero.webp\n---\n"
            "no bold mentions here.\n",
            encoding="utf-8",
        )
        result = _run(
            capsys,
            ["build-brief", "--article", str(article), "--assets-root", str(assets_root)],
        )
        assert result["products"] == []
        assert any("No products detected" in b for b in result["blockers"])

    def test_generic_only_yields_blocker(
        self, capsys, generic_article, assets_root
    ):
        result = _run(
            capsys,
            [
                "build-brief",
                "--article",
                str(generic_article),
                "--assets-root",
                str(assets_root),
            ],
        )
        # The detector picks up "Tratamente Specifice" but it slugifies to a
        # generic category, so the brief must refuse with a blocker.
        assert any("generic-category" in b for b in result["blockers"])

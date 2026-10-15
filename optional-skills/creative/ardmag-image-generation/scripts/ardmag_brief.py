#!/usr/bin/env python3
"""Build an image-generation brief for an ARDmag blog article.

The brief encodes the article's real products, maps them to real asset
directories, drafts an organic workshop-scene prompt, and emits a
cache-busting filename plus a production-validation checklist.

Subcommands:
    detect-products   — extract product names from an article (tags + body)
    resolve-assets    — map detected products to asset dirs / sample images
    suggest-filename  — propose a cache-busting hero filename
    build-brief       — do all of the above and emit a full JSON brief

All output is JSON on stdout. Errors exit non-zero with a JSON error.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
PROMPT_TEMPLATE_PATH = SKILL_DIR / "prompts" / "hero_image.md"

# Slugs that indicate generic-category mappings; these are a fallback only.
GENERIC_CATEGORY_SLUGS = {
    "solutii-delta",
    "solutii-tenax",
    "tratamente-specifice",
    "tratamente-piatra",
    "categorii-generale",
}

# Tokens commonly bolded in articles that aren't products. Lowercase, slugged.
NON_PRODUCT_TOKENS = {
    "interior",
    "exterior",
    "natural",
    "umed",
    "baza-de-apa",
    "baza-de-solvent",
    "apa",
    "solvent",
    "verticala",
    "orizontala",
    "blat",
    "scari",
    "pavaj",
    "fatada",
    "gard",
    "test",
    "plus",
    "ce-trebuie-sa-stii",
    "important",
    "atentie",
    "nota",
    "pentru",
    "caracteristici-tehnice",
    "intaritorul-este-incluse",
    "1-kit",
}

BRAND_TAGS = {
    "delta research": "delta-research",
    "delta-research": "delta-research",
    "delta": "delta-research",
    "tenax": "tenax",
}


def _err(msg: str, **extra: Any) -> None:
    payload = {"ok": False, "error": msg}
    payload.update(extra)
    print(json.dumps(payload))
    sys.exit(1)


def _ok(payload: dict[str, Any]) -> None:
    payload = {"ok": True, **payload}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# ── slug / normalize ─────────────────────────────────────────────────────────


def slugify(value: str) -> str:
    """Lowercase, strip diacritics, replace non-alnum with single hyphen."""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value


# ── frontmatter parser ───────────────────────────────────────────────────────


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter_dict, body). Frontmatter must be the YAML-ish
    block at the top of the file between '---' lines. We avoid pulling in
    PyYAML by parsing the handful of keys we need (title, tags, heroImage,
    kicker, description, publishedAt, author)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")

    fm: dict[str, Any] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            items = [
                _strip_quotes(item.strip())
                for item in re.split(r",(?![^\[]*\])", inner)
                if item.strip()
            ]
            fm[key] = items
        else:
            fm[key] = _strip_quotes(value)
    return fm, body


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


# ── product detector ────────────────────────────────────────────────────────


# Capture `**Foo**` or `**Foo Bar**` runs that look like product names.
# We require the inner text to start with a letter and not contain inline
# punctuation that signals a sentence fragment.
BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")


def extract_bold_mentions(body: str) -> list[str]:
    mentions: list[str] = []
    seen: set[str] = set()
    for match in BOLD_RE.finditer(body):
        candidate = match.group(1).strip()
        if not candidate or len(candidate) < 2:
            continue
        if candidate.endswith("?") or candidate.endswith(":"):
            continue
        # Skip sentence-like fragments (multiple words with lowercase
        # connectors like "și", "sau", "de la", commas, etc.).
        if "," in candidate:
            continue
        words = candidate.split()
        if len(words) > 4:
            continue
        slug = slugify(candidate)
        if not slug or slug in NON_PRODUCT_TOKENS:
            continue
        if slug in seen:
            continue
        seen.add(slug)
        mentions.append(candidate)
    return mentions


def infer_brand(tags: Iterable[str], body: str) -> str:
    found: set[str] = set()
    for tag in tags:
        key = tag.strip().lower()
        if key in BRAND_TAGS:
            found.add(BRAND_TAGS[key])
    low = body.lower()
    if "delta research" in low or "delta-research" in low:
        found.add("delta-research")
    if " tenax" in low or "tenax " in low:
        found.add("tenax")
    if not found:
        return "none"
    if len(found) == 1:
        return next(iter(found))
    return "mixed"


def load_catalog(catalog_path: Path) -> list[dict[str, str]]:
    """Load the Wix catalog CSV. We only need handleId, name, brand."""
    rows: list[dict[str, str]] = []
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            rows.append(
                {
                    "handleId": (row.get("handleId") or "").strip(),
                    "name": name,
                    "brand": (row.get("brand") or "").strip(),
                }
            )
    return rows


def catalog_match(mention: str, catalog: list[dict[str, str]]) -> dict[str, str] | None:
    """Find a catalog row whose name matches the mention (case-insensitive,
    diacritic-insensitive). Prefers exact match over substring."""
    target = slugify(mention)
    if not target:
        return None
    exact = None
    sub = None
    for row in catalog:
        name_slug = slugify(row["name"])
        if name_slug == target:
            exact = row
            break
        if name_slug and (target in name_slug.split("-") or name_slug == target):
            sub = sub or row
        elif name_slug and target in name_slug:
            sub = sub or row
    return exact or sub


def detect_products(
    article_path: Path,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    text = article_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    bold = extract_bold_mentions(body)
    brand = infer_brand(tags, body)

    catalog = load_catalog(catalog_path) if catalog_path else []

    products: list[dict[str, Any]] = []
    for mention in bold:
        entry: dict[str, Any] = {
            "name": mention,
            "slug": slugify(mention),
        }
        if catalog:
            match = catalog_match(mention, catalog)
            if match:
                entry["catalog_name"] = match["name"]
                entry["catalog_handle"] = match["handleId"]
                entry["catalog_brand"] = match["brand"]
                entry["in_catalog"] = True
            else:
                entry["in_catalog"] = False
        products.append(entry)

    return {
        "article": str(article_path),
        "title": fm.get("title", ""),
        "tags": tags,
        "brand": brand,
        "products": products,
        "frontmatter_hero_image": fm.get("heroImage", ""),
    }


# ── asset resolver ──────────────────────────────────────────────────────────


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def resolve_assets(
    products: list[dict[str, Any]],
    assets_root: Path,
) -> list[dict[str, Any]]:
    if not assets_root.exists():
        return [
            {**p, "asset_dir": None, "sample_image": None, "is_generic": False}
            for p in products
        ]
    available = {d.name: d for d in assets_root.iterdir() if d.is_dir()}
    resolved: list[dict[str, Any]] = []
    for product in products:
        slug = product["slug"]
        candidates = [slug, slug.replace("-", "_")]
        # Try a couple of variants for multi-word names.
        if "-" in slug:
            candidates.append(slug.split("-")[0])
        asset_dir = None
        for cand in candidates:
            if cand in available:
                asset_dir = available[cand]
                break
        sample = None
        if asset_dir is not None:
            for child in sorted(asset_dir.iterdir()):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                    sample = child
                    break
        resolved.append(
            {
                **product,
                "asset_dir": str(asset_dir) if asset_dir else None,
                "sample_image": str(sample) if sample else None,
                "is_generic": slug in GENERIC_CATEGORY_SLUGS,
            }
        )
    return resolved


# ── filename suggester ──────────────────────────────────────────────────────


GENERIC_HERO_NAMES = {"hero", "cover", "thumbnail", "main", "image"}


def suggest_filename(
    article_slug: str,
    brand: str,
    products: list[dict[str, Any]],
    current_hero: str = "",
) -> dict[str, Any]:
    current_stem = Path(current_hero).stem if current_hero else ""
    current_stem_lower = current_stem.lower()
    is_generic_current = (
        not current_stem
        or current_stem_lower in GENERIC_HERO_NAMES
        or current_stem_lower.startswith("hero.")
    )

    parts = ["hero"]
    if brand and brand not in ("none", "mixed"):
        parts.append(brand)
    topic = slugify(article_slug)
    if topic:
        topic_short = "-".join(topic.split("-")[:4])
        parts.append(topic_short)
    elif products:
        parts.append(products[0]["slug"])

    name = "-".join(p for p in parts if p)
    filename = f"{name}.webp"
    if current_hero:
        current_filename = Path(current_hero).name
        if filename == current_filename:
            filename = f"{name}-v2.webp"

    return {
        "suggested_filename": filename,
        "needs_cache_busting": is_generic_current,
        "current_hero": current_hero,
    }


# ── prompt builder ──────────────────────────────────────────────────────────


def _scene_surfaces_for(brand: str, products: list[dict[str, Any]]) -> str:
    # Default surfaces for stone-treatment / polishing topics.
    return "granite, marble, travertine and andesite stone"


def build_prompt(
    title: str,
    brand: str,
    products: list[dict[str, Any]],
    aspect_ratio: str = "landscape",
) -> str:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    # Strip the markdown frontmatter / heading block — keep only the body
    # after the first `---` separator.
    body_start = template.find("---\n")
    body = template[body_start + 4 :] if body_start != -1 else template

    product_lines: list[str] = []
    for product in products:
        if product.get("is_generic"):
            continue
        line = f"- {product['name']}"
        if product.get("catalog_name") and product["catalog_name"].upper() != product["name"].upper():
            line += f" ({product['catalog_name']})"
        if product.get("asset_dir"):
            line += f"  [reference: {product['asset_dir']}]"
        product_lines.append(line)

    if not product_lines:
        product_lines.append(
            "- (NONE — refuse to generate until at least one specific product is identified)"
        )

    brand_label = {
        "delta-research": "Delta Research",
        "tenax": "Tenax",
        "mixed": "Delta Research and Tenax",
        "none": "ARDmag",
    }.get(brand, "ARDmag")

    return body.format(
        article_title=title or "(untitled)",
        scene_surfaces=_scene_surfaces_for(brand, products),
        product_list_block="\n".join(product_lines),
        brand_label=brand_label,
        aspect_ratio=aspect_ratio,
    )


# ── full brief ──────────────────────────────────────────────────────────────


def build_brief(
    article_path: Path,
    assets_root: Path | None,
    catalog_path: Path | None,
    aspect_ratio: str = "landscape",
) -> dict[str, Any]:
    detection = detect_products(article_path, catalog_path)
    products = detection["products"]
    if assets_root is not None:
        products = resolve_assets(products, assets_root)

    article_slug = article_path.stem
    fn = suggest_filename(
        article_slug=article_slug,
        brand=detection["brand"],
        products=products,
        current_hero=detection["frontmatter_hero_image"],
    )

    specific_products = [p for p in products if not p.get("is_generic")]
    only_generic = bool(products) and not specific_products
    no_products = not products

    blockers: list[str] = []
    if no_products:
        blockers.append(
            "No products detected in article body. Skill is configured to "
            "refuse generation without at least one specific product."
        )
    if only_generic:
        blockers.append(
            "Only generic-category matches found "
            f"({', '.join(p['slug'] for p in products)}). Refusing — the "
            "article likely names specific products that were missed by "
            "the detector. Re-run after tightening detection or naming the "
            "products manually."
        )

    prompt = build_prompt(
        title=detection["title"],
        brand=detection["brand"],
        products=products,
        aspect_ratio=aspect_ratio,
    )

    new_filename = fn["suggested_filename"]
    article_dir_url = f"/blog/{article_slug}/"
    frontmatter_patch = f"{article_dir_url}{new_filename}"

    checklist = [
        "Local Next build succeeds with the new heroImage path",
        "CI on GitHub is green",
        f"Production article HTML contains '{new_filename}'",
        "Production blog list shows the new hero card",
        f"GET on the production image URL ({frontmatter_patch}) returns 200",
        "Visual confirmation: labels readable, products integrated, no collage",
    ]

    return {
        "article": str(article_path),
        "article_slug": article_slug,
        "title": detection["title"],
        "tags": detection["tags"],
        "brand": detection["brand"],
        "products": products,
        "prompt": prompt,
        "suggested_filename": new_filename,
        "frontmatter_patch": frontmatter_patch,
        "needs_cache_busting": fn["needs_cache_busting"],
        "current_hero": fn["current_hero"],
        "blockers": blockers,
        "validation_checklist": checklist,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_detect = sub.add_parser("detect-products")
    p_detect.add_argument("--article", required=True)
    p_detect.add_argument("--catalog", default=None)

    p_resolve = sub.add_parser("resolve-assets")
    p_resolve.add_argument("--article", required=True)
    p_resolve.add_argument("--assets-root", required=True)
    p_resolve.add_argument("--catalog", default=None)

    p_filename = sub.add_parser("suggest-filename")
    p_filename.add_argument("--article", required=True)
    p_filename.add_argument("--catalog", default=None)

    p_brief = sub.add_parser("build-brief")
    p_brief.add_argument("--article", required=True)
    p_brief.add_argument("--assets-root", default=None)
    p_brief.add_argument("--catalog", default=None)
    p_brief.add_argument("--aspect-ratio", default="landscape")

    args = parser.parse_args(argv)

    article = Path(args.article)
    if not article.exists():
        _err(f"article not found: {article}")
    catalog = Path(args.catalog) if getattr(args, "catalog", None) else None
    if catalog and not catalog.exists():
        _err(f"catalog not found: {catalog}")

    if args.cmd == "detect-products":
        _ok(detect_products(article, catalog))
    elif args.cmd == "resolve-assets":
        assets_root = Path(args.assets_root)
        if not assets_root.exists():
            _err(f"assets-root not found: {assets_root}")
        detection = detect_products(article, catalog)
        resolved = resolve_assets(detection["products"], assets_root)
        _ok({**detection, "products": resolved})
    elif args.cmd == "suggest-filename":
        detection = detect_products(article, catalog)
        result = suggest_filename(
            article_slug=article.stem,
            brand=detection["brand"],
            products=detection["products"],
            current_hero=detection["frontmatter_hero_image"],
        )
        _ok(result)
    elif args.cmd == "build-brief":
        assets_root = Path(args.assets_root) if args.assets_root else None
        if assets_root and not assets_root.exists():
            _err(f"assets-root not found: {assets_root}")
        _ok(
            build_brief(
                article_path=article,
                assets_root=assets_root,
                catalog_path=catalog,
                aspect_ratio=args.aspect_ratio,
            )
        )


if __name__ == "__main__":
    main()

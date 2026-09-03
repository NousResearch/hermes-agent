#!/usr/bin/env python3
"""Generate Lunar City masked reference cards and silhouette prep artifacts.

This is an intake/prep stage for high-poly asset generation. It deliberately
does not promote the existing scene-crop image-to-3D meshes to production.
Instead, it creates deterministic masked sources and silhouettes that can guide
future free/local 2D-to-3D runs toward the approved reference silhouettes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
REFERENCE_MANIFEST = PUBLIC / "lunar-city" / "generated-3d" / "reference-crops" / "reference-crops-manifest.json"
MASTER_MANIFEST = PUBLIC / "lunar-city" / "master-assets" / "master-asset-manifest.json"
OUTPUT = PUBLIC / "lunar-city" / "master-assets" / "masks"
MASK_MANIFEST = OUTPUT / "mask-manifest.json"
REVIEW_PREVIEW = OUTPUT / "mask-review-contact-sheet.png"
CACHE_OUTPUT = Path("/private/tmp/lunar-city-master-asset-masked-sources")
REMBG_CACHE = Path("/private/tmp/lunar-city-rembg-cache")

BACKGROUND_RGBA = (210, 210, 210, 255)

TARGET_MASTER_ASSET_IDS = {
    "building-engineering": "building-engineering-workshop",
    "prop-break-garden": "building-break-garden",
    "worker-bot-round": "worker-research",
    "worker-bot-carrying": "worker-release",
    "worker-bot-review": "worker-review",
    "child-bot-garden": "child-curious",
}


@dataclass(frozen=True)
class MaskResult:
    coverage_ratio: float
    generation_input_status: str
    method: str
    mask: Image.Image
    quality_flags: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _try_rembg(image: Image.Image) -> Image.Image | None:
    REMBG_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("U2NET_HOME", str(REMBG_CACHE))
    os.environ.setdefault("XDG_CACHE_HOME", str(REMBG_CACHE))

    try:
        from rembg import remove
    except Exception:
        return None

    try:
        removed = remove(image.convert("RGBA"))
    except Exception:
        return None

    return removed.convert("RGBA").getchannel("A")


def _background_delta_mask(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, BACKGROUND_RGBA)
    delta = ImageChops.difference(rgba, background).convert("L")
    mask = delta.point(lambda value: 255 if value > 18 else 0)
    return mask


def _clean_mask(mask: Image.Image) -> Image.Image:
    cleaned = mask.convert("L")
    cleaned = cleaned.filter(ImageFilter.MedianFilter(size=5))
    cleaned = cleaned.filter(ImageFilter.MaxFilter(size=5))
    cleaned = cleaned.filter(ImageFilter.GaussianBlur(radius=1.1))
    return cleaned.point(lambda value: 255 if value > 24 else 0)


def _coverage(mask: Image.Image) -> float:
    alpha = mask.convert("L")
    histogram = alpha.histogram()
    opaque = sum(count for value, count in enumerate(histogram) if value >= 128)
    return opaque / float(alpha.width * alpha.height)


def make_mask(image: Image.Image) -> MaskResult:
    rembg_mask = _try_rembg(image)
    fallback_mask = _background_delta_mask(image)

    if rembg_mask is not None and 0.03 <= _coverage(rembg_mask) <= 0.96:
        method = "rembg_alpha"
        mask = rembg_mask
    else:
        method = "background_delta_fallback"
        mask = fallback_mask

    mask = _clean_mask(mask)
    coverage_ratio = _coverage(mask)
    quality_flags: list[str] = ["requires_human_silhouette_review"]
    if coverage_ratio < 0.04:
        quality_flags.append("low_coverage")
    if coverage_ratio > 0.92:
        quality_flags.append("high_coverage_possible_background_leak")
    generation_input_status = "ready_for_generation_review"
    return MaskResult(
        coverage_ratio=coverage_ratio,
        generation_input_status=generation_input_status,
        mask=mask,
        method=method,
        quality_flags=quality_flags,
    )


def mask_quality_flags(kind: str, coverage_ratio: float, method: str) -> list[str]:
    flags: list[str] = []
    expected_ranges = {
        "building": (0.08, 0.78),
        "child": (0.04, 0.30),
        "leader": (0.04, 0.48),
        "prop": (0.08, 0.72),
        "vehicle": (0.08, 0.42),
        "worker": (0.04, 0.30),
    }
    low, high = expected_ranges.get(kind, (0.04, 0.80))
    if coverage_ratio < low:
        flags.append("coverage_below_expected_subject_silhouette")
    if coverage_ratio > high:
        flags.append("broad_scene_crop_not_subject_isolated")
    if method == "background_delta_fallback":
        flags.append("rembg_alpha_unavailable_or_unusable")
    return flags


def masked_source(image: Image.Image, mask: Image.Image) -> Image.Image:
    source = image.convert("RGBA")
    transparent = Image.new("RGBA", source.size, (0, 0, 0, 0))
    transparent.alpha_composite(source)
    transparent.putalpha(mask.convert("L"))
    return transparent


def silhouette_preview(mask: Image.Image) -> Image.Image:
    alpha = mask.convert("L")
    silhouette = Image.new("RGBA", alpha.size, (14, 18, 24, 255))
    white = Image.new("RGBA", alpha.size, (230, 244, 255, 255))
    silhouette.paste(white, (0, 0), alpha)
    return silhouette


def _fit_thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumbnail = ImageOps.contain(image.convert("RGBA"), size)
    canvas = Image.new("RGBA", size, (10, 14, 20, 255))
    canvas.alpha_composite(thumbnail, ((size[0] - thumbnail.width) // 2, (size[1] - thumbnail.height) // 2))
    return canvas


def _draw_label(image: Image.Image, x: int, y: int, label: str) -> None:
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small = font
    draw.text((x, y), label[:31], fill=(230, 244, 255, 255), font=font)
    draw.text((x, y + 18), "crop / mask / silhouette", fill=(130, 196, 255, 255), font=small)


def write_review_contact_sheet(mask_entries: list[dict[str, Any]]) -> None:
    columns = 4
    tile_width = 360
    tile_height = 250
    rows = (len(mask_entries) + columns - 1) // columns
    sheet = Image.new("RGBA", (columns * tile_width, rows * tile_height), (5, 8, 14, 255))

    for index, entry in enumerate(mask_entries):
        col = index % columns
        row = index // columns
        x = col * tile_width
        y = row * tile_height
        source = Image.open(PUBLIC / entry["sourceReferenceCrop"]).convert("RGBA")
        mask = Image.open(PUBLIC / entry["mask"]).convert("L")
        silhouette = Image.open(PUBLIC / entry["silhouettePreview"]).convert("RGBA")
        mask_rgba = Image.merge("RGBA", (mask, mask, mask, Image.new("L", mask.size, 255)))

        sheet.alpha_composite(_fit_thumbnail(source, (108, 108)), (x + 18, y + 52))
        sheet.alpha_composite(_fit_thumbnail(mask_rgba, (108, 108)), (x + 126, y + 52))
        sheet.alpha_composite(_fit_thumbnail(silhouette, (108, 108)), (x + 234, y + 52))
        _draw_label(sheet, x + 18, y + 18, entry["id"])

    sheet.convert("RGB").save(REVIEW_PREVIEW, optimize=True, quality=88)


def main() -> None:
    reference = _load_json(REFERENCE_MANIFEST)
    master = _load_json(MASTER_MANIFEST)
    required_master_ids = {asset["id"] for asset in master["requiredAssets"]}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    masks: list[dict[str, Any]] = []

    for card in reference["cards"]:
        card_id = card["id"]
        target_master_asset_id = TARGET_MASTER_ASSET_IDS.get(card_id, card_id)
        crop_path = PUBLIC / card["uri"]
        image = Image.open(crop_path).convert("RGBA")
        result = make_mask(image)

        mask_uri = f"lunar-city/master-assets/masks/{card_id}-mask.png"
        silhouette_uri = f"lunar-city/master-assets/masks/{card_id}-silhouette.png"
        cached_masked_source = CACHE_OUTPUT / f"{card_id}-masked.png"

        result.mask.save(PUBLIC / mask_uri)
        silhouette_preview(result.mask).save(PUBLIC / silhouette_uri)
        CACHE_OUTPUT.mkdir(parents=True, exist_ok=True)
        masked_source(image, result.mask).save(cached_masked_source)

        quality_flags = [
            *result.quality_flags,
            *mask_quality_flags(card["kind"], result.coverage_ratio, result.method),
        ]
        if target_master_asset_id not in required_master_ids:
            quality_flags.append("target_master_asset_missing")

        generation_input_status = (
            "needs_refined_subject_crop"
            if any(
                flag
                in {
                    "broad_scene_crop_not_subject_isolated",
                    "coverage_below_expected_subject_silhouette",
                    "rembg_alpha_unavailable_or_unusable",
                    "target_master_asset_missing",
                }
                for flag in quality_flags
            )
            else result.generation_input_status
        )

        masks.append(
            {
                "approvedForGeneration": generation_input_status == "ready_for_generation_review",
                "id": card_id,
                "kind": card["kind"],
                "role": card["role"],
                "sourceReferenceCrop": card["uri"],
                "targetMasterAssetId": target_master_asset_id,
                "targetMasterAssetExists": target_master_asset_id in required_master_ids,
                "mask": mask_uri,
                "maskedSourceCachePath": str(cached_masked_source),
                "silhouettePreview": silhouette_uri,
                "coverageRatio": round(result.coverage_ratio, 4),
                "method": result.method,
                "productionUse": "silhouette_prep_only",
                "generationInputStatus": generation_input_status,
                "requiresHumanMaskReview": True,
                "qualityFlags": quality_flags,
            }
        )

    write_review_contact_sheet(masks)

    manifest = {
        "schemaVersion": 1,
        "source": "approved_lunar_city_reference_crop_masks",
        "productionUse": "silhouette_prep_only",
        "productionEligibility": "not_production_master_asset",
        "reviewPreview": "lunar-city/master-assets/masks/mask-review-contact-sheet.png",
        "maskingPolicy": {
            "requiredBeforeImageTo3DGeneration": True,
            "generationMustUseMaskedSource": True,
            "generationMustPreserveSilhouette": True,
            "rejectIfSilhouetteMismatch": True,
            "humanReviewRequiredBeforeMasterPromotion": True,
        },
        "privacy": {
            "usesRawSoulContent": False,
            "containsPrivateProfileIdentifiers": False,
        },
        "sourceManifest": "lunar-city/generated-3d/reference-crops/reference-crops-manifest.json",
        "targetMasterAssetManifest": "lunar-city/master-assets/master-asset-manifest.json",
        "maskCount": len(masks),
        "masks": masks,
    }
    MASK_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"maskCount": len(masks), "manifest": str(MASK_MANIFEST)}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Lunar City masked reference cards and silhouette prep artifacts.

This is an intake/prep stage for high-poly asset generation. It deliberately
does not promote the existing scene-crop image-to-3D meshes to production.
Instead, it creates deterministic masked sources and silhouettes that can guide
future free/local 2D-to-3D runs toward the approved reference silhouettes.
"""

from __future__ import annotations

import json
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
CACHE_OUTPUT = Path("/private/tmp/lunar-city-master-asset-masked-sources")

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
    method: str
    mask: Image.Image
    quality_flags: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _try_rembg(image: Image.Image) -> Image.Image | None:
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
        method = "rembg_alpha_then_background_delta_union"
        mask = ImageChops.lighter(rembg_mask, fallback_mask)
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
    return MaskResult(coverage_ratio=coverage_ratio, mask=mask, method=method, quality_flags=quality_flags)


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

        if target_master_asset_id not in required_master_ids:
            quality_flags = [*result.quality_flags, "target_master_asset_missing"]
        else:
            quality_flags = result.quality_flags

        masks.append(
            {
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
                "requiresHumanMaskReview": True,
                "qualityFlags": quality_flags,
            }
        )

    manifest = {
        "schemaVersion": 1,
        "source": "approved_lunar_city_reference_crop_masks",
        "productionUse": "silhouette_prep_only",
        "productionEligibility": "not_production_master_asset",
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

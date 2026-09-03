#!/usr/bin/env python3
"""Build the Lunar City high-poly master asset intake manifest.

This is intentionally not a generator for final art. It defines the production
gate for real master assets:

1. Full-resolution/high-poly master asset first.
2. Retopology and LODs derived from that master.
3. 2K default / 4K hero PBR bakes derived from that master.

Raw scene-crop image-to-3D meshes and simple placeholder mascots are not valid
production sources.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "public" / "lunar-city" / "master-assets"
SOURCE_DIR = OUT_DIR / "sources"
MANIFEST = OUT_DIR / "master-asset-manifest.json"
MASTER_SCENE = SOURCE_DIR / "lunar-city-sculpted-master-assets.blend"
MASTER_SCENE_METADATA = SOURCE_DIR / "lunar-city-sculpted-master-assets-metadata.json"

ACCEPTED_FORMATS = (".blend", ".glb", ".fbx", ".obj")


def asset(asset_id: str, kind: str, role: str, display_name: str, hero: bool = False) -> dict:
    return {
        "id": asset_id,
        "kind": kind,
        "role": role,
        "displayName": display_name,
        "heroAsset": hero,
        "sourceCandidates": [f"lunar-city/master-assets/sources/{asset_id}{ext}" for ext in ACCEPTED_FORMATS],
        "status": "missing",
        "selectedSource": None,
        "acceptance": {
            "sourceQuality": "full_resolution_high_poly_master",
            "minimumTriangleCount": 120000 if hero else 45000,
            "requiresVisualApproval": True,
            "requiresRecognizableSilhouette": True,
            "requiresPbrBake": True,
            "requiresRetopology": True,
            "requiresLods": ["hero", "high", "medium", "low"] if hero else ["high", "medium", "low"],
            "rejectIf": [
                "raw_scene_crop_relief_mesh",
                "floating_blob",
                "simple_mascot_placeholder",
                "flat_billboard_or_reference_plane",
                "unriggable_single_lump",
                "high_poly_cube_or_wrong_silhouette",
            ],
        },
    }


REQUIRED_ASSETS = [
    asset("terrain-colony-basin", "terrain", "environment", "Concave lunar colony basin", True),
    asset("road-network-primary", "road", "navigation", "Ground-conforming city road network", False),
    asset("skybox-lunar-orbit", "skybox", "environment", "Lunar orbit skybox/background", False),
    asset("building-library", "building", "knowledge", "Library", True),
    asset("building-research-lab", "building", "research", "Research Lab", True),
    asset("building-arts-studio", "building", "creative", "Arts Studio", True),
    asset("building-engineering-workshop", "building", "engineering", "Engineering Workshop", True),
    asset("building-operations-depot", "building", "operations", "Operations Depot", True),
    asset("building-release-gatehouse", "building", "release", "Release Gatehouse", True),
    asset("building-triage-clinic", "building", "medical", "Triage Clinic", True),
    asset("building-council-hall", "building", "governance", "Council Hall", True),
    asset("building-review-office", "building", "review", "Review Office", True),
    asset("building-archive", "building", "archive", "Archive", True),
    asset("building-break-garden", "building", "rest", "Break Garden", True),
    asset("leader-owl-archivist", "leader", "knowledge", "Owl archivist leader", True),
    asset("leader-fox-scientist", "leader", "research", "Fox scientist leader", True),
    asset("leader-raccoon-artist", "leader", "creative", "Raccoon artist leader", True),
    asset("leader-eagle-councillor", "leader", "governance", "Eagle councillor leader", True),
    asset("leader-badger-engineer", "leader", "engineering", "Badger engineer leader", True),
    asset("leader-gold-medic", "leader", "medical", "Gold medic leader", True),
    asset("leader-hawk-reviewer", "leader", "review", "Hawk reviewer leader", True),
    asset("leader-owl-historian", "leader", "archive", "Owl historian leader", True),
    asset("worker-audit", "worker", "audit", "Audit worker robot", False),
    asset("worker-operations", "worker", "operations", "Operations worker robot", False),
    asset("worker-release", "worker", "release", "Release worker robot", False),
    asset("worker-research", "worker", "research", "Research worker robot", False),
    asset("worker-review", "worker", "review", "Review worker robot", False),
    asset("worker-support", "worker", "support", "Support worker robot", False),
    asset("child-curious", "child", "child", "Curious child robot", False),
    asset("child-social", "child", "child", "Social child robot", False),
    asset("child-bold", "child", "child", "Bold child robot", False),
    asset("child-cautious", "child", "child", "Cautious child robot", False),
    asset("dispatcher-cube", "dispatcher", "dispatcher", "Dispatcher companion cube", True),
    asset("vehicle-bus", "vehicle", "transport", "Colony bus / tram", False),
    asset("prop-status-signage", "prop", "state", "In-world status signage set", False),
    asset("prop-repair-tools", "prop", "recovery", "Repair and recovery tool set", False),
]


def mark_present_sources(required_assets: list[dict]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    master_metadata = {}
    master_assets = {}
    if MASTER_SCENE.exists() and MASTER_SCENE_METADATA.exists():
        master_metadata = json.loads(MASTER_SCENE_METADATA.read_text())
        master_assets = {entry["id"]: entry for entry in master_metadata.get("assets", [])}
    for entry in required_assets:
        master_asset = master_assets.get(entry["id"])
        if master_asset:
            entry["status"] = "sculpted_master_scene_present_unvalidated"
            entry["selectedSource"] = "lunar-city/master-assets/sources/lunar-city-sculpted-master-assets.blend"
            entry["sourceBytes"] = MASTER_SCENE.stat().st_size
            entry["sourceCollection"] = master_asset["collection"]
            entry["sourceMetadata"] = "lunar-city/master-assets/sources/lunar-city-sculpted-master-assets-metadata.json"
            entry["evaluatedTriangleCount"] = master_asset["evaluatedTriangleCount"]
            entry["meshObjectCount"] = master_asset["meshObjectCount"]
            entry["sculptedSurfaceCount"] = master_asset["sculptedSurfaceCount"]
            entry["animationRigWireCount"] = master_asset["animationRigWireCount"]
            entry["textureResolutionTarget"] = master_asset["textureResolutionTarget"]
            entry["retopologyTarget"] = master_asset["retopologyTarget"]
            entry["sourceCandidateStatus"] = "authoritative_master_scene_collection"
            continue
        for ext in ACCEPTED_FORMATS:
            candidate = SOURCE_DIR / f"{entry['id']}{ext}"
            if candidate.exists():
                entry["status"] = "source_present_unvalidated"
                entry["selectedSource"] = f"lunar-city/master-assets/sources/{entry['id']}{ext}"
                entry["sourceBytes"] = candidate.stat().st_size
                break


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required_assets = [deepcopy(item) for item in REQUIRED_ASSETS]
    mark_present_sources(required_assets)
    present_count = sum(1 for item in required_assets if item["status"] != "missing")
    missing_count = len(required_assets) - present_count
    manifest = {
        "schemaVersion": 1,
        "productionUse": "production_source_intake",
        "productionReady": False,
        "sourceDirectory": "lunar-city/master-assets/sources",
        "authoritativeMasterScene": (
            "lunar-city/master-assets/sources/lunar-city-sculpted-master-assets.blend"
            if MASTER_SCENE.exists()
            else None
        ),
        "authoritativeMasterSceneMetadata": (
            "lunar-city/master-assets/sources/lunar-city-sculpted-master-assets-metadata.json"
            if MASTER_SCENE_METADATA.exists()
            else None
        ),
        "acceptedFormats": list(ACCEPTED_FORMATS),
        "pipeline": {
            "source": "full_resolution_high_poly_master_assets",
            "retopology": "derive_smart_low_poly_lods_from_master",
            "textureBake": "bake_2k_default_4k_hero_pbr_from_master",
            "animation": "rig_and_animate_after_master_validation",
        },
        "rejectedProductionSources": [
            "raw_scene_crop_image_to_3d",
            "floating_blob_meshes",
            "simple_mascot_generator",
            "flat_reference_planes",
        ],
        "counts": {
            "required": len(required_assets),
            "present": present_count,
            "missing": missing_count,
        },
        "requiredAssets": required_assets,
        "validation": {
            "failsClosedUntilEveryRequiredMasterExists": missing_count > 0,
            "usesSingleAuthoritativeMasterScene": MASTER_SCENE.exists() and MASTER_SCENE_METADATA.exists(),
            "usesPerAssetCollections": bool(present_count) and all(
                "sourceCollection" in item for item in required_assets if item["status"] != "missing"
            ),
            "requiresNoRawSoulContent": True,
            "requiresNoPrivateProfileIdentifiers": True,
            "requiresRecognizableReferenceSilhouette": True,
            "requiresPerAssetRetopologyPlan": True,
            "requiresPerAssetLods": True,
            "requiresPbrTextureBake": True,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(MANIFEST), "required": len(required_assets), "present": present_count, "missing": missing_count}))


if __name__ == "__main__":
    main()

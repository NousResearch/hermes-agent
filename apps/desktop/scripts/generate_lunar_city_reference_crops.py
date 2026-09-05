#!/usr/bin/env python3
"""Build named Lunar City reference cards for local image-to-3D generation.

The cards are derived only from the approved design-reference images supplied by
the operator. They intentionally contain visual archetypes and building roles,
not private profile identifiers or raw SOUL.md content.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "lunar-city" / "generated-3d" / "reference-crops"
MANIFEST = OUTPUT / "reference-crops-manifest.json"

CLEAN_REFERENCE = Path("/Users/mikedemott/Downloads/exec-b753ec5c-e213-413f-886a-c87cb14298f7.png")
UI_REFERENCE = Path("/Users/mikedemott/Downloads/Codex Image Aug 30, 2026, 06_42_48 PM.png")


# Coordinates are hand-selected from the approved clean concept image. They are
# deliberately generous so image-to-3D models receive enough surrounding context
# to reconstruct walls, props, leaders, and silhouettes.
CROPS = [
    {"id": "building-library", "kind": "building", "role": "knowledge", "source": "clean", "box": [88, 38, 535, 372]},
    {"id": "building-research-lab", "kind": "building", "role": "research", "source": "clean", "box": [720, 42, 1255, 360]},
    {"id": "building-arts-studio", "kind": "building", "role": "creative", "source": "clean", "box": [42, 392, 455, 690]},
    {"id": "building-engineering", "kind": "building", "role": "engineering", "source": "clean", "box": [12, 642, 424, 928]},
    {"id": "building-operations-depot", "kind": "building", "role": "operations", "source": "clean", "box": [0, 760, 396, 1012]},
    {"id": "building-release-gatehouse", "kind": "building", "role": "release", "source": "clean", "box": [555, 420, 826, 655]},
    {"id": "building-triage-clinic", "kind": "building", "role": "medical", "source": "clean", "box": [716, 568, 960, 770]},
    {"id": "building-council-hall", "kind": "building", "role": "governance", "source": "clean", "box": [1000, 360, 1456, 662]},
    {"id": "building-review-office", "kind": "building", "role": "review", "source": "clean", "box": [1016, 560, 1456, 820]},
    {"id": "building-archive", "kind": "building", "role": "archive", "source": "clean", "box": [1000, 760, 1395, 1042]},
    {"id": "leader-owl-archivist", "kind": "leader", "role": "knowledge", "source": "clean", "box": [300, 168, 438, 355]},
    {"id": "leader-fox-scientist", "kind": "leader", "role": "research", "source": "clean", "box": [888, 92, 1054, 304]},
    {"id": "leader-raccoon-artist", "kind": "leader", "role": "creative", "source": "clean", "box": [160, 500, 306, 678]},
    {"id": "leader-eagle-councillor", "kind": "leader", "role": "governance", "source": "clean", "box": [1124, 470, 1288, 660]},
    {"id": "leader-badger-engineer", "kind": "leader", "role": "engineering", "source": "clean", "box": [194, 700, 340, 900]},
    {"id": "leader-hawk-reviewer", "kind": "leader", "role": "review", "source": "clean", "box": [1212, 642, 1380, 820]},
    {"id": "leader-owl-historian", "kind": "leader", "role": "archive", "source": "clean", "box": [1134, 818, 1288, 1006]},
    {"id": "worker-bot-round", "kind": "worker", "role": "generic", "source": "clean", "box": [604, 548, 684, 656]},
    {"id": "worker-bot-carrying", "kind": "worker", "role": "delivery", "source": "clean", "box": [676, 472, 760, 600]},
    {"id": "worker-bot-review", "kind": "worker", "role": "review", "source": "clean", "box": [1028, 490, 1110, 602]},
    {"id": "child-bot-garden", "kind": "child", "role": "child", "source": "clean", "box": [602, 744, 684, 856]},
    {"id": "vehicle-bus", "kind": "vehicle", "role": "transit", "source": "clean", "box": [574, 300, 828, 432]},
    {"id": "prop-break-garden", "kind": "prop", "role": "resting", "source": "clean", "box": [438, 720, 804, 990]},
]


def square_card(image: Image.Image, box: list[int], size: int = 768) -> Image.Image:
    crop = image.crop(tuple(box)).convert("RGBA")
    crop = ImageOps.contain(crop, (size, size))
    card = Image.new("RGBA", (size, size), (210, 210, 210, 255))
    x = (size - crop.width) // 2
    y = (size - crop.height) // 2
    card.alpha_composite(crop, (x, y))
    return card.convert("RGB")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "clean": Image.open(CLEAN_REFERENCE),
        "ui": Image.open(UI_REFERENCE),
    }
    manifest = {
        "schemaVersion": 1,
        "source": "approved_lunar_city_reference_images",
        "privacy": {
            "usesRawSoulContent": False,
            "containsPrivateProfileIdentifiers": False,
        },
        "cards": [],
    }
    for crop in CROPS:
        card = square_card(sources[crop["source"]], crop["box"])
        path = OUTPUT / f"{crop['id']}.png"
        card.save(path)
        manifest["cards"].append(
            {
                "id": crop["id"],
                "kind": crop["kind"],
                "role": crop["role"],
                "sourceImage": crop["source"],
                "box": crop["box"],
                "uri": f"lunar-city/generated-3d/reference-crops/{path.name}",
                "targetMesh": f"lunar-city/generated-3d/meshes/{crop['id']}.glb",
            }
        )
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cards": len(manifest["cards"]), "manifest": str(MANIFEST)}, sort_keys=True))


if __name__ == "__main__":
    main()

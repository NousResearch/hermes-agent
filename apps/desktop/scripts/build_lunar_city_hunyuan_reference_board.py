#!/usr/bin/env python3
"""Build a dated Blender review board for the local Hunyuan2MV references."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "public" / "lunar-city" / "generated-3d"
BUILDINGS_DIR = GENERATED / "hunyuan2mv-reference-2026-09-04"
WORKERS_DIR = GENERATED / "hunyuan2mv-worker-reference-2026-09-04"
OUT_DIR = GENERATED / "hunyuan2mv-reference-board-2026-09-04"
BOARD_BLEND = OUT_DIR / "lunar-city-hunyuan2mv-reference-board.blend"
BOARD_GLB = OUT_DIR / "lunar-city-hunyuan2mv-reference-board.glb"
BOARD_RENDER = OUT_DIR / "lunar-city-hunyuan2mv-reference-board.png"
BOARD_METADATA = OUT_DIR / "lunar-city-hunyuan2mv-reference-board.json"

BUILDINGS = [
    ("building-owl-library", "building", "owl.glb"),
    ("building-elephant-memory", "building", "elephant.glb"),
    ("building-cat-arts", "building", "cat.glb"),
    ("building-fox-observatory", "building", "fox.glb"),
    ("building-capybara-revenue", "building", "capybara.glb"),
    ("building-lion-civic", "building", "lion.glb"),
    ("building-beaver-damworks", "building", "beaver.glb"),
    ("building-monkey-publication", "building", "monkey.glb"),
]
WORKERS = [
    ("worker-audit", "worker", "worker-audit.glb"),
    ("worker-operations", "worker", "worker-operations.glb"),
    ("worker-release", "worker", "worker-release.glb"),
    ("worker-research", "worker", "worker-research.glb"),
    ("worker-review", "worker", "worker-review.glb"),
    ("worker-support", "worker", "worker-support.glb"),
]


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in list(bpy.data.collections):
        if coll.users == 0:
            bpy.data.collections.remove(coll)


def get_collection(name: str) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    return coll


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector, Vector]:
    points = []
    for obj in objects:
        if obj.type == "MESH":
            points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        return Vector((0, 0, 0)), Vector((1, 1, 1)), Vector((1, 1, 1))
    low = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    return low, high, high - low


def normalize(objects: list[bpy.types.Object], target: float, location: tuple[float, float, float]) -> None:
    low, high, dims = bounds(objects)
    scale = target / max(dims.x, dims.y, dims.z, 0.001)
    center = (low + high) * 0.5
    for obj in objects:
        obj.location = (obj.location - center) * scale + Vector(location)
        obj.scale = tuple(component * scale for component in obj.scale)
    bpy.context.view_layer.update()
    low, _, _ = bounds(objects)
    for obj in objects:
        obj.location.z -= low.z


def label(text: str, location: tuple[float, float, float], size: float = 0.18) -> None:
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.name = f"label_{text.lower().replace('-', '_').replace(' ', '_')}"
    obj.data.body = text.replace("-", " ").upper()
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    material = bpy.data.materials.get("review_label") or bpy.data.materials.new("review_label")
    material.diffuse_color = (0.65, 0.9, 1.0, 1)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.45, 0.85, 1.0, 1)
        bsdf.inputs["Emission Color"].default_value = (0.1, 0.55, 1.0, 1)
        bsdf.inputs["Emission Strength"].default_value = 0.35
    obj.data.materials.append(material)
    direction = Vector((0, 0, 1)) - Vector(location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def import_reference(asset_id: str, kind: str, filename: str, index: int) -> dict:
    source_dir = BUILDINGS_DIR if kind == "building" else WORKERS_DIR
    path = source_dir / filename
    if not path.exists():
        return {"id": asset_id, "kind": kind, "status": "missing", "source": str(path)}

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    row_y = 3.2 if kind == "building" else -2.0
    step = 2.45 if kind == "building" else 2.65
    count = 8 if kind == "building" else 6
    x = (index - (count - 1) / 2) * step
    target = 2.9 if kind == "building" else 1.7
    normalize(imported, target, (x, row_y, 0))
    coll = get_collection(f"Hunyuan {kind.title()} References")
    for obj in imported:
        if obj.name not in coll.objects:
            coll.objects.link(obj)
        for existing in list(obj.users_collection):
            if existing != coll:
                existing.objects.unlink(obj)
        obj["lunar_city_asset_id"] = asset_id
        obj["reference_only"] = True
    label(asset_id, (x, row_y - 1.35, 0.05), 0.14 if kind == "building" else 0.16)
    low, high, dims = bounds(meshes)
    return {
        "id": asset_id,
        "kind": kind,
        "status": "imported",
        "source": str(path.relative_to(GENERATED)),
        "objects": len(imported),
        "meshObjects": len(meshes),
        "vertices": sum(len(obj.data.vertices) for obj in meshes),
        "faces": sum(len(obj.data.polygons) for obj in meshes),
        "bounds": [round(value, 4) for value in dims],
        "referenceOnly": True,
    }


def setup_scene() -> None:
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0.55, -0.08))
    floor = bpy.context.object
    floor.name = "hunyuan_reference_board_floor"
    floor.dimensions = (22, 11, 0.12)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    material = bpy.data.materials.new("lunar_review_floor")
    material.diffuse_color = (0.025, 0.04, 0.065, 1)
    floor.data.materials.append(material)

    label("HUNYUAN2MV LUNAR CITY REFERENCE BOARD", (0, 6.4, 0.05), 0.3)
    label("BUILDINGS", (-10.2, 3.2, 0.05), 0.2)
    label("WORKERS", (-10.2, -2.0, 0.05), 0.2)

    bpy.ops.object.light_add(type="AREA", location=(0, -2, 10))
    key = bpy.context.object
    key.data.energy = 1300
    key.data.shape = "DISK"
    key.data.size = 8
    bpy.ops.object.light_add(type="AREA", location=(-8, 4, 5))
    fill = bpy.context.object
    fill.data.energy = 650
    fill.data.color = (0.25, 0.55, 1.0)
    fill.data.size = 6
    bpy.ops.object.light_add(type="AREA", location=(8, 2, 4))
    rim = bpy.context.object
    rim.data.energy = 500
    rim.data.color = (1.0, 0.35, 0.12)
    rim.data.size = 5

    bpy.ops.object.camera_add(location=(0, -16, 9))
    camera = bpy.context.object
    camera.data.lens = 48
    camera.data.sensor_width = 36
    camera.rotation_euler = (Vector((0, 0.7, 1.5)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 2200
    scene.render.resolution_y = 1250
    scene.render.resolution_percentage = 50
    scene.render.image_settings.file_format = "PNG"
    scene.world.color = (0.008, 0.012, 0.025)


def export_board() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type in {"MESH", "EMPTY"} and not obj.name.startswith("label_"):
            obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(BOARD_GLB), export_format="GLB", use_selection=True)
    bpy.ops.object.select_all(action="DESELECT")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_scene()
    setup_scene()
    assets = []
    for index, (asset_id, kind, filename) in enumerate(BUILDINGS):
        assets.append(import_reference(asset_id, kind, filename, index))
    for index, (asset_id, kind, filename) in enumerate(WORKERS):
        assets.append(import_reference(asset_id, kind, filename, index))
    metadata = {
        "schemaVersion": 1,
        "generator": "build_lunar_city_hunyuan_reference_board.py",
        "status": "reference_only",
        "blend": str(BOARD_BLEND.relative_to(GENERATED)),
        "glb": str(BOARD_GLB.relative_to(GENERATED)),
        "preview": str(BOARD_RENDER.relative_to(GENERATED)),
        "assets": assets,
        "notes": [
            "Hunyuan3D-2mv shape-only reference pass from approved 2x2 turnarounds.",
            "Plan views were retained as QA references but not mislabeled as elevations.",
            "Meshes require visual review, retopology, PBR baking, scale/collision checks, and LODs before production promotion.",
        ],
    }
    BOARD_METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(BOARD_BLEND))
    export_board()
    bpy.context.scene.render.filepath = str(BOARD_RENDER)
    bpy.ops.render.render(write_still=True)
    print(json.dumps({"imported": sum(item["status"] == "imported" for item in assets), "missing": sum(item["status"] == "missing" for item in assets), "blend": str(BOARD_BLEND), "preview": str(BOARD_RENDER)}))


if __name__ == "__main__":
    main()

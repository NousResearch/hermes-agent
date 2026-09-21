#!/usr/bin/env python3
"""Build a Blender review board for image-to-3D Lunar City assets.

Run with Blender:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python apps/desktop/scripts/build_lunar_city_generated_asset_board.py

The script imports the generated GLBs created from the approved reference crops,
normalizes their scale, groups them into clean collections, adds a non-destructive
wireframe inspection overlay, and writes a review .blend/.glb/.png plus mesh
metadata for desktop tests and human review.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
LUNAR_CITY = PUBLIC / "lunar-city"
GENERATED = LUNAR_CITY / "generated-3d"
REFERENCE_MANIFEST = GENERATED / "reference-crops" / "reference-crops-manifest.json"
MESH_DIR = GENERATED / "meshes"
BOARD_BLEND = GENERATED / "lunar-city-generated-assets-board.blend"
BOARD_GLB = GENERATED / "lunar-city-generated-assets-board.glb"
BOARD_RENDER = GENERATED / "lunar-city-generated-assets-board.png"
BOARD_METADATA = GENERATED / "generated-assets-metadata.json"


COLLECTION_ORDER = ("building", "leader", "worker", "child", "vehicle", "prop")
ROW_Y = {
    "building": 6.0,
    "leader": 2.2,
    "worker": -1.0,
    "child": -3.5,
    "vehicle": -5.6,
    "prop": -7.5,
}
TARGET_SIZE = {
    "building": 2.8,
    "leader": 1.55,
    "worker": 1.1,
    "child": 0.9,
    "vehicle": 1.8,
    "prop": 1.3,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    owner = parent or bpy.context.scene.collection
    if coll.name not in owner.children:
        owner.children.link(coll)
    return coll


def link_to_collection(obj: bpy.types.Object, coll: bpy.types.Collection) -> None:
    if obj.name not in coll.objects:
        coll.objects.link(obj)
    for existing in list(obj.users_collection):
        if existing != coll:
            existing.objects.unlink(obj)


def object_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector, Vector]:
    points: list[Vector] = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        return Vector((0, 0, 0)), Vector((1, 1, 1)), Vector((1, 1, 1))
    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_v, max_v, max_v - min_v


def center_and_scale(objects: list[bpy.types.Object], target: float, location: tuple[float, float, float]) -> None:
    min_v, max_v, dims = object_bounds(objects)
    max_dim = max(dims.x, dims.y, dims.z, 0.001)
    scale = target / max_dim
    center = (min_v + max_v) * 0.5
    for obj in objects:
        obj.location = (obj.location - center) * scale + Vector(location)
        obj.scale = tuple(component * scale for component in obj.scale)
    bpy.context.view_layer.update()
    min_v, _, _ = object_bounds(objects)
    lift = -min_v.z
    for obj in objects:
        obj.location.z += lift


def add_materials() -> tuple[bpy.types.Material, bpy.types.Material, bpy.types.Material]:
    skin = bpy.data.materials.new("generated_asset_skin_reference")
    skin.use_nodes = True
    skin.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.72, 0.82, 0.92, 1)
    skin.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.52
    skin.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.18

    wire = bpy.data.materials.new("cyan_wireframe_inspection_overlay")
    wire.use_nodes = True
    wire.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.0, 0.85, 1.0, 1)
    wire.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (0.0, 0.55, 1.0, 1)
    wire.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.5

    floor = bpy.data.materials.new("matte_lunar_review_floor")
    floor.diffuse_color = (0.08, 0.09, 0.105, 1)
    return skin, wire, floor


def add_label(text: str, loc: tuple[float, float, float], size: float = 0.22) -> bpy.types.Object:
    bpy.ops.object.text_add(location=loc, rotation=(math.radians(65), 0, 0))
    obj = bpy.context.object
    obj.name = f"label_{text.lower().replace(' ', '_')}"
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    mat = bpy.data.materials.get("label_white") or bpy.data.materials.new("label_white")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.95, 0.98, 1.0, 1)
    bsdf.inputs["Emission Color"].default_value = (0.4, 0.75, 1.0, 1)
    bsdf.inputs["Emission Strength"].default_value = 0.25
    mat.diffuse_color = (0.85, 0.95, 1.0, 1)
    obj.data.materials.append(mat)
    return obj


def add_reference_card(
    card: dict,
    loc: tuple[float, float, float],
    size: float,
    reference_collection: bpy.types.Collection,
) -> bpy.types.Object | None:
    image_path = PUBLIC / card["uri"]
    if not image_path.exists():
        return None
    image = bpy.data.images.load(str(image_path))
    mat = bpy.data.materials.new(f"reference_crop_{card['id']}")
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = image
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mat.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])

    bpy.ops.mesh.primitive_plane_add(size=size, location=loc, rotation=(math.radians(90), 0, 0))
    obj = bpy.context.object
    obj.name = f"reference_crop_card_{card['id']}"
    obj.data.materials.append(mat)
    obj["lunar_city_asset_id"] = card["id"]
    obj["reference_only"] = True
    obj.hide_render = True
    link_to_collection(obj, reference_collection)
    return obj


def export_runtime_board_glb() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.get("reference_only") or obj.get("inspection_only"):
            continue
        obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(BOARD_GLB),
        export_format="GLB",
        use_selection=True,
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=6,
        export_draco_position_quantization=14,
        export_draco_normal_quantization=10,
        export_draco_texcoord_quantization=12,
    )
    bpy.ops.object.select_all(action="DESELECT")


def add_floor(floor_material: bpy.types.Material) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.8, -0.045))
    floor = bpy.context.object
    floor.name = "lunar_city_generated_asset_review_floor"
    floor.dimensions = (24, 17, 0.08)
    floor.data.materials.append(floor_material)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def add_wire_overlay(obj: bpy.types.Object, wire_material: bpy.types.Material) -> bpy.types.Object:
    overlay = obj.copy()
    overlay.data = obj.data.copy()
    overlay.name = f"{obj.name}_wire_overlay"
    bpy.context.scene.collection.objects.link(overlay)
    overlay.display_type = "WIRE"
    overlay.show_in_front = True
    overlay.hide_render = True
    overlay.data.materials.clear()
    overlay.data.materials.append(wire_material)
    modifier = overlay.modifiers.new("inspection_wire_skin", "WIREFRAME")
    modifier.thickness = 0.006
    modifier.use_even_offset = True
    return overlay


def import_asset(
    card: dict,
    index_by_kind: dict[str, int],
    root_collection: bpy.types.Collection,
    reference_collection: bpy.types.Collection,
) -> dict:
    asset_id = card["id"]
    kind = card["kind"]
    mesh_path = PUBLIC / card["targetMesh"]
    if not mesh_path.exists():
        return {"id": asset_id, "kind": kind, "status": "missing", "mesh": str(mesh_path)}

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]

    slot = index_by_kind.get(kind, 0)
    index_by_kind[kind] = slot + 1
    x = (slot - 4.5) * 2.35 if kind == "building" else (slot - 3) * 2.0
    y = ROW_Y.get(kind, -7.5)
    center_and_scale(imported, TARGET_SIZE.get(kind, 1.2), (x, y, 0))
    add_reference_card(
        card,
        (x, y + 1.05, TARGET_SIZE.get(kind, 1.2) * 0.5),
        TARGET_SIZE.get(kind, 1.2),
        reference_collection,
    )

    kind_collection = collection(f"{kind}s", root_collection)
    for obj in imported:
        obj.name = f"{asset_id}_{obj.name}"
        link_to_collection(obj, kind_collection)
        if obj.type == "MESH" and not obj.data.materials:
            obj.data.materials.append(bpy.data.materials["generated_asset_skin_reference"])
            obj["needs_pbr_rebake"] = True
        obj["lunar_city_asset_id"] = asset_id
        obj["lunar_city_asset_kind"] = kind
        obj["source_reference_crop"] = card["uri"]

    overlays = []
    for obj in mesh_objects:
        overlay = add_wire_overlay(obj, bpy.data.materials["cyan_wireframe_inspection_overlay"])
        overlay["lunar_city_asset_id"] = asset_id
        overlay["inspection_only"] = True
        link_to_collection(overlay, kind_collection)
        overlays.append(overlay)

    add_label(asset_id.replace("-", " "), (x, y - 1.12, 0.18), size=0.18)

    _, _, dims = object_bounds(mesh_objects)
    vertices = sum(len(obj.data.vertices) for obj in mesh_objects)
    faces = sum(len(obj.data.polygons) for obj in mesh_objects)
    return {
        "id": asset_id,
        "kind": kind,
        "status": "imported",
        "mesh": card["targetMesh"],
        "sourceReferenceCrop": card["uri"],
        "objectCount": len(imported),
        "meshObjectCount": len(mesh_objects),
        "wireOverlayCount": len(overlays),
        "vertices": vertices,
        "faces": faces,
        "bounds": [round(dims.x, 4), round(dims.y, 4), round(dims.z, 4)],
        "pbrStatus": "needs_rebake" if any(obj.get("needs_pbr_rebake") for obj in mesh_objects) else "source_materials",
    }


def setup_camera_and_lighting() -> None:
    bpy.ops.object.light_add(type="AREA", location=(0, -4, 8))
    key = bpy.context.object
    key.name = "large_softbox_asset_review_key"
    key.data.energy = 950
    key.data.size = 8
    bpy.ops.object.light_add(type="POINT", location=(-7, 5, 3))
    rim = bpy.context.object
    rim.name = "cyan_rim_light"
    rim.data.color = (0.2, 0.75, 1.0)
    rim.data.energy = 160

    bpy.ops.object.camera_add(location=(0, -15.5, 9.0), rotation=(math.radians(60), 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.name = "lunar_city_generated_assets_camera"
    camera.data.lens = 22
    camera.data.dof.use_dof = False

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.015, 0.018, 0.027)
    scene.render.resolution_x = 1800
    scene.render.resolution_y = 1200


def main() -> None:
    ensure_dir(GENERATED)
    clear_scene()
    skin, wire, floor_material = add_materials()
    _ = skin, wire
    root_collection = collection("Lunar City Generated 3D Asset Board")
    reference_collection = collection("Reference Crop Cards Hidden From Render", root_collection)
    add_floor(floor_material)
    add_label("Lunar City image-to-3D mesh assets only", (0, 7.9, 0.25), size=0.38)
    for kind, y in ROW_Y.items():
        add_label(kind.upper(), (-10.6, y, 0.2), size=0.24)

    data = json.loads(REFERENCE_MANIFEST.read_text())
    index_by_kind: dict[str, int] = {}
    assets = [import_asset(card, index_by_kind, root_collection, reference_collection) for card in data["cards"]]

    setup_camera_and_lighting()
    metadata = {
        "schemaVersion": 1,
        "generator": "build_lunar_city_generated_asset_board.py",
        "sourceManifest": "lunar-city/generated-3d/reference-crops/reference-crops-manifest.json",
        "blend": "lunar-city/generated-3d/lunar-city-generated-assets-board.blend",
        "glb": "lunar-city/generated-3d/lunar-city-generated-assets-board.glb",
        "preview": "lunar-city/generated-3d/lunar-city-generated-assets-board.png",
        "assetCount": len(assets),
        "importedCount": sum(1 for asset in assets if asset["status"] == "imported"),
        "missingCount": sum(1 for asset in assets if asset["status"] == "missing"),
        "assets": assets,
        "privacy": data["privacy"],
        "notes": [
            "GLBs are generated from approved visual references with local TripoSR.",
            "Wireframe overlays are inspection-only, non-destructive, and hidden from renders.",
            "Reference crop cards are included in a separate Blender collection but hidden from renders.",
            "The exported review GLB excludes reference cards and inspection overlays to keep PR size bounded.",
            "PBR fields are marked needs_rebake until texture baking is complete.",
        ],
    }
    BOARD_METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    bpy.ops.wm.save_as_mainfile(filepath=str(BOARD_BLEND))
    export_runtime_board_glb()
    bpy.context.scene.render.filepath = str(BOARD_RENDER)
    bpy.ops.render.render(write_still=True)
    print(json.dumps({"imported": metadata["importedCount"], "missing": metadata["missingCount"], "blend": str(BOARD_BLEND), "glb": str(BOARD_GLB), "preview": str(BOARD_RENDER)}))


if __name__ == "__main__":
    main()

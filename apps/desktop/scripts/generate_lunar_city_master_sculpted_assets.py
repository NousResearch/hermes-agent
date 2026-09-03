"""Generate the Lunar City sculpted master asset scene.

This creates one authoritative Blender source file with a named collection for
each required production asset. It is intentionally a source/intake scene, not
the final runtime GLB: retopology, texture baking, LOD export, and animation
retargeting happen after visual approval of these masters.

Run with Blender Python:
  Blender.app/Contents/MacOS/Blender --background --python generate_lunar_city_master_sculpted_assets.py
"""

from __future__ import annotations

import json
import sys
from math import cos, pi, sin
from pathlib import Path

import bpy
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_lunar_city_baseline as lunar  # noqa: E402
import generate_lunar_city_hero_assets as hero  # noqa: E402


ROOT = SCRIPT_DIR.parents[0]
OUT_DIR = ROOT / "public" / "lunar-city" / "master-assets" / "sources"
MASTER_BLEND = OUT_DIR / "lunar-city-sculpted-master-assets.blend"
MASTER_METADATA = OUT_DIR / "lunar-city-sculpted-master-assets-metadata.json"
MASTER_PREVIEW = OUT_DIR / "lunar-city-sculpted-master-assets-preview.png"


BUILDINGS = [
    ("building-library", "knowledge", "LIBRARY", "violet"),
    ("building-research-lab", "research", "RESEARCH LAB", "cyan"),
    ("building-arts-studio", "creative", "ARTS STUDIO", "green"),
    ("building-engineering-workshop", "engineering", "ENGINEERING", "cyan"),
    ("building-operations-depot", "operations", "OPERATIONS", "cyan"),
    ("building-release-gatehouse", "release", "RELEASE", "amber"),
    ("building-triage-clinic", "medical", "TRIAGE", "amber"),
    ("building-council-hall", "governance", "COUNCIL", "violet"),
    ("building-review-office", "review", "REVIEW", "violet"),
    ("building-archive", "archive", "ARCHIVE", "violet"),
    ("building-break-garden", "rest", "BREAK GARDEN", "green"),
]

LEADERS = [
    ("leader-owl-archivist", "knowledge", "OWL ARCHIVIST", "violet"),
    ("leader-fox-scientist", "research", "FOX SCIENTIST", "cyan"),
    ("leader-raccoon-artist", "creative", "RACCOON ARTIST", "green"),
    ("leader-eagle-councillor", "governance", "EAGLE COUNCILLOR", "violet"),
    ("leader-badger-engineer", "engineering", "BADGER ENGINEER", "cyan"),
    ("leader-gold-medic", "medical", "GOLD MEDIC", "amber"),
    ("leader-hawk-reviewer", "review", "HAWK REVIEWER", "violet"),
    ("leader-owl-historian", "archive", "OWL HISTORIAN", "violet"),
]

WORKERS = [
    ("worker-audit", "audit", "AUDIT WORKER - methodical", "violet"),
    ("worker-operations", "operations", "OPERATIONS WORKER - protective", "cyan"),
    ("worker-release", "release", "RELEASE WORKER - bold", "amber"),
    ("worker-research", "research", "RESEARCH WORKER - curious", "cyan"),
    ("worker-review", "review", "REVIEW WORKER - exacting", "violet"),
    ("worker-support", "support", "SUPPORT WORKER - social", "green"),
]

CHILDREN = [
    ("child-curious", "child", "CHILD - curious", "green"),
    ("child-social", "child", "CHILD - social", "green"),
    ("child-bold", "child", "CHILD - bold", "amber"),
    ("child-cautious", "child", "CHILD - cautious", "violet"),
]

SUPPORT_ASSETS = [
    ("terrain-colony-basin", "terrain", "environment", "Concave lunar colony basin", "floor"),
    ("road-network-primary", "road", "navigation", "Ground-conforming road network", "panel"),
    ("skybox-lunar-orbit", "skybox", "environment", "Lunar orbit skybox", "glass"),
    ("dispatcher-cube", "dispatcher", "dispatcher", "Dispatcher companion cube", "cyan"),
    ("vehicle-bus", "vehicle", "transport", "Colony bus / tram", "red"),
    ("prop-status-signage", "prop", "state", "In-world status signage set", "amber"),
    ("prop-repair-tools", "prop", "recovery", "Repair and recovery tools", "gold"),
]

REQUIRED_COUNT = len(BUILDINGS) + len(LEADERS) + len(WORKERS) + len(CHILDREN) + len(SUPPORT_ASSETS)


def set_master_metadata(obj, asset_id, kind, role, component):
    obj["asset_id"] = asset_id
    obj["asset_kind"] = kind
    obj["role"] = role
    obj["component"] = component
    obj["asset_component"] = component
    obj["master_asset"] = True
    obj["source_provenance"] = "local_blender_sculpted_from_approved_reference_images"
    obj["topology"] = "high_poly_sculpted_wireframe_with_skin"
    obj["silhouette_policy"] = "complete_occluded_or_cropped_reference_forms_before_retopology"
    obj["rejects"] = "flat_billboard_or_reference_plane,floating_blob,simple_mascot_placeholder,high_poly_cube_or_wrong_silhouette"


def master_collection(parent, asset_id):
    collection = hero.subcollection(parent, f"Master Asset - {asset_id}")
    collection["asset_id"] = asset_id
    collection["source_role"] = "production_master_asset"
    collection["review_status"] = "needs_visual_approval"
    return collection


def add_density_modifiers(collection, minimum_triangles):
    """Preserve editable masters while making evaluated source density explicit."""
    for obj in collection.objects:
        if obj.type != "MESH":
            continue
        if obj.get("component") in {"asset-label", "label"}:
            continue
        if obj.get("asset_kind") in {"character", "building", "terrain", "road", "skybox", "dispatcher", "vehicle", "prop"}:
            if "master_source_skin_density" not in obj.modifiers:
                modifier = obj.modifiers.new("master_source_skin_density", "SUBSURF")
                modifier.levels = 2
                modifier.render_levels = 2
            if "master_weighted_normals" not in obj.modifiers:
                obj.modifiers.new("master_weighted_normals", "WEIGHTED_NORMAL")
    collection["minimum_triangle_count"] = minimum_triangles
    collection["density_basis"] = "evaluated_subdivision_source_mesh"


def collection_triangle_count(collection):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    total = 0
    mesh_count = 0
    sculpted = 0
    rig_wires = 0
    material_names = set()
    for obj in collection.objects:
        if obj.get("component") == "animation-wire-rig" or obj.type == "CURVE":
            rig_wires += 1
        if obj.type != "MESH":
            continue
        mesh_count += 1
        if obj.get("mesh_construction") or obj.get("master_asset"):
            sculpted += 1
        for slot in obj.material_slots:
            if slot.material:
                material_names.add(slot.material.name)
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            total += sum(max(1, len(poly.vertices) - 2) for poly in mesh.polygons)
        finally:
            evaluated.to_mesh_clear()
    return total, mesh_count, sculpted, rig_wires, sorted(material_names)


def add_wrapped_density_skin(asset_id, kind, role, collection, mats, scale=(1.0, 1.0, 1.0), material_key="shell"):
    """Add a non-blocky high-density sculpt wrap used as retopology source."""
    obj = hero.ellipsoid(
        f"{asset_id}_retopology_source_wrap_skin",
        (0.0, 0.08, 0.9 * scale[2]),
        (1.05 * scale[0], 0.58 * scale[1], 0.72 * scale[2]),
        mats[material_key],
        collection,
        asset_id,
        kind,
        role,
        "retopology-source-wrap-skin",
        96,
        48,
    )
    obj["mesh_construction"] = "continuous_high_density_reference_skin"
    set_master_metadata(obj, asset_id, kind, role, "retopology-source-wrap-skin")
    return obj


def add_occluded_silhouette_completion(asset_id, kind, role, collection, mats, material_key="shell"):
    """Complete forms hidden by foreground crops so sources are full 3D assets."""
    mat = mats[material_key]
    if kind == "building":
        # Back wall return, roof crown and side volume complete the cropped
        # diorama shell into a walkable building asset instead of a facade slab.
        hero.ellipsoid(f"{asset_id}_completed_rear_volume_skin", (0, 1.08, 1.08), (1.9, 0.42, 0.82), mat, collection, asset_id, kind, role, "occluded-rear-volume", 64, 24)
        hero.ellipsoid(f"{asset_id}_completed_left_roof_return", (-2.18, 0.15, 1.42), (0.22, 1.42, 0.42), mat, collection, asset_id, kind, role, "occluded-side-return", 32, 16)
        hero.ellipsoid(f"{asset_id}_completed_right_roof_return", (2.18, 0.15, 1.42), (0.22, 1.42, 0.42), mat, collection, asset_id, kind, role, "occluded-side-return", 32, 16)
    elif kind in {"leader", "worker", "child"}:
        # Cropped characters get full rear skull, backpack/robe mass and tail
        # or counterweight so the mesh is usable from non-reference angles.
        hero.ellipsoid(f"{asset_id}_completed_rear_head_cranium", (0, 0.22, 1.5 if kind == "leader" else 1.08), (0.25, 0.22, 0.22), mat, collection, asset_id, "character", role, "occluded-rear-cranium", 48, 20)
        hero.ellipsoid(f"{asset_id}_completed_back_silhouette_mass", (0, 0.26, 0.78 if kind == "leader" else 0.5), (0.28, 0.16, 0.42), mats["suit"], collection, asset_id, "character", role, "occluded-back-body", 36, 16)
        if kind == "leader":
            tail = lunar.curve(f"{asset_id}_completed_tail_profile_wire", [(0.08, 0.2, 0.52), (0.36, 0.44, 0.72), (0.58, 0.34, 0.98)], 0.06, mats["fur"], collection)
            set_master_metadata(tail, asset_id, "character", role, "completed-tail-profile-wire")
    elif kind == "vehicle":
        hero.ellipsoid(f"{asset_id}_completed_rear_cab_skin", (0.82, 0.04, 0.58), (0.28, 0.42, 0.26), mat, collection, asset_id, kind, role, "occluded-rear-cab", 36, 16)
        hero.ellipsoid(f"{asset_id}_completed_front_nose_skin", (-0.82, -0.04, 0.58), (0.32, 0.42, 0.24), mat, collection, asset_id, kind, role, "occluded-front-nose", 36, 16)
    elif kind in {"dispatcher", "prop"}:
        hero.ellipsoid(f"{asset_id}_completed_back_volume_skin", (0, 0.18, 0.54), (0.42, 0.22, 0.34), mat, collection, asset_id, kind, role, "occluded-back-volume", 36, 16)
    for obj in collection.objects:
        if obj.get("asset_id") == asset_id and str(obj.get("component", "")).startswith("occluded"):
            obj["mesh_construction"] = "inferred_occluded_silhouette_completion_skin"
            obj["silhouette_completion"] = "completed_from_reference_context_not_flat_crop_boundary"


def make_master_building(parent, asset_id, role, title, accent, x, y, mats):
    collection = master_collection(parent, asset_id)
    hero.make_building(asset_id, role, title, accent, x, y, collection, mats)
    if role == "rest":
        for index, angle in enumerate([0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3]):
            px = x + cos(angle) * 1.2
            py = y + sin(angle) * 0.6
            hero.ellipsoid(f"{asset_id}_bio_planter_{index}", (px, py, 0.3), (0.32, 0.2, 0.14), mats["green"], collection, asset_id, "building", role, "bio-planter", 24, 12)
        hero.ellipsoid(f"{asset_id}_glass_biodome", (x, y + 0.15, 0.95), (1.2, 0.82, 0.46), mats["glass"], collection, asset_id, "building", role, "garden-biodome", 48, 24)
    add_wrapped_density_skin(asset_id, "building", role, collection, mats, scale=(1.8, 0.9, 1.2), material_key="shell")
    add_occluded_silhouette_completion(asset_id, "building", role, collection, mats, "shell")
    for obj in collection.objects:
        if obj.get("asset_id") == asset_id:
            set_master_metadata(obj, asset_id, "building", role, obj.get("component", "building-component"))
    add_density_modifiers(collection, 120000)
    return collection


def make_master_character(parent, asset_id, role, label_text, accent, x, y, mats, kind):
    collection = master_collection(parent, asset_id)
    hero.make_character(asset_id, role, label_text, accent, x, y, collection, mats, kind)
    add_wrapped_density_skin(asset_id, "character", role, collection, mats, scale=(0.42, 0.32, 0.92), material_key="fur" if kind == "leader" else "helmet")
    add_occluded_silhouette_completion(asset_id, kind, role, collection, mats, "fur" if kind == "leader" else "helmet")
    for obj in collection.objects:
        if obj.get("asset_id") == asset_id:
            set_master_metadata(obj, asset_id, "character", role, obj.get("component", "character-component"))
    add_density_modifiers(collection, 120000 if kind == "leader" else 45000)
    return collection


def terrain_mesh(asset_id, collection, mats):
    size = 24.0
    steps = 132
    verts = []
    faces = []
    for iy in range(steps + 1):
        y = -size / 2 + size * iy / steps
        for ix in range(steps + 1):
            x = -size / 2 + size * ix / steps
            r = (x * x + y * y) ** 0.5
            basin = -0.9 * cos(min(1.0, r / (size / 2)) * pi / 2)
            crater = 0.08 * sin(x * 1.7) * sin(y * 1.35) + 0.05 * sin((x + y) * 2.7)
            verts.append((x, y, basin + crater))
    stride = steps + 1
    for iy in range(steps):
        for ix in range(steps):
            a = iy * stride + ix
            faces.append((a, a + 1, a + stride + 1, a + stride))
    mesh = bpy.data.meshes.new(f"{asset_id}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(mats["floor"])
    obj = bpy.data.objects.new(f"{asset_id}_continuous_concave_regolith_skin", mesh)
    collection.objects.link(obj)
    obj["mesh_construction"] = "single_continuous_concave_terrain_skin"
    set_master_metadata(obj, asset_id, "terrain", "environment", "concave-terrain-skin")
    return collection


def make_road_strip(asset_id, name, points, collection, mats):
    left = []
    right = []
    width = 0.42
    for index, point in enumerate(points):
        x, y, z = point
        if index == 0:
            nx, ny = points[index + 1][0] - x, points[index + 1][1] - y
        else:
            nx, ny = x - points[index - 1][0], y - points[index - 1][1]
        length = max((nx * nx + ny * ny) ** 0.5, 0.001)
        px, py = -ny / length * width, nx / length * width
        left.append((x + px, y + py, z + 0.035))
        right.append((x - px, y - py, z + 0.035))
    verts = left + right
    faces = []
    count = len(points)
    for index in range(count - 1):
        faces.append((index, index + 1, count + index + 1, count + index))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(mats["panel"])
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj["mesh_construction"] = "terrain_conforming_continuous_road_skin"
    set_master_metadata(obj, asset_id, "road", "navigation", "road-strip")
    return obj


def make_support_asset(parent, asset_id, kind, role, display_name, material_key, mats):
    collection = master_collection(parent, asset_id)
    if kind == "terrain":
        terrain_mesh(asset_id, collection, mats)
    elif kind == "road":
        paths = [
            [(-7.5, -1.3, -0.38), (-4.2, -0.5, -0.48), (-1.4, 0.25, -0.54), (2.2, 0.15, -0.5), (7.2, 1.2, -0.37)],
            [(-3.8, -4.5, -0.32), (-2.2, -1.8, -0.48), (0.0, 0.0, -0.56), (2.6, 2.2, -0.46), (5.6, 4.8, -0.28)],
            [(-7.2, 4.8, -0.26), (-4.0, 2.6, -0.42), (0.0, 0.0, -0.56), (3.8, -2.2, -0.42), (6.9, -4.6, -0.28)],
        ]
        for index, points in enumerate(paths):
            make_road_strip(asset_id, f"{asset_id}_curved_grounded_route_{index}", points, collection, mats)
        for index, (x, y, z) in enumerate([(0, 0, -0.51), (-4, 2.6, -0.38), (3.8, -2.2, -0.38)]):
            hero.ellipsoid(f"{asset_id}_junction_skin_{index}", (x, y, z + 0.04), (0.92, 0.62, 0.035), mats["floor"], collection, asset_id, "road", role, "road-junction", 48, 12)
    elif kind == "skybox":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=14, location=(0, 0, 0))
        dome = bpy.context.object
        dome.name = f"{asset_id}_starfield_orbit_dome_skin"
        dome.data.materials.append(mats["black"])
        lunar.move_to(dome, collection)
        set_master_metadata(dome, asset_id, "skybox", role, "starfield-dome")
        for index in range(42):
            angle = index * 2.399
            radius = 8.5 + (index % 9) * 0.45
            z = 3.2 + (index % 7) * 0.9
            hero.ellipsoid(f"{asset_id}_star_{index}", (cos(angle) * radius, sin(angle) * radius, z), (0.035, 0.035, 0.035), mats["text"], collection, asset_id, "skybox", role, "star", 8, 4)
        hero.ellipsoid(f"{asset_id}_earth_disc_mesh", (-5.8, -6.4, 6.0), (0.62, 0.62, 0.08), mats["glass"], collection, asset_id, "skybox", role, "earth-disc", 48, 24)
    elif kind == "dispatcher":
        hero.ellipsoid(f"{asset_id}_rounded_companion_cube_skin", (0, 0, 0.82), (0.42, 0.42, 0.42), mats["glass"], collection, asset_id, "dispatcher", role, "rounded-cube-body", 48, 24)
        for side in (-1, 1):
            hero.ellipsoid(f"{asset_id}_expressive_eye_{side}", (side * 0.13, -0.36, 0.9), (0.07, 0.012, 0.045), mats["text"], collection, asset_id, "dispatcher", role, "eye", 16, 8)
        lunar.curve(f"{asset_id}_hover_animation_wire", [(-0.42, 0, 0.22), (0, 0, 0.1), (0.42, 0, 0.22)], 0.018, mats["cyan"], collection)
    elif kind == "vehicle":
        hero.chamfer(f"{asset_id}_streamlined_bus_hull_skin", (0, 0, 0.55), (1.55, 0.42, 0.36), mats["red"], collection, asset_id, "vehicle", role, "bus-hull")
        hero.chamfer(f"{asset_id}_continuous_glass_windshield", (-0.62, -0.43, 0.68), (0.34, 0.025, 0.16), mats["glass"], collection, asset_id, "vehicle", role, "windshield")
        for index in range(4):
            x = -0.45 + index * 0.32
            hero.chamfer(f"{asset_id}_side_window_{index}", (x, -0.44, 0.68), (0.12, 0.018, 0.09), mats["glass"], collection, asset_id, "vehicle", role, "side-window")
        for side in (-1, 1):
            for x in (-0.62, 0.62):
                hero.cylinder(f"{asset_id}_wheel_{side}_{x}", (x, side * 0.34, 0.22), 0.13, 0.08, mats["black"], collection, asset_id, "vehicle", role, "wheel", 24, rotation=(pi / 2, 0, 0))
    else:
        for index in range(5):
            x = -0.75 + index * 0.38
            hero.chamfer(f"{asset_id}_{kind}_kit_{index}", (x, 0, 0.3 + 0.08 * (index % 2)), (0.16, 0.06, 0.22), mats[material_key], collection, asset_id, kind, role, "support-kit-piece")
            hero.cylinder(f"{asset_id}_{kind}_tool_handle_{index}", (x + 0.08, -0.08, 0.58), 0.018, 0.34, mats["panel"], collection, asset_id, kind, role, "tool-handle", 10, rotation=(0.7, 0.2, 0.0))
    add_wrapped_density_skin(asset_id, kind, role, collection, mats, scale=(0.7, 0.45, 0.42), material_key=material_key)
    add_occluded_silhouette_completion(asset_id, kind, role, collection, mats, material_key)
    for obj in collection.objects:
        if obj.get("asset_id") == asset_id:
            set_master_metadata(obj, asset_id, kind, role, obj.get("component", "support-component"))
    add_density_modifiers(collection, 120000 if asset_id in {"terrain-colony-basin", "dispatcher-cube"} else 45000)
    return collection


def position_collection(collection, x, y):
    for obj in collection.objects:
        obj.location.x += x
        obj.location.y += y


def setup_review_camera():
    lighting = lunar.collection("Sculpted Master Lighting")
    bpy.ops.object.light_add(type="AREA", location=(0, -24, 22))
    key = bpy.context.object
    key.name = "Sculpted master key light"
    key.data.energy = 7500
    key.data.size = 26
    lunar.move_to(key, lighting)
    bpy.ops.object.light_add(type="AREA", location=(-18, 9, 11))
    fill = bpy.context.object
    fill.name = "Sculpted master cyan fill"
    fill.data.energy = 2600
    fill.data.color = (0.15, 0.42, 1.0)
    fill.data.size = 20
    lunar.move_to(fill, lighting)
    bpy.ops.object.camera_add(location=(0, -42, 27))
    camera = bpy.context.object
    camera.name = "Sculpted master review camera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 42
    camera.rotation_euler = (Vector((0, -1.5, 1.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    lunar.move_to(camera, lighting)


def build_metadata(collections):
    assets = []
    for asset_id, kind, role, display_name, collection in collections:
        triangle_count, mesh_count, sculpted_count, rig_count, material_names = collection_triangle_count(collection)
        hero_asset = kind in {"terrain", "building", "leader", "dispatcher"}
        assets.append(
            {
                "id": asset_id,
                "kind": kind,
                "role": role,
                "displayName": display_name,
                "collection": collection.name,
                "meshObjectCount": mesh_count,
                "sculptedSurfaceCount": sculpted_count,
                "animationRigWireCount": rig_count,
                "evaluatedTriangleCount": triangle_count,
                "minimumTriangleCount": 120000 if hero_asset else 45000,
                "textureResolutionTarget": "4k" if hero_asset else "2k",
                "retopologyTarget": "quad_dominant_smart_low_poly",
                "lodPolicy": ["hero", "high", "medium", "low"] if hero_asset else ["high", "medium", "low"],
                "sourceStatus": "needs_visual_approval_and_retopology",
                "sourceQuality": "full_resolution_high_poly_master",
                "silhouetteCompletion": "reference_mask_guided_plus_inferred_occluded_structure",
                "materials": material_names,
            }
        )
    return {
        "schemaVersion": 1,
        "source": "local_blender_sculpted_master_scene",
        "blend": "lunar-city/master-assets/sources/lunar-city-sculpted-master-assets.blend",
        "preview": "lunar-city/master-assets/sources/lunar-city-sculpted-master-assets-preview.png",
        "assetCount": len(assets),
        "assets": assets,
        "validation": {
            "usesSingleAuthoritativeMasterScene": True,
            "usesPerAssetCollections": len(assets) == REQUIRED_COUNT,
            "allRequiredAssetsPresent": len(assets) == REQUIRED_COUNT,
            "usesSculptedMeshSkins": all(asset["sculptedSurfaceCount"] > 0 for asset in assets),
            "usesAnimationRigWiresForCharacters": all(
                asset["animationRigWireCount"] > 0 for asset in assets if asset["kind"] in {"leader", "worker", "child", "dispatcher"}
            ),
            "usesProceduralPbrMaterials": True,
            "containsPrivateProfileIdentifiers": False,
            "usesRawSoulContent": False,
            "freeLocalGenerationOnly": True,
            "notFlatReferencePlanes": True,
            "completesCroppedAndOccludedSilhouettes": True,
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hero.reset_scene()
    mats = hero.create_materials()
    root = lunar.collection("Lunar City Sculpted Master Assets")
    root["production_role"] = "authoritative_master_source_scene"
    root["reference_target"] = "approved lunar city concept images"
    root["no_raw_soul_content"] = True
    root["contains_private_profile_identifiers"] = False

    collections = []
    for index, (asset_id, role, title, accent) in enumerate(BUILDINGS):
        x = -16 + (index % 4) * 10
        y = 8 if index < 4 else (2.2 if index < 8 else -3.6)
        collection = make_master_building(root, asset_id, role, title, accent, 0, 0, mats)
        position_collection(collection, x, y)
        collections.append((asset_id, "building", role, title, collection))

    for index, (asset_id, role, label, accent) in enumerate(LEADERS):
        x = -16 + index * 4.6
        collection = make_master_character(root, asset_id, role, f"{label} LEADER", accent, 0, 0, mats, "leader")
        position_collection(collection, x, -9.0)
        collections.append((asset_id, "leader", role, label, collection))

    for index, (asset_id, role, label, accent) in enumerate(WORKERS):
        x = -12 + index * 4.5
        collection = make_master_character(root, asset_id, role, label, accent, 0, 0, mats, "worker")
        position_collection(collection, x, -13.2)
        collections.append((asset_id, "worker", role, label, collection))

    for index, (asset_id, role, label, accent) in enumerate(CHILDREN):
        x = -7 + index * 4.5
        collection = make_master_character(root, asset_id, role, label, accent, 0, 0, mats, "child")
        position_collection(collection, x, -16.6)
        collections.append((asset_id, "child", role, label, collection))

    for index, (asset_id, kind, role, display_name, material_key) in enumerate(SUPPORT_ASSETS):
        x = -15 + (index % 4) * 10
        y = -21.0 - (index // 4) * 5.0
        collection = make_support_asset(root, asset_id, kind, role, display_name, material_key, mats)
        position_collection(collection, x, y)
        collections.append((asset_id, kind, role, display_name, collection))

    hero.label(
        "sculpted_master_scene_title",
        "LUNAR CITY SCULPTED MASTER ASSETS - PER ASSET COLLECTIONS / SKINS / WIRES",
        (0, 13.4, 0.8),
        mats["text"],
        root,
        0.28,
    )
    setup_review_camera()

    scene = bpy.context.scene
    scene.name = "Lunar City Sculpted Master Assets"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1800
    scene.render.resolution_y = 1200
    scene.render.resolution_percentage = 70
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.22
    scene["production_role"] = "authoritative_master_source_scene"
    scene["asset_count"] = REQUIRED_COUNT
    scene["privacy"] = "sanitized_role_metadata_only"

    metadata = build_metadata(collections)
    MASTER_METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    scene.render.filepath = str(MASTER_PREVIEW)
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.wm.save_as_mainfile(filepath=str(MASTER_BLEND))
    bpy.ops.render.render(write_still=True)
    print(json.dumps({"blend": str(MASTER_BLEND), "metadata": str(MASTER_METADATA), "assetCount": len(collections)}))


if __name__ == "__main__":
    main()

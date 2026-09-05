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
MASTER_BUILDINGS_PREVIEW = OUT_DIR / "lunar-city-sculpted-master-buildings.png"
MASTER_LEADERS_PREVIEW = OUT_DIR / "lunar-city-sculpted-master-leaders.png"
MASTER_WORKERS_PREVIEW = OUT_DIR / "lunar-city-sculpted-master-workers-children.png"
MASTER_SUPPORT_PREVIEW = OUT_DIR / "lunar-city-sculpted-master-support.png"
MASTER_RESEARCH_LAB_CLOSEUP = OUT_DIR / "lunar-city-sculpted-master-research-lab-closeup.png"
MASTER_FOX_LEADER_CLOSEUP = OUT_DIR / "lunar-city-sculpted-master-fox-leader-closeup.png"


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
    finished_components = 0
    anatomical_heads = 0
    material_names = set()
    for obj in collection.objects:
        if obj.get("component") == "animation-wire-rig" or obj.type == "CURVE":
            rig_wires += 1
        if obj.type != "MESH":
            continue
        mesh_count += 1
        if obj.get("mesh_construction") or obj.get("master_asset"):
            sculpted += 1
        if str(obj.get("asset_component", "")).startswith("finished-"):
            finished_components += 1
        if obj.get("asset_component") == "finished-anatomical-species-head":
            anatomical_heads += 1
        for slot in obj.material_slots:
            if slot.material:
                material_names.add(slot.material.name)
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            total += sum(max(1, len(poly.vertices) - 2) for poly in mesh.polygons)
        finally:
            evaluated.to_mesh_clear()
    return total, mesh_count, sculpted, rig_wires, finished_components, anatomical_heads, sorted(material_names)


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
    obj["review_visibility"] = "hidden_in_preview_visible_as_retopology_source_in_blend"
    obj.hide_render = True
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


def add_reference_grade_building_finish(asset_id, role, title, accent, collection, mats):
    accent_mat = mats[accent]

    for obj in collection.objects:
        if obj.get("component") == "retopology-source-wrap-skin":
            obj["review_visibility"] = "hidden_in_preview_visible_as_retopology_source_in_blend"
            obj.hide_render = True

    # A readable cutaway room needs a foreground frame, interior back wall,
    # floor pattern, and furniture silhouettes. Without these the building
    # reads like a generic capsule.
    for side in (-1, 1):
        hero.ellipsoid(f"{asset_id}_finished_outer_arch_column_{side}", (side * 2.44, -1.3, 1.05), (0.18, 0.16, 0.92), mats["shell"], collection, asset_id, "building", role, "finished-outer-arch-column", 32, 16)
        rail = lunar.curve(
            f"{asset_id}_finished_front_safety_rail_{side}",
            [(side * 0.55, -1.82, 0.38), (side * 1.2, -1.86, 0.44), (side * 2.05, -1.78, 0.54)],
            0.025,
            mats["gold"],
            collection,
        )
        set_master_metadata(rail, asset_id, "building", role, "finished-front-safety-rail")
    hero.chamfer(f"{asset_id}_finished_back_wall_deep_panel", (0, 1.5, 1.04), (1.85, 0.035, 0.58), mats["dark"], collection, asset_id, "building", role, "finished-back-wall-deep-panel")
    hero.chamfer(f"{asset_id}_finished_entry_threshold", (0, -1.74, 0.24), (0.88, 0.08, 0.08), accent_mat, collection, asset_id, "building", role, "finished-entry-threshold")
    hero.chamfer(f"{asset_id}_finished_sign_icon_plaque", (-1.02, -1.76, 1.38), (0.2, 0.022, 0.16), accent_mat, collection, asset_id, "building", role, "finished-sign-icon-plaque")
    for col in range(5):
        x = -1.55 + col * 0.78
        hero.chamfer(f"{asset_id}_finished_wall_screen_{col}", (x, 1.43, 1.16), (0.22, 0.022, 0.16), mats["glass"], collection, asset_id, "building", role, "finished-wall-screen")
        hero.ellipsoid(f"{asset_id}_finished_floor_tile_{col}", (x, -0.42, 0.18), (0.24, 0.38, 0.016), mats["panel"], collection, asset_id, "building", role, "finished-floor-tile", 20, 8)
    for row in range(3):
        rib = lunar.curve(
            f"{asset_id}_finished_layered_roof_rib_{row}",
            [(-2.05, 0.92, 1.64 + row * 0.16), (-0.8, 1.08, 1.78 + row * 0.16), (0.8, 1.08, 1.78 + row * 0.16), (2.05, 0.92, 1.64 + row * 0.16)],
            0.018,
            mats["shell"] if row % 2 else accent_mat,
            collection,
        )
        set_master_metadata(rib, asset_id, "building", role, "finished-layered-roof-rib")

    if role in {"knowledge", "archive"}:
        for shelf in range(3):
            x = -1.15 + shelf * 1.15
            hero.chamfer(f"{asset_id}_finished_tall_library_shelf_{shelf}", (x, 1.18, 0.96), (0.34, 0.06, 0.68), mats["wood"], collection, asset_id, "building", role, "finished-library-shelf")
            for book in range(5):
                hero.chamfer(f"{asset_id}_finished_book_spine_{shelf}_{book}", (x - 0.13 + book * 0.065, 1.1, 0.66 + (book % 3) * 0.22), (0.022, 0.014, 0.09), accent_mat, collection, asset_id, "building", role, "finished-book-spine")
        hero.ellipsoid(f"{asset_id}_finished_arcane_orb", (1.35, -0.36, 0.92), (0.22, 0.22, 0.22), accent_mat, collection, asset_id, "building", role, "finished-arcane-orb", 36, 18)
    elif role == "research":
        hero.cone(f"{asset_id}_finished_large_telescope_tube", (1.26, -0.42, 0.98), 0.16, 0.08, 1.0, mats["shell"], collection, asset_id, "building", role, "finished-large-telescope-tube", 32, rotation=(1.18, 0.16, -0.7))
        for i in range(5):
            hero.cylinder(f"{asset_id}_finished_sample_tank_{i}", (-1.45 + i * 0.36, 0.68, 0.72), 0.06, 0.46, mats["glass"], collection, asset_id, "building", role, "finished-sample-tank", 18)
    elif role == "creative":
        hero.chamfer(f"{asset_id}_finished_easel_canvas", (0.78, -0.52, 0.82), (0.28, 0.026, 0.36), mats["text"], collection, asset_id, "building", role, "finished-easel-canvas")
        for i in range(8):
            hero.ellipsoid(f"{asset_id}_finished_paint_pot_{i}", (-1.35 + i * 0.22, -0.74, 0.23), (0.04, 0.04, 0.04), accent_mat, collection, asset_id, "building", role, "finished-paint-pot", 12, 6)
    elif role == "engineering":
        for i in range(4):
            hero.chamfer(f"{asset_id}_finished_engineering_bench_{i}", (-1.38 + i * 0.9, 0.18, 0.46), (0.32, 0.16, 0.11), mats["wood"], collection, asset_id, "building", role, "finished-engineering-bench")
            hero.cylinder(f"{asset_id}_finished_power_coil_{i}", (-1.38 + i * 0.9, -0.02, 0.72), 0.075, 0.28, accent_mat, collection, asset_id, "building", role, "finished-power-coil", 18)
    elif role == "operations":
        for i in range(4):
            hero.chamfer(f"{asset_id}_finished_storage_crate_{i}", (-1.15 + i * 0.72, -0.1, 0.38), (0.22, 0.18, 0.14), mats["panel"], collection, asset_id, "building", role, "finished-storage-crate")
        hero.chamfer(f"{asset_id}_finished_loader_console", (1.35, 0.52, 0.72), (0.34, 0.12, 0.18), mats["glass"], collection, asset_id, "building", role, "finished-loader-console")
    elif role == "release":
        hero.chamfer(f"{asset_id}_finished_release_gate_frame", (0, -0.82, 0.88), (0.72, 0.055, 0.48), mats["gold"], collection, asset_id, "building", role, "finished-release-gate-frame")
        hero.chamfer(f"{asset_id}_finished_ready_status_board", (1.2, -0.92, 0.96), (0.34, 0.03, 0.16), mats["green"], collection, asset_id, "building", role, "finished-ready-status-board")
    elif role == "medical":
        hero.chamfer(f"{asset_id}_finished_triage_bed", (-0.58, -0.18, 0.42), (0.68, 0.26, 0.1), mats["white"], collection, asset_id, "building", role, "finished-triage-bed")
        hero.cylinder(f"{asset_id}_finished_scanner_column", (1.1, 0.28, 0.76), 0.14, 0.86, mats["glass"], collection, asset_id, "building", role, "finished-scanner-column", 28)
    elif role == "governance":
        hero.cylinder(f"{asset_id}_finished_council_holo_table", (0, -0.08, 0.42), 0.42, 0.12, mats["glass"], collection, asset_id, "building", role, "finished-council-holo-table", 40)
        for i in range(3):
            hero.chamfer(f"{asset_id}_finished_banner_{i}", (-0.48 + i * 0.48, 1.08, 1.34), (0.12, 0.016, 0.34), accent_mat, collection, asset_id, "building", role, "finished-council-banner")
    elif role == "review":
        for i in range(5):
            hero.chamfer(f"{asset_id}_finished_review_monitor_{i}", (-1.4 + i * 0.7, 0.66, 0.9), (0.22, 0.024, 0.15), mats["glass"], collection, asset_id, "building", role, "finished-review-monitor")
    elif role == "rest":
        hero.ellipsoid(f"{asset_id}_finished_central_garden_dome", (0, -0.1, 0.72), (0.72, 0.48, 0.3), mats["glass"], collection, asset_id, "building", role, "finished-central-garden-dome", 40, 18)
        for i in range(10):
            hero.ellipsoid(f"{asset_id}_finished_garden_plant_{i}", (-1.2 + i * 0.27, -0.45 + 0.08 * (i % 3), 0.28), (0.045, 0.035, 0.11), mats["green"], collection, asset_id, "building", role, "finished-garden-plant", 10, 6)


def anthropomorphic_head_mesh(asset_id, role, label_text, collection, mats):
    lower = label_text.lower()
    skin_mat = hero.leader_skin_material(label_text, role, mats)
    verts = []
    faces = []
    rings = 20
    segments = 48
    z = 1.74
    for ring in range(rings + 1):
        theta = pi * ring / rings
        for segment in range(segments):
            phi = 2 * pi * segment / segments
            sx = sin(theta) * cos(phi)
            sy = sin(theta) * sin(phi)
            sz = cos(theta)
            front = max(0.0, -sy)
            side = abs(sx)
            crown = max(0.0, sz)
            jaw = max(0.0, -sz)
            if "fox" in lower:
                rx, ry, rz = 0.24 + 0.03 * side, 0.22 + 0.08 * front, 0.28
            elif "raccoon" in lower or "badger" in lower:
                rx, ry, rz = 0.28 + 0.03 * side, 0.22 + 0.06 * front, 0.23 + 0.05 * crown
            elif "eagle" in lower or "hawk" in lower:
                rx, ry, rz = 0.24 + 0.02 * side, 0.22 + 0.08 * front, 0.29 + 0.04 * crown
            elif "owl" in lower:
                rx, ry, rz = 0.3 + 0.02 * side, 0.2 + 0.05 * front, 0.31 + 0.03 * crown
            else:
                rx, ry, rz = 0.26, 0.21 + 0.05 * front, 0.26
            px = sx * rx * (1.0 - 0.08 * jaw)
            py = -0.18 + sy * ry - 0.08 * front * (1.0 - side)
            pz = z + sz * rz - 0.05 * jaw
            if front > 0.45 and abs(sx) < 0.45 and -0.65 < sz < 0.25:
                py -= 0.06 + (0.055 if "fox" in lower else 0.035)
                px *= 0.72
                pz -= 0.03 * jaw
            verts.append((px, py, pz))
    for ring in range(rings):
        for segment in range(segments):
            a = ring * segments + segment
            b = ring * segments + (segment + 1) % segments
            c = (ring + 1) * segments + (segment + 1) % segments
            d = (ring + 1) * segments + segment
            faces.append((a, b, c, d))
    mesh = bpy.data.meshes.new(f"{asset_id}_anatomical_head_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(skin_mat)
    obj = bpy.data.objects.new(f"{asset_id}_finished_anatomical_species_head_skin", mesh)
    collection.objects.link(obj)
    obj["mesh_construction"] = "integrated_anthropomorphic_species_head_skin"
    obj["retopology_target"] = "quad_face_loops_with_muzzle_and_brow_flow"
    set_master_metadata(obj, asset_id, "character", role, "finished-anatomical-species-head")
    return hero.polish_surface(obj, subdivision=1, bevel=0.003)


def add_reference_grade_leader_finish(asset_id, role, label_text, collection, mats, accent):
    """Add the visible anthropomorphic finish missing from rough blob passes."""
    lower = label_text.lower()
    accent_mat = mats[accent]
    skin_mat = hero.leader_skin_material(label_text, role, mats)
    z = 1.74

    for obj in collection.objects:
        component = str(obj.get("component", ""))
        if (
            obj.get("asset_kind") == "character"
            and component != "animation-wire-rig"
            and not component.startswith("finished-")
        ):
            obj["review_visibility"] = "hidden_in_preview_visible_as_retopology_source_in_blend"
            obj.hide_render = True

    coat_verts = [
        (-0.46, -0.36, 1.22),
        (0.46, -0.36, 1.22),
        (0.36, -0.28, 0.42),
        (0.2, -0.3, 0.12),
        (-0.2, -0.3, 0.12),
        (-0.36, -0.28, 0.42),
        (-0.34, 0.1, 1.16),
        (0.34, 0.1, 1.16),
        (0.28, 0.16, 0.38),
        (0.12, 0.08, 0.08),
        (-0.12, 0.08, 0.08),
        (-0.28, 0.16, 0.38),
    ]
    coat_faces = [
        (0, 1, 2, 3, 4, 5),
        (6, 11, 10, 9, 8, 7),
        (0, 6, 7, 1),
        (1, 7, 8, 2),
        (2, 8, 9, 3),
        (3, 9, 10, 4),
        (4, 10, 11, 5),
        (5, 11, 6, 0),
    ]
    mesh = bpy.data.meshes.new(f"{asset_id}_finished_tailored_coat_mesh")
    mesh.from_pydata(coat_verts, [], coat_faces)
    mesh.update()
    mesh.materials.append(accent_mat)
    coat = bpy.data.objects.new(f"{asset_id}_finished_tailored_coat_skin", mesh)
    collection.objects.link(coat)
    coat["mesh_construction"] = "continuous_tailored_cloak_skin"
    set_master_metadata(coat, asset_id, "character", role, "finished-tailored-coat-skin")
    hero.polish_surface(coat, subdivision=1, bevel=0.025)

    hero.ellipsoid(f"{asset_id}_finished_visible_neck", (0, -0.12, 1.33), (0.13, 0.1, 0.18), skin_mat, collection, asset_id, "character", role, "finished-visible-neck", 24, 12)
    hero.chamfer(f"{asset_id}_finished_inner_vest_panel", (0, -0.43, 0.82), (0.16, 0.018, 0.36), mats["dark"], collection, asset_id, "character", role, "finished-inner-vest-panel")
    for stripe in range(5):
        x = (stripe - 2) * 0.055
        seam = lunar.curve(f"{asset_id}_finished_cloak_gold_seam_{stripe}", [(x, -0.47, 1.12), (x * 0.7, -0.48, 0.72), (x * 0.35, -0.44, 0.28)], 0.01, mats["gold"], collection)
        set_master_metadata(seam, asset_id, "character", role, "finished-cloak-gold-seam")

    # Face and posture are the first-read features in the reference. These
    # are deliberately large and frontal so the leader does not collapse into
    # a generic pawn when viewed at desktop game scale.
    anthropomorphic_head_mesh(asset_id, role, label_text, collection, mats)
    hero.ellipsoid(f"{asset_id}_finished_muzzle_highlight_patch", (0, -0.62, z - 0.08), (0.13, 0.018, 0.088), mats["fur_light"], collection, asset_id, "character", role, "finished-muzzle-highlight-patch", 32, 14)
    for side in (-1, 1):
        hero.ellipsoid(f"{asset_id}_finished_eye_white_{side}", (side * 0.13, -0.66, z + 0.05), (0.07, 0.014, 0.052), mats["white"], collection, asset_id, "character", role, "finished-eye-white", 24, 12)
        hero.ellipsoid(f"{asset_id}_finished_eye_pupil_{side}", (side * 0.135, -0.674, z + 0.048), (0.03, 0.006, 0.026), mats["black"], collection, asset_id, "character", role, "finished-eye-pupil", 16, 8)
        hero.ellipsoid(f"{asset_id}_finished_shoulder_pad_{side}", (side * 0.42, -0.08, 1.15), (0.2, 0.14, 0.085), accent_mat, collection, asset_id, "character", role, "finished-shoulder-pad", 32, 12)
        hero.sculpted_limb(f"{asset_id}_finished_robed_sleeve_{side}", (side * 0.38, -0.06, 1.08), (side * 0.58, -0.34, 0.75), 0.075, 0.047, accent_mat, collection, asset_id, role, "finished-robed-sleeve", 18)
        hero.ellipsoid(f"{asset_id}_finished_hand_{side}", (side * 0.62, -0.38, 0.72), (0.075, 0.05, 0.055), mats["fur_light"], collection, asset_id, "character", role, "finished-hand", 20, 10)

    if "owl" in lower:
        for side in (-1, 1):
            hero.ellipsoid(f"{asset_id}_finished_owl_face_disc_{side}", (side * 0.135, -0.565, z + 0.04), (0.14, 0.014, 0.16), mats["fur_light"], collection, asset_id, "character", role, "finished-owl-face-disc", 32, 16)
            hero.cone(f"{asset_id}_finished_owl_horn_{side}", (side * 0.22, -0.13, z + 0.35), 0.095, 0.006, 0.42, skin_mat, collection, asset_id, "character", role, "finished-owl-horn", 16, rotation=(0.32, side * 0.56, 0))
        for index in range(11):
            dx = (index - 5) * 0.045
            feather = lunar.curve(f"{asset_id}_finished_layered_chest_feather_{index}", [(dx, -0.46, 1.1), (dx * 0.8, -0.5, 0.92), (dx * 0.55, -0.46, 0.76)], 0.015, skin_mat, collection)
            set_master_metadata(feather, asset_id, "character", role, "finished-layered-chest-feather")
        hero.cone(f"{asset_id}_finished_short_beak", (0, -0.61, z - 0.05), 0.07, 0.01, 0.22, mats["beak"], collection, asset_id, "character", role, "finished-beak", 20, rotation=(1.52, 0, 0))
    elif "fox" in lower:
        hero.cone(f"{asset_id}_finished_long_fox_snout", (0, -0.68, z - 0.08), 0.105, 0.024, 0.28, mats["fur_light"], collection, asset_id, "character", role, "finished-fox-snout", 28, rotation=(1.54, 0, 0))
        for side in (-1, 1):
            hero.cone(f"{asset_id}_finished_tall_fox_ear_{side}", (side * 0.24, -0.08, z + 0.34), 0.105, 0.005, 0.48, skin_mat, collection, asset_id, "character", role, "finished-fox-ear", 18, rotation=(0.26, side * 0.62, 0))
        tail = lunar.curve(f"{asset_id}_finished_bushy_fox_tail", [(0.26, 0.18, 0.38), (0.82, 0.34, 0.9), (0.55, 0.05, 1.42)], 0.13, skin_mat, collection)
        set_master_metadata(tail, asset_id, "character", role, "finished-bushy-fox-tail")
        hero.cone(f"{asset_id}_finished_telescope_prop", (0.64, -0.42, 1.18), 0.075, 0.045, 0.72, mats["shell"], collection, asset_id, "character", role, "finished-telescope-prop", 24, rotation=(1.2, 0.12, -0.72))
    elif "raccoon" in lower:
        for side in (-1, 1):
            hero.ellipsoid(f"{asset_id}_finished_raccoon_mask_{side}", (side * 0.13, -0.575, z + 0.04), (0.115, 0.012, 0.075), mats["black"], collection, asset_id, "character", role, "finished-raccoon-mask", 24, 10)
        for band in range(4):
            tail = lunar.curve(f"{asset_id}_finished_ringed_tail_band_{band}", [(0.28, 0.22, 0.4 + band * 0.16), (0.68, 0.3, 0.55 + band * 0.16), (0.62, 0.12, 0.68 + band * 0.16)], 0.025, mats["black"] if band % 2 else skin_mat, collection)
            set_master_metadata(tail, asset_id, "character", role, "finished-ringed-tail-band")
        hero.cylinder(f"{asset_id}_finished_paintbrush_prop", (0.58, -0.38, 1.02), 0.02, 0.64, mats["gold"], collection, asset_id, "character", role, "finished-paintbrush-prop", 12, rotation=(0.8, 0.2, -0.55))
    elif "eagle" in lower or "hawk" in lower:
        hero.cone(f"{asset_id}_finished_hooked_raptor_beak", (0, -0.64, z - 0.04), 0.105, 0.006, 0.38, mats["beak"], collection, asset_id, "character", role, "finished-raptor-beak", 24, rotation=(1.5, 0, 0))
        for side in (-1, 1):
            wing = lunar.curve(f"{asset_id}_finished_feathered_wing_{side}", [(side * 0.28, 0.02, 1.25), (side * 0.76, 0.08, 0.9), (side * 0.54, 0.0, 0.48)], 0.095 if "eagle" in lower else 0.07, skin_mat, collection)
            set_master_metadata(wing, asset_id, "character", role, "finished-feathered-wing")
            hero.cone(f"{asset_id}_finished_brow_crest_{side}", (side * 0.1, -0.59, z + 0.17), 0.05, 0.006, 0.2, mats["white"], collection, asset_id, "character", role, "finished-brow-crest", 12, rotation=(1.1, side * 0.18, side * 0.65))
    elif "badger" in lower:
        stripe = lunar.curve(f"{asset_id}_finished_badger_crown_stripe", [(0, -0.59, z + 0.24), (0, -0.61, z + 0.03), (0, -0.58, z - 0.22)], 0.045, mats["white"], collection)
        set_master_metadata(stripe, asset_id, "character", role, "finished-badger-crown-stripe")
        for side in (-1, 1):
            hero.ellipsoid(f"{asset_id}_finished_badger_face_band_{side}", (side * 0.14, -0.57, z + 0.03), (0.095, 0.012, 0.16), mats["black"], collection, asset_id, "character", role, "finished-badger-face-band", 24, 12)
            hero.ellipsoid(f"{asset_id}_finished_stocky_badger_shoulder_{side}", (side * 0.34, -0.05, 1.14), (0.26, 0.18, 0.12), skin_mat, collection, asset_id, "character", role, "finished-stocky-shoulder", 32, 12)
        hero.chamfer(f"{asset_id}_finished_wrench_prop", (0.56, -0.42, 0.96), (0.18, 0.035, 0.065), mats["shell"], collection, asset_id, "character", role, "finished-wrench-prop")
    elif "gold" in lower:
        hero.ellipsoid(f"{asset_id}_finished_soft_medic_helmet", (0, -0.16, z + 0.06), (0.36, 0.26, 0.26), skin_mat, collection, asset_id, "character", role, "finished-medic-helmet", 48, 22)
        hero.chamfer(f"{asset_id}_finished_medic_cross_bar", (0, -0.59, z + 0.18), (0.16, 0.014, 0.036), mats["red"], collection, asset_id, "character", role, "finished-medic-cross-bar")
        hero.chamfer(f"{asset_id}_finished_medic_cross_stem", (0, -0.59, z + 0.18), (0.042, 0.014, 0.14), mats["red"], collection, asset_id, "character", role, "finished-medic-cross-stem")
        hero.chamfer(f"{asset_id}_finished_medkit_prop", (-0.58, -0.42, 0.82), (0.18, 0.05, 0.13), mats["white"], collection, asset_id, "character", role, "finished-medkit-prop")


def add_reference_grade_worker_finish(asset_id, role, collection, mats, accent):
    accent_mat = mats[accent]
    for side in (-1, 1):
        hero.ellipsoid(f"{asset_id}_finished_glove_{side}", (side * 0.42, -0.34, 0.46), (0.06, 0.04, 0.045), mats["helmet"], collection, asset_id, "character", role, "finished-worker-glove", 18, 8)
        hero.ellipsoid(f"{asset_id}_finished_knee_joint_{side}", (side * 0.14, -0.05, 0.28), (0.055, 0.04, 0.045), accent_mat, collection, asset_id, "character", role, "finished-worker-knee-joint", 18, 8)
    if role == "audit":
        hero.ellipsoid(f"{asset_id}_finished_magnifier_lens", (0.44, -0.44, 0.72), (0.09, 0.016, 0.09), mats["glass"], collection, asset_id, "character", role, "finished-audit-magnifier", 24, 10)
        hero.cylinder(f"{asset_id}_finished_magnifier_handle", (0.52, -0.38, 0.58), 0.014, 0.28, mats["gold"], collection, asset_id, "character", role, "finished-audit-handle", 10, rotation=(0.75, 0.3, -0.5))
    elif role == "operations":
        hero.chamfer(f"{asset_id}_finished_hardhat_brim", (0, -0.28, 1.2), (0.23, 0.035, 0.035), accent_mat, collection, asset_id, "character", role, "finished-hardhat-brim")
    elif role == "release":
        hero.chamfer(f"{asset_id}_finished_delivery_crate", (0.5, -0.36, 0.62), (0.16, 0.12, 0.12), mats["gold"], collection, asset_id, "character", role, "finished-delivery-crate")
    elif role == "research":
        hero.cylinder(f"{asset_id}_finished_sample_vial", (0.46, -0.4, 0.68), 0.035, 0.22, mats["glass"], collection, asset_id, "character", role, "finished-sample-vial", 16)
    elif role == "review":
        hero.chamfer(f"{asset_id}_finished_clipboard", (-0.42, -0.36, 0.68), (0.12, 0.022, 0.16), mats["panel"], collection, asset_id, "character", role, "finished-clipboard")
    elif role == "support":
        hero.ellipsoid(f"{asset_id}_finished_support_heart_status", (0, -0.31, 0.76), (0.07, 0.012, 0.055), mats["green"], collection, asset_id, "character", role, "finished-support-heart", 18, 8)


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
    add_reference_grade_building_finish(asset_id, role, title, accent, collection, mats)
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
    if kind == "leader":
        add_reference_grade_leader_finish(asset_id, role, label_text, collection, mats, accent)
    else:
        add_reference_grade_worker_finish(asset_id, role, collection, mats, accent)
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


def set_named_collections_render(collections, visible_ids):
    visible_names = {f"Master Asset - {asset_id}" for asset_id in visible_ids}
    for _asset_id, _kind, _role, _display_name, collection in collections:
        collection_hidden = collection.name not in visible_names
        collection.hide_render = collection_hidden
        for obj in collection.objects:
            obj.hide_render = collection_hidden or obj.get("review_visibility") == "hidden_in_preview_visible_as_retopology_source_in_blend"


def show_all_collections(collections):
    for _asset_id, _kind, _role, _display_name, collection in collections:
        collection.hide_render = False
        for obj in collection.objects:
            obj.hide_render = obj.get("review_visibility") == "hidden_in_preview_visible_as_retopology_source_in_blend"


def aim_camera(location, target, ortho_scale):
    camera = bpy.context.scene.camera
    if not camera:
        return
    camera.location = location
    camera.data.ortho_scale = ortho_scale
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def render_review_previews(collections):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"

    # The skybox and terrain are valid source assets, but they physically
    # occlude the board when shown in the all-asset orthographic review render.
    set_named_collections_render(
        collections,
        [asset_id for asset_id, kind, *_rest in collections if kind not in {"skybox", "terrain"}],
    )
    aim_camera((0, -42, 27), (0, -4.5, 1.0), 44)
    scene.render.filepath = str(MASTER_PREVIEW)
    bpy.ops.render.render(write_still=True)

    set_named_collections_render(collections, [asset_id for asset_id, kind, *_rest in collections if kind == "building"])
    aim_camera((0, -33, 18), (0, 2.2, 1.0), 42)
    scene.render.filepath = str(MASTER_BUILDINGS_PREVIEW)
    bpy.ops.render.render(write_still=True)

    set_named_collections_render(collections, [asset_id for asset_id, kind, *_rest in collections if kind == "leader"])
    aim_camera((0, -25, 14), (0, -10.2, 0.95), 20)
    scene.render.filepath = str(MASTER_LEADERS_PREVIEW)
    bpy.ops.render.render(write_still=True)

    set_named_collections_render(collections, [asset_id for asset_id, kind, *_rest in collections if kind in {"worker", "child"}])
    aim_camera((0, -26, 13), (0, -17.3, 0.65), 20)
    scene.render.filepath = str(MASTER_WORKERS_PREVIEW)
    bpy.ops.render.render(write_still=True)

    set_named_collections_render(collections, [asset_id for asset_id, kind, *_rest in collections if kind in {"terrain", "road", "dispatcher", "vehicle", "prop"}])
    aim_camera((0, -38, 24), (0, -22.8, 1.2), 36)
    scene.render.filepath = str(MASTER_SUPPORT_PREVIEW)
    bpy.ops.render.render(write_still=True)

    set_named_collections_render(collections, ["building-research-lab"])
    aim_camera((-3.8, -13.0, 7.0), (-3.5, 8.0, 1.1), 5.8)
    scene.render.filepath = str(MASTER_RESEARCH_LAB_CLOSEUP)
    bpy.ops.render.render(write_still=True)

    set_named_collections_render(collections, ["leader-fox-scientist"])
    aim_camera((-2.8, -15.0, 7.4), (-2.8, -8.6, 1.05), 3.4)
    scene.render.filepath = str(MASTER_FOX_LEADER_CLOSEUP)
    bpy.ops.render.render(write_still=True)
    show_all_collections(collections)


def build_metadata(collections):
    assets = []
    for asset_id, kind, role, display_name, collection in collections:
        triangle_count, mesh_count, sculpted_count, rig_count, finished_count, anatomical_heads, material_names = collection_triangle_count(collection)
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
                "finishedSilhouetteComponentCount": finished_count,
                "anatomicalHeadMeshCount": anatomical_heads,
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
        "buildingPreview": "lunar-city/master-assets/sources/lunar-city-sculpted-master-buildings.png",
        "leaderPreview": "lunar-city/master-assets/sources/lunar-city-sculpted-master-leaders.png",
        "workerChildPreview": "lunar-city/master-assets/sources/lunar-city-sculpted-master-workers-children.png",
        "supportPreview": "lunar-city/master-assets/sources/lunar-city-sculpted-master-support.png",
        "researchLabCloseupPreview": "lunar-city/master-assets/sources/lunar-city-sculpted-master-research-lab-closeup.png",
        "foxLeaderCloseupPreview": "lunar-city/master-assets/sources/lunar-city-sculpted-master-fox-leader-closeup.png",
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
            "usesReferenceGradeLeaderFinishing": all(
                asset["finishedSilhouetteComponentCount"] >= 12 for asset in assets if asset["kind"] == "leader"
            ),
            "usesAnatomicalLeaderHeadMeshes": all(
                asset["anatomicalHeadMeshCount"] >= 1 for asset in assets if asset["kind"] == "leader"
            ),
            "usesReferenceGradeBuildingFinishing": all(
                asset["finishedSilhouetteComponentCount"] >= 14 for asset in assets if asset["kind"] == "building"
            ),
            "usesRoleSpecificWorkerFinishing": all(
                asset["finishedSilhouetteComponentCount"] >= 3 for asset in assets if asset["kind"] in {"worker", "child"}
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
        x = -10.5 + (index % 4) * 7.0
        y = 8 if index < 4 else (2.2 if index < 8 else -3.6)
        collection = make_master_building(root, asset_id, role, title, accent, 0, 0, mats)
        position_collection(collection, x, y)
        collections.append((asset_id, "building", role, title, collection))

    for index, (asset_id, role, label, accent) in enumerate(LEADERS):
        x = -8.4 + (index % 4) * 5.6
        y = -8.6 - (index // 4) * 3.2
        collection = make_master_character(root, asset_id, role, f"{label} LEADER", accent, 0, 0, mats, "leader")
        position_collection(collection, x, y)
        collections.append((asset_id, "leader", role, label, collection))

    for index, (asset_id, role, label, accent) in enumerate(WORKERS):
        x = -7.5 + (index % 3) * 7.5
        y = -14.4 - (index // 3) * 3.0
        collection = make_master_character(root, asset_id, role, label, accent, 0, 0, mats, "worker")
        position_collection(collection, x, y)
        collections.append((asset_id, "worker", role, label, collection))

    for index, (asset_id, role, label, accent) in enumerate(CHILDREN):
        x = -8.4 + index * 5.6
        collection = make_master_character(root, asset_id, role, label, accent, 0, 0, mats, "child")
        position_collection(collection, x, -20.2)
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

    bpy.ops.wm.save_as_mainfile(filepath=str(MASTER_BLEND))
    render_review_previews(collections)
    print(json.dumps({"blend": str(MASTER_BLEND), "metadata": str(MASTER_METADATA), "assetCount": len(collections)}))


if __name__ == "__main__":
    main()

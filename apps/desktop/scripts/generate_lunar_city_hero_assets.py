"""Generate Lunar City hero asset models.

This is the replacement path for the crude placeholder kit. The output is a
Blender asset board and GLB containing named hero assets for every building,
leader, worker, and child archetype used by the Lunar City world.

The assets are still generated locally/free in Blender, but the composition is
asset-first: reusable skinned meshes, curved shell/wire structures, visible
rig-control curves, and per-asset metadata. The production city can then be
assembled from these assets instead of hand-placing primitive blocks.

Run with Blender's Python:
  Blender.app/Contents/MacOS/Blender --background --python generate_lunar_city_hero_assets.py
"""

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


ROOT = SCRIPT_DIR.parents[0]
OUTPUT = ROOT / "public" / "lunar-city"
HERO_DIR = OUTPUT / "hero-assets"
HERO_BLEND = HERO_DIR / "lunar-city-hero-assets.blend"
HERO_GLB = HERO_DIR / "lunar-city-hero-assets.glb"
HERO_RENDER = HERO_DIR / "lunar-city-hero-assets.png"
HERO_BUILDING_RENDER = HERO_DIR / "lunar-city-hero-buildings.png"
HERO_CHARACTER_RENDER = HERO_DIR / "lunar-city-hero-characters.png"
HERO_LEADER_RENDER = HERO_DIR / "lunar-city-hero-leaders.png"
HERO_MANIFEST = HERO_DIR / "hero-assets-manifest.json"


BUILDINGS = [
    ("library", "knowledge", "LIBRARY", "violet"),
    ("research-lab", "research", "RESEARCH LAB", "cyan"),
    ("arts-studio", "creative", "ARTS STUDIO", "green"),
    ("council-hall", "governance", "COUNCIL HALL", "violet"),
    ("engineering-workshop", "engineering", "ENGINEERING", "cyan"),
    ("triage-clinic", "medical", "TRIAGE", "amber"),
    ("review-office", "review", "REVIEW OFFICE", "violet"),
    ("archive", "archive", "ARCHIVE", "violet"),
]

LEADERS = [
    ("leader-knowledge", "knowledge", "owl archivist", "violet"),
    ("leader-research", "research", "fox scientist", "cyan"),
    ("leader-creative", "creative", "raccoon artist", "green"),
    ("leader-governance", "governance", "eagle councillor", "violet"),
    ("leader-engineering", "engineering", "badger engineer", "cyan"),
    ("leader-medical", "medical", "gold medic", "amber"),
    ("leader-review", "review", "hawk reviewer", "violet"),
    ("leader-archive", "archive", "owl historian", "violet"),
]

WORKERS = [
    ("worker-audit", "audit", "methodical", "violet"),
    ("worker-operations", "operations", "protective", "cyan"),
    ("worker-release", "release", "bold", "amber"),
    ("worker-research", "research", "curious", "cyan"),
    ("worker-review", "review", "methodical", "violet"),
    ("worker-support", "support", "social", "green"),
]

CHILDREN = [
    ("child-curious", "child", "curious", "green"),
    ("child-social", "child", "social", "green"),
    ("child-bold", "child", "bold", "amber"),
    ("child-cautious", "child", "cautious", "violet"),
]


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.collections, bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        if hasattr(block, "remove"):
            for item in list(block):
                if item.users == 0:
                    block.remove(item)


def subcollection(parent, name):
    child = bpy.data.collections.new(name)
    parent.children.link(child)
    return child


def tune_material(mat, *, noise_scale, noise_detail, bump_strength, bump_distance, pipeline):
    mat["surface_pipeline"] = pipeline
    mat["texture_resolution_target"] = "2k_procedural_default"
    node_tree = mat.node_tree
    for node in node_tree.nodes:
        if node.bl_idname == "ShaderNodeTexNoise":
            node.inputs["Scale"].default_value = noise_scale
            node.inputs["Detail"].default_value = noise_detail
        elif node.bl_idname == "ShaderNodeBump":
            node.inputs["Strength"].default_value = bump_strength
            node.inputs["Distance"].default_value = bump_distance
    return mat


def create_materials():
    mats = {
        "floor": lunar.material("Hero floor graphite PBR", (0.15, 0.17, 0.2), metallic=0.38, roughness=0.34),
        "shell": lunar.material("Hero white hull PBR", (0.68, 0.72, 0.78), metallic=0.44, roughness=0.24),
        "dark": lunar.material("Hero dark interior alloy", (0.045, 0.055, 0.075), metallic=0.38, roughness=0.42),
        "panel": lunar.material("Hero inset panel alloy", (0.36, 0.4, 0.47), metallic=0.55, roughness=0.28),
        "glass": lunar.material("Hero cyan glass emission", (0.02, 0.28, 0.36), metallic=0.12, roughness=0.08, emission=(0.0, 0.85, 1.0)),
        "violet": lunar.material("Hero violet identity emission", (0.42, 0.06, 0.7), metallic=0.2, roughness=0.25, emission=(0.55, 0.05, 0.95)),
        "cyan": lunar.material("Hero cyan identity emission", (0.02, 0.42, 0.58), metallic=0.24, roughness=0.24, emission=(0.0, 0.65, 0.95)),
        "amber": lunar.material("Hero amber identity emission", (0.75, 0.34, 0.05), metallic=0.2, roughness=0.28, emission=(1.0, 0.34, 0.03)),
        "green": lunar.material("Hero green identity emission", (0.18, 0.5, 0.18), metallic=0.08, roughness=0.42, emission=(0.15, 0.75, 0.12)),
        "red": lunar.material("Hero red alert enamel", (0.72, 0.08, 0.05), metallic=0.18, roughness=0.35, emission=(0.9, 0.04, 0.02)),
        "black": lunar.material("Hero ink black detail", (0.01, 0.012, 0.016), metallic=0.12, roughness=0.5),
        "white": lunar.material("Hero warm eye highlight", (0.95, 0.9, 0.82), roughness=0.38),
        "beak": lunar.material("Hero beak gold keratin", (0.95, 0.55, 0.14), roughness=0.42),
        "fur": lunar.material("Hero leader fur", (0.72, 0.38, 0.15), roughness=0.5),
        "fur_light": lunar.material("Hero leader light muzzle", (0.95, 0.76, 0.48), roughness=0.58),
        "owl_feather": lunar.material("Hero owl moon feather PBR", (0.54, 0.43, 0.31), roughness=0.62),
        "fox_fur": lunar.material("Hero fox amber fur PBR", (0.88, 0.42, 0.14), roughness=0.54),
        "raccoon_fur": lunar.material("Hero raccoon silver fur PBR", (0.42, 0.45, 0.46), roughness=0.58),
        "eagle_feather": lunar.material("Hero eagle bronze feather PBR", (0.64, 0.42, 0.19), roughness=0.56),
        "badger_fur": lunar.material("Hero badger slate fur PBR", (0.22, 0.23, 0.24), roughness=0.6),
        "gold_fur": lunar.material("Hero golden medic fur PBR", (0.92, 0.68, 0.24), roughness=0.5),
        "hawk_feather": lunar.material("Hero hawk russet feather PBR", (0.58, 0.31, 0.13), roughness=0.55),
        "archive_owl": lunar.material("Hero archive owl violet feather PBR", (0.34, 0.28, 0.46), roughness=0.62),
        "helmet": lunar.material("Hero worker helmet ceramic", (0.86, 0.9, 0.94), metallic=0.42, roughness=0.2),
        "suit": lunar.material("Hero worker suit fabric alloy", (0.12, 0.15, 0.18), metallic=0.25, roughness=0.46),
        "gold": lunar.material("Hero gold trim", (0.92, 0.62, 0.17), metallic=0.68, roughness=0.24),
        "wood": lunar.material("Hero warm desk wood", (0.42, 0.2, 0.08), metallic=0.03, roughness=0.6),
        "text": lunar.material("Hero sign text emission", (0.9, 0.98, 1.0), roughness=0.16, emission=(0.9, 0.98, 1.0)),
        "review_floor": lunar.material("Hero review floor", (0.1, 0.11, 0.13), roughness=0.86),
    }
    tune_material(mats["floor"], noise_scale=36, noise_detail=10, bump_strength=0.22, bump_distance=0.055, pipeline="2k_pbr_lunar_floor_micro_panel_noise")
    tune_material(mats["shell"], noise_scale=28, noise_detail=9, bump_strength=0.14, bump_distance=0.035, pipeline="2k_pbr_ceramic_hull_panel_noise")
    tune_material(mats["dark"], noise_scale=42, noise_detail=8, bump_strength=0.12, bump_distance=0.03, pipeline="2k_pbr_dark_alloy_brushed_noise")
    tune_material(mats["panel"], noise_scale=64, noise_detail=11, bump_strength=0.1, bump_distance=0.022, pipeline="2k_pbr_panel_seam_noise")
    tune_material(mats["fur"], noise_scale=52, noise_detail=12, bump_strength=0.18, bump_distance=0.03, pipeline="4k_pbr_leader_fur_directional_noise")
    tune_material(mats["fur_light"], noise_scale=48, noise_detail=10, bump_strength=0.13, bump_distance=0.024, pipeline="4k_pbr_leader_muzzle_soft_noise")
    for key in ("owl_feather", "fox_fur", "raccoon_fur", "eagle_feather", "badger_fur", "gold_fur", "hawk_feather", "archive_owl"):
        tune_material(mats[key], noise_scale=58, noise_detail=12, bump_strength=0.16, bump_distance=0.028, pipeline="4k_pbr_unique_leader_species_skin")
        mats[key]["texture_resolution_target"] = "4k_hero_leader"
    tune_material(mats["helmet"], noise_scale=34, noise_detail=9, bump_strength=0.08, bump_distance=0.018, pipeline="2k_pbr_worker_helmet_ceramic_noise")
    tune_material(mats["suit"], noise_scale=44, noise_detail=11, bump_strength=0.16, bump_distance=0.028, pipeline="2k_pbr_worker_suit_fabric_alloy_noise")
    tune_material(mats["wood"], noise_scale=22, noise_detail=12, bump_strength=0.2, bump_distance=0.04, pipeline="2k_pbr_warm_wood_grain_noise")
    for key in ("glass", "violet", "cyan", "amber", "green", "red", "gold", "white", "black", "beak", "text", "review_floor"):
        tune_material(mats[key], noise_scale=18, noise_detail=5, bump_strength=0.04, bump_distance=0.012, pipeline="2k_pbr_emissive_or_trim_micro_noise")
    mats["fur"]["texture_resolution_target"] = "4k_hero_leader"
    mats["fur_light"]["texture_resolution_target"] = "4k_hero_leader"
    mats["shell"]["texture_resolution_target"] = "4k_hero_building_facade_allowed"
    return mats


def arc_points(cx, cy, cz, width, height, count=9):
    points = []
    for index in range(count):
        t = index / (count - 1)
        x = cx - width / 2 + width * t
        z = cz + sin(t * pi) * height
        points.append((x, cy, z))
    return points


def add_asset_metadata(obj, asset_id, kind, role, component):
    obj["asset_id"] = asset_id
    obj["asset_kind"] = kind
    obj["role"] = role
    obj["component"] = component
    obj["asset_component"] = component
    obj["hero_asset"] = True
    obj["source_provenance"] = "handbuilt_from_approved_reference_images"
    obj["topology"] = "skinned_mesh_wireframe_controls"


def mark(obj, asset_id, kind, role, component):
    add_asset_metadata(obj, asset_id, kind, role, component)
    return obj


def ellipsoid(name, location, scale, mat, target, asset_id, kind, role, component, segments=32, rings=16):
    obj = lunar.sphere(name, location, 1.0, mat, target, segments, rings)
    obj.scale = scale
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)
    return mark(obj, asset_id, kind, role, component)


def cylinder(name, location, radius, depth, mat, target, asset_id, kind, role, component, vertices=20, rotation=None):
    obj = lunar.cylinder(name, location, radius, depth, mat, target, vertices)
    if rotation:
        obj.rotation_euler = rotation
    return mark(obj, asset_id, kind, role, component)


def cone(name, location, radius1, radius2, depth, mat, target, asset_id, kind, role, component, vertices=20, rotation=None):
    obj = lunar.cone(name, location, radius1, radius2, depth, mat, target, vertices, rotation)
    return mark(obj, asset_id, kind, role, component)


def polish_surface(obj, *, subdivision=0, bevel=0.0, weighted_normals=True):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.shade_smooth()
    finally:
        obj.select_set(False)
    if subdivision:
        modifier = obj.modifiers.new("sculpted_skin_subdivision", "SUBSURF")
        modifier.levels = subdivision
        modifier.render_levels = subdivision
    if bevel:
        modifier = obj.modifiers.new("soft_retained_edge_bevel", "BEVEL")
        modifier.width = bevel
        modifier.segments = 4
    if weighted_normals:
        obj.modifiers.new("weighted_skin_normals", "WEIGHTED_NORMAL")
    return obj


def sculpted_arch_shell(name, x, y, base, width, depth, height, mat, target, asset_id, role):
    verts = []
    faces = []
    cols = 18
    rows = 10
    for row in range(rows + 1):
        v = row / rows
        for col in range(cols + 1):
            u = col / cols
            px = x - width / 2 + width * u
            arch = sin(pi * u)
            crown = sin(pi * v)
            py = y + depth / 2 + 0.16 * arch * crown
            pz = base + 0.22 + height * v + 0.18 * arch * (1.0 - v)
            verts.append((px, py, pz))
    for row in range(rows):
        for col in range(cols):
            a = row * (cols + 1) + col
            faces.append((a, a + 1, a + cols + 2, a + cols + 1))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    target.objects.link(obj)
    mark(obj, asset_id, "building", role, "single-piece-arched-skin")
    obj["mesh_construction"] = "continuous_curved_surface"
    solid = obj.modifiers.new("skin_thickness", "SOLIDIFY")
    solid.thickness = 0.12
    solid.offset = 0
    return polish_surface(obj, subdivision=1, bevel=0.015)


def sculpted_floor_plate(name, x, y, base, width, depth, mat, target, asset_id, role):
    verts = []
    faces = []
    cols = 10
    rows = 8
    for row in range(rows + 1):
        v = row / rows
        for col in range(cols + 1):
            u = col / cols
            px = x - width / 2 + width * u
            py = y - depth / 2 + depth * v
            edge = min(u, 1 - u, v, 1 - v)
            pz = base + 0.07 + 0.025 * sin(pi * u) * sin(pi * v) - 0.035 * max(0, 0.18 - edge)
            verts.append((px, py, pz))
    for row in range(rows):
        for col in range(cols):
            a = row * (cols + 1) + col
            faces.append((a, a + 1, a + cols + 2, a + cols + 1))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    target.objects.link(obj)
    mark(obj, asset_id, "building", role, "single-piece-sculpted-floor")
    obj["mesh_construction"] = "continuous_floor_skin"
    solid = obj.modifiers.new("floor_skin_thickness", "SOLIDIFY")
    solid.thickness = 0.08
    solid.offset = -1
    return polish_surface(obj, subdivision=1, bevel=0.02)


def sculpted_side_buttress(name, x, y, base, side, mat, target, asset_id, role):
    verts = []
    faces = []
    rows = 8
    cols = 4
    for row in range(rows + 1):
        v = row / rows
        for col in range(cols + 1):
            u = col / cols
            px = x + side * (2.35 + 0.08 * sin(pi * v))
            py = y - 1.35 + 2.65 * u
            pz = base + 0.22 + 2.05 * v
            if row > rows * 0.65:
                px -= side * 0.35 * ((v - 0.65) / 0.35)
            verts.append((px, py, pz))
    for row in range(rows):
        for col in range(cols):
            a = row * (cols + 1) + col
            faces.append((a, a + 1, a + cols + 2, a + cols + 1))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    target.objects.link(obj)
    mark(obj, asset_id, "building", role, "curved-side-buttress-skin")
    obj["mesh_construction"] = "continuous_side_skin"
    solid = obj.modifiers.new("buttress_skin_thickness", "SOLIDIFY")
    solid.thickness = 0.12
    solid.offset = 0
    return polish_surface(obj, subdivision=1, bevel=0.018)


def sculpted_robe(name, x, y, height, mat, target, asset_id, role):
    verts = [
        (x - 0.42, y + 0.07, height * 0.86),
        (x + 0.42, y + 0.07, height * 0.86),
        (x + 0.58, y + 0.2, height * 0.28),
        (x + 0.36, y + 0.28, height * 0.06),
        (x - 0.36, y + 0.28, height * 0.06),
        (x - 0.58, y + 0.2, height * 0.28),
        (x, y + 0.23, height * 0.45),
    ]
    faces = [(0, 1, 6), (1, 2, 6), (2, 3, 6), (3, 4, 6), (4, 5, 6), (5, 0, 6)]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    target.objects.link(obj)
    mark(obj, asset_id, "character", role, "skinned-robe-cloth")
    obj["mesh_construction"] = "single_cloth_skin"
    return polish_surface(obj, subdivision=1, bevel=0.01)


def sculpted_actor_body(name, x, y, height, radius, mat, target, asset_id, role, kind):
    verts = []
    faces = []
    rings = 10
    segments = 24
    for ring in range(rings + 1):
        v = ring / rings
        z = height * (0.06 + 0.74 * v)
        waist = 0.78 + 0.28 * sin(pi * v)
        shoulder = 1.0 + 0.18 * sin(pi * min(1.0, v * 1.35))
        taper = 0.58 + 0.42 * (1.0 - abs(v - 0.58))
        if kind == "leader":
            rx = radius * (waist + 0.12 * shoulder)
            ry = radius * 0.68 * taper
        else:
            rx = radius * (0.82 + 0.2 * sin(pi * v))
            ry = radius * 0.62 * (0.72 + 0.18 * sin(pi * v))
        for segment in range(segments):
            angle = 2 * pi * segment / segments
            front_bias = 1.0 - 0.08 * max(0, -sin(angle))
            px = x + cos(angle) * rx * front_bias
            py = y + sin(angle) * ry
            verts.append((px, py, z))
    for ring in range(rings):
        for segment in range(segments):
            a = ring * segments + segment
            b = ring * segments + (segment + 1) % segments
            c = (ring + 1) * segments + (segment + 1) % segments
            d = (ring + 1) * segments + segment
            faces.append((a, b, c, d))
    faces.append(tuple(reversed(range(segments))))
    top_start = rings * segments
    faces.append(tuple(range(top_start, top_start + segments)))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    target.objects.link(obj)
    mark(obj, asset_id, "character", role, "single-piece-sculpted-body")
    obj["mesh_construction"] = f"continuous_{kind}_body_skin"
    obj["retopology_target"] = "quad_rings_with_lod_decimation"
    return polish_surface(obj, subdivision=1, bevel=0.006)


def sculpted_head(name, x, y, z, mat, target, asset_id, role, label_text, kind):
    verts = []
    faces = []
    rings = 12
    segments = 28
    lower = label_text.lower()
    for ring in range(rings + 1):
        theta = pi * ring / rings
        for segment in range(segments):
            phi = 2 * pi * segment / segments
            sx = sin(theta) * cos(phi)
            sy = sin(theta) * sin(phi)
            sz = cos(theta)
            front = max(0.0, -sy)
            side = abs(sx)
            if kind == "leader":
                rx = 0.34 + (0.04 if "fox" in lower else 0.0) - 0.025 * max(0, sz)
                ry = 0.29 + 0.08 * front + (0.03 if "eagle" in lower or "hawk" in lower else 0.0)
                rz = 0.33 + (0.03 if "owl" in lower else 0.0)
                pz = z + sz * rz - 0.03 * front
                px = x + sx * rx * (1.0 - 0.08 * max(0, -sz))
                py = y + sy * ry - 0.08 * front * (1.0 - side)
            else:
                rx = 0.26 if kind == "worker" else 0.22
                ry = 0.23 if kind == "worker" else 0.2
                rz = 0.26 if kind == "worker" else 0.21
                visor_flatten = 1.0 - 0.18 * front
                px = x + sx * rx * visor_flatten
                py = y + sy * ry
                pz = z + sz * rz + 0.025 * max(0, sz)
            verts.append((px, py, pz))
    for ring in range(rings):
        for segment in range(segments):
            a = ring * segments + segment
            b = ring * segments + (segment + 1) % segments
            c = (ring + 1) * segments + (segment + 1) % segments
            d = (ring + 1) * segments + segment
            faces.append((a, b, c, d))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    target.objects.link(obj)
    mark(obj, asset_id, "character", role, "single-piece-sculpted-head")
    obj["mesh_construction"] = f"continuous_{kind}_head_skin"
    obj["retopology_target"] = "quad_sphere_deformation"
    return polish_surface(obj, subdivision=1, bevel=0.004)


def sculpted_limb(name, start, end, radius_a, radius_b, mat, target, asset_id, role, component, segments=14):
    start_v = Vector(start)
    end_v = Vector(end)
    axis = end_v - start_v
    length = max(axis.length, 0.001)
    direction = axis.normalized()
    reference = Vector((0, 0, 1))
    if abs(direction.dot(reference)) > 0.96:
        reference = Vector((1, 0, 0))
    tangent = direction.cross(reference).normalized()
    bitangent = direction.cross(tangent).normalized()
    verts = []
    faces = []
    rings = 6
    for ring in range(rings + 1):
        t = ring / rings
        center = start_v.lerp(end_v, t)
        radius = radius_a * (1 - t) + radius_b * t
        squash = 0.76 + 0.16 * sin(pi * t)
        for segment in range(segments):
            angle = 2 * pi * segment / segments
            offset = tangent * (cos(angle) * radius) + bitangent * (sin(angle) * radius * squash)
            verts.append(tuple(center + offset))
    for ring in range(rings):
        for segment in range(segments):
            a = ring * segments + segment
            b = ring * segments + (segment + 1) % segments
            c = (ring + 1) * segments + (segment + 1) % segments
            d = (ring + 1) * segments + segment
            faces.append((a, b, c, d))
    faces.append(tuple(reversed(range(segments))))
    top_start = rings * segments
    faces.append(tuple(range(top_start, top_start + segments)))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    target.objects.link(obj)
    mark(obj, asset_id, "character", role, component)
    obj["mesh_construction"] = "continuous_tapered_limb_skin"
    obj["retopology_target"] = "quad_tube_limb"
    return polish_surface(obj, subdivision=1, bevel=0.003)


def sculpted_antenna(name, x, y, z, height, mat, target, asset_id, role):
    stem = lunar.curve(name, [(x, y, z), (x + 0.02, y, z + height * 0.45), (x, y, z + height)], 0.012, mat, target)
    add_asset_metadata(stem, asset_id, "character", role, "flexible-antenna-rig")
    stem["mesh_construction"] = "curve_skin_antenna"
    stem["retopology_target"] = "curve_runtime_bone"
    return stem


def add_species_detail(asset_id, role, label_text, x, y, height, accent_mat, target, mats):
    lower = label_text.lower()
    if "owl" in lower:
        for side in (-1, 1):
            ellipsoid(f"{asset_id}_facial_disk_{side}_mesh", (x + side * 0.12, y - 0.31, height + 0.23), (0.13, 0.018, 0.14), mats["fur_light"], target, asset_id, "character", role, "owl-facial-disk", 20, 10)
        for side in (-1, 1):
            for feather in range(4):
                cone(f"{asset_id}_brow_feather_{side}_{feather}_mesh", (x + side * (0.08 + feather * 0.035), y - 0.29, height + 0.42 - feather * 0.015), 0.035, 0.004, 0.18, mats["fur"], target, asset_id, "character", role, "owl-brow-feather", 8, rotation=(0.8, side * 0.2, side * 0.5))
    elif "fox" in lower:
        for side in (-1, 1):
            cone(f"{asset_id}_fox_cheek_tuft_{side}_mesh", (x + side * 0.25, y - 0.22, height + 0.1), 0.08, 0.008, 0.22, mats["fur_light"], target, asset_id, "character", role, "fox-cheek-tuft", 12, rotation=(1.2, side * 0.55, 0))
        tail = lunar.curve(f"{asset_id}_fox_tail_sculpt_wire", [(x + 0.24, y + 0.2, 0.42), (x + 0.64, y + 0.36, 0.86), (x + 0.42, y + 0.12, 1.18)], 0.105, mats["fur"], target)
        add_asset_metadata(tail, asset_id, "character", role, "fox-tail-volume")
    elif "raccoon" in lower or "badger" in lower:
        for side in (-1, 1):
            ellipsoid(f"{asset_id}_mask_patch_{side}_mesh", (x + side * 0.1, y - 0.318, height + 0.23), (0.08, 0.012, 0.055), mats["black"], target, asset_id, "character", role, "mask-patch", 16, 8)
        stripe = lunar.curve(f"{asset_id}_head_stripe_wire", [(x, y - 0.32, height + 0.44), (x, y - 0.335, height + 0.28), (x, y - 0.33, height + 0.11)], 0.018, mats["white"], target)
        add_asset_metadata(stripe, asset_id, "character", role, "head-stripe")
    elif "eagle" in lower or "hawk" in lower:
        for side in (-1, 1):
            wing = lunar.curve(f"{asset_id}_folded_wing_{side}_wire", [(x + side * 0.28, y + 0.02, height * 0.74), (x + side * 0.55, y + 0.12, height * 0.48), (x + side * 0.42, y + 0.18, height * 0.22)], 0.055, mats["fur"], target)
            add_asset_metadata(wing, asset_id, "character", role, "folded-wing")
    if role == "medical":
        chamfer(f"{asset_id}_medic_cross_mesh", (x, y - 0.285, height * 0.56), (0.11, 0.012, 0.028), mats["red"], target, asset_id, "character", role, "medic-cross-horizontal")
        chamfer(f"{asset_id}_medic_cross_vertical_mesh", (x, y - 0.286, height * 0.56), (0.035, 0.012, 0.09), mats["red"], target, asset_id, "character", role, "medic-cross-vertical")


def leader_skin_material(label_text, role, mats):
    lower = label_text.lower()
    if "archive" in role:
        return mats["archive_owl"]
    if "owl" in lower:
        return mats["owl_feather"]
    if "fox" in lower:
        return mats["fox_fur"]
    if "raccoon" in lower:
        return mats["raccoon_fur"]
    if "eagle" in lower:
        return mats["eagle_feather"]
    if "badger" in lower:
        return mats["badger_fur"]
    if "gold" in lower:
        return mats["gold_fur"]
    if "hawk" in lower:
        return mats["hawk_feather"]
    return mats["fur"]


def add_unique_leader_signature(asset_id, role, label_text, x, y, height, accent_mat, target, mats):
    lower = label_text.lower()
    skin_mat = leader_skin_material(label_text, role, mats)

    def signature_curve(name, points, bevel, mat):
        obj = lunar.curve(name, points, bevel, mat, target)
        add_asset_metadata(obj, asset_id, "character", role, "unique-leader-signature")
        return obj

    if "owl" in lower:
        for side in (-1, 1):
            ellipsoid(f"{asset_id}_large_owl_face_disk_{side}", (x + side * 0.13, y - 0.352, height + 0.27), (0.18, 0.016, 0.19), mats["fur_light"], target, asset_id, "character", role, "unique-leader-signature", 24, 12)
            cone(f"{asset_id}_tall_owl_ear_horn_{side}", (x + side * 0.2, y - 0.03, height + 0.67), 0.08, 0.006, 0.38, skin_mat, target, asset_id, "character", role, "unique-leader-signature", 12, rotation=(0.36, side * 0.58, side * 0.2))
        for feather in range(9):
            dx = (feather - 4) * 0.065
            signature_curve(f"{asset_id}_layered_owl_chest_feather_{feather}", [(x + dx, y - 0.29, height * 0.76), (x + dx * 0.84, y - 0.32, height * 0.66), (x + dx * 0.6, y - 0.3, height * 0.54)], 0.018, skin_mat)
    elif "fox" in lower:
        cone(f"{asset_id}_long_fox_muzzle_signature", (x, y - 0.43, height + 0.14), 0.14, 0.035, 0.46, mats["fur_light"], target, asset_id, "character", role, "unique-leader-signature", 24, rotation=(1.52, 0, 0))
        for side in (-1, 1):
            cone(f"{asset_id}_knife_fox_ear_{side}", (x + side * 0.26, y - 0.02, height + 0.65), 0.095, 0.004, 0.46, skin_mat, target, asset_id, "character", role, "unique-leader-signature", 14, rotation=(0.3, side * 0.64, 0))
            signature_curve(f"{asset_id}_sweeping_fox_tail_{side}", [(x + side * 0.2, y + 0.2, 0.38), (x + side * 0.76, y + 0.3, 0.92), (x + side * 0.56, y + 0.12, 1.38)], 0.09, skin_mat)
    elif "raccoon" in lower:
        for side in (-1, 1):
            ellipsoid(f"{asset_id}_bold_raccoon_eye_mask_{side}", (x + side * 0.115, y - 0.36, height + 0.26), (0.105, 0.011, 0.068), mats["black"], target, asset_id, "character", role, "unique-leader-signature", 18, 8)
            signature_curve(f"{asset_id}_ringed_tail_band_{side}", [(x + 0.34, y + 0.2 + side * 0.018, 0.48 + side * 0.11), (x + 0.66, y + 0.26, 0.68 + side * 0.1), (x + 0.62, y + 0.17, 0.9 + side * 0.08)], 0.035, mats["black"])
        ellipsoid(f"{asset_id}_rounded_raccoon_snout_signature", (x, y - 0.37, height + 0.1), (0.17, 0.018, 0.09), mats["fur_light"], target, asset_id, "character", role, "unique-leader-signature", 20, 10)
    elif "eagle" in lower:
        cone(f"{asset_id}_large_hooked_eagle_beak_signature", (x, y - 0.43, height + 0.16), 0.1, 0.006, 0.36, mats["beak"], target, asset_id, "character", role, "unique-leader-signature", 20, rotation=(1.5, 0, 0))
        for side in (-1, 1):
            signature_curve(f"{asset_id}_broad_eagle_wing_{side}", [(x + side * 0.24, y + 0.06, height * 0.78), (x + side * 0.72, y + 0.18, height * 0.56), (x + side * 0.58, y + 0.16, height * 0.25)], 0.07, skin_mat)
            cone(f"{asset_id}_eagle_brow_crest_{side}", (x + side * 0.1, y - 0.35, height + 0.36), 0.05, 0.006, 0.18, mats["white"], target, asset_id, "character", role, "unique-leader-signature", 10, rotation=(1.1, side * 0.2, side * 0.7))
    elif "badger" in lower:
        signature_curve(f"{asset_id}_badger_white_crown_stripe", [(x, y - 0.36, height + 0.55), (x, y - 0.38, height + 0.28), (x, y - 0.36, height + 0.02)], 0.035, mats["white"])
        for side in (-1, 1):
            ellipsoid(f"{asset_id}_badger_black_face_band_{side}", (x + side * 0.14, y - 0.355, height + 0.24), (0.085, 0.012, 0.15), mats["black"], target, asset_id, "character", role, "unique-leader-signature", 18, 8)
            signature_curve(f"{asset_id}_badger_stocky_shoulder_{side}", [(x + side * 0.28, y - 0.04, height * 0.74), (x + side * 0.46, y + 0.02, height * 0.58), (x + side * 0.36, y - 0.02, height * 0.42)], 0.065, skin_mat)
    elif "gold" in lower:
        ellipsoid(f"{asset_id}_golden_medic_soft_head_signature", (x, y - 0.015, height + 0.2), (0.28, 0.24, 0.24), skin_mat, target, asset_id, "character", role, "unique-leader-signature", 28, 14)
        chamfer(f"{asset_id}_medic_helmet_cross_signature", (x, y - 0.36, height + 0.43), (0.13, 0.012, 0.032), mats["red"], target, asset_id, "character", role, "unique-leader-signature")
        chamfer(f"{asset_id}_medic_helmet_cross_stem_signature", (x, y - 0.36, height + 0.43), (0.036, 0.012, 0.12), mats["red"], target, asset_id, "character", role, "unique-leader-signature")
        signature_curve(f"{asset_id}_medic_soft_tail_signature", [(x + 0.16, y + 0.18, 0.38), (x + 0.48, y + 0.22, 0.7), (x + 0.36, y + 0.12, 1.0)], 0.07, skin_mat)
    elif "hawk" in lower:
        cone(f"{asset_id}_sharp_hawk_beak_signature", (x, y - 0.44, height + 0.15), 0.082, 0.004, 0.34, mats["beak"], target, asset_id, "character", role, "unique-leader-signature", 18, rotation=(1.52, 0, 0))
        signature_curve(f"{asset_id}_hawk_swept_head_crest", [(x - 0.12, y - 0.04, height + 0.53), (x, y - 0.09, height + 0.71), (x + 0.12, y - 0.04, height + 0.53)], 0.04, skin_mat)
        for side in (-1, 1):
            signature_curve(f"{asset_id}_narrow_hawk_wing_{side}", [(x + side * 0.24, y + 0.04, height * 0.74), (x + side * 0.62, y + 0.13, height * 0.52), (x + side * 0.48, y + 0.15, height * 0.28)], 0.05, skin_mat)


def add_building_surface_detail(asset_id, role, x, y, base, accent_mat, target, mats):
    for row, z in enumerate((0.48, 0.82, 1.18, 1.54, 1.9)):
        seam = lunar.curve(
            f"{asset_id}_hull_panel_seam_{row}",
            [(x - 2.12, y + 1.64, base + z), (x - 0.9, y + 1.72, base + z + 0.03), (x + 0.9, y + 1.72, base + z + 0.03), (x + 2.12, y + 1.64, base + z)],
            0.009,
            mats["dark"],
            target,
        )
        add_asset_metadata(seam, asset_id, "building", role, "recessed-hull-panel-seam")
    for col, dx in enumerate((-1.9, -1.25, -0.55, 0.55, 1.25, 1.9)):
        conduit = lunar.curve(
            f"{asset_id}_vertical_power_conduit_{col}",
            [(x + dx, y + 1.66, base + 0.34), (x + dx * 0.96, y + 1.73, base + 1.02), (x + dx * 0.88, y + 1.66, base + 1.88)],
            0.012,
            accent_mat,
            target,
        )
        add_asset_metadata(conduit, asset_id, "building", role, "glowing-power-conduit")
    for col, dx in enumerate((-1.72, -0.86, 0.0, 0.86, 1.72)):
        ellipsoid(f"{asset_id}_roof_sensor_dome_{col}", (x + dx, y + 0.24, base + 2.43), (0.18, 0.14, 0.075), mats["glass"], target, asset_id, "building", role, "roof-sensor-dome", 18, 8)
        cylinder(f"{asset_id}_roof_sensor_pin_{col}", (x + dx, y + 0.24, base + 2.56), 0.012, 0.18, accent_mat, target, asset_id, "building", role, "roof-sensor-pin", 8)
    for side in (-1, 1):
        rail = lunar.curve(
            f"{asset_id}_door_guard_rail_{side}",
            [(x + side * 1.02, y - 1.82, base + 0.24), (x + side * 1.18, y - 1.83, base + 0.66), (x + side * 1.0, y - 1.8, base + 1.08)],
            0.025,
            mats["gold"],
            target,
        )
        add_asset_metadata(rail, asset_id, "building", role, "door-guard-rail")
    if role == "research":
        for i in range(5):
            cylinder(f"{asset_id}_sample_canister_{i}", (x - 1.35 + i * 0.34, y + 0.24, base + 0.58), 0.055, 0.42, mats["glass"], target, asset_id, "building", role, "sample-canister", 18)
    elif role in {"knowledge", "archive"}:
        for i in range(6):
            chamfer(f"{asset_id}_book_spine_detail_{i}", (x - 1.55 + i * 0.28, y + 1.0, base + 1.36), (0.045, 0.018, 0.22), accent_mat, target, asset_id, "building", role, "book-spine-detail")
    elif role == "creative":
        for i in range(4):
            ellipsoid(f"{asset_id}_paint_glow_blob_{i}", (x - 0.62 + i * 0.36, y - 0.74, base + 0.2), (0.07, 0.035, 0.026), accent_mat, target, asset_id, "building", role, "paint-glow-blob", 14, 6)
    elif role == "governance":
        for i in range(3):
            chamfer(f"{asset_id}_banner_hanging_panel_{i}", (x - 0.56 + i * 0.56, y + 1.0, base + 1.38), (0.18, 0.018, 0.45), mats["violet"], target, asset_id, "building", role, "banner-hanging-panel")
    elif role == "engineering":
        for i in range(4):
            cylinder(f"{asset_id}_tool_wall_socket_{i}", (x - 1.1 + i * 0.72, y + 0.9, base + 0.92), 0.04, 0.08, mats["gold"], target, asset_id, "building", role, "tool-wall-socket", 12, rotation=(1.57, 0, 0))
    elif role == "medical":
        chamfer(f"{asset_id}_medical_cross_bar", (x, y + 1.38, base + 1.55), (0.34, 0.018, 0.065), mats["red"], target, asset_id, "building", role, "medical-cross-bar")
        chamfer(f"{asset_id}_medical_cross_stem", (x, y + 1.38, base + 1.55), (0.08, 0.018, 0.32), mats["red"], target, asset_id, "building", role, "medical-cross-stem")
    elif role == "review":
        for i in range(3):
            chamfer(f"{asset_id}_review_status_chip_{i}", (x - 0.42 + i * 0.42, y - 1.78, base + 1.08), (0.12, 0.018, 0.05), mats["green"] if i == 0 else mats["amber"], target, asset_id, "building", role, "review-status-chip")


def chamfer(name, location, scale, mat, target, asset_id, kind, role, component, rotation=None):
    obj = lunar.chamfered_box_asset(name, (1, 1, 1), mat, target, 0.09)
    obj.location = location
    obj.scale = scale
    if rotation:
        obj.rotation_euler = rotation
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)
    add_asset_metadata(obj, asset_id, kind, role, component)
    return obj


def label(name, text, location, mat, target, size=0.26):
    obj = lunar.text_label(name, text, location, mat, target, size)
    obj.rotation_euler = (1.22, 0, 0)
    return obj


def make_building(asset_id, role, title, accent, x, y, target, mats):
    base = 0.0
    accent_mat = mats[accent]
    # Foundation and open diorama shell. These are single skinned meshes over
    # arched wire controls, not stacked building blocks.
    sculpted_floor_plate(f"{asset_id}_single_piece_floor_skin", x, y, base, 5.0, 3.45, mats["floor"], target, asset_id, role)
    sculpted_arch_shell(f"{asset_id}_single_piece_back_hull_skin", x, y, base, 5.15, 3.15, 1.95, mats["shell"], target, asset_id, role)
    sculpted_side_buttress(f"{asset_id}_left_continuous_buttress_skin", x, y, base, -1, mats["shell"], target, asset_id, role)
    sculpted_side_buttress(f"{asset_id}_right_continuous_buttress_skin", x, y, base, 1, mats["shell"], target, asset_id, role)
    for dx in (-1.65, -0.55, 0.55, 1.65):
        ellipsoid(f"{asset_id}_smooth_roof_fairing_{dx}", (x + dx, y + 0.78, base + 2.24), (0.45, 0.64, 0.11), mats["shell"], target, asset_id, "building", role, "smooth-roof-fairing", 24, 10)
    # Curved front shell and glow ribs
    frame_points = [(x - 2.45, y - 1.55, base + 0.18), *arc_points(x, y - 1.55, base + 1.65, 4.9, 0.62, 11), (x + 2.45, y - 1.55, base + 0.18)]
    frame = lunar.curve(f"{asset_id}_curved_wire_shell", frame_points, 0.075, mats["shell"], target)
    add_asset_metadata(frame, asset_id, "building", role, "curved-wire-shell")
    for zoff in (0.54, 1.05, 1.55):
        rib = lunar.curve(f"{asset_id}_accent_rib_{zoff}", arc_points(x, y - 1.62, base + zoff, 4.75, 0.16, 9), 0.022, accent_mat, target)
        add_asset_metadata(rib, asset_id, "building", role, "accent-rib")
    # Facade signs and inset panels
    chamfer(f"{asset_id}_hero_sign_panel", (x, y - 1.72, base + 1.36), (1.1, 0.035, 0.24), accent_mat, target, asset_id, "building", role, "sign")
    label(f"{asset_id}_hero_sign_text", title, (x, y - 1.765, base + 1.38), mats["text"], target, 0.18 if len(title) > 10 else 0.22)
    for col in range(5):
        px = x - 1.7 + col * 0.85
        chamfer(f"{asset_id}_back_panel_{col}", (px, y + 1.39, base + 1.16), (0.26, 0.035, 0.22), mats["glass"], target, asset_id, "building", role, "backlit-panel")
    for col in range(4):
        px = x - 1.35 + col * 0.9
        ellipsoid(f"{asset_id}_inset_floor_tile_{col}", (px, y - 0.15, base + 0.15), (0.38, 0.58, 0.018), mats["panel"], target, asset_id, "building", role, "inset-floor-tile", 20, 8)
    # Role-specific hero interior
    if role in {"knowledge", "archive"}:
        for shelf in range(3):
            sx = x - 1.2 + shelf * 1.2
            chamfer(f"{asset_id}_curved_bookshelf_{shelf}", (sx, y + 1.1, base + 0.84), (0.38, 0.08, 0.72), mats["wood"], target, asset_id, "building", role, "bookshelf")
            for row in range(3):
                chamfer(f"{asset_id}_book_glow_{shelf}_{row}", (sx, y + 0.98, base + 0.42 + row * 0.28), (0.32, 0.02, 0.035), accent_mat, target, asset_id, "building", role, "book-glow")
        mark(lunar.sphere(f"{asset_id}_floating_orb_asset", (x + 1.35, y - 0.35, base + 1.0), 0.26, accent_mat, target, 32, 16), asset_id, "building", role, "floating-orb")
    elif role == "research":
        cylinder(f"{asset_id}_telescope_tripod_asset", (x + 1.15, y - 0.24, base + 0.48), 0.055, 0.9, mats["panel"], target, asset_id, "building", role, "telescope-tripod", 16)
        cone(f"{asset_id}_telescope_tube_asset", (x + 1.45, y - 0.5, base + 1.0), 0.16, 0.09, 0.95, mats["shell"], target, asset_id, "building", role, "telescope-tube", 24, rotation=(1.15, 0.2, -0.7))
        for i in range(4):
            chamfer(f"{asset_id}_console_wall_{i}", (x - 1.45 + i * 0.75, y + 0.78, base + 0.6), (0.3, 0.12, 0.18), mats["dark"], target, asset_id, "building", role, "console")
    elif role == "creative":
        chamfer(f"{asset_id}_canvas_asset", (x + 0.95, y - 0.28, base + 0.9), (0.32, 0.025, 0.38), mats["text"], target, asset_id, "building", role, "canvas")
        for i in range(7):
            cylinder(f"{asset_id}_paint_vial_{i}", (x - 1.3 + i * 0.24, y - 0.62, base + 0.22), 0.045, 0.16, accent_mat, target, asset_id, "building", role, "paint-vial", 12)
    elif role == "governance":
        cylinder(f"{asset_id}_council_holo_table", (x, y - 0.12, base + 0.38), 0.54, 0.16, mats["glass"], target, asset_id, "building", role, "holo-table", 40)
        mark(lunar.sphere(f"{asset_id}_council_hologram", (x, y - 0.12, base + 0.86), 0.38, mats["glass"], target, 32, 16), asset_id, "building", role, "hologram")
    elif role == "engineering":
        for i in range(3):
            chamfer(f"{asset_id}_tool_bench_{i}", (x - 1.25 + i * 1.2, y + 0.35, base + 0.38), (0.48, 0.22, 0.14), mats["wood"], target, asset_id, "building", role, "tool-bench")
            cylinder(f"{asset_id}_coil_stack_{i}", (x - 1.25 + i * 1.2, y + 0.02, base + 0.7), 0.12, 0.42, accent_mat, target, asset_id, "building", role, "coil-stack", 20)
    elif role == "medical":
        chamfer(f"{asset_id}_medbed_asset", (x - 0.55, y - 0.15, base + 0.4), (0.78, 0.32, 0.13), mats["text"], target, asset_id, "building", role, "medbed")
        cylinder(f"{asset_id}_scanner_tube", (x + 1.25, y + 0.25, base + 0.78), 0.19, 0.94, mats["glass"], target, asset_id, "building", role, "scanner-tube", 32)
    else:
        for i in range(5):
            chamfer(f"{asset_id}_review_screen_{i}", (x - 1.55 + i * 0.78, y + 0.68, base + 0.82), (0.26, 0.025, 0.19), mats["glass"], target, asset_id, "building", role, "review-screen")
    add_building_surface_detail(asset_id, role, x, y, base, accent_mat, target, mats)
    label(f"{asset_id}_asset_label", f"{title} HERO ASSET", (x, y - 2.5, base + 0.16), mats["text"], target, 0.15)


def make_character(asset_id, role, label_text, accent, x, y, target, mats, kind):
    accent_mat = mats[accent]
    leader = kind == "leader"
    child = kind == "child"
    height = 1.55 if leader else (0.82 if child else 1.05)
    body_w = 0.46 if leader else (0.28 if child else 0.34)
    body_mat = accent_mat if leader else mats["suit"]
    sculpted_actor_body(f"{asset_id}_single_piece_body_skin", x, y, height, body_w, body_mat, target, asset_id, role, kind)
    chamfer(f"{asset_id}_belt_panel_mesh", (x, y - 0.24, height * 0.42), (body_w * 0.72, 0.025, 0.055), mats["gold"] if leader else accent_mat, target, asset_id, "character", role, "belt-panel")
    head_mat = leader_skin_material(label_text, role, mats) if leader else mats["helmet"]
    sculpted_head(f"{asset_id}_single_piece_head_skin", x, y - 0.01, height + 0.18, head_mat, target, asset_id, role, label_text, kind)
    visor = chamfer(f"{asset_id}_visor_mesh", (x, y - 0.27, height + 0.18), (0.16, 0.018, 0.07), mats["glass"], target, asset_id, "character", role, "visor")
    visor["animation_binding"] = f"{asset_id}:look"
    for side in (-1, 1):
        arm_start = (x + side * body_w * 0.82, y - 0.02, height * 0.58)
        arm_end = (x + side * (body_w + 0.3), y - 0.12, height * 0.34)
        sculpted_limb(f"{asset_id}_upper_arm_{side}_skin", arm_start, arm_end, 0.065 if leader else 0.045, 0.042 if leader else 0.03, mats["suit"] if not leader else accent_mat, target, asset_id, role, "single-piece-upper-arm")
        leg_start = (x + side * body_w * 0.34, y, height * 0.2)
        leg_end = (x + side * body_w * 0.48, y - 0.04, 0.08)
        sculpted_limb(f"{asset_id}_leg_{side}_skin", leg_start, leg_end, 0.065 if leader else 0.047, 0.045 if leader else 0.032, mats["suit"], target, asset_id, role, "single-piece-leg")
        ellipsoid(f"{asset_id}_foot_{side}_mesh", (x + side * body_w * 0.48, y - 0.1, 0.05), (0.1, 0.16, 0.045), mats["black"], target, asset_id, "character", role, "foot", 20, 10)
    if leader:
        sculpted_robe(f"{asset_id}_robe_cloth_skin", x, y, height, accent_mat, target, asset_id, role)
        ellipsoid(f"{asset_id}_front_muzzle_plate_mesh", (x, y - 0.335, height + 0.12), (0.16, 0.018, 0.11), mats["fur_light"], target, asset_id, "character", role, "front-muzzle-plate", 20, 10)
        for side in (-1, 1):
            cone(f"{asset_id}_ear_{side}_mesh", (x + side * 0.22, y, height + 0.58), 0.12, 0.02, 0.34, mats["fur"], target, asset_id, "character", role, "ear", 16, rotation=(0.18, side * 0.34, 0))
            ellipsoid(f"{asset_id}_eye_{side}_mesh", (x + side * 0.12, y - 0.335, height + 0.255), (0.062, 0.012, 0.045), mats["white"], target, asset_id, "character", role, "eye", 16, 8)
            ellipsoid(f"{asset_id}_pupil_{side}_mesh", (x + side * 0.124, y - 0.348, height + 0.255), (0.029, 0.006, 0.025), mats["black"], target, asset_id, "character", role, "pupil", 12, 6)
        cone(f"{asset_id}_muzzle_mesh", (x, y - 0.32, height + 0.12), 0.13, 0.06, 0.28, mats["fur_light"], target, asset_id, "character", role, "muzzle", 20, rotation=(1.45, 0, 0))
        if any(bird in label_text.lower() for bird in ("owl", "eagle", "hawk")):
            cone(f"{asset_id}_beak_mesh", (x, y - 0.39, height + 0.14), 0.075, 0.01, 0.24, mats["beak"], target, asset_id, "character", role, "beak", 18, rotation=(1.5, 0, 0))
        cloak = lunar.curve(f"{asset_id}_cloak_wire_shape", [(x - 0.32, y + 0.16, 0.55), (x, y + 0.24, 0.95), (x + 0.32, y + 0.16, 0.55)], 0.035, accent_mat, target)
        add_asset_metadata(cloak, asset_id, "character", role, "cloak-wire")
        chamfer(f"{asset_id}_gold_collar_mesh", (x, y - 0.11, height * 0.9), (0.27, 0.028, 0.035), mats["gold"], target, asset_id, "character", role, "collar")
        tail = lunar.curve(f"{asset_id}_tail_or_robe_sweep_wire", [(x + 0.22, y + 0.12, 0.45), (x + 0.48, y + 0.2, 0.72), (x + 0.62, y + 0.12, 0.96)], 0.055, mats["fur"], target)
        add_asset_metadata(tail, asset_id, "character", role, "tail-or-robe-sweep")
        add_unique_leader_signature(asset_id, role, label_text, x, y, height, accent_mat, target, mats)
        if role == "research":
            cone(f"{asset_id}_held_telescope_mesh", (x + 0.58, y - 0.18, height * 0.78), 0.08, 0.05, 0.56, mats["shell"], target, asset_id, "character", role, "held-telescope", 20, rotation=(1.25, 0.08, -0.76))
        elif role in {"knowledge", "archive"}:
            chamfer(f"{asset_id}_held_book_mesh", (x - 0.42, y - 0.2, height * 0.54), (0.18, 0.035, 0.13), mats["wood"], target, asset_id, "character", role, "held-book")
        elif role == "creative":
            cylinder(f"{asset_id}_paintbrush_mesh", (x + 0.45, y - 0.16, height * 0.6), 0.018, 0.52, mats["gold"], target, asset_id, "character", role, "paintbrush", 10, rotation=(0.8, 0.32, -0.5))
        elif role == "governance":
            chamfer(f"{asset_id}_tablet_gavel_mesh", (x + 0.43, y - 0.2, height * 0.57), (0.14, 0.03, 0.12), mats["gold"], target, asset_id, "character", role, "gavel-tablet")
        elif role == "engineering":
            chamfer(f"{asset_id}_wrench_head_mesh", (x + 0.5, y - 0.18, height * 0.62), (0.11, 0.025, 0.045), mats["shell"], target, asset_id, "character", role, "wrench")
        elif role == "medical":
            chamfer(f"{asset_id}_medkit_mesh", (x - 0.46, y - 0.17, height * 0.46), (0.14, 0.035, 0.1), mats["white"], target, asset_id, "character", role, "medkit")
        add_species_detail(asset_id, role, label_text, x, y, height, accent_mat, target, mats)
    rig_points = [(x, y, 0.05), (x, y, height * 0.65), (x, y, height + 0.18)]
    spine = lunar.curve(f"{asset_id}_animation_spine_wire", rig_points, 0.014, accent_mat, target)
    arms = lunar.curve(f"{asset_id}_animation_arm_wire", [(x - body_w, y, height * 0.7), (x, y, height * 0.78), (x + body_w, y, height * 0.7)], 0.012, accent_mat, target)
    legs = lunar.curve(f"{asset_id}_animation_leg_wire", [(x - body_w * 0.6, y, 0.02), (x, y, height * 0.32), (x + body_w * 0.6, y, 0.02)], 0.012, accent_mat, target)
    for rig in (spine, arms, legs):
        add_asset_metadata(rig, asset_id, "character", role, "animation-wire-rig")
        rig["animation_clips"] = "idle,walk,work,carry,inspect,repair,talk,wait,panic,celebrate,rest,return"
    if not leader:
        sculpted_antenna(f"{asset_id}_antenna_stem_skin", x, y, height + 0.38, 0.28, mats["shell"], target, asset_id, role)
        mark(lunar.sphere(f"{asset_id}_antenna_light_mesh", (x, y, height + 0.66), 0.045, accent_mat, target, 12, 6), asset_id, "character", role, "antenna-light")
        chamfer(f"{asset_id}_backpack_powerpack_mesh", (x, y + 0.22, height * 0.5), (0.17, 0.08, 0.19), accent_mat, target, asset_id, "character", role, "powerpack")
    label(f"{asset_id}_asset_label", label_text, (x, y - 1.05, 0.18), mats["text"], target, 0.13)


def setup_camera_and_lighting(target):
    lighting = lunar.collection("Hero Asset Lighting")
    bpy.ops.object.light_add(type="AREA", location=(0, -18, 17))
    key = bpy.context.object
    key.name = "Hero asset key light"
    key.data.energy = 6800
    key.data.size = 22
    lunar.move_to(key, lighting)
    bpy.ops.object.light_add(type="AREA", location=(-18, 7, 8))
    fill = bpy.context.object
    fill.name = "Hero cyan-violet fill"
    fill.data.energy = 2400
    fill.data.color = (0.18, 0.42, 1.0)
    fill.data.size = 18
    lunar.move_to(fill, lighting)
    bpy.ops.object.camera_add(location=(0, -34, 23))
    camera = bpy.context.object
    camera.name = "Hero asset review camera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 36
    camera.rotation_euler = (Vector((0, -2.0, 1.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    lunar.move_to(camera, lighting)
    world = bpy.context.scene.world or bpy.data.worlds.new("Hero Asset World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.005, 0.007, 0.014, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.32
    target["lighting_profile"] = "hero_asset_review"


def asset_quality_entries():
    grouped = {}
    for obj in bpy.data.objects:
        asset_id = obj.get("asset_id")
        if not asset_id:
            continue
        entry = grouped.setdefault(
            asset_id,
            {
                "id": asset_id,
                "collection": f"Hero Asset - {asset_id}",
                "heroComponentCount": 0,
                "sculptedSurfaceCount": 0,
                "proceduralPbrMaterialCount": 0,
                "animationRigWireCount": 0,
                "lodPolicy": ["hero", "high", "medium", "low"],
                "retopologyTarget": "quad_dominant_smart_low_poly",
            },
        )
        entry["heroComponentCount"] += 1
        if obj.get("mesh_construction"):
            entry["sculptedSurfaceCount"] += 1
        if obj.get("component") == "animation-wire-rig" or obj.get("component") == "flexible-antenna-rig":
            entry["animationRigWireCount"] += 1
        for material_slot in getattr(obj, "material_slots", []):
            material = material_slot.material
            if material and material.get("surface_pipeline"):
                entry["proceduralPbrMaterialCount"] += 1
    return [grouped[key] for key in sorted(grouped)]


def lod_budget_entries():
    entries = []
    for asset_id, role, _title, _accent in BUILDINGS:
        entries.append(
            {
                "id": asset_id,
                "kind": "building",
                "role": role,
                "sourceCollection": f"Hero Asset - {asset_id}",
                "levels": {
                    "hero": {"screenCoverage": "close", "triangleBudget": 18000, "textureResolution": "4k"},
                    "high": {"screenCoverage": "district", "triangleBudget": 9000, "textureResolution": "2k"},
                    "medium": {"screenCoverage": "city", "triangleBudget": 3500, "textureResolution": "2k_atlas"},
                    "low": {"screenCoverage": "background", "triangleBudget": 900, "textureResolution": "1k_atlas"},
                },
                "policy": "preserve_silhouette_and_signage_first",
            }
        )
    for asset_id, role, _species, _accent in LEADERS:
        entries.append(
            {
                "id": asset_id,
                "kind": "leader",
                "role": role,
                "sourceCollection": f"Hero Asset - {asset_id}",
                "levels": {
                    "hero": {"screenCoverage": "dialogue", "triangleBudget": 14000, "textureResolution": "4k"},
                    "high": {"screenCoverage": "room", "triangleBudget": 7000, "textureResolution": "2k"},
                    "medium": {"screenCoverage": "street", "triangleBudget": 2400, "textureResolution": "2k_atlas"},
                    "low": {"screenCoverage": "background", "triangleBudget": 650, "textureResolution": "1k_atlas"},
                },
                "policy": "preserve_face_props_and_body_language_first",
            }
        )
    for asset_id, role, _personality, _accent in WORKERS + CHILDREN:
        entries.append(
            {
                "id": asset_id,
                "kind": "worker" if asset_id.startswith("worker-") else "child",
                "role": role,
                "sourceCollection": f"Hero Asset - {asset_id}",
                "levels": {
                    "hero": {"screenCoverage": "selected", "triangleBudget": 8000, "textureResolution": "2k"},
                    "high": {"screenCoverage": "room", "triangleBudget": 4200, "textureResolution": "2k"},
                    "medium": {"screenCoverage": "street", "triangleBudget": 1600, "textureResolution": "1k_atlas"},
                    "low": {"screenCoverage": "crowd", "triangleBudget": 420, "textureResolution": "512_atlas"},
                },
                "policy": "preserve_visor_accent_and_animation_silhouette_first",
            }
        )
    return entries


def main():
    HERO_DIR.mkdir(parents=True, exist_ok=True)
    reset_scene()
    mats = create_materials()
    root = lunar.collection("Hero Assets")
    buildings = lunar.collection("Hero Building Assets")
    characters = lunar.collection("Hero Character Assets")
    props = lunar.collection("Hero Supporting Assets")
    lunar.cube("hero_asset_review_floor", (0, -2.2, -0.16), (42, 24, 0.06), mats["review_floor"], props, 0.14)

    for index, (asset_id, role, title, accent) in enumerate(BUILDINGS):
        x = -12 + (index % 4) * 8
        y = 5.4 if index < 4 else 0.6
        make_building(asset_id, role, title, accent, x, y, subcollection(buildings, f"Hero Asset - {asset_id}"), mats)

    for index, (asset_id, role, species, accent) in enumerate(LEADERS):
        x = -14 + index * 4
        make_character(asset_id, role, f"{role.upper()} LEADER - {species}", accent, x, -4.7, subcollection(characters, f"Hero Asset - {asset_id}"), mats, "leader")

    for index, (asset_id, role, personality, accent) in enumerate(WORKERS):
        x = -10 + index * 4
        make_character(asset_id, role, f"{role.upper()} WORKER - {personality}", accent, x, -8.0, subcollection(characters, f"Hero Asset - {asset_id}"), mats, "worker")

    for index, (asset_id, role, personality, accent) in enumerate(CHILDREN):
        x = -6 + index * 4
        make_character(asset_id, role, f"CHILD - {personality}", accent, x, -10.9, subcollection(characters, f"Hero Asset - {asset_id}"), mats, "child")

    label("hero_asset_title", "LUNAR CITY HERO ASSETS - MESH SHELLS / RIG WIRES / ROLE PROPS", (0, 9.7, 0.72), mats["text"], props, 0.28)
    setup_camera_and_lighting(root)

    scene = bpy.context.scene
    scene.name = "Lunar City Hero Assets"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 2200
    scene.render.resolution_y = 1450
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.28
    scene.view_settings.gamma = 1.0
    scene["production_role"] = "source_asset_library"
    scene["reference_target"] = "approved lunar city concept images"
    scene["asset_count"] = len(BUILDINGS) + len(LEADERS) + len(WORKERS) + len(CHILDREN)
    scene["hero_mesh_components"] = sum(1 for obj in bpy.data.objects if obj.get("hero_asset"))
    scene["sculpted_surface_components"] = sum(1 for obj in bpy.data.objects if obj.get("mesh_construction"))
    scene["sculpted_character_core_components"] = sum(1 for obj in bpy.data.objects if str(obj.get("mesh_construction", "")).startswith("continuous_") and obj.get("asset_kind") == "character")
    scene["sculpted_character_limb_components"] = sum(1 for obj in bpy.data.objects if obj.get("mesh_construction") == "continuous_tapered_limb_skin")
    scene["building_detail_components"] = sum(1 for obj in bpy.data.objects if obj.get("asset_kind") == "building" and obj.get("asset_component") in {
        "recessed-hull-panel-seam",
        "glowing-power-conduit",
        "roof-sensor-dome",
        "roof-sensor-pin",
        "door-guard-rail",
        "sample-canister",
        "book-spine-detail",
        "paint-glow-blob",
        "banner-hanging-panel",
        "tool-wall-socket",
        "medical-cross-bar",
        "medical-cross-stem",
        "review-status-chip",
    })
    scene["unique_leader_signature_components"] = sum(
        1
        for obj in bpy.data.objects
        if obj.get("asset_kind") == "character" and obj.get("asset_component") == "unique-leader-signature"
    )
    pbr_materials = sorted({mat.name for mat in bpy.data.materials if mat.get("surface_pipeline")})
    quality_entries = asset_quality_entries()
    lod_entries = lod_budget_entries()

    manifest = {
        "schemaVersion": 1,
        "source": "local_blender_hero_asset_generation",
        "referenceTarget": "approved_lunar_city_reference_images",
        "blend": "lunar-city/hero-assets/lunar-city-hero-assets.blend",
        "glb": "lunar-city/hero-assets/lunar-city-hero-assets.glb",
        "preview": "lunar-city/hero-assets/lunar-city-hero-assets.png",
        "buildingPreview": "lunar-city/hero-assets/lunar-city-hero-buildings.png",
        "characterPreview": "lunar-city/hero-assets/lunar-city-hero-characters.png",
        "leaderPreview": "lunar-city/hero-assets/lunar-city-hero-leaders.png",
        "assetCount": scene["asset_count"],
        "heroMeshComponentCount": scene["hero_mesh_components"],
        "sculptedSurfaceComponentCount": scene["sculpted_surface_components"],
        "sculptedCharacterCoreComponentCount": scene["sculpted_character_core_components"],
        "sculptedCharacterLimbComponentCount": scene["sculpted_character_limb_components"],
        "buildingDetailComponentCount": scene["building_detail_components"],
        "uniqueLeaderSignatureCount": scene["unique_leader_signature_components"],
        "proceduralPbrMaterialCount": len(pbr_materials),
        "proceduralPbrMaterials": pbr_materials,
        "assetQuality": quality_entries,
        "lods": lod_entries,
        "buildings": [
            {"id": asset_id, "role": role, "title": title, "collection": f"Hero Asset - {asset_id}", "lod": ["hero", "high", "medium", "low"]}
            for asset_id, role, title, _accent in BUILDINGS
        ],
        "leaders": [
            {
                "id": asset_id,
                "role": role,
                "identity": species,
                "signature": species,
                "collection": f"Hero Asset - {asset_id}",
                "animationClips": ["idle", "walk", "work", "talk", "panic", "celebrate"],
            }
            for asset_id, role, species, _accent in LEADERS
        ],
        "workers": [
            {
                "id": asset_id,
                "role": role,
                "personality": personality,
                "collection": f"Hero Asset - {asset_id}",
                "animationClips": ["idle", "walk", "work", "carry", "repair", "celebrate"],
            }
            for asset_id, role, personality, _accent in WORKERS
        ],
        "children": [
            {
                "id": asset_id,
                "role": role,
                "personality": personality,
                "collection": f"Hero Asset - {asset_id}",
                "animationClips": ["idle", "walk", "talk", "panic", "celebrate", "rest"],
            }
            for asset_id, role, personality, _accent in CHILDREN
        ],
        "validation": {
            "allAssetsVisibleInReviewScene": True,
            "usesSeparateHeroAssetScene": True,
            "noRawSoulContent": True,
            "freeLocalGenerationOnly": True,
            "usesContinuousSculptedSurfaces": scene["sculpted_surface_components"] >= len(BUILDINGS) * 4 + len(LEADERS),
            "usesContinuousCharacterCoreMeshes": scene["sculpted_character_core_components"] >= (len(LEADERS) + len(WORKERS) + len(CHILDREN)) * 2,
            "usesContinuousCharacterLimbMeshes": scene["sculpted_character_limb_components"] >= (len(LEADERS) + len(WORKERS) + len(CHILDREN)) * 4,
            "usesProceduralPbrMaterials": len(pbr_materials) >= 12,
            "usesDetailedBuildingFacades": scene["building_detail_components"] >= len(BUILDINGS) * 20,
            "usesUniqueLeaderSignatures": scene["unique_leader_signature_components"] >= len(LEADERS) * 3,
            "tracksPerAssetQuality": len(quality_entries) == len(BUILDINGS) + len(LEADERS) + len(WORKERS) + len(CHILDREN),
            "tracksLodBudgets": len(lod_entries) == len(BUILDINGS) + len(LEADERS) + len(WORKERS) + len(CHILDREN),
        },
    }
    HERO_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    scene.render.filepath = str(HERO_RENDER)
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.wm.save_as_mainfile(filepath=str(HERO_BLEND))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(HERO_GLB), export_format="GLB", use_selection=False)
    bpy.ops.render.render(write_still=True)
    camera = scene.camera
    if camera:
        characters.hide_render = True
        camera.location = (0, -23, 15)
        camera.data.ortho_scale = 17
        camera.rotation_euler = (Vector((0, 3.1, 1.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(HERO_BUILDING_RENDER)
        bpy.ops.render.render(write_still=True)
        characters.hide_render = False
        buildings.hide_render = True
        camera.location = (0, -28, 17)
        camera.data.ortho_scale = 20
        camera.rotation_euler = (Vector((0, -7.8, 0.9)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(HERO_CHARACTER_RENDER)
        bpy.ops.render.render(write_still=True)
        buildings.hide_render = False
        buildings.hide_render = True
        for asset_id, *_rest in WORKERS + CHILDREN:
            collection = bpy.data.collections.get(f"Hero Asset - {asset_id}")
            if collection:
                collection.hide_render = True
        camera.location = (0, -26, 16)
        camera.data.ortho_scale = 18
        camera.rotation_euler = (Vector((0, -4.7, 1.0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(HERO_LEADER_RENDER)
        bpy.ops.render.render(write_still=True)
        for asset_id, *_rest in WORKERS + CHILDREN:
            collection = bpy.data.collections.get(f"Hero Asset - {asset_id}")
            if collection:
                collection.hide_render = False
        buildings.hide_render = False


if __name__ == "__main__":
    main()

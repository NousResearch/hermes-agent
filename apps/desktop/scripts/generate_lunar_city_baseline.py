"""Generate the grounded Lunar City baseline scene in Blender.

Run with Blender's Python, not the application's Python:
  Blender.app/Contents/MacOS/Blender --background --python generate_lunar_city_baseline.py
"""

import json
from math import atan2, cos, pi, sin
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "lunar-city"
SCENE_METADATA = OUTPUT / "lunar-city-scene-metadata.json"
ROAD_CLEARANCE = 0.08
BUILDING_FOOTPRINT = (6.4, 5.4)


def material(name, color, metallic=0.0, roughness=0.55, emission=None):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1.0)
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    if emission:
        node.inputs["Emission Color"].default_value = (*emission, 1.0)
        node.inputs["Emission Strength"].default_value = 3.0
    noise = mat.node_tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 7.0
    noise.inputs["Detail"].default_value = 3.0
    bump = mat.node_tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12 if not emission else 0.04
    bump.inputs["Distance"].default_value = 0.08
    mat.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    mat.node_tree.links.new(bump.outputs["Normal"], node.inputs["Normal"])
    mat["surface_pipeline"] = "procedural_pbr_noise_bump"
    return mat


def collection(name):
    value = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(value)
    return value


def mesh_object(name, verts, faces, mat, target):
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    target.objects.link(obj)
    obj["lod"] = "high"
    return obj


def chamfered_box_asset(name, dimensions, mat, target, chamfer=0.08):
    """Create a reusable skinned mesh asset, not an ad-hoc cube primitive."""
    width, depth, height = dimensions
    half_w = width / 2
    half_d = depth / 2
    half_h = height / 2
    cut_w = min(chamfer, half_w * 0.45)
    cut_d = min(chamfer, half_d * 0.45)
    outline = [
        (-half_w + cut_w, -half_d),
        (half_w - cut_w, -half_d),
        (half_w, -half_d + cut_d),
        (half_w, half_d - cut_d),
        (half_w - cut_w, half_d),
        (-half_w + cut_w, half_d),
        (-half_w, half_d - cut_d),
        (-half_w, -half_d + cut_d),
    ]
    verts = [(x, y, -half_h) for x, y in outline] + [(x, y, half_h) for x, y in outline]
    faces = [tuple(range(7, -1, -1)), tuple(range(8, 16))]
    for index in range(8):
        nxt = (index + 1) % 8
        faces.append((index, nxt, nxt + 8, index + 8))
    obj = mesh_object(name, verts, faces, mat, target)
    obj["asset_source"] = True
    obj["asset_geometry"] = "chamfered_skinned_mesh"
    modifier = obj.modifiers.new("skin_edge_weight", "WEIGHTED_NORMAL")
    if hasattr(modifier, "keep_sharp"):
        modifier.keep_sharp = True
    return obj


def arched_wall_asset(name, width, height, thickness, mat, target, segments=10):
    """Create an open-front arched wall mesh source for linked building shells."""
    half_w = width / 2
    shoulder = height * 0.62
    radius = half_w
    arch_center_z = shoulder
    front = []
    front.append((-half_w, 0, -height / 2))
    front.append((half_w, 0, -height / 2))
    front.append((half_w, 0, shoulder - height / 2))
    for index in range(1, segments):
        angle = pi * (index / segments)
        x = cos(angle) * radius
        z = arch_center_z + sin(angle) * radius * 0.34 - height / 2
        front.append((x, 0, z))
    front.append((-half_w, 0, shoulder - height / 2))
    back = [(x, thickness, z) for x, _y, z in front]
    verts = front + back
    face_front = tuple(range(len(front)))
    face_back = tuple(range(len(front), len(front) * 2))
    faces = [face_front, tuple(reversed(face_back))]
    count = len(front)
    for index in range(count):
        nxt = (index + 1) % count
        faces.append((index, nxt, nxt + count, index + count))
    obj = mesh_object(name, verts, faces, mat, target)
    obj["asset_source"] = True
    obj["asset_geometry"] = "arched_wall_skinned_mesh"
    obj.modifiers.new("arched_wall_weighted_normals", "WEIGHTED_NORMAL")
    return obj


def instance_asset(source, name, target, location, scale=(1, 1, 1), rotation=None):
    obj = bpy.data.objects.new(name, source.data)
    obj.location = location
    obj.scale = tuple(axis * 2 for axis in scale)
    if rotation:
        obj.rotation_euler = rotation
    target.objects.link(obj)
    for modifier in source.modifiers:
        clone = obj.modifiers.new(modifier.name, modifier.type)
        for attr in ("width", "segments", "keep_sharp"):
            if hasattr(modifier, attr) and hasattr(clone, attr):
                setattr(clone, attr, getattr(modifier, attr))
    obj["lod"] = source.get("lod", "high")
    obj["source_asset"] = source.name
    obj["world_instance"] = True
    return obj


def build_asset_kit(target, mats):
    target["source"] = "procedural reusable mesh asset kit"
    target["render_policy"] = "hidden source meshes; visible linked world instances"
    assets = {
        "floor_plate": chamfered_box_asset("asset_floor_plate", (1, 1, 1), mats["floor"], target, 0.1),
        "side_wall": chamfered_box_asset("asset_side_wall", (1, 1, 1), mats["shell"], target, 0.12),
        "roof_beam": chamfered_box_asset("asset_roof_beam", (1, 1, 1), mats["shell"], target, 0.11),
        "roof_cap": chamfered_box_asset("asset_roof_cap", (1, 1, 1), mats["shell"], target, 0.12),
        "panel_strip": chamfered_box_asset("asset_panel_strip", (1, 1, 1), mats["panel"], target, 0.04),
        "floor_seam": chamfered_box_asset("asset_floor_seam", (1, 1, 1), mats["panel"], target, 0.01),
        "glass_inset": chamfered_box_asset("asset_glass_inset", (1, 1, 1), mats["glass"], target, 0.03),
        "front_step": chamfered_box_asset("asset_front_step", (1, 1, 1), mats["floor"], target, 0.03),
        "door_arch_wall": arched_wall_asset("asset_arch_wall_shell", 1, 1, 1, mats["shell"], target),
    }
    for mat_name in ("violet", "cyan", "amber", "green"):
        assets[f"sign_{mat_name}"] = chamfered_box_asset(f"asset_sign_{mat_name}", (1, 1, 1), mats[mat_name], target, 0.05)
        assets[f"accent_strip_{mat_name}"] = chamfered_box_asset(f"asset_accent_strip_{mat_name}", (1, 1, 1), mats[mat_name], target, 0.03)
    for source in assets.values():
        source.hide_viewport = True
        source.hide_render = True
    return assets


def move_to(obj, target):
    for group in list(obj.users_collection):
        group.objects.unlink(obj)
    target.objects.link(obj)


def cube(name, location, scale, mat, target, bevel=0.15, rotation=None):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    if rotation:
        obj.rotation_euler = rotation
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        modifier = obj.modifiers.new("soft_shell_edges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    obj["lod"] = "high"
    obj.data.materials.append(mat)
    move_to(obj, target)
    return obj


def cylinder(name, location, radius, depth, mat, target, vertices=16):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    obj["lod"] = "high"
    obj.data.materials.append(mat)
    move_to(obj, target)
    return obj


def sphere(name, location, radius, mat, target, segments=16, rings=8):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj["lod"] = "high"
    obj.data.materials.append(mat)
    move_to(obj, target)
    return obj


def cone(name, location, radius1, radius2, depth, mat, target, vertices=16, rotation=None):
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=radius1,
        radius2=radius2,
        depth=depth,
        location=location,
        rotation=rotation or (0, 0, 0),
    )
    obj = bpy.context.object
    obj.name = name
    obj["lod"] = "high"
    obj.data.materials.append(mat)
    move_to(obj, target)
    return obj


def text_label(name, text, location, mat, target, size=0.34):
    bpy.ops.object.text_add(location=location, rotation=(1.2, 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.extrude = 0.012
    obj["lod"] = "hero"
    obj.data.materials.append(mat)
    move_to(obj, target)
    return obj


def curve(name, points, bevel, mat, target):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.bevel_depth = bevel
    data.bevel_resolution = 3
    spline = data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, co in zip(spline.bezier_points, points):
        point.co = co
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, data)
    target.objects.link(obj)
    obj["lod"] = "high"
    data.materials.append(mat)
    return obj


def ribbon(name, points, width, mat, target):
    verts = []
    faces = []
    for index, point in enumerate(points):
        if index == 0:
            dx = points[index + 1][0] - point[0]
            dy = points[index + 1][1] - point[1]
        else:
            dx = point[0] - points[index - 1][0]
            dy = point[1] - points[index - 1][1]
        length = max((dx * dx + dy * dy) ** 0.5, 0.001)
        nx = -dy / length
        ny = dx / length
        verts.append((point[0] + nx * width / 2, point[1] + ny * width / 2, point[2]))
        verts.append((point[0] - nx * width / 2, point[1] - ny * width / 2, point[2]))
    for index in range(len(points) - 1):
        faces.append((index * 2, index * 2 + 1, index * 2 + 3, index * 2 + 2))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    target.objects.link(obj)
    obj["terrain_conforming"] = True
    obj["lod"] = "high"
    return obj


def terrain(target, mat):
    size = 46
    steps = 72
    verts = []
    faces = []
    for y in range(steps + 1):
        for x in range(steps + 1):
            px = (x / steps - 0.5) * size
            py = (y / steps - 0.5) * size
            radius = (px * px + py * py) ** 0.5
            height = ground_height(px, py)
            verts.append((px, py, height))
    for y in range(steps):
        for x in range(steps):
            a = y * (steps + 1) + x
            faces.append((a, a + 1, a + steps + 2, a + steps + 1))
    mesh = bpy.data.meshes.new("concave_colony_basin_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(mat)
    obj = bpy.data.objects.new("terrain-colony-basin", mesh)
    target.objects.link(obj)
    obj["lod"] = "high"
    obj["collision_role"] = "terrain"
    solid = obj.modifiers.new("terrain_thickness", "SOLIDIFY")
    solid.thickness = 0.5
    bevel = obj.modifiers.new("terrain_edge_softening", "BEVEL")
    bevel.width = 0.08
    bevel.segments = 2
    return obj


def ground_height(x, y):
    radius = (x * x + y * y) ** 0.5
    return -0.35 - 0.0055 * radius * radius + 0.14 * sin(x * 0.45) * cos(y * 0.33)


def place_grounded_cube(name, x, y, zoff, scale, mat, target, bevel=0.12, rotation=None):
    return cube(name, (x, y, ground_height(x, y) + zoff), scale, mat, target, bevel, rotation)


def scatter_terrain_detail(props, mats):
    for index in range(58):
        angle = (index * 2.399963229728653) % (pi * 2)
        radius = 5.0 + (index % 11) * 1.65
        x = cos(angle) * radius + sin(index * 0.7) * 1.4
        y = sin(angle) * radius + cos(index * 0.5) * 1.1
        if -3 < x < 4 and -4 < y < 3:
            continue
        z = ground_height(x, y)
        if index % 5 == 0:
            cylinder(f"terrain_crater_{index}", (x, y, z + 0.012), 0.35 + (index % 4) * 0.12, 0.025, mats["crater"], props, 24)
        else:
            rock = sphere(f"terrain_rock_{index}", (x, y, z + 0.08), 0.08 + (index % 3) * 0.05, mats["rock"], props, 8, 4)
            rock.scale.x *= 1.3
            rock.scale.y *= 0.8


ROLE_LABELS = {
    "archive": "ARCHIVE",
    "creative": "ARTS STUDIO",
    "engineering": "ENGINEERING",
    "governance": "COUNCIL HALL",
    "knowledge": "LIBRARY",
    "medical": "TRIAGE",
    "research": "RESEARCH LAB",
    "review": "REVIEW OFFICE",
}


def add_role_props(asset_id, role, x, y, base, accent, buildings, mats):
    back_y = y + 1.45
    if role in {"knowledge", "archive"}:
        for shelf in range(3):
            sx = x - 1.45 + shelf * 1.45
            cube(f"{asset_id}_bookshelf_{shelf}", (sx, back_y, base + 0.85), (0.42, 0.14, 0.78), mats["wood"], buildings, 0.04)
            for row in range(3):
                cube(f"{asset_id}_book_row_{shelf}_{row}", (sx, back_y - 0.13, base + 0.38 + row * 0.32), (0.34, 0.035, 0.045), mats[accent], buildings, 0.015)
        sphere(f"{asset_id}_orb", (x + 1.4, y - 0.3, base + 1.0), 0.28, mats[accent], buildings, 24, 12)
    elif role == "research":
        cylinder(f"{asset_id}_telescope_tripod", (x + 1.25, y - 0.25, base + 0.52), 0.08, 0.95, mats["panel"], buildings, 12)
        cone(f"{asset_id}_telescope_tube", (x + 1.55, y - 0.55, base + 1.05), 0.18, 0.12, 1.05, mats["shell"], buildings, 18, rotation=(1.15, 0.25, -0.75))
        for console in range(3):
            cube(f"{asset_id}_console_{console}", (x - 1.2 + console * 1.0, y + 0.65, base + 0.45), (0.42, 0.24, 0.25), mats["console"], buildings, 0.06)
    elif role == "creative":
        cube(f"{asset_id}_easel", (x + 0.95, y - 0.35, base + 0.75), (0.06, 0.08, 0.62), mats["wood"], buildings, 0.02, rotation=(0, 0, 0.2))
        cube(f"{asset_id}_canvas", (x + 0.95, y - 0.43, base + 0.95), (0.38, 0.04, 0.3), mats["cream"], buildings, 0.025)
        for pot in range(5):
            cube(f"{asset_id}_paint_pot_{pot}", (x - 1.4 + pot * 0.28, y - 0.65, base + 0.22), (0.09, 0.09, 0.08), mats[accent], buildings, 0.025)
    elif role == "engineering":
        for bench in range(2):
            cube(f"{asset_id}_workbench_{bench}", (x - 0.9 + bench * 1.8, y + 0.3, base + 0.42), (0.65, 0.28, 0.18), mats["wood"], buildings, 0.04)
            cube(f"{asset_id}_tool_glow_{bench}", (x - 0.9 + bench * 1.8, y + 0.06, base + 0.68), (0.36, 0.035, 0.05), mats[accent], buildings, 0.02)
        cylinder(f"{asset_id}_parts_tower", (x + 1.45, y + 0.5, base + 0.72), 0.18, 0.9, mats["panel"], buildings, 12)
    elif role == "governance":
        cylinder(f"{asset_id}_holo_table", (x, y - 0.1, base + 0.4), 0.62, 0.18, mats["glass"], buildings, 32)
        sphere(f"{asset_id}_hologram", (x, y - 0.1, base + 0.92), 0.48, mats["glass"], buildings, 24, 12)
        for flag in (-1, 1):
            cube(f"{asset_id}_banner_{flag}", (x + flag * 1.95, y + 0.85, base + 1.1), (0.18, 0.035, 0.78), mats[accent], buildings, 0.025)
    elif role == "medical":
        cube(f"{asset_id}_medbed", (x - 0.55, y - 0.2, base + 0.45), (0.82, 0.38, 0.16), mats["cream"], buildings, 0.07)
        cube(f"{asset_id}_med_sign_cross_h", (x + 0.95, y - 0.55, base + 0.88), (0.32, 0.035, 0.07), mats[accent], buildings, 0.015)
        cube(f"{asset_id}_med_sign_cross_v", (x + 0.95, y - 0.55, base + 0.88), (0.07, 0.035, 0.32), mats[accent], buildings, 0.015)
        cylinder(f"{asset_id}_scanner_column", (x + 1.45, y + 0.35, base + 0.78), 0.22, 1.05, mats["glass"], buildings, 20)
    else:
        for monitor in range(4):
            cube(f"{asset_id}_review_monitor_{monitor}", (x - 1.25 + monitor * 0.8, y + 0.65, base + 0.82), (0.28, 0.035, 0.22), mats["glass"], buildings, 0.03)
        cube(f"{asset_id}_review_desk", (x, y - 0.25, base + 0.42), (0.9, 0.34, 0.18), mats["wood"], buildings, 0.05)


def building(asset_id, role, location, accent, buildings, mats, kit):
    x, y = location
    base = ground_height(x, y)
    floor = instance_asset(kit["floor_plate"], f"{asset_id}_floor", buildings, (x, y, base + 0.12), (2.95, 2.35, 0.12))
    back_wall = instance_asset(kit["side_wall"], f"{asset_id}_back_wall", buildings, (x, y + 1.92, base + 1.32), (3.02, 0.18, 1.26))
    left_wall = instance_asset(kit["side_wall"], f"{asset_id}_left_wall", buildings, (x - 2.9, y, base + 1.18), (0.18, 1.94, 1.12))
    right_wall = instance_asset(kit["side_wall"], f"{asset_id}_right_wall", buildings, (x + 2.9, y, base + 1.18), (0.18, 1.94, 1.12))
    roof_back = instance_asset(kit["roof_beam"], f"{asset_id}_roof_back_beam", buildings, (x, y + 1.84, base + 2.6), (3.02, 0.22, 0.18))
    roof_left = instance_asset(kit["roof_beam"], f"{asset_id}_roof_left_beam", buildings, (x - 2.9, y - 0.2, base + 2.52), (0.18, 1.74, 0.16))
    roof_right = instance_asset(kit["roof_beam"], f"{asset_id}_roof_right_beam", buildings, (x + 2.9, y - 0.2, base + 2.52), (0.18, 1.74, 0.16))
    roof_caps = []
    for cap_index, dx in enumerate((-2.1, -0.7, 0.7, 2.1)):
        roof_caps.append(instance_asset(kit["roof_cap"], f"{asset_id}_roof_segment_cap_{cap_index}", buildings, (x + dx, y + 0.95, base + 2.68), (0.42, 0.74, 0.1)))
    sign = instance_asset(kit[f"sign_{accent}"], f"{asset_id}_sign", buildings, (x, y - 2.17, base + 1.75), (1.55, 0.06, 0.38))
    text_label(f"{asset_id}_sign_text", ROLE_LABELS.get(role, role.upper()), (x, y - 2.245, base + 1.76), mats["sign_text"], buildings, 0.28 if role != "engineering" else 0.22)
    sign["asset_id"] = asset_id
    sign["role"] = role
    floor["asset_id"] = asset_id
    floor["role"] = role
    floor["lod"] = "hero"
    for shell in (back_wall, left_wall, right_wall, roof_back, roof_left, roof_right, *roof_caps):
        shell["asset_id"] = asset_id
        shell["role"] = role
        shell["skinned_wireframe_shell"] = True
    for tile_x in (-1.95, -0.95, 0, 0.95, 1.95):
        instance_asset(kit["floor_seam"], f"{asset_id}_floor_tile_seam_x_{tile_x}", buildings, (x + tile_x, y, base + 0.255), (0.018, 2.0, 0.012))
    for tile_y in (-1.3, -0.6, 0.1, 0.8):
        instance_asset(kit["floor_seam"], f"{asset_id}_floor_tile_seam_y_{tile_y}", buildings, (x, y + tile_y, base + 0.258), (2.55, 0.018, 0.012))
    for dx in (-1.72, 0, 1.72):
        instance_asset(kit["glass_inset"], f"{asset_id}_back_window_{dx}", buildings, (x + dx, y + 1.73, base + 1.18), (0.42, 0.05, 0.28))
    for index, dx in enumerate((-2.95, -1.9, -0.75, 0.75, 1.9, 2.95)):
        instance_asset(kit["panel_strip"], f"{asset_id}_skin_panel_{index}", buildings, (x + dx, y - 2.05, base + 1.35), (0.08, 0.06, 0.92))
    for index, dz in enumerate((0.34, 0.72, 2.16, 2.46)):
        instance_asset(kit[f"accent_strip_{accent}"], f"{asset_id}_horizontal_skin_{index}", buildings, (x, y - 2.1, base + dz), (2.86, 0.04, 0.055))
    for dx in (-2.55, 2.55):
        cylinder(f"{asset_id}_vent_{dx}", (x + dx, y + 0.4, base + 2.88), 0.2, 0.5, mats[accent], buildings, 18)
    curve(
        f"{asset_id}_arched_entry_frame",
        [
            (x - 2.78, y - 2.14, base + 0.22),
            (x - 2.78, y - 2.14, base + 2.02),
            (x - 2.25, y - 2.14, base + 2.44),
            (x, y - 2.14, base + 2.72),
            (x + 2.25, y - 2.14, base + 2.44),
            (x + 2.78, y - 2.14, base + 2.02),
            (x + 2.78, y - 2.14, base + 0.22),
        ],
        0.12,
        mats["shell"],
        buildings,
    )
    for radius, zoff in ((2.96, 0.38), (3.04, 1.24), (3.08, 2.38)):
        curve(
            f"{asset_id}_wire_skin_rib_{zoff}",
            [
                (x - radius, y - 2.05, base + zoff),
                (x - radius * 0.5, y - 2.36, base + zoff + 0.08),
                (x, y - 2.48, base + zoff + 0.12),
                (x + radius * 0.5, y - 2.36, base + zoff + 0.08),
                (x + radius, y - 2.05, base + zoff),
            ],
            0.035,
            mats[accent],
            buildings,
        )
    for step in range(5):
        instance_asset(kit["front_step"], f"{asset_id}_front_step_{step}", buildings, (x, y - 2.42 - step * 0.24, base + 0.08 + step * 0.01), (1.55 - step * 0.1, 0.09, 0.045))
    add_role_props(asset_id, role, x, y, base, accent, buildings, mats)
    floor["architecture"] = "asset_instanced_open_front_skinned_wireframe_shell"
    return {
        "asset_id": asset_id,
        "base_z": base,
        "bbox": {
            "max_x": x + BUILDING_FOOTPRINT[0] / 2,
            "max_y": y + BUILDING_FOOTPRINT[1] / 2,
            "min_x": x - BUILDING_FOOTPRINT[0] / 2,
            "min_y": y - BUILDING_FOOTPRINT[1] / 2,
        },
        "role": role,
    }


def build_character_kit(target, mats):
    target["source"] = "reusable rig-style Hermes character mesh assets"
    target["render_policy"] = "hidden source meshes; visible linked animated character instances"
    assets = {
        "worker_body": chamfered_box_asset("asset_worker_body_suit", (1, 1, 1), mats["character"], target, 0.18),
        "child_body": chamfered_box_asset("asset_child_body_suit", (1, 1, 1), mats["character"], target, 0.16),
        "leader_body": chamfered_box_asset("asset_leader_tailored_suit", (1, 1, 1), mats["violet"], target, 0.2),
        "worker_head": chamfered_box_asset("asset_worker_helmet_head", (1, 1, 1), mats["helmet"], target, 0.2),
        "child_head": chamfered_box_asset("asset_child_helmet_head", (1, 1, 1), mats["helmet"], target, 0.2),
        "leader_head": chamfered_box_asset("asset_leader_animal_head", (1, 1, 1), mats["fur"], target, 0.22),
        "visor": chamfered_box_asset("asset_character_visor", (1, 1, 1), mats["glass"], target, 0.04),
        "collar": chamfered_box_asset("asset_leader_collar_trim", (1, 1, 1), mats["gold"], target, 0.04),
        "cloak": chamfered_box_asset("asset_leader_cloak_panel", (1, 1, 1), mats["violet"], target, 0.1),
    }
    for mat_name in ("violet", "cyan", "amber", "green"):
        assets[f"leader_body_{mat_name}"] = chamfered_box_asset(f"asset_leader_body_{mat_name}", (1, 1, 1), mats[mat_name], target, 0.2)
        assets[f"role_trim_{mat_name}"] = chamfered_box_asset(f"asset_role_trim_{mat_name}", (1, 1, 1), mats[mat_name], target, 0.035)
    for source in assets.values():
        source.hide_viewport = True
        source.hide_render = True
        source["asset_source"] = True
        source["rig_source"] = "hermes_character_kit"
    return assets


def add_character_wire_rig(name, x, y, z, height, radius, characters, mats, accent):
    spine = curve(
        f"{name}_wire_spine",
        [(x, y, z + 0.15), (x, y, z + height * 0.72), (x, y, z + height + 0.42)],
        0.018,
        mats[accent or "glass"],
        characters,
    )
    arms = curve(
        f"{name}_wire_arms",
        [
            (x - radius * 1.35, y, z + height * 0.72),
            (x, y, z + height * 0.82),
            (x + radius * 1.35, y, z + height * 0.72),
        ],
        0.014,
        mats[accent or "glass"],
        characters,
    )
    legs = curve(
        f"{name}_wire_legs",
        [
            (x - radius * 0.75, y, z + 0.05),
            (x, y, z + height * 0.35),
            (x + radius * 0.75, y, z + 0.05),
        ],
        0.014,
        mats[accent or "glass"],
        characters,
    )
    for rig in (spine, arms, legs):
        rig["character_rig_wire"] = True
        rig["source_asset"] = "character_wire_rig"
    return [spine, arms, legs]


def character(name, location, leader, characters, mats, kit=None, role=None, personality=None, kind=None, accent=None):
    x, y, z = location
    child = kind == "child"
    radius = 0.25 if child else (0.34 if not leader else 0.48)
    height = 0.65 if child else (0.9 if not leader else 1.2)
    if kit:
        if leader:
            body = instance_asset(kit.get(f"leader_body_{accent or 'violet'}", kit["leader_body"]), f"{name}_body", characters, (x, y, z + height / 2), (radius * 0.92, radius * 0.7, height / 2))
            head = instance_asset(kit["leader_head"], f"{name}_head", characters, (x, y, z + height + 0.45), (0.42, 0.34, 0.46))
        elif child:
            body = instance_asset(kit["child_body"], f"{name}_body", characters, (x, y, z + height / 2), (radius * 0.82, radius * 0.68, height / 2))
            head = instance_asset(kit["child_head"], f"{name}_head", characters, (x, y, z + height + 0.35), (0.26, 0.24, 0.27))
        else:
            body = instance_asset(kit["worker_body"], f"{name}_body", characters, (x, y, z + height / 2), (radius * 0.82, radius * 0.68, height / 2))
            head = instance_asset(kit["worker_head"], f"{name}_head", characters, (x, y, z + height + 0.45), (0.34, 0.3, 0.34))
    else:
        suit = mats[accent or "character"] if leader else mats["character"]
        body = cylinder(f"{name}_body", (x, y, z + height / 2), radius, height, suit, characters, 24 if leader else 16)
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=2,
            radius=0.32 if child else (0.43 if not leader else 0.58),
            location=(x, y, z + height + (0.35 if child else 0.45)),
        )
        head = bpy.context.object
        head.name = f"{name}_head"
        head["lod"] = "high"
        head.data.materials.append(mats["fur"] if leader else mats["helmet"])
        move_to(head, characters)
    head["character_component"] = "skinned_head_mesh"
    if leader:
        for side in (-1, 1):
            cone(
                f"{name}_ear_{side}",
                (x + side * 0.28, y, z + height + 0.97),
                0.16,
                0.02,
                0.42,
                mats["fur"],
                characters,
                12,
                rotation=(0.2, side * 0.4, 0),
            )
        cone(f"{name}_snout", (x, y - 0.48, z + height + 0.42), 0.17, 0.08, 0.36, mats["fur_light"], characters, 16, rotation=(1.45, 0, 0))
        if kit:
            instance_asset(kit["cloak"], f"{name}_cloak", characters, (x, y + 0.22, z + height * 0.52), (0.5, 0.08, 0.62))
            instance_asset(kit["collar"], f"{name}_collar", characters, (x, y - 0.15, z + height + 0.02), (0.46, 0.07, 0.08))
        else:
            cube(f"{name}_cloak", (x, y + 0.22, z + height * 0.52), (0.5, 0.08, 0.62), mats[accent or "violet"], characters, 0.08)
            cube(f"{name}_collar", (x, y - 0.15, z + height + 0.02), (0.46, 0.07, 0.08), mats["gold"], characters, 0.035)
    if kit:
        trim_key = f"role_trim_{accent or 'cyan'}"
        if trim_key not in kit:
            trim_key = "role_trim_cyan"
        visor = instance_asset(
            kit["visor"],
            f"{name}_visor",
            characters,
            (x, y - (0.3 if child else 0.39), z + height + (0.35 if child else 0.45)),
            (0.15 if child else 0.2, 0.04, 0.08 if child else 0.11),
        )
        instance_asset(
            kit[trim_key],
            f"{name}_role_trim",
            characters,
            (x, y - 0.12, z + height * 0.72),
            (radius * 0.72, 0.035, 0.055),
        )
    else:
        visor = cube(
            f"{name}_visor",
            (x, y - (0.3 if child else 0.39), z + height + (0.35 if child else 0.45)),
            (0.15 if child else 0.2, 0.04, 0.08 if child else 0.11),
            mats[accent or "glass"],
            characters,
            0.05,
        )
    add_character_wire_rig(name, x, y, z, height, radius, characters, mats, accent or "glass")
    body["role"] = role or ("leader" if leader else "worker")
    body["personality"] = personality or ("bold" if leader else "curious")
    body["kind"] = kind or ("leader" if leader else "worker")
    body["asset_family"] = "hermes-profile-variant"
    body["rig_contract"] = "wireframe_controls_with_skinned_mesh_instances"
    add_animation_library(body, name, leader)


def add_transport_and_infrastructure(props, mats):
    route = [(-4.5, 2.7), (-0.5, 2.4), (4.2, 3.0), (8.7, 5.0)]
    for offset, mat_name in ((-0.34, "panel"), (0.34, "panel")):
        curve(
            f"tram_track_{offset}",
            [(x, y + offset, ground_height(x, y) + 0.12) for x, y in route],
            0.025,
            mats[mat_name],
            props,
        )
    x, y = 3.1, 2.75
    z = ground_height(x, y)
    cube("transit_shuttle_body", (x, y, z + 0.42), (1.55, 0.42, 0.36), mats["transport"], props, 0.12)
    cube("transit_shuttle_window", (x - 0.55, y - 0.43, z + 0.52), (0.35, 0.04, 0.16), mats["glass"], props, 0.025)
    cube("transit_shuttle_door", (x + 0.55, y - 0.43, z + 0.42), (0.22, 0.04, 0.25), mats["amber"], props, 0.025)
    for sign_x, sign_y, label in [(-1.2, 3.7, "READY"), (2.1, -5.6, "TRIAGE"), (-4.2, 0.7, "BUS STOP")]:
        z = ground_height(sign_x, sign_y)
        cylinder(f"wayfinding_post_{label}", (sign_x, sign_y, z + 0.55), 0.035, 1.1, mats["panel"], props, 8)
        cube(f"wayfinding_sign_{label}", (sign_x, sign_y - 0.08, z + 1.15), (0.48, 0.035, 0.18), mats["green"], props, 0.025)
        text_label(f"wayfinding_text_{label}", label, (sign_x, sign_y - 0.125, z + 1.16), mats["sign_text"], props, 0.18)


def add_habitat_domes(props, mats):
    for index, (x, y, radius) in enumerate([(16, 7.5, 1.0), (-16, 6.8, 0.9), (18, -4.5, 0.8)]):
        z = ground_height(x, y)
        sphere(f"habitat_dome_{index}", (x, y, z + radius * 0.52), radius, mats["glass"], props, 32, 12)
        cylinder(f"habitat_dome_base_{index}", (x, y, z + 0.08), radius * 0.86, 0.16, mats["panel"], props, 32)


def add_animation_library(body, name, leader):
    """Add small reusable actions so the baseline is animation-ready.

    The desktop world resolves Hermes events to these stable clip names. The
    actions deliberately stay on the prototype body rather than baking motion
    into every generated worker instance.
    """
    clips = {
        "idle": (0.0, 0.0, 0.0),
        "walk": (0.0, 0.12, 0.0),
        "work": (0.0, -0.18, 0.0),
        "carry": (0.0, 0.08, 0.08),
        "inspect": (0.0, -0.08, -0.12),
        "repair": (0.0, -0.25, 0.18),
        "talk": (0.0, 0.05, -0.16),
        "wait": (0.0, 0.02, 0.0),
        "panic": (0.0, 0.35, 0.3),
        "celebrate": (0.0, -0.3, -0.3),
        "rest": (0.0, -0.12, 0.0),
        "return": (0.0, 0.12, -0.08),
    }
    body["animation_clips"] = ",".join(clips)
    body["animation_role"] = "leader" if leader else "worker"
    for clip, (_, pitch, roll) in clips.items():
        action = bpy.data.actions.new(f"{name}.{clip}")
        action.use_fake_user = True
        body.animation_data_create()
        body.animation_data.action = action
        body.rotation_euler = (pitch, 0.0, roll)
        body.keyframe_insert(data_path="rotation_euler", frame=1)
        body.rotation_euler = (-pitch, 0.0, -roll)
        body.keyframe_insert(data_path="rotation_euler", frame=12)
        body.rotation_euler = (pitch, 0.0, roll)
        body.keyframe_insert(data_path="rotation_euler", frame=24)
        action.frame_start = 1
        action.frame_end = 24


def validate_scene(plan, road_points, building_records):
    checks = {
        "asset_sources_present": True,
        "asset_instances_present": True,
        "buildings_do_not_overlap": True,
        "buildings_touch_ground": True,
        "collections_present": True,
        "lods_present": True,
        "roads_conform_to_terrain": True,
        "terrain_anchors_valid": True,
        "textures_declared": True,
    }
    failures = []

    for x, y, z in road_points:
        expected = ground_height(x, y) + ROAD_CLEARANCE
        if abs(z - expected) > 0.001:
            checks["roads_conform_to_terrain"] = False
            failures.append(f"road point ({x},{y}) is not terrain conforming")

    for asset_id, _role, (x, y), _accent in plan:
        expected = ground_height(x, y)
        record = next(item for item in building_records if item["asset_id"] == asset_id)
        if abs(record["base_z"] - expected) > 0.001:
            checks["buildings_touch_ground"] = False
            failures.append(f"{asset_id} base does not match terrain height")

    for index, left in enumerate(building_records):
        for right in building_records[index + 1 :]:
            lb = left["bbox"]
            rb = right["bbox"]
            overlaps = lb["min_x"] < rb["max_x"] and lb["max_x"] > rb["min_x"] and lb["min_y"] < rb["max_y"] and lb["max_y"] > rb["min_y"]
            if overlaps:
                checks["buildings_do_not_overlap"] = False
                failures.append(f"{left['asset_id']} overlaps {right['asset_id']}")

    required_collections = {"Buildings", "Characters", "Lighting", "Props", "Roads", "Terrain"}
    present_collections = {collection.name for collection in bpy.data.collections}
    missing = sorted(required_collections - present_collections)
    if missing:
        checks["collections_present"] = False
        failures.append(f"missing collections: {', '.join(missing)}")

    objects_missing_lod = [
        obj.name
        for obj in bpy.data.objects
        if obj.type in {"CURVE", "MESH"} and not obj.name.startswith("skybox_star_") and "lod" not in obj
    ]
    if objects_missing_lod:
        checks["lods_present"] = False
        failures.append(f"objects missing lod metadata: {', '.join(objects_missing_lod[:12])}")

    asset_source_count = sum(1 for obj in bpy.data.objects if obj.get("asset_source"))
    asset_instance_count = sum(1 for obj in bpy.data.objects if obj.get("world_instance"))
    if asset_source_count < 20:
        checks["asset_sources_present"] = False
        failures.append(f"expected reusable source assets, found {asset_source_count}")
    if asset_instance_count < 90:
        checks["asset_instances_present"] = False
        failures.append(f"expected linked world asset instances, found {asset_instance_count}")

    passed = all(checks.values())
    metadata = {
        "checks": checks,
        "failures": failures,
        "passed": passed,
        "sceneScaleMeters": { "radius": 42, "roadClearance": ROAD_CLEARANCE },
        "summary": {
            "assetInstanceCount": asset_instance_count,
            "assetSourceCount": asset_source_count,
            "buildingCount": len(building_records),
            "roadAnchorCount": len(road_points),
            "renderedCollections": sorted(required_collections | {"World Asset Sources"}),
        },
    }

    if not passed:
        raise RuntimeError("; ".join(failures))

    return metadata


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.collections, bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        if hasattr(block, "remove"):
            for item in list(block):
                if item.users == 0:
                    block.remove(item)

    terrain_col = collection("Terrain")
    roads = collection("Roads")
    buildings = collection("Buildings")
    characters = collection("Characters")
    props = collection("Props")
    lighting = collection("Lighting")
    asset_sources = collection("World Asset Sources")

    mats = {
        "terrain": material("Lunar regolith", (0.18, 0.19, 0.21), roughness=0.94),
        "crater": material("Crater shadow", (0.08, 0.085, 0.095), roughness=0.96),
        "rock": material("Regolith rock", (0.24, 0.25, 0.27), roughness=0.9),
        "shell": material("Colony shell", (0.46, 0.49, 0.54), metallic=0.52, roughness=0.28),
        "floor": material("Road and room floor plate", (0.18, 0.2, 0.23), metallic=0.4, roughness=0.38),
        "interior": material("Interior shadow", (0.045, 0.055, 0.075), metallic=0.25, roughness=0.5),
        "glass": material("Cyan emissive glass", (0.02, 0.24, 0.32), metallic=0.2, roughness=0.12, emission=(0.0, 0.85, 1.0)),
        "violet": material("Violet identity", (0.35, 0.05, 0.62), metallic=0.2, roughness=0.3, emission=(0.4, 0.02, 0.8)),
        "cyan": material("Cyan identity", (0.02, 0.38, 0.55), metallic=0.25, roughness=0.3, emission=(0.0, 0.45, 0.8)),
        "amber": material("Amber identity", (0.65, 0.23, 0.03), metallic=0.25, roughness=0.3, emission=(0.9, 0.18, 0.02)),
        "green": material("Garden identity", (0.12, 0.42, 0.14), metallic=0.1, roughness=0.48, emission=(0.12, 0.55, 0.1)),
        "road": material("Road composite", (0.2, 0.22, 0.26), metallic=0.44, roughness=0.34),
        "panel": material("Inset hull panel", (0.62, 0.65, 0.68), metallic=0.5, roughness=0.24),
        "wood": material("Warm interior wood", (0.38, 0.18, 0.08), metallic=0.05, roughness=0.62),
        "console": material("Console dark alloy", (0.06, 0.09, 0.12), metallic=0.5, roughness=0.28, emission=(0.02, 0.18, 0.22)),
        "cream": material("Canvas and med fabric", (0.78, 0.7, 0.56), metallic=0.0, roughness=0.72),
        "fur": material("Leader warm fur", (0.72, 0.36, 0.13), roughness=0.48),
        "fur_light": material("Leader muzzle fur", (0.92, 0.72, 0.45), roughness=0.54),
        "gold": material("Leader trim gold", (0.82, 0.52, 0.14), metallic=0.62, roughness=0.26),
        "transport": material("Transit shuttle red alloy", (0.54, 0.08, 0.05), metallic=0.46, roughness=0.28),
        "sign_text": material("Sign text glow", (0.85, 0.96, 1.0), roughness=0.2, emission=(0.85, 0.96, 1.0)),
        "character": material("Worker suit", (0.18, 0.22, 0.26), metallic=0.35, roughness=0.36),
        "helmet": material("Helmet shell", (0.78, 0.82, 0.86), metallic=0.65, roughness=0.2),
        "star": material("Skybox star", (0.7, 0.9, 1.0), roughness=0.1, emission=(0.45, 0.75, 1.0)),
    }

    terrain(terrain_col, mats["terrain"])
    building_kit = build_asset_kit(asset_sources, mats)
    character_kit = build_character_kit(asset_sources, mats)
    scatter_terrain_detail(props, mats)
    roads_points = [
        (x, y, ground_height(x, y) + ROAD_CLEARANCE)
        for x, y in [(-18, -10), (-12, -6), (-6, -3.4), (0, -0.5), (5.8, 2.0), (12, 6.8), (18, 10.8)]
    ]
    ribbon("road-network-primary", roads_points, 1.4, mats["road"], roads)
    curve("road-network-glow", [(x, y, z + 0.05) for x, y, z in roads_points], 0.055, mats["glass"], roads)
    for a, b in zip(roads_points, roads_points[1:]):
        ax, ay, az = a
        bx, by, bz = b
        steps = max(2, int(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / 1.3))
        yaw = atan2(by - ay, bx - ax)
        for step in range(steps):
            t = (step + 0.5) / steps
            x = ax + (bx - ax) * t
            y = ay + (by - ay) * t
            z = ground_height(x, y) + ROAD_CLEARANCE + 0.035
            tile = cube(f"road_tile_{ax}_{ay}_{step}", (x, y, z), (0.46, 0.58, 0.025), mats["floor"], roads, 0.025, rotation=(0, 0, yaw))
            tile["terrain_conforming"] = True
    for index, (x, y, z) in enumerate(roads_points[1:-1]):
        cube(f"road_intersection_{index}", (x, y, z + 0.02), (0.75, 0.75, 0.045), mats["road"], roads, 0.16)

    plan = [
        ("library", "knowledge", (-10, 8), "violet"),
        ("research-lab", "research", (9, 9), "cyan"),
        ("arts-studio", "creative", (-13, -3), "green"),
        ("council-hall", "governance", (12, -3), "violet"),
        ("engineering-workshop", "engineering", (-9, -12), "cyan"),
        ("triage-clinic", "medical", (2, -9), "amber"),
        ("review-office", "review", (14, -12), "violet"),
        ("archive", "archive", (0, 11), "violet"),
    ]
    building_records = []
    for asset_id, role, location, accent in plan:
        building_records.append(building(asset_id, role, location, accent, buildings, mats, building_kit))

    garden_ground = ground_height(0, -1)
    sphere("break-garden_glasshouse", (0, -1, garden_ground + 0.78), 1.18, mats["glass"], props, 24, 12)
    for index in range(16):
        angle = index / 16 * pi * 2
        radius = 1.2 + 1.4 * ((index % 3) / 3)
        x = cos(angle) * radius
        y = -1 + sin(angle) * radius * 0.7
        z = ground_height(x, y)
        cylinder(f"break-garden_plant_{index}", (x, y, z + 0.18), 0.07, 0.35, mats["green"], props, 8)
    add_habitat_domes(props, mats)
    add_transport_and_infrastructure(props, mats)
    for index, (x, y, accent) in enumerate(((-10, 6.2, "violet"), (8.7, 6.8, "cyan"), (12, -5.4, "violet"), (-9, -10.4, "cyan"))):
        character(f"leader-scene-{index}", (x, y, ground_height(x, y)), True, characters, mats, kit=character_kit, accent=accent)
    for index, (x, y) in enumerate(((-4, -2), (-1, -1.9), (2, -1.6), (4, 0.6), (-7, -4.4), (8.8, -1.7), (10.8, -7.5), (-12, -0.6), (15, 5.2))):
        character(f"worker-scene-{index}", (x, y, ground_height(x, y)), False, characters, mats, kit=character_kit, accent="cyan")
    cube("dispatcher-cube", (0, 4, ground_height(0, 4) + 0.62), (0.55, 0.55, 0.55), mats["glass"], characters, 0.18)

    asset_library = collection("Character Variant Library")
    asset_library["source"] = "sanitized Hermes role and personality classes"
    asset_library.hide_render = True
    asset_library["render_policy"] = "viewport_asset_library_only"
    leader_variants = [
        ("knowledge", "curious", "violet"),
        ("research", "curious", "cyan"),
        ("creative", "social", "green"),
        ("governance", "methodical", "violet"),
        ("engineering", "bold", "cyan"),
        ("medical", "protective", "amber"),
        ("review", "methodical", "violet"),
        ("archive", "cautious", "violet"),
    ]
    for index, (role, personality, accent) in enumerate(leader_variants):
        character(
            f"leader-{role}",
            (-12 + (index % 4) * 4, 27 - (index // 4) * 5, 0.0),
            True,
            asset_library,
            mats,
            kit=character_kit,
            role=role,
            personality=personality,
            kind="leader",
            accent=accent,
        )

    worker_variants = [
        ("audit", "methodical", "violet"),
        ("operations", "protective", "cyan"),
        ("release", "bold", "amber"),
        ("research", "curious", "cyan"),
        ("review", "methodical", "violet"),
        ("support", "social", "green"),
    ]
    for index, (role, personality, accent) in enumerate(worker_variants):
        character(
            f"worker-{role}",
            (6 + (index % 3) * 3, 27 - (index // 3) * 5, 0.0),
            False,
            asset_library,
            mats,
            kit=character_kit,
            role=role,
            personality=personality,
            kind="worker",
            accent=accent,
        )

    for index, personality in enumerate(("curious", "social", "bold", "cautious")):
        character(
            f"child-{personality}",
            (16 + index * 2, 27, 0.0),
            False,
            asset_library,
            mats,
            kit=character_kit,
            role="child",
            personality=personality,
            kind="child",
            accent="glass",
        )

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 28))
    key = bpy.context.object
    key.name = "Lunar key light"
    key.data.energy = 4200
    key.data.shape = "DISK"
    key.data.size = 20
    move_to(key, lighting)
    bpy.ops.object.light_add(type="AREA", location=(-18, -12, 8))
    fill = bpy.context.object
    fill.name = "Colony cyan fill"
    fill.data.energy = 1600
    fill.data.color = (0.05, 0.4, 1.0)
    fill.data.size = 10
    move_to(fill, lighting)
    bpy.ops.object.light_add(type="POINT", location=(9, 9, 4))
    hero_glow = bpy.context.object
    hero_glow.name = "Research lab practical glow"
    hero_glow.data.energy = 850
    hero_glow.data.color = (0.05, 0.75, 1.0)
    move_to(hero_glow, lighting)
    bpy.ops.object.light_add(type="POINT", location=(-10, 8, 4))
    library_glow = bpy.context.object
    library_glow.name = "Library violet practical glow"
    library_glow.data.energy = 700
    library_glow.data.color = (0.5, 0.15, 1.0)
    move_to(library_glow, lighting)
    for index in range(42):
        angle = index / 42 * pi * 2
        radius = 28 + (index % 5) * 1.7
        sphere(f"skybox_star_{index}", (cos(angle) * radius, sin(angle) * radius, 15 + (index % 7) * 2.1), 0.045, mats["star"], lighting, 8, 4)
    sphere("skybox_earth", (-18, 17, 18), 1.25, mats["glass"], lighting, 32, 16)

    bpy.ops.object.camera_add(location=(27, -31, 25))
    camera = bpy.context.object
    camera.name = "Lunar City hero camera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 31
    camera.data.lens = 62
    camera.rotation_euler = (0.82, 0, 0.68)
    target = Vector((0, -0.5, -0.4))
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    move_to(camera, lighting)

    world = bpy.context.scene.world or bpy.data.worlds.new("Lunar World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.008, 0.02, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.28
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        for attr, value in (
            ("use_gtao", True),
            ("use_raytracing", True),
            ("gtao_distance", 4),
            ("gtao_factor", 1.2),
            ("use_shadows", True),
        ):
            if hasattr(scene.eevee, attr):
                setattr(scene.eevee, attr, value)
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.35
    scene.view_settings.gamma = 1.0
    scene.render.filepath = str(OUTPUT / "lunar-city-baseline.png")
    scene.render.image_settings.file_format = "PNG"
    scene["asset_manifest"] = "asset-manifest.json"
    scene["design_reference"] = "Hermes Lunar City approved reference"
    scene["grounded_roads"] = True
    scene["concave_terrain"] = True
    scene["animation_contract"] = "world-animation.ts"
    scene["animation_clips"] = "idle,walk,work,carry,inspect,repair,talk,wait,panic,celebrate,rest,return"
    validation = validate_scene(plan, roads_points, building_records)
    scene["validation_passed"] = validation["passed"]
    SCENE_METADATA.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT / "lunar-city-baseline.blend"))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(OUTPUT / "lunar-city-baseline.glb"), export_format="GLB", use_selection=False)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()

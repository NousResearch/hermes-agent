"""Generate the grounded Lunar City baseline scene in Blender.

Run with Blender's Python, not the application's Python:
  Blender.app/Contents/MacOS/Blender --background --python generate_lunar_city_baseline.py
"""

from math import cos, pi, sin
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "lunar-city"


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


def move_to(obj, target):
    for group in list(obj.users_collection):
        group.objects.unlink(obj)
    target.objects.link(obj)


def cube(name, location, scale, mat, target, bevel=0.15):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        modifier = obj.modifiers.new("soft_shell_edges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    obj.data.materials.append(mat)
    move_to(obj, target)
    return obj


def cylinder(name, location, radius, depth, mat, target, vertices=16):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
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
    data.materials.append(mat)
    return obj


def terrain(target, mat):
    size = 42
    steps = 48
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
    solid = obj.modifiers.new("terrain_thickness", "SOLIDIFY")
    solid.thickness = 0.5
    bevel = obj.modifiers.new("terrain_edge_softening", "BEVEL")
    bevel.width = 0.08
    bevel.segments = 2
    return obj


def ground_height(x, y):
    radius = (x * x + y * y) ** 0.5
    return -0.35 - 0.007 * radius * radius + 0.16 * sin(x * 0.45) * cos(y * 0.33)


def building(asset_id, role, location, accent, buildings, mats):
    x, y = location
    base = ground_height(x, y)
    shell = cube(f"{asset_id}_shell", (x, y, base + 1.2), (2.9, 2.25, 1.2), mats["shell"], buildings, 0.28)
    inner = cube(f"{asset_id}_inner", (x, y - 0.18, base + 1.0), (2.45, 1.76, 1.0), mats["interior"], buildings, 0.2)
    roof = cube(f"{asset_id}_roof", (x, y, base + 2.55), (2.6, 1.95, 0.18), mats["shell"], buildings, 0.16)
    sign = cube(f"{asset_id}_sign", (x, y - 2.34, base + 1.65), (1.35, 0.06, 0.35), mats[accent], buildings, 0.06)
    sign["asset_id"] = asset_id
    sign["role"] = role
    for dx in (-1.75, 0, 1.75):
        cube(f"{asset_id}_window_{dx}", (x + dx, y - 2.31, base + 0.85), (0.42, 0.05, 0.28), mats["glass"], buildings, 0.04)
    for dx in (-2.55, 2.55):
        cylinder(f"{asset_id}_vent_{dx}", (x + dx, y, base + 2.7), 0.24, 0.35, mats[accent], buildings)
    curve(
        f"{asset_id}_arched_entry_frame",
        [
            (x - 2.72, y - 2.28, base + 0.22),
            (x - 2.72, y - 2.28, base + 2.05),
            (x - 2.25, y - 2.28, base + 2.48),
            (x, y - 2.28, base + 2.62),
            (x + 2.25, y - 2.28, base + 2.48),
            (x + 2.72, y - 2.28, base + 2.05),
            (x + 2.72, y - 2.28, base + 0.22),
        ],
        0.12,
        mats["shell"],
        buildings,
    )
    shell["architecture"] = "arched_entry_frame"
    return shell


def character(name, location, leader, characters, mats, role=None, personality=None, kind=None, accent=None):
    x, y, z = location
    child = kind == "child"
    radius = 0.25 if child else (0.34 if not leader else 0.48)
    height = 0.65 if child else (0.9 if not leader else 1.2)
    body = cylinder(f"{name}_body", (x, y, z + height / 2), radius, height, mats[accent or "character"], characters)
    head = bpy.data.objects.new(f"{name}_head", bpy.data.meshes.new(f"{name}_head_mesh"))
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=2,
        radius=0.32 if child else (0.43 if not leader else 0.58),
        location=(x, y, z + height + (0.35 if child else 0.45)),
    )
    head = bpy.context.object
    head.name = f"{name}_head"
    head.data.materials.append(mats["helmet"])
    move_to(head, characters)
    visor = cube(
        f"{name}_visor",
        (x, y - (0.3 if child else 0.39), z + height + (0.35 if child else 0.45)),
        (0.15 if child else 0.2, 0.04, 0.08 if child else 0.11),
        mats[accent or "glass"],
        characters,
        0.05,
    )
    body["role"] = role or ("leader" if leader else "worker")
    body["personality"] = personality or ("bold" if leader else "curious")
    body["kind"] = kind or ("leader" if leader else "worker")
    body["asset_family"] = "hermes-profile-variant"
    add_animation_library(body, name, leader)


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


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.collections, bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
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

    mats = {
        "terrain": material("Lunar regolith", (0.08, 0.09, 0.11), roughness=0.93),
        "shell": material("Colony shell", (0.28, 0.32, 0.36), metallic=0.55, roughness=0.32),
        "interior": material("Interior shadow", (0.025, 0.04, 0.06), metallic=0.25, roughness=0.5),
        "glass": material("Cyan emissive glass", (0.02, 0.18, 0.25), metallic=0.2, roughness=0.16, emission=(0.0, 0.8, 1.0)),
        "violet": material("Violet identity", (0.35, 0.05, 0.62), metallic=0.2, roughness=0.3, emission=(0.4, 0.02, 0.8)),
        "cyan": material("Cyan identity", (0.02, 0.38, 0.55), metallic=0.25, roughness=0.3, emission=(0.0, 0.45, 0.8)),
        "amber": material("Amber identity", (0.65, 0.23, 0.03), metallic=0.25, roughness=0.3, emission=(0.9, 0.18, 0.02)),
        "green": material("Garden identity", (0.1, 0.35, 0.12), metallic=0.1, roughness=0.55),
        "road": material("Road composite", (0.12, 0.14, 0.17), metallic=0.45, roughness=0.42),
        "character": material("Worker suit", (0.18, 0.22, 0.26), metallic=0.35, roughness=0.36),
        "helmet": material("Helmet shell", (0.78, 0.82, 0.86), metallic=0.65, roughness=0.2),
    }

    terrain(terrain_col, mats["terrain"])
    roads_points = [
        (x, y, ground_height(x, y) + 0.18)
        for x, y in [(-17, -9), (-10, -5), (-4, -3), (0, 0), (6, 2), (12, 7), (18, 11)]
    ]
    curve("road-network-primary", roads_points, 0.55, mats["road"], roads)
    curve("road-network-glow", [(x, y, z + 0.12) for x, y, z in roads_points], 0.07, mats["glass"], roads)
    for index, (x, y, z) in enumerate(roads_points[1:-1]):
        cube(f"road_intersection_{index}", (x, y, z + 0.03), (0.8, 0.8, 0.08), mats["road"], roads, 0.2)

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
    for asset_id, role, location, accent in plan:
        building(asset_id, role, location, accent, buildings, mats)

    cylinder("break_garden", (0, -1, 0.25), 3.2, 0.25, mats["green"], props, 32)
    cylinder("break_garden_glasshouse", (0, -1, 0.7), 1.2, 0.65, mats["glass"], props, 24)
    character("leader-prototype", (-2, -1, 0.5), True, characters, mats)
    for index, location in enumerate(((-1, -2, 0.4), (1, -2, 0.4), (2, 0, 0.4), (-2, 0, 0.4))):
        character(f"worker-prototype-{index}", location, False, characters, mats)
    cube("dispatcher-cube", (0, 4, 0.5), (0.55, 0.55, 0.55), mats["glass"], characters, 0.18)

    asset_library = collection("Character Asset Library")
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
            role="child",
            personality=personality,
            kind="child",
            accent="glass",
        )

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 28))
    key = bpy.context.object
    key.name = "Lunar key light"
    key.data.energy = 2400
    key.data.shape = "DISK"
    key.data.size = 20
    move_to(key, lighting)
    bpy.ops.object.light_add(type="AREA", location=(-18, -12, 8))
    fill = bpy.context.object
    fill.name = "Colony cyan fill"
    fill.data.energy = 900
    fill.data.color = (0.05, 0.4, 1.0)
    fill.data.size = 10
    move_to(fill, lighting)

    bpy.ops.object.camera_add(location=(32, -38, 34))
    camera = bpy.context.object
    camera.name = "Lunar City hero camera"
    camera.data.lens = 52
    camera.rotation_euler = (0.82, 0, 0.68)
    target = Vector((0, 0, 0))
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    move_to(camera, lighting)

    world = bpy.context.scene.world or bpy.data.worlds.new("Lunar World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.008, 0.02, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.18
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(OUTPUT / "lunar-city-baseline.png")
    scene.render.image_settings.file_format = "PNG"
    scene["asset_manifest"] = "asset-manifest.json"
    scene["design_reference"] = "Hermes Lunar City approved reference"
    scene["grounded_roads"] = True
    scene["concave_terrain"] = True
    scene["animation_contract"] = "world-animation.ts"
    scene["animation_clips"] = "idle,walk,work,carry,inspect,repair,talk,wait,panic,celebrate,rest,return"

    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT / "lunar-city-baseline.blend"))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(OUTPUT / "lunar-city-baseline.glb"), export_format="GLB", use_selection=False)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()

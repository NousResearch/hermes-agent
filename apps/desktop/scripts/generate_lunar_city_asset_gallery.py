"""Generate a Blender-only Lunar City asset review gallery.

This creates a separate .blend file for visual inspection of the reusable
building and character assets. It is intentionally not exported as the
production desktop GLB.

Run with Blender's Python:
  Blender.app/Contents/MacOS/Blender --background --python generate_lunar_city_asset_gallery.py
"""

import sys
from math import pi
from pathlib import Path

import bpy
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_lunar_city_baseline as lunar  # noqa: E402


OUTPUT = SCRIPT_DIR.parents[0] / "public" / "lunar-city"
GALLERY_BLEND = OUTPUT / "lunar-city-asset-review.blend"
GALLERY_RENDER = OUTPUT / "lunar-city-asset-review.png"


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.collections, bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        if hasattr(block, "remove"):
            for item in list(block):
                if item.users == 0:
                    block.remove(item)


def add_label(name, body, location, mat, target, size=0.34):
    label = lunar.text_label(name, body, location, mat, target, size)
    label.rotation_euler = (pi / 2.25, 0, 0)
    label["asset_review_label"] = True
    return label


def create_materials():
    return {
        "terrain": lunar.material("Review floor", (0.12, 0.13, 0.15), roughness=0.9),
        "crater": lunar.material("Crater shadow", (0.08, 0.085, 0.095), roughness=0.96),
        "rock": lunar.material("Regolith rock", (0.24, 0.25, 0.27), roughness=0.9),
        "shell": lunar.material("Colony shell", (0.52, 0.55, 0.6), metallic=0.52, roughness=0.25),
        "floor": lunar.material("Road and room floor plate", (0.18, 0.2, 0.23), metallic=0.4, roughness=0.38),
        "interior": lunar.material("Interior shadow", (0.045, 0.055, 0.075), metallic=0.25, roughness=0.5),
        "glass": lunar.material("Cyan emissive glass", (0.02, 0.24, 0.32), metallic=0.2, roughness=0.12, emission=(0.0, 0.85, 1.0)),
        "violet": lunar.material("Violet identity", (0.35, 0.05, 0.62), metallic=0.2, roughness=0.3, emission=(0.4, 0.02, 0.8)),
        "cyan": lunar.material("Cyan identity", (0.02, 0.38, 0.55), metallic=0.25, roughness=0.3, emission=(0.0, 0.45, 0.8)),
        "amber": lunar.material("Amber identity", (0.65, 0.23, 0.03), metallic=0.25, roughness=0.3, emission=(0.9, 0.18, 0.02)),
        "green": lunar.material("Garden identity", (0.12, 0.42, 0.14), metallic=0.1, roughness=0.48, emission=(0.12, 0.55, 0.1)),
        "road": lunar.material("Road composite", (0.2, 0.22, 0.26), metallic=0.44, roughness=0.34),
        "panel": lunar.material("Inset hull panel", (0.62, 0.65, 0.68), metallic=0.5, roughness=0.24),
        "wood": lunar.material("Warm interior wood", (0.38, 0.18, 0.08), metallic=0.05, roughness=0.62),
        "console": lunar.material("Console dark alloy", (0.06, 0.09, 0.12), metallic=0.5, roughness=0.28, emission=(0.02, 0.18, 0.22)),
        "cream": lunar.material("Canvas and med fabric", (0.78, 0.7, 0.56), metallic=0.0, roughness=0.72),
        "fur": lunar.material("Leader warm fur", (0.72, 0.36, 0.13), roughness=0.48),
        "fur_light": lunar.material("Leader muzzle fur", (0.92, 0.72, 0.45), roughness=0.54),
        "gold": lunar.material("Leader trim gold", (0.82, 0.52, 0.14), metallic=0.62, roughness=0.26),
        "transport": lunar.material("Transit shuttle red alloy", (0.54, 0.08, 0.05), metallic=0.46, roughness=0.28),
        "sign_text": lunar.material("Sign text glow", (0.85, 0.96, 1.0), roughness=0.2, emission=(0.85, 0.96, 1.0)),
        "character": lunar.material("Worker suit", (0.18, 0.22, 0.26), metallic=0.35, roughness=0.36),
        "helmet": lunar.material("Helmet shell", (0.78, 0.82, 0.86), metallic=0.65, roughness=0.2),
        "star": lunar.material("Skybox star", (0.7, 0.9, 1.0), roughness=0.1, emission=(0.45, 0.75, 1.0)),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reset_scene()

    source_assets = lunar.collection("Reusable Source Assets")
    buildings = lunar.collection("Building Models")
    leaders = lunar.collection("Leader Models")
    workers = lunar.collection("Worker Models")
    children = lunar.collection("Child Models")
    props = lunar.collection("Supporting Assets")
    lighting = lunar.collection("Lighting")

    mats = create_materials()
    building_kit = lunar.build_asset_kit(source_assets, mats)
    character_kit = lunar.build_character_kit(source_assets, mats)

    source_assets.hide_viewport = False
    source_assets.hide_render = True
    source_assets["note"] = "Hidden source meshes are linked into visible review models."

    lunar.cube("asset_review_floor", (0, 0, -0.16), (33, 23, 0.08), mats["terrain"], props, 0.18)

    building_plan = [
        ("library", "knowledge", (-21, 8), "violet"),
        ("research-lab", "research", (-14, 8), "cyan"),
        ("arts-studio", "creative", (-7, 8), "green"),
        ("council-hall", "governance", (0, 8), "violet"),
        ("engineering-workshop", "engineering", (7, 8), "cyan"),
        ("triage-clinic", "medical", (14, 8), "amber"),
        ("review-office", "review", (21, 8), "violet"),
        ("archive", "archive", (0, 1), "violet"),
    ]
    for asset_id, role, location, accent in building_plan:
        record = lunar.building(asset_id, role, location, accent, buildings, mats, building_kit)
        record["review_gallery"] = True
        add_label(f"{asset_id}_review_label", lunar.ROLE_LABELS[role], (location[0], location[1] - 3.15, lunar.ground_height(*location) + 0.35), mats["sign_text"], buildings, 0.22)

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
        x = -18 + index * 5.1
        y = -6.0
        lunar.character(f"leader-{role}", (x, y, 0.0), True, leaders, mats, kit=character_kit, role=role, personality=personality, kind="leader", accent=accent)
        add_label(f"leader-{role}_label", f"LEADER: {role}", (x, y - 1.5, 0.3), mats["sign_text"], leaders, 0.18)

    worker_variants = [
        ("audit", "methodical", "violet"),
        ("operations", "protective", "cyan"),
        ("release", "bold", "amber"),
        ("research", "curious", "cyan"),
        ("review", "methodical", "violet"),
        ("support", "social", "green"),
    ]
    for index, (role, personality, accent) in enumerate(worker_variants):
        x = -13 + index * 5.1
        y = -11.0
        lunar.character(f"worker-{role}", (x, y, 0.0), False, workers, mats, kit=character_kit, role=role, personality=personality, kind="worker", accent=accent)
        add_label(f"worker-{role}_label", f"WORKER: {role}", (x, y - 1.25, 0.25), mats["sign_text"], workers, 0.16)

    for index, personality in enumerate(("curious", "social", "bold", "cautious")):
        x = -7.5 + index * 5.0
        y = -15.5
        lunar.character(f"child-{personality}", (x, y, 0.0), False, children, mats, kit=character_kit, role="child", personality=personality, kind="child", accent="green")
        add_label(f"child-{personality}_label", f"CHILD: {personality}", (x, y - 1.0, 0.22), mats["sign_text"], children, 0.15)

    lunar.add_transport_and_infrastructure(props, mats)
    lunar.add_habitat_domes(props, mats)
    add_label("gallery_title", "LUNAR CITY ASSET REVIEW - BUILDINGS / LEADERS / WORKERS / CHILDREN", (0, 14.1, 0.7), mats["sign_text"], props, 0.34)

    bpy.ops.object.light_add(type="AREA", location=(0, -10, 18))
    key = bpy.context.object
    key.name = "Asset review key light"
    key.data.energy = 5500
    key.data.size = 24
    lunar.move_to(key, lighting)

    bpy.ops.object.light_add(type="AREA", location=(-18, 6, 8))
    fill = bpy.context.object
    fill.name = "Asset review cyan fill"
    fill.data.energy = 1800
    fill.data.color = (0.04, 0.46, 1.0)
    fill.data.size = 14
    lunar.move_to(fill, lighting)

    bpy.ops.object.camera_add(location=(0, -29, 21))
    camera = bpy.context.object
    camera.name = "Asset review camera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 38
    target = Vector((0, 0, 0.8))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    lunar.move_to(camera, lighting)

    world = bpy.context.scene.world or bpy.data.worlds.new("Asset Review World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.006, 0.008, 0.015, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.32

    scene = bpy.context.scene
    scene.name = "Lunar City Asset Review"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.35
    scene.view_settings.gamma = 1.0
    scene["purpose"] = "visual inspection of reusable Lunar City building and character asset models"
    scene["production_export"] = False
    scene["asset_source_count"] = sum(1 for obj in bpy.data.objects if obj.get("asset_source"))
    scene["asset_instance_count"] = sum(1 for obj in bpy.data.objects if obj.get("world_instance"))
    scene["character_wire_rigs"] = sum(1 for obj in bpy.data.objects if obj.get("character_rig_wire"))

    scene.render.filepath = str(GALLERY_RENDER)
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.wm.save_as_mainfile(filepath=str(GALLERY_BLEND))
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()

"""Import this finite generation batch into the open Blender workshop."""
from pathlib import Path
import hashlib
import json
import math
import traceback
import bpy
from mathutils import Vector

FOLDER = Path(ASSET_ROOT).resolve()
ASSETS = tuple(ASSET_IDS)
if not ASSETS or any(Path(a).name != a for a in ASSETS):
    raise ValueError('ASSET_IDS must be nonempty folder names')
WORKSHOP = FOLDER.parent / 'multiview-2026-09-07/leader-rig-workshop.blend'
STATE = FOLDER / 'blender-import-status.json'
# Re-running this script replaces only its own callback, never scene contents.
old = bpy.app.driver_namespace.get('lunar_live_import_callback')
if old and bpy.app.timers.is_registered(old):
    bpy.app.timers.unregister(old)


def import_asset(asset):
    folder = FOLDER / asset
    receipt = json.loads((folder / 'generation.json').read_text())
    glb = folder / 'shape-metres.glb'
    if hashlib.sha256(glb.read_bytes()).hexdigest() != receipt['outputSha256']:
        raise ValueError(f'{asset}: mesh does not match generation receipt')
    scene = bpy.data.scenes.new('Workshop | ' + asset)
    scene['lunar_asset'] = asset
    bpy.context.window.scene = scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = 'METERS'
    source = bpy.data.collections.new('01 Preserved generation | ' + asset)
    working = bpy.data.collections.new('02 Cleanup working copy | ' + asset)
    scene.collection.children.link(source)
    scene.collection.children.link(working)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    bpy.context.view_layer.update()
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == 'MESH']
    points = [obj.matrix_world @ Vector(c) for obj in meshes for c in obj.bound_box]
    height = max(p.z for p in points) - min(p.z for p in points)
    if abs(height - receipt['targetHeightMetres']) > .001:
        raise ValueError(f'{asset}: imported height {height} differs from receipt')
    copies = []
    for obj in imported:
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        source.objects.link(obj)
        if obj.type == 'MESH':
            copy = obj.copy()
            copy.data = obj.data.copy()
            copy.parent = None
            copy.matrix_world = obj.matrix_world.copy()
            copy.name = asset + ' | cleanup'
            working.objects.link(copy)
            copy['source_sha256'] = receipt['outputSha256']
            copy['stage'] = 'cleanup_preview_materials_and_rig_pending'
            for polygon in copy.data.polygons:
                polygon.use_smooth = True
            # A reversible light smoothing preview; the original remains intact.
            modifier = copy.modifiers.new('Gentle surface cleanup - reversible', 'SMOOTH')
            modifier.factor = .18
            modifier.iterations = 2
            copies.append(copy)
        obj.hide_render = True
        obj.hide_set(True)
    scene['verified_source_height_metres'] = height
    scene['source_sha256'] = receipt['outputSha256']
    bpy.ops.object.camera_add(location=(height * 1.4, -height * 3.5, height * 1.7))
    camera = bpy.context.object
    target = Vector((0, 0, height * .5))
    camera.rotation_euler = (target-camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = height * 1.4
    scene.camera = camera
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
    scene.render.resolution_percentage = 100
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'SINGLE'
    scene.display.shading.single_color = (.63, .65, .68)
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = 'BOTH'
    scene.display.shading.background_type = 'WORLD'
    scene.world = bpy.data.worlds.new('Workshop background | '+asset)
    scene.world.color = (.045,.045,.045)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in copies:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = copies[0]
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
    scene['lunar_import_complete'] = True
    bpy.ops.wm.save_as_mainfile(filepath=str(WORKSHOP))
    return {'asset':asset, 'scene':scene.name, 'heightMetres':height,
            'sourceFaces':sum(len(o.data.polygons) for o in meshes),
            'status':'imported_with_reversible_smoothing_preview'}


def tick():
    imported = {s.get('lunar_asset') for s in bpy.data.scenes if s.get('lunar_import_complete')}
    pending = [asset for asset in ASSETS if asset not in imported]
    try:
        if bpy.context.mode != 'OBJECT':
            return 5.0
        for asset in pending:
            if (FOLDER / asset / 'generation.json').exists():
                result = import_asset(asset)
                STATE.write_text(json.dumps({'latest':result, 'imported':sorted(imported | {asset}),
                    'expected':list(ASSETS), 'workshop':str(WORKSHOP)},indent=2)+'\n')
                print('LUNAR IMPORT', result)
                return 5.0
    except Exception:
        (FOLDER / 'blender-import-error.txt').write_text(traceback.format_exc())
        raise
    return 5.0 if pending else None

bpy.app.driver_namespace['lunar_live_import_callback'] = tick
bpy.app.timers.register(tick, first_interval=1.0)
print('Lunar live import queue registered for the specified assets.')

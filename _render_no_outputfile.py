import bpy, os, time, json, glob

project_dir = 'C:/Users/downl/Documents/New project/hermes-agent'
frames_dir = project_dir + '/render_frames'
scene = bpy.context.scene

# === COMPOSITOR WITHOUT OUTPUTFILE ===
if scene.compositing_node_group:
    bpy.data.node_groups.remove(scene.compositing_node_group)

cg = bpy.data.node_groups.new('FinalComp', 'CompositorNodeTree')
scene.compositing_node_group = cg
scene.render.use_compositing = True

# Nodes (NO OutputFile - just chain to scene output)
rl = cg.nodes.new('CompositorNodeRLayers'); rl.location = (0, 100)
glare = cg.nodes.new('CompositorNodeGlare'); glare.location = (250, 200)
glare.inputs['Strength'].default_value = 3.0
glare.inputs['Size'].default_value = 10.0
glare.inputs['Iterations'].default_value = 6
glare.inputs['Highlights Threshold'].default_value = 0.2

rgb = cg.nodes.new('CompositorNodeRGB'); rgb.location = (250, -200)
rgb.outputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

alpha = cg.nodes.new('CompositorNodeAlphaOver'); alpha.location = (500, 100)
alpha.inputs['Fac'].default_value = 0.0

# LINKS
cg.links.new(rl.outputs['Image'], glare.inputs['Image'])
cg.links.new(glare.outputs['Image'], alpha.inputs['Background'])
cg.links.new(rgb.outputs['Color'], alpha.inputs['Foreground'])
# Alpha output goes to scene render result (no terminal node needed in 5.2)

# Keyframe
fac = alpha.inputs['Fac']
fac.default_value = 0.0; fac.keyframe_insert('default_value', frame=1)
fac.keyframe_insert('default_value', frame=130)
fac.default_value = 1.0; fac.keyframe_insert('default_value', frame=180)

# Scene settings (PNG output)
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.frame_start = 1; scene.frame_end = 180
scene.render.fps = 24; scene.render.film_transparent = False
scene.render.use_motion_blur = False
scene.render.filepath = frames_dir + '/frame_'
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

bpy.ops.wm.save_as_mainfile(filepath=project_dir + '/city_destruction.blend')

# === RENDER ===
print('RENDERING 180 frames (no OutputFile node)...')
t0 = time.time()
bpy.ops.render.render(animation=True)
t1 = time.time()

pngs = sorted(glob.glob(frames_dir + '/frame_[0-9][0-9][0-9][0-9].png'))
result = {
    'frames': len(pngs),
    'duration': round(t1 - t0, 1),
}
if pngs:
    sizes = set(os.path.getsize(f) for f in pngs)
    result['unique_sizes'] = len(sizes)
    result['size_range'] = [min(sizes), max(sizes)]
    result['sample'] = [(os.path.basename(f), os.path.getsize(f)) for f in [pngs[0], pngs[len(pngs)//2], pngs[-1]]]

print('DONE:' + json.dumps(result))

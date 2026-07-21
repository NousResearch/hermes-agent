import bpy, os, json

scene = bpy.context.scene

# Remove old compositor and create simplified one WITHOUT OutputFile
if scene.compositing_node_group:
    bpy.data.node_groups.remove(scene.compositing_node_group)

cg = bpy.data.node_groups.new('CityDestructionComp', 'CompositorNodeTree')
scene.compositing_node_group = cg
scene.render.use_compositing = True

# Nodes
rl = cg.nodes.new('CompositorNodeRLayers')
rl.location = (0, 100)

glare = cg.nodes.new('CompositorNodeGlare')
glare.location = (250, 200)
glare.inputs['Strength'].default_value = 3.0
glare.inputs['Size'].default_value = 10.0
glare.inputs['Iterations'].default_value = 6
glare.inputs['Highlights Threshold'].default_value = 0.2

rgb = cg.nodes.new('CompositorNodeRGB')
rgb.location = (250, -200)
rgb.outputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

alpha = cg.nodes.new('CompositorNodeAlphaOver')
alpha.location = (500, 100)
alpha.inputs['Fac'].default_value = 0.0

# NO OutputFile - just connect alpha output to... nothing?
# In Blender 5.2, the last node in chain defines compositor output
# But without Composite node, how does it work?

# Actually, in Blender 5.2 compositor auto-applies the node tree
# to the render result. Let's add a Viewer node as terminal
# instead of OutputFile.
viewer = cg.nodes.new('CompositorNodeViewer')
viewer.location = (750, 100)

# Links
cg.links.new(rl.outputs['Image'], glare.inputs['Image'])
cg.links.new(glare.outputs['Image'], alpha.inputs['Background'])
cg.links.new(rgb.outputs['Color'], alpha.inputs['Foreground'])
cg.links.new(alpha.outputs['Image'], viewer.inputs['Image'])

# Keyframe Fac
fac = alpha.inputs['Fac']
fac.default_value = 0.0
fac.keyframe_insert('default_value', frame=1)
fac.keyframe_insert('default_value', frame=130)
fac.keyframe_insert('default_value', frame=131)  # hold at 0
fac.default_value = 1.0
fac.keyframe_insert('default_value', frame=180)

result = {
    'nodes': [n.name for n in cg.nodes],
    'links_count': len(cg.links),
    'success': True
}

print('COMPOSITOR:' + json.dumps(result, default=str))

# Now render a test frame
scene.render.filepath = 'render_frames/frame_'
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.frame_set(90)

# Quick Eevee optimization
scene.eevee.use_gtao = False
scene.eevee.use_bloom = False  # using compositor glare instead
scene.render.use_motion_blur = False

bpy.ops.render.render(write_still=True, scene=scene.name)

# Check output
out_path = 'render_frames/frame_0090.png'
if os.path.exists(out_path):
    result['test_frame_size'] = os.path.getsize(out_path)
    result['test_frame_path'] = out_path
else:
    # Check what files appeared
    import glob
    pngs = sorted(glob.glob('render_frames/frame_0*.png'))
    result['new_files'] = [f for f in pngs[-3:]]
    if result['new_files']:
        result['test_frame_size'] = os.path.getsize(result['new_files'][-1])

print('TEST:' + json.dumps(result, default=str))

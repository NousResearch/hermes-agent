"""Reversible static helmet paint repair; preserves source and excludes animation edits."""
import bpy,json
from pathlib import Path
scene=bpy.context.scene
body=next(o for o in scene.objects if o.name.startswith('Core wheel partition'))
assert not body.get('static_helmet_paint_study'), 'Study already installed'
source=list(body.data.materials);body.data.materials.clear()
for old in source:
 old.use_fake_user=True;m=old.copy();m.name='Core static paint study | '+old.name;body.data.materials.append(m)
 nodes=m.node_tree.nodes;links=m.node_tree.links
 shader=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
 for node in nodes:
  if node.type=='NORMAL_MAP':node.inputs['Strength'].default_value=.12
 prior=shader.inputs['Base Color'].links[0].from_socket
 geo=nodes.new('ShaderNodeNewGeometry');pos=nodes.new('ShaderNodeSeparateXYZ');normal=nodes.new('ShaderNodeSeparateXYZ')
 links.new(geo.outputs['Position'],pos.inputs[0]);links.new(geo.outputs['Normal'],normal.inputs[0])
 def smooth(socket,a,b):
  n=nodes.new('ShaderNodeMapRange');n.interpolation_type='SMOOTHSTEP';n.clamp=True
  n.inputs['From Min'].default_value=a;n.inputs['From Max'].default_value=b
  links.new(socket,n.inputs['Value']);return n.outputs['Result']
 # Only upper upward-facing helmet paint. Front visor, ears and rear access face retained.
 z=smooth(pos.outputs['Z'],.91,.97);nz=smooth(normal.outputs['Z'],.25,.65)
 mask=nodes.new('ShaderNodeMath');mask.operation='MULTIPLY';links.new(z,mask.inputs[0]);links.new(nz,mask.inputs[1])
 x=nodes.new('ShaderNodeMath');x.operation='ABSOLUTE';links.new(pos.outputs['X'],x.inputs[0])
 stripe=smooth(x.outputs[0],.14,.17)
 paint=nodes.new('ShaderNodeMixRGB');paint.blend_type='MIX';links.new(stripe,paint.inputs[0]);paint.inputs[1].default_value=(.58,.56,.52,1);paint.inputs[2].default_value=(.25,.18,.38,1)
 mix=nodes.new('ShaderNodeMixRGB');mix.blend_type='MIX';links.new(mask.outputs[0],mix.inputs[0]);links.new(prior,mix.inputs[1]);links.new(paint.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],shader.inputs['Base Color'])
body['static_helmet_paint_study']=True
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs'
(folder/'static-surface-study.json').write_text(json.dumps({'stage':'Static paint study, not final geometry or export','sourceMaterials':[m.name for m in source],'normalStrength':.12,'repair':'Suppress top-facing duplicate visor projection with cream armor and purple side trim','pending':['helmet gear emblem and panel details','shell geometry cleanup','arm and tool topology','all-view inspection and portable texture bake'],'animation':'deferred by user'},indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(folder.parents[1]/'multiview-2026-09-07/leader-rig-workshop.blend'))

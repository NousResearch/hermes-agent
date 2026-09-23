"""Editable owl archivist study from the preserved four-view card.

Run in Blender's visible Python console. Adds a scene; preserves existing scenes.
This is a modeling study, not an approved production asset.
"""
from pathlib import Path
import math
import random
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/lunar-city/production-studies/owl-archivist'
CARD = ROOT / 'public/lunar-city/source-cards/downloads-2026-09-05/Codex Image Sep 4, 2026, 12_37_24 AM.png'

def material(name, color, metallic=0, roughness=.5):
    mat = bpy.data.materials.new('Owl - ' + name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    return mat

def finish(obj, name, mat):
    obj.name = name
    obj.data.materials.append(mat)
    if obj.type == 'MESH':
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    return obj

def oval(name, loc, scale, mat, segments=24, rings=16):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=loc)
    obj = finish(bpy.context.object, name, mat)
    obj.scale = scale
    return obj

def box(name, loc, scale, mat, bevel=.04):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = finish(bpy.context.object, name, mat)
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mod = obj.modifiers.new('Soft crafted edges', 'BEVEL')
    mod.width = bevel
    mod.segments = 3
    obj.modifiers.new('Weighted corner normals', 'WEIGHTED_NORMAL')
    return obj

def cord(name, points, radius, mat):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 12
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    spline = curve.splines.new('BEZIER')
    spline.bezier_points.add(len(points)-1)
    for point, xyz in zip(spline.bezier_points, points):
        point.co = xyz
        point.handle_left_type = point.handle_right_type = 'AUTO'
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    return finish(obj, name, mat)

def ring(name, center, rx, rz, mat, width=.014):
    x,y,z = center
    pts = [(x+rx*math.cos(t*math.tau/64), y, z+rz*math.sin(t*math.tau/64)) for t in range(65)]
    return cord(name, pts, width, mat)

def panel(name, points, mat, thickness=.016):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], [tuple(range(len(points)))])
    mesh.update()
    obj=bpy.data.objects.new(name,mesh)
    bpy.context.scene.collection.objects.link(obj)
    finish(obj,name,mat)
    obj.modifiers.new('Tailored thickness','SOLIDIFY').thickness=thickness
    bevel=obj.modifiers.new('Rounded seams','BEVEL'); bevel.width=.012;bevel.segments=3
    return obj

def text(name, words, loc, size, mat):
    bpy.ops.object.text_add(location=loc, rotation=(math.pi/2,0,0))
    obj = finish(bpy.context.object,name,mat)
    obj.data.body=words;obj.data.align_x='CENTER';obj.data.size=size;obj.data.extrude=.001
    return obj

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    scene=bpy.data.scenes.new('Owl Archivist - card study')
    bpy.context.window.scene=scene
    brown=material('umber feathers',(.19,.095,.035))
    feather=material('warm feather edges',(.29,.16,.063))
    cream=material('ivory facial plumage',(.72,.65,.48))
    teal=material('teal wool',(.035,.105,.105),roughness=.84)
    leather=material('aged leather',(.115,.05,.019),roughness=.52)
    gold=material('antique brass',(.52,.31,.095),metallic=.78,roughness=.3)
    black=material('charcoal beak and talons',(.019,.023,.02),roughness=.34)
    eye=material('obsidian eyes',(.004,.008,.007),roughness=.09)
    iris=material('amber iris',(.36,.18,.038),roughness=.23)
    paper=material('catalog paper',(.77,.70,.51),roughness=.9)
    white=material('eye catchlight',(.95,.92,.8),roughness=.1)
    # Body and the coat have independent geometry for subsequent retopology.
    oval('Body plumage',(0,0,1.0),(.64,.42,.8),brown)
    rows=[(.4,.61,.40),(.46,.68,.44),(.9,.65,.42),(1.3,.60,.39),(1.55,.49,.33),(1.62,.38,.29)]
    verts=[];faces=[];steps=64
    for z,rx,ry in rows:
        verts.extend((rx*math.cos(i*math.tau/steps),ry*math.sin(i*math.tau/steps),z) for i in range(steps))
    for j in range(len(rows)-1):
        for i in range(steps):
            a=j*steps+i;b=j*steps+(i+1)%steps;faces.append((a,b,b+steps,a+steps))
    mesh=bpy.data.meshes.new('Coat quad shell');mesh.from_pydata(verts,[],faces);mesh.update()
    coat=bpy.data.objects.new('Tailored coat shell',mesh);scene.collection.objects.link(coat);finish(coat,coat.name,teal)
    coat.modifiers.new('Cloth shell thickness','SOLIDIFY').thickness=.025
    coat.modifiers.new('Tailored smooth surface','SUBSURF').levels=2
    for side in (-1,1):
        sleeve=oval('Coat sleeve', (side*.65,0,1.05),(.235,.28,.49),teal)
        sleeve.rotation_euler.y=side*.18
        oval('Leather elbow patch',(side*.83,.09,.94),(.032,.17,.18),leather)
        oval('Feathered hand',(side*.74,-.065,.62),(.18,.18,.23),brown)
        for i in range(4):
            digit=oval('Layered hand feather',(side*(.64+i*.052),-.218,.62-i*.02),(.045,.053,.14),feather,16,10)
            digit.rotation_euler.y=-side*.24
        oval('Taloned foot',(side*.32,-.08,.21),(.23,.3,.13),brown)
        for i in range(3):
            x=side*.32+(i-1)*.12
            oval('Toe',(x,-.29,.16),(.065,.18,.065),feather,16,10)
            claw=oval('Curved claw',(x,-.44,.145),(.042,.09,.045),black,16,10)
            claw.rotation_euler.x=-.25
        panel('Ivory collar',[(side*.08,-.33,1.57),(side*.38,-.30,1.62),(side*.28,-.43,1.36)],cream)
        points=[(side*.39,-.33,1.56),(side*.52,-.34,1.37),(side*.36,-.43,1.30),(side*.42,-.43,1.20),(0,-.46,.99)]
        panel('Broad tailored lapel',points,teal)
        cord('Lapel brass piping',points,.009,gold)
        cord('Coat cuff seam',[(side*.85,-.20,.79),(side*.72,-.28,.77),(side*.58,-.21,.79)],.012,gold)
    cord('Coat hem piping',[(.65*math.cos(t*math.tau/64),.435*math.sin(t*math.tau/64),.43) for t in range(65)],.012,gold)
    cord('Coat center seam',[(0,-.45,.44),(0,-.445,.75),(0,-.43,1.03)],.009,gold)
    for z in (.57,.77,.97):
        oval('Brass coat button',(.07,-.462,z),(.038,.018,.038),gold,16,10)
    panel('Archivist tie',[(-.075,-.39,1.46),(.075,-.39,1.46),(.09,-.43,1.22),(0,-.46,1.1),(-.09,-.43,1.22)],leather)
    # Head. Front is negative Y; facial disks and glasses remain separate.
    oval('Owl head',(0,0,1.96),(.60,.43,.56),brown,48,32)
    for side in (-1,1):
        for row in range(3):
            for i in range(22):
                angle=i*math.tau/22;rx=.21+row*.042;rz=.24+row*.038
                x=side*.255+rx*math.cos(angle);z=2.01+rz*math.sin(angle)
                f=oval('Facial disk feather',(x,-.393-row*.004,z),(.043,.047,.086),cream,12,8)
                f.rotation_euler.y=math.pi/2-angle
        oval('Eye socket',(side*.25,-.443,2.025),(.18,.07,.207),leather)
        oval('Amber iris',(side*.25,-.503,2.025),(.139,.038,.161),iris)
        oval('Glossy eye',(side*.25,-.537,2.035),(.108,.036,.134),eye)
        oval('Eye reflection',(side*.216,-.574,2.09),(.027,.008,.033),white,16,10)
        ring('Brass spectacle rim',(side*.25,-.59,2.025),.193,.203,gold)
        cord('Spectacle temple',[(side*.443,-.585,2.05),(side*.56,-.30,2.08),(side*.55,-.04,2.04)],.012,gold)
        for j in range(4):
            tuft=oval('Ear tuft',(side*(.43+j*.027),.02,2.36+j*.039),(.058,.07,.20-j*.02),brown,16,10)
            tuft.rotation_euler.y=side*(.38+j*.12)
    cord('Spectacle bridge',[(-.059,-.59,2.04),(0,-.615,2.073),(.059,-.59,2.04)],.014,gold)
    beak=oval('Hooked beak',(0,-.51,1.86),(.094,.113,.151),black)
    # Layered crown and rear plumage, with deterministic slight variation.
    rng=random.Random(41)
    for row in range(9):
        phi=.18+row*.26
        for col in range(24):
            theta=(col+(row%2)*.5)*math.tau/24
            x=.606*math.sin(phi)*math.cos(theta);y=.438*math.sin(phi)*math.sin(theta);z=1.96+.567*math.cos(phi)
            if y<-.17 and z<2.32: continue
            f=oval('Crown feather',(x,y,z),(.046,.021,.088),feather if rng.random()>.35 else brown,12,8)
            normal=Vector((x/.606,y/.438,(z-1.96)/.567));f.rotation_euler=normal.to_track_quat('Y','Z').to_euler()
    # Book and satchel are independent attachments.
    box('Catalog pages',(.47,-.51,.99),(.34,.095,.45),paper,.015)
    for y in (-.573,-.449): box('Catalog leather cover',(.47,y,.99),(.38,.025,.49),leather,.015)
    box('Catalog spine',(.285,-.51,.99),(.035,.145,.49),leather,.014)
    cord('Catalog gold border',[(.315,-.591,.77),(.63,-.591,.77),(.63,-.591,1.21),(.315,-.591,1.21),(.315,-.591,.77)],.006,gold)
    text('Catalog title','CATALOG',(.475,-.598,1.11),.042,gold)
    box('Archive satchel',(.79,.10,.53),(.34,.27,.40),leather,.065)
    box('Satchel flap',(.79,-.05,.62),(.34,.045,.20),leather,.035)
    ring('Satchel buckle',(.79,-.085,.57),.052,.045,gold,.009)
    cord('Leather shoulder strap',[(.79,.07,.58),(.58,.14,1.52),(.37,-.02,1.62),(.58,-.34,1.34),(.79,-.08,.62)],.035,leather)
    for i in range(3): box('Archive index cards',(.73+i*.05,.08,.77),(.14,.018,.17+i*.035),paper,.006)
    cord('Pencil', [(-.75,-.26,.68),(-.80,-.26,1.01)],.019,leather)
    oval('Pencil ferrule',(-.797,-.26,.985),(.021,.022,.029),gold,16,10)
    # The reference is an image empty: never exported as character geometry.
    bpy.ops.object.empty_add(type='IMAGE',location=(-2.2,.7,1.4),rotation=(math.pi/2,0,0))
    reference=bpy.context.object;reference.name='APPROVED CARD - comparison only';reference.data=bpy.data.images.load(str(CARD));reference.empty_display_size=2.7
    reference['reference_only']=True
    scene.world=bpy.data.worlds.new('Owl studio world');scene.world.color=(.15,.15,.15)
    bpy.ops.object.camera_add(location=(4,-8,3.4));camera=bpy.context.object
    camera.rotation_euler=(Vector((0,0,1.3))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=3.3;scene.camera=camera
    scene.render.engine='BLENDER_WORKBENCH';scene.display.shading.light='STUDIO';scene.display.shading.color_type='MATERIAL';scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True
    scene.render.resolution_x=1200;scene.render.resolution_y=1200;scene.render.resolution_percentage=100
    scene['production_status']='modeling_study_unapproved_not_rigged'
    scene['source_card']=str(CARD)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'owl-archivist-study.blend'))
    print('Owl study saved:',OUT)
    return scene

if __name__=='__main__':
    build()

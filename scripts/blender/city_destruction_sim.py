import bpy
import bmesh
import math
import random
import os

# ============================================================
# シーン初期化
# ============================================================
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 180
scene.render.fps = 24
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'
scene.use_nodes = True

# 出力ディレクトリ
output_dir = os.path.join(os.path.dirname(bpy.data.filepath), "render_frames")
os.makedirs(output_dir, exist_ok=True)
scene.render.filepath = os.path.join(output_dir, "frame_")

# Eevee設定 (BloomはCompositorのGlareノードで実装)
scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.taa_render_samples = 64
# scene.eevee.use_bloom = False  # Blender 5.2では存在しないためコメントアウト

# ============================================================
# Compositor設定 (Blender 5.2 API準拠)
# ============================================================
cg = bpy.data.node_groups.new("CityDestructionComp", "CompositorNodeTree")
scene.compositing_node_group = cg

# ノードクリア
for node in cg.nodes:
    cg.nodes.remove(node)

# Render Layers
rl = cg.nodes.new("CompositorNodeRLayers")
rl.location = (-600, 0)

# Glare (Bloom用)
glare = cg.nodes.new("CompositorNodeGlare")
glare.location = (-200, 0)
glare.inputs["Type"].default_value = "Bloom"
glare.inputs["Quality"].default_value = "High"
glare.inputs["Strength"].default_value = 1.5
glare.inputs["Threshold"].default_value = 0.8
glare.inputs["Size"].default_value = 0.6

# RGB node for white color (constant)
rgb_node = cg.nodes.new("CompositorNodeRGB")
rgb_node.location = (-200, -300)
rgb_node.outputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # White RGBA

# AlphaOver node for fade to white
alpha_over = cg.nodes.new("CompositorNodeAlphaOver")
alpha_over.location = (200, 0)
# Factor will be animated

# リンク
# Glare takes render layer
cg.links.new(rl.outputs["Image"], glare.inputs["Image"])
# AlphaOver: Background = glare output, Foreground = white RGB
cg.links.new(glare.outputs["Image"], alpha_over.inputs["Background"])
cg.links.new(rgb_node.outputs["Color"], alpha_over.inputs["Foreground"])

# ============================================================
# マテリアル定義
# ============================================================
def create_emission_material(name, color, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    emit = nodes.new('ShaderNodeEmission')
    emit.inputs['Color'].default_value = (*color, 1.0)
    emit.inputs['Strength'].default_value = strength
    out.location = (300, 0)
    links.new(emit.outputs['Emission'], out.inputs['Surface'])
    return mat

def create_principled_material(name, base_color, roughness=0.7, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*base_color, 1.0)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    return mat

# マテリアル作成
mat_ground = create_principled_material("Ground", (0.15, 0.12, 0.1), 0.9)
mat_building = create_principled_material("Building", (0.25, 0.22, 0.2), 0.8)
mat_roof = create_principled_material("Roof", (0.15, 0.1, 0.08), 0.9)
mat_debris = create_principled_material("Debris", (0.2, 0.18, 0.15), 0.85)
mat_fireball = create_emission_material("Fireball", (1.0, 0.4, 0.05), 50.0)
mat_shockwave = create_emission_material("Shockwave", (1.0, 0.6, 0.2), 20.0)
mat_mushroom = create_principled_material("Mushroom", (0.15, 0.12, 0.1), 0.7)
mat_mushroom_emit = create_emission_material("MushroomEmit", (0.3, 0.2, 0.1), 5.0)

# ============================================================
# 光源
# ============================================================
bpy.ops.object.light_add(type='SUN', location=(100, -100, 300))
sun = bpy.context.active_object
sun.name = "Sun"
sun.data.energy = 5.0
sun.data.angle = 0.01
# 少し斜めから照らして影をつける
sun.rotation_euler = (0.8, 0.2, -0.5)

# 補助環境光＆空の色
world = bpy.context.scene.world
if world:
    world.use_nodes = True
    # スカイカラーを設定
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Strength'].default_value = 0.5
        bg_node.inputs['Color'].default_value = (0.1, 0.1, 0.15, 1.0)  # やや明るい空

# ============================================================
# 地面
# ============================================================
bpy.ops.mesh.primitive_plane_add(size=2000, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
ground.data.materials.append(mat_ground)

# ============================================================
# 木造家屋グリッド (5x5 = 25棟)
# ============================================================
buildings = []
grid_size = 5
spacing = 40
half_extent = (grid_size - 1) * spacing / 2

for gx in range(grid_size):
    for gy in range(grid_size):
        x = gx * spacing - half_extent
        y = gy * spacing - half_extent
        
        # 建物サイズにバリエーション
        w = random.uniform(8, 14)
        d = random.uniform(8, 14)
        h = random.uniform(6, 12)
        
        # 箱本体
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, h/2))
        box = bpy.context.active_object
        box.name = f"House_{gx}_{gy}_Box"
        box.scale = (w/2, d/2, h/2)
        box.data.materials.append(mat_building)
        
        # 三角屋根
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=max(w,d)*0.75, radius2=0, depth=h*0.4)
        roof_mesh = bpy.data.meshes.new(f"House_{gx}_{gy}_Roof")
        bm.to_mesh(roof_mesh)
        bm.free()
        roof = bpy.data.objects.new(f"House_{gx}_{gy}_Roof", roof_mesh)
        roof.location = (x, y, h + h*0.2)
        roof.data.materials.append(mat_roof)
        bpy.context.collection.objects.link(roof)
        
        buildings.append({
            'box': box, 'roof': roof,
            'base_pos': (x, y, 0),
            'center': (x, y, h/2),
            'height': h,
            'size': (w, d),
            'destroyed': False
        })

# ============================================================
# 爆弾 (小さな球体)
# ============================================================
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.8, location=(0, 0, 200))
bomb = bpy.context.active_object
bomb.name = "Bomb"
mat_bomb = create_principled_material("Bomb", (0.05, 0.05, 0.05), 0.1)
bomb.data.materials.append(mat_bomb)

# ============================================================
# 火球 (爆発火球) - 半透明で発光
# ============================================================
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, -100))
fireball = bpy.context.active_object
fireball.name = "Fireball"
# 半透明発光マテリアルを作成
mat_fireball_glow = bpy.data.materials.new("Fireball_Glow")
mat_fireball_glow.use_nodes = True
nodes_fb = mat_fireball_glow.node_tree.nodes
links_fb = mat_fireball_glow.node_tree.links
nodes_fb.clear()
out_fb = nodes_fb.new('ShaderNodeOutputMaterial')
mix_fb = nodes_fb.new('ShaderNodeMixShader')
trans_fb = nodes_fb.new('ShaderNodeBsdfTransparent')
emit_fb = nodes_fb.new('ShaderNodeEmission')
emit_fb.inputs['Color'].default_value = (1.0, 0.6, 0.1, 1.0)
emit_fb.inputs['Strength'].default_value = 50.0
mix_fb.inputs['Fac'].default_value = 0.4  # 40% emission, 60% transparent
links_fb.new(trans_fb.outputs['BSDF'], mix_fb.inputs[1])
links_fb.new(emit_fb.outputs['Emission'], mix_fb.inputs[2])
links_fb.new(mix_fb.outputs['Shader'], out_fb.inputs['Surface'])
fireball.data.materials.append(mat_fireball_glow)
fireball.scale = (0.01, 0.01, 0.01)

# 火球用の強度アニメーション用変数（後でEmission Strengthをアニメート）
fireball_mix_factor = mix_fb

# ============================================================
# 衝撃波リング
# ============================================================
bpy.ops.mesh.primitive_circle_add(radius=1, fill_type='NGON', location=(0, 0, 2))
shockwave = bpy.context.active_object
shockwave.name = "ShockwaveRing"
shockwave.data.materials.append(mat_shockwave)
shockwave.scale = (0.01, 0.01, 1)

# ============================================================
# キノコ雲 (複数の球体 + パーティクル)
# ============================================================
mushroom_parts = []
for i in range(8):
    r = random.uniform(8, 15)
    z = 50 + i * 15
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=16, ring_count=8, location=(0, 0, z))
    part = bpy.context.active_object
    part.name = f"Mushroom_{i}"
    part.data.materials.append(mat_mushroom)
    part.scale = (0.01, 0.01, 0.01)
    mushroom_parts.append(part)

# パーティクルシステム (煙)
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 30))
smoke_emitter = bpy.context.active_object
smoke_emitter.name = "SmokeEmitter"
ps = smoke_emitter.modifiers.new("SmokeParticles", 'PARTICLE_SYSTEM')
psys = smoke_emitter.particle_systems[0]
psys.settings.count = 2000
psys.settings.frame_start = 50
psys.settings.frame_end = 130
psys.settings.lifetime = 80
psys.settings.emit_from = 'VOLUME'
psys.settings.particle_size = 0.5
psys.settings.size_random = 0.5
psys.settings.use_emit_random = True
psys.settings.physics_type = 'NEWTON'
psys.settings.effector_weights.gravity = 0.05
psys.settings.effector_weights.all = 0.1
psys.settings.brownian_factor = 2.0
psys.settings.drag_factor = 0.1

# パーティクルマテリアル
mat_smoke = create_principled_material("Smoke", (0.1, 0.08, 0.06), 0.9)
smoke_emitter.data.materials.append(mat_smoke)

# ============================================================
# カメラアニメーション
# ============================================================
cam = bpy.data.objects.get("Camera")
if not cam:
    bpy.ops.object.camera_add(location=(0, -150, 80), rotation=(1.1, 0, 0))
    cam = bpy.context.active_object
cam.name = "MainCam"
scene.camera = cam

# フレームごとのキーフレーム設定
def set_keyframe(obj, prop, frame, value):
    setattr(obj, prop, value)
    obj.keyframe_insert(data_path=prop, frame=frame)

# Phase 1 (0-30): 爆弾落下追跡 → 引き
set_keyframe(cam, "location", 1, (0, -100, 60))
set_keyframe(cam, "rotation_euler", 1, (1.1, 0, 0))
set_keyframe(cam, "location", 15, (0, -80, 80))
set_keyframe(cam, "rotation_euler", 15, (1.0, 0, 0))
set_keyframe(cam, "location", 30, (0, -120, 120))
set_keyframe(cam, "rotation_euler", 30, (0.85, 0, 0))

# Phase 2 (30-50): 爆発火球
set_keyframe(cam, "location", 40, (0, -140, 140))
set_keyframe(cam, "rotation_euler", 40, (0.8, 0, 0))
set_keyframe(cam, "location", 50, (0, -160, 160))
set_keyframe(cam, "rotation_euler", 50, (0.75, 0, 0))

# Phase 3 (50-90): 衝撃波・家屋崩壊
set_keyframe(cam, "location", 70, (0, -180, 180))
set_keyframe(cam, "rotation_euler", 70, (0.7, 0, 0))

# Phase 4 (90-130): キノコ雲成長
set_keyframe(cam, "location", 110, (0, -200, 220))
set_keyframe(cam, "rotation_euler", 110, (0.65, 0, 0))
set_keyframe(cam, "location", 130, (0, -220, 240))
set_keyframe(cam, "rotation_euler", 130, (0.6, 0, 0))

# Phase 5 (130-180): 白蒸発・ホワイトアウト
set_keyframe(cam, "location", 150, (0, -240, 250))
set_keyframe(cam, "rotation_euler", 150, (0.55, 0, 0))
set_keyframe(cam, "location", 180, (0, -260, 260))
set_keyframe(cam, "rotation_euler", 180, (0.5, 0, 0))

# カメラ補間をベジェに（アニメーションデータが作成された後で実行）
# We'll do this after all keyframes are set

# ============================================================
# 爆弾落下アニメーション
# ============================================================
set_keyframe(bomb, "location", 1, (0, 0, 200))
set_keyframe(bomb, "location", 30, (0, 0, 2))
set_keyframe(bomb, "location", 31, (0, 0, -10))  # 地中へ

# ============================================================
# 火球アニメーション
# ============================================================
# スケール & 発光強度
for frame, scale, strength in [
    (30, 1, 0),
    (35, 20, 60),
    (40, 50, 80),
    (45, 80, 60),
    (50, 100, 30),
    (60, 120, 10),
    (70, 140, 1),
]:
    fireball.scale = (scale, scale, scale)
    fireball.keyframe_insert("scale", frame=frame)
    emit_fb.inputs['Strength'].default_value = strength
    emit_fb.inputs['Strength'].keyframe_insert("default_value", frame=frame)
    # 透明度もアニメート: 最大でも0.3まで（70%透明）
    t = min(strength / 200.0, 0.3)
    mix_fb.inputs['Fac'].default_value = t
    mix_fb.inputs['Fac'].keyframe_insert("default_value", frame=frame)

# ============================================================
# 衝撃波アニメーション
# ============================================================
shockwave_frames = [
    (30, 1, 25),
    (40, 100, 30),
    (50, 250, 25),
    (60, 400, 15),
    (70, 550, 5),
    (80, 700, 1),
    (90, 850, 0),
]
for frame, radius, strength in shockwave_frames:
    shockwave.scale = (radius, radius, 1)
    shockwave.keyframe_insert("scale", frame=frame)
    mat_shockwave.node_tree.nodes['Emission'].inputs['Strength'].default_value = strength
    mat_shockwave.node_tree.nodes['Emission'].inputs['Strength'].keyframe_insert("default_value", frame=frame)

# ============================================================
# 家屋崩壊 (衝撃波到達で破壊)
# ============================================================
shockwave_speed = 10.0  # m/frame
for i, bld in enumerate(buildings):
    dist = math.sqrt(bld['center'][0]**2 + bld['center'][1]**2)
    arrival_frame = 30 + int(dist / shockwave_speed)
    collapse_frame = arrival_frame + random.randint(2, 8)
    end_frame = collapse_frame + random.randint(10, 20)
    
    box, roof = bld['box'], bld['roof']
    orig_loc = box.location
    orig_rot = box.rotation_euler
    
    # 到着前は元の位置で静止
    box.location = orig_loc
    box.keyframe_insert("location", frame=arrival_frame)
    box.rotation_euler = orig_rot
    box.keyframe_insert("rotation_euler", frame=arrival_frame)
    
    # 崩壊: 位置オフセット + 回転 + スケール縮小
    offset = (
        random.uniform(-bld['size'][0]*0.5, bld['size'][0]*0.5),
        random.uniform(-bld['size'][1]*0.5, bld['size'][1]*0.5),
        -bld['height']*0.3
    )
    rot = (
        random.uniform(-0.8, 0.8),
        random.uniform(-0.8, 0.8),
        random.uniform(-0.3, 0.3)
    )
    
    # 崩壊開始フレーム
    box.location = (orig_loc[0] + offset[0], orig_loc[1] + offset[1], orig_loc[2] + offset[2])
    box.keyframe_insert("location", frame=collapse_frame)
    box.rotation_euler = rot
    box.keyframe_insert("rotation_euler", frame=collapse_frame)
    
    # 完全崩壊フレーム: 小さくなり平らに
    box.scale = (bld['size'][0]*0.2, bld['size'][1]*0.2, bld['height']*0.1)
    box.keyframe_insert("scale", frame=end_frame)
    
    # 屋根も同様に処理
    roof_loc = roof.location
    roof.location = roof_loc
    roof.keyframe_insert("location", frame=arrival_frame)
    roof.rotation_euler = roof.rotation_euler
    roof.keyframe_insert("rotation_euler", frame=arrival_frame)
    
    # 屋根の崩壊
    roof.location = (roof_loc[0] + offset[0]*1.2, roof_loc[1] + offset[1]*1.2, roof_loc[2] + offset[2] - 2.0)
    roof.keyframe_insert("location", frame=collapse_frame)
    roof.rotation_euler = (rot[0]*1.5, rot[1]*1.5, rot[2]*0.5)
    roof.keyframe_insert("rotation_euler", frame=collapse_frame)
    roof.scale = (0.3, 0.3, 0.05)
    roof.keyframe_insert("scale", frame=end_frame)

# ============================================================
# キノコ雲成長
# ============================================================
for i, part in enumerate(mushroom_parts):
    base_z = 50 + i * 15
    for frame, scale_factor in [
        (50, 0.01), (60, 0.2), (70, 0.5), (80, 0.8),
        (90, 1.0), (100, 1.2), (110, 1.4), (120, 1.5),
        (130, 1.5), (180, 1.5)
    ]:
        s = scale_factor
        part.scale = (s, s, s)
        part.keyframe_insert("scale", frame=frame)
        # 上昇アニメーション (70フレーム以降)
        if frame >= 70:
            part.location.z = base_z + (frame - 70) * 0.5
            part.keyframe_insert("location", frame=frame)

# 煙パーティクル発生キーフレーム
psys.settings.frame_start = 50
psys.settings.frame_end = 130
# psys.settings.keyframe_insert("frame_start", frame=50)
# psys.settings.keyframe_insert("frame_end", frame=50)

# カメラ補間をベジェに（アニメーションデータ作成後に実行）
# Blender 5.2 uses action slots for fcurves
if cam.animation_data and cam.animation_data.action:
    try:
        action = cam.animation_data.action
        for slot in action.slots:
            for fcu in slot.fcurves:
                for kf in fcu.keyframe_points:
                    kf.interpolation = 'BEZIER'
                    kf.handle_left_type = 'AUTO_CLAMPED'
                    kf.handle_right_type = 'AUTO_CLAMPED'
    except Exception as e:
        print(f"Camera interpolation setup failed: {e}")
        pass

# ============================================================
# ホワイトアウト (Compositor AlphaOver Factor アニメート)
# ============================================================
factor_input = alpha_over.inputs["Fac"]
factor_input.default_value = 0.0
factor_input.keyframe_insert("default_value", frame=120)
factor_input.default_value = 0.0
factor_input.keyframe_insert("default_value", frame=130)
factor_input.default_value = 0.3
factor_input.keyframe_insert("default_value", frame=140)
factor_input.default_value = 0.6
factor_input.keyframe_insert("default_value", frame=150)
factor_input.default_value = 0.85
factor_input.keyframe_insert("default_value", frame=160)
factor_input.default_value = 1.0
factor_input.keyframe_insert("default_value", frame=180)

# ============================================================
# グレア強度も Phase 2-4 で強く
# ============================================================
glare_strength = glare.inputs["Strength"]
glare_strength.default_value = 0.5
glare_strength.keyframe_insert("default_value", frame=1)
glare_strength.default_value = 2.0
glare_strength.keyframe_insert("default_value", frame=35)
glare_strength.default_value = 3.0
glare_strength.keyframe_insert("default_value", frame=45)
glare_strength.default_value = 1.5
glare_strength.keyframe_insert("default_value", frame=60)
glare_strength.default_value = 0.5
glare_strength.keyframe_insert("default_value", frame=90)
glare_strength.default_value = 0.0
glare_strength.keyframe_insert("default_value", frame=130)

# ============================================================
# レンダリング実行
# ============================================================
print("Starting render...")
print(f"Output directory: {output_dir}")
bpy.ops.render.render(animation=True, write_still=False)
print("Render complete!")
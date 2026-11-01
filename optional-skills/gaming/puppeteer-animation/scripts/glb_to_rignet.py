#!/usr/bin/env python3
"""
glb_to_rignet.py — Convert a rigged GLB (e.g. SkinTokens output) into the
RigNet-style rig.txt + mesh.obj layout that Seed3D/Puppeteer's animation
module expects:

    out_dir/
      objs/
        mesh.obj      (+ material.mtl / texture.png when the GLB is textured)
        rig.txt       # joints / root / hier / skin lines

Runs inside Blender's Python (bpy) — no system deps beyond Blender:

    blender -b --factory-startup -noaudio --python glb_to_rignet.py -- \
        input_rigged.glb -o out_dir/

Vertex/global transforms are baked to world space so joint positions and
mesh vertices share one coordinate frame (Puppeteer requirement).
"""
import argparse
import os
import re
import sys

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("glb")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--decimate", type=int, default=0,
                   help="Target face count per mesh (0 = keep). Decimate "
                   "BEFORE rig.txt export so skin weights match — Puppeteer "
                   "OOMs on 200k+ vert meshes and raw GLB skin indices won't "
                   "match an externally decimated OBJ.")
    return p.parse_args(argv)


def clean(name):
    return re.sub(r"\W+", "_", name)


def main():
    args = parse_args()
    objs_dir = os.path.join(args.out, "objs")
    os.makedirs(objs_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.glb)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not meshes or not arms:
        raise SystemExit(f"ERROR: need >=1 mesh and >=1 armature, got "
                         f"{len(meshes)} meshes, {len(arms)} armatures")
    arm = arms[0]

    deps = bpy.context.evaluated_depsgraph_get()

    if args.decimate:
        for mesh in meshes:
            faces = len(mesh.data.polygons)
            if faces > args.decimate:
                mod = mesh.modifiers.new("decim", "DECIMATE")
                mod.ratio = args.decimate / faces
                bpy.context.view_layer.objects.active = mesh
                bpy.ops.object.modifier_apply(modifier=mod.name)
                print(f"decimated {mesh.name}: {faces} -> "
                      f"{len(mesh.data.polygons)} faces")
        deps = bpy.context.evaluated_depsgraph_get()

    # --- rig.txt ---
    joints = {}
    root = None
    for bone in arm.data.bones:
        world_head = arm.matrix_world @ bone.head_local
        joints[bone.name] = {
            "pos": world_head,
            "parent": bone.parent.name if bone.parent else None,
        }
        if bone.parent is None:
            root = bone.name

    rig_path = os.path.join(objs_dir, "rig.txt")
    with open(rig_path, "w") as f:
        f.writelines(
            f"joints {clean(name)} {j['pos'][0]:.8f} {j['pos'][1]:.8f} {j['pos'][2]:.8f}\n"
            for name, j in joints.items()
        )
        f.write(f"root {clean(root)}\n")

        vert_offset = 0
        for mesh in meshes:
            vg_names = {g.index: clean(g.name) for g in mesh.vertex_groups}
            eval_mesh = mesh.evaluated_get(deps).to_mesh()
            for vtx in eval_mesh.vertices:
                weights = [(vg_names[g.group], g.weight) for g in vtx.groups
                           if g.group in vg_names and g.weight > 1e-4]
                if not weights:
                    continue
                total = sum(w for _, w in weights)
                pairs = " ".join(f"{b} {w / total:.4f}" for b, w in weights)
                f.write(f"skin {vert_offset + vtx.index} {pairs}\n")
            vert_offset += len(eval_mesh.vertices)

        for name, j in joints.items():
            if j["parent"]:
                f.write(f"hier {clean(j['parent'])} {clean(name)}\n")

    # --- mesh.obj (world-space, merged, with materials/textures) ---
    for o in bpy.context.scene.objects:
        o.select_set(o.type == "MESH")
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    obj_path = os.path.join(objs_dir, "mesh.obj")
    bpy.ops.wm.obj_export(filepath=obj_path, export_selected_objects=True,
                          export_materials=True, export_triangulated_mesh=False,
                          path_mode="COPY")
    # OBJ export already wrote material.mtl + copied textures next to mesh.obj.

    print(f"OK wrote {rig_path} and {obj_path} "
          f"({len(joints)} joints, {vert_offset} verts)")


if __name__ == "__main__":
    main()

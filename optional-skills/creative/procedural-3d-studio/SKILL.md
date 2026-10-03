---
name: procedural-3d-studio
description: "Synthesize 3D meshes, GLB/OBJ models, and game assets."
version: 1.0.0
author: 0xAlyDev
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [3d, modeling, gamedev, gltf, mesh, blender]
    category: creative
    homepage: https://github.com/0xalydev/procedural-3d-studio
---

# procedural-3d-studio

Generate 3D meshes, low-poly game assets, and environment props with pure Python standard library.
Exports directly to binary glTF 2.0 (`.glb`), Wavefront (`.obj`/`.mtl`), Godot 4 scenes (`.tscn`),
interactive WebGL Three.js HTML previews, and headless Blender automation scripts.

Zero external dependencies required.

## When to Use

- User asks to create a 3D model, mesh, or game asset (trees, rocks, castle towers, weapons, crates, dungeons, terrain).
- User wants assets exported for Godot 4, Three.js, Unity, or Blender.
- User wants an interactive WebGL preview of a generated 3D model.
- User asks to automate 3D renders using headless Blender.

## Quick CLI Usage

The core engine is located at `scripts/mesh_studio.py`:

```bash
# 1. Procedural stylized low-poly tree
python scripts/mesh_studio.py tree -o models/pine_tree --tiers 4 --height 1.5

# 2. Procedural rock / asteroid with facet roughness
python scripts/mesh_studio.py rock -o models/boulder --radius 1.0 --roughness 0.4

# 3. Medieval castle tower with battlements & roof
python scripts/mesh_studio.py tower -o models/watchtower --height 4.0 --battlements 10

# 4. Low-poly fantasy sword
python scripts/mesh_studio.py sword -o models/hero_sword --length 2.5

# 5. Modular dungeon floor tile with wall flags (north/west walls)
python scripts/mesh_studio.py dungeon -o models/dungeon_corner --walls nw

# 6. Procedural heightmap terrain
python scripts/mesh_studio.py terrain -o models/island_terrain --grid-size 24 --scale 1.8

# 7. Reinforced cargo crate
python scripts/mesh_studio.py crate -o models/sci_fi_crate --size 1.2

# 8. Standard primitives (cube, sphere, cylinder, cone)
python scripts/mesh_studio.py primitive sphere -o models/orb --size 1.0
```

## Generated Outputs per Asset

Each generation automatically outputs:
1. `*.glb`: Self-contained binary glTF 2.0 mesh with vertex colors, normals, and PBR material.
2. `*.obj` & `*.mtl`: Wavefront geometry and material file.
3. `*.tscn`: Ready-to-import Godot 4 scene node tree.
4. `*.html`: Interactive Three.js WebGL orbit-camera preview. Open directly in any browser.
5. `*.blender.py`: Headless Blender automation script for Cycles/Eevee render snapshots.

## Headless Blender Automation

If Blender is installed on the host machine, render high-resolution 1024x1024 image snapshots:

```bash
blender --background --python models/pine_tree.blender.py
```

See `references/blender-automation.md` for batch rendering and physics simulation recipes.
See `references/game-engine-recipes.md` for Godot 4, Three.js, and Unity import guidelines.

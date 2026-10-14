# GLB pipeline (Godot side)

Blender authoring is covered by the optional `blender-mcp` skill:

```bash
hermes skills install official/creative/blender-mcp
hermes mcp install blender
```

This reference is only the handoff into Godot for Quest.

## Export from DCC

- Prefer **GLB** (binary glTF) over glTF+bin folders for simpler imports.
- Apply transforms / freeze scale in the DCC before export when possible.
- Keep real-world scale (1 unit = 1 meter) for comfortable XR.

## Optional compression

```bash
gltfpack -i raw.glb -o optimized.glb -cc -tc
```

Avoid mesh quantization extensions that the Godot importer rejects. If import
fails after gltfpack, retry with milder flags or raw GLB.

## Godot import

1. Copy GLB under `res://` (e.g. `res://models/`).
2. Let Godot generate `.import` — open the Import dock and confirm.
3. Instance as scene or load at runtime.

### Stale import cache

If geometry looks wrong after re-export:

1. Delete `model.glb.import`
2. Delete matching entries under `.godot/imported/`
3. Reopen/reimport the project

Deleting only one of the two can leave stale meshes.

## Mobile renderer constraints

Quest uses the **Mobile** renderer:

- Prefer baked or simple lights; avoid heavy shadow cascades
- Glow supported; SSAO generally not — design materials accordingly
- Watch triangle count and overdraw; profile on-device

## Materials

- Keep PBR simple (base color, roughness, normal)
- Huge 4K atlases waste VRAM; 1K–2K is usually enough for Quest 3
- Emission for accents is fine; don't rely on expensive multi-bounce GI

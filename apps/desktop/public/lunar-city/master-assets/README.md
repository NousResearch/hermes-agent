# Lunar City master assets

Production Lunar City art starts here.

The accepted pipeline is:

1. Create or import a full-resolution/high-poly master asset.
2. Validate that the master clearly matches the approved Lunar City reference style.
3. Retopologize into smart low-poly runtime LODs.
4. Bake PBR textures from the master: 2K default, 4K only for hero leaders and building facades.
5. Rig and animate characters after master validation.

Do not use these as production sources:

- raw scene-crop image-to-3D outputs
- floating blobs
- simple mascot placeholders
- flat billboard/reference planes
- unriggable single-lump meshes
- high-poly meshes with the wrong silhouette, such as cube/default primitive failures

Drop candidate source files into `sources/` using one of the exact ids from
`master-asset-manifest.json`, for example:

- `sources/leader-fox-scientist.blend`
- `sources/worker-review.glb`
- `sources/building-research-lab.fbx`

Supported source formats are `.blend`, `.glb`, `.fbx`, and `.obj`.

Run the manifest builder after adding sources:

```bash
python3 apps/desktop/scripts/build_lunar_city_master_asset_manifest.py
```

The current manifest fails closed until every required high-poly master exists
and passes validation.

Note: a previous local Hunyuan3D research-lab candidate produced a high-poly
cube/default primitive. That clears a triangle-count check but fails visual
silhouette validation, so it is not a production master.

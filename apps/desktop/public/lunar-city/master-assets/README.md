# Lunar City master assets

Production Lunar City art starts here.

The accepted pipeline is:

1. Generate or hand-author masked source cards and silhouette previews from the
   approved reference images.
2. Create or import a full-resolution/high-poly master asset from the masked
   source, not from the raw scene crop.
3. Validate that the master clearly matches the approved Lunar City reference
   silhouette and style.
4. Retopologize into smart low-poly runtime LODs.
5. Bake PBR textures from the master: 2K default, 4K only for hero leaders and building facades.
6. Rig and animate characters after master validation.

Mask artifacts live in `masks/` and are indexed by `masks/mask-manifest.json`:

- `*-mask.png` is the grayscale silhouette mask.
- `*-silhouette.png` is the visual review card for silhouette approval.
- `mask-review-contact-sheet.png` shows every crop/mask/silhouette side by
  side so broad or wrong silhouettes are visible before generation.

Run mask prep before any image-to-3D generation:

```bash
/private/tmp/TripoSR/.venv/bin/python apps/desktop/scripts/generate_lunar_city_asset_masks.py
```

The script also writes transparent masked source cards to
`/private/tmp/lunar-city-master-asset-masked-sources/`. Those cache files are
used as image-to-3D inputs and are intentionally not checked into the desktop
asset tree because they are large binary prep artifacts.

These masks are prep artifacts only. They require human silhouette review before
any generated asset can be promoted to a production master.

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

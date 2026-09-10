# Lunar City generated 3D assets

This folder contains the first local image-to-3D asset pass for Lunar City.
It is now explicitly rejected for production use.

- Source images: the two approved Lunar City design references supplied by the operator.
- Crop manifest: `reference-crops/reference-crops-manifest.json`.
- Mesh generator used for this pass: local TripoSR from `/private/tmp/TripoSR`.
- Stable Fast 3D was installed and evaluated, but actual inference is blocked until the local Hugging Face account has access to the gated `stabilityai/stable-fast-3d` model.
- Privacy boundary: no raw `SOUL.md` content or private profile identifiers are written into these public assets.

The current GLBs are real generated meshes, not procedural block placeholders, but they are rejected as production assets. Scene-level 2D crops produce incomplete floating/blobby relief geometry because the model receives occluded objects mixed with backgrounds, walls, workers, UI, and props.

Production Lunar City assets must instead start from full-resolution/high-poly master assets, then retopologize, bake PBR textures, rig, animate, and produce LODs from those masters. These raw TripoSR scene-crop outputs are retained only as evidence and reference material so future work does not repeat this failed path.

Review artifacts:

- `meshes/*.glb`: one generated mesh per selected building, leader, worker/child, vehicle, and prop.
- `lunar-city-generated-assets-board.blend`: Blender scene containing all generated assets grouped for inspection.
- `lunar-city-generated-assets-board.glb`: exported review board.
- `lunar-city-generated-assets-board.png`: rendered preview of the board.
- `generated-assets-metadata.json`: import status, mesh counts, source crop provenance, and PBR status.

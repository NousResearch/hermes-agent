# Blender Scripts

Fork-specific Blender 5.2 LTS city destruction demo scene and compositor
pipeline. Includes scene setup scripts, render pipeline, and VOICEVOX
narration generation for the educational urban demolition simulation.

## Requirements

- Blender 5.2 LTS (Eevee renderer)
- Python 3.13+ (for narration scripts)
- VOICEVOX Engine (optional, for narration)

## Usage

```bash
# Render the scene
blender -b city_destruction.blend -P scripts/blender/render_no_outputfile.py

# Generate narration (requires VOICEVOX on localhost:50021)
uv run python scripts/blender/make_narration.py
```

## Notes

- Blender compositor API diverges by version — these scripts target Blender
  5.2's `scene.compositing_node_group` (not `scene.node_tree`).
- All paths in Blender scripts must be absolute (CWD is the install dir).
- See `docs/architecture/city_destruction_sim.md` for design documentation.

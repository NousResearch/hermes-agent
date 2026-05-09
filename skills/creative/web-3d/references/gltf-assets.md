# glTF / GLB Assets

If the request involves real 3D assets, treat asset hygiene as part of the implementation, not an afterthought.

## Before loading

Check:

- file format: `.glb` vs `.gltf`
- texture locations
- approximate file size
- whether the asset is already web-ready
- whether the model needs authoring fixes before integration

If the asset is not ready, the right move may be `blender-mcp` first.

## Common problems

- model origin is far from center
- scale is wildly off
- textures fail due to wrong relative paths
- asset is too large for interactive web use
- material setup looks wrong under web lighting
- mobile perf collapses due to geometry/texture weight

## Integration rules

- normalize scale and framing
- establish a default camera that shows the asset immediately
- keep lights simple first
- do not blame the web layer for broken source assets

## Viewer defaults

- sane camera distance
- at least one key light + fill
- neutral environment/background
- optional controls only when appropriate
- loading / fallback state

## Performance posture

Be suspicious of:

- huge textures
- too many draw calls
- baked detail that should have been simplified

If the model is public-facing, optimize before polishing the page around it.

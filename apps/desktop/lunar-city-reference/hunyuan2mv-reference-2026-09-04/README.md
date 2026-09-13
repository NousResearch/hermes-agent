# Lunar City Hunyuan3D-2mv Reference Set

These eight GLBs are reference candidates generated locally from the approved
2x2 building turnarounds using Tencent Hunyuan3D-2mv shape generation.

The front, side, and rear panels were extracted from each sheet. The plan
panel was retained as a modeling reference but was not mislabeled as a fourth
elevation because Hunyuan3D-2mv expects canonical side views.

Generation settings: local `tencent/Hunyuan3D-2mv`, turbo variant, 20
inference steps, octree resolution 256, Metal/MPS on Apple M4 Pro.

This directory is reference-only. The meshes still require visual review,
cleanup/retopology, PBR texture work, collision/scale checks, and LODs before
being promoted into the production Lunar City asset set.

| File | Leader building |
| --- | --- |
| `owl.glb` | Owl Arabic-style woven library |
| `elephant.glb` | Elephant Tree of Life memory graph |
| `cat.glb` | Cat graffiti/mosaic arts studio |
| `fox.glb` | Fox grass-covered planetarium observatory with telescope |
| `capybara.glb` | Capybara lazy-river revenue lab |
| `lion.glb` | Lion Pride Rock / civic headquarters |
| `beaver.glb` | Beaver damworks architecture building |
| `monkey.glb` | Monkey publication/fraternity house |
